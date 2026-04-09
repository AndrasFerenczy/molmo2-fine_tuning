"""
Sliding window attention model.
"""

from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor
import torch.nn as nn

from gumbel_model.model import ModelConfig
from gumbel_model.full_attention_model import (
    MLP,
    RMSNorm,
    Model,
    TritonFullAttention,
    triton_keybias_attention,
)
from gumbel_model.utils.masking import sliding_window_mask_factory_method
from gumbel_model.utils.segmented_ops import (
    global_margin_clamped_excess,
    doc_relative_positions,
    is_doc_start_from_doc_idx,
)
from torch.nn.attention.flex_attention import and_masks


@dataclass
class SlidingAttentionModelConfig(ModelConfig):
    window_size: int = 16
    use_triton_full_attention: bool = False
    warp_specialize: bool = False


class TritonSlidingWindowAttention(TritonFullAttention):
    """Sliding-window attention backed by the Triton key-bias kernel."""

    def __init__(self, config):
        super().__init__(config)
        self.window_size = config.window_size

    def _triton_key_bias(self, *, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        # Outside the local window, add a large negative per-key bias so attention mass
        # is numerically suppressed while still reusing the Triton key-bias kernel.
        return torch.full(
            (batch_size, self.n_head, seq_len),
            fill_value=-1.0e6 / 1.44269504,
            device=device,
            dtype=torch.bfloat16,
        )

    def _triton_key_bias_window(self) -> int:
        # Keep previous `window_size` normal tokens plus self.
        return self.window_size + 1


class TritonSlidingWindowAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = TritonSlidingWindowAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: torch.Tensor | None = None,
        past_key_values=None,
        documents_idx_BxT: torch.Tensor | None = None,
    ):
        attn_output, past_key_values = self.attn(
            self.attention_norm(x),
            freqs_cis,
            attn_block_mask=attn_block_mask,
            past_key_values=past_key_values,
            documents_idx_BxT=documents_idx_BxT,
        )
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, past_key_values


class SlidingAttentionModel(Model):
    """Model with sliding window attention."""

    def __init__(self, config: SlidingAttentionModelConfig):
        super().__init__(config)

    def _get_block_cls(self):
        return TritonSlidingWindowAttentionBlock if self.use_triton_full_attention else super()._get_block_cls()

    def _can_use_triton_prefill(
        self,
        idx_BxT: Tensor,
        is_real_BxT: Tensor,
    ) -> bool:
        return (
            self.use_triton_full_attention
            and triton_keybias_attention is not None
            and idx_BxT.is_cuda
        )

    def get_prefilling_mask_function(
        self,
        idx_BxT: Tensor,
        documents_idx_BxT: Tensor | None = None,
    ) -> Callable:
        """Get attention mask for prefilling: causal + document + sliding window."""
        base_mask_fn = super().get_prefilling_mask_function(idx_BxT, documents_idx_BxT=documents_idx_BxT)
        sliding_mask_fn = sliding_window_mask_factory_method(self.config.window_size)
        return and_masks(base_mask_fn, sliding_mask_fn)

    def create_attention_mask(
        self,
        idx_BxT: Tensor,
        cache_lengths: Tensor,
        is_real_BxT: Tensor | None = None,
        documents_idx_BxT: Tensor | None = None,
    ) -> Callable | None:
        mask_fn = super().create_attention_mask(
            idx_BxT,
            cache_lengths,
            is_real_BxT=is_real_BxT,
            documents_idx_BxT=documents_idx_BxT,
        )
        if int(cache_lengths.max().item()) > 0:
            mask_fn = and_masks(mask_fn, sliding_window_mask_factory_method(self.config.window_size))
        return mask_fn

    def forward(self, idx_BxT: Tensor, targets_BxT: Tensor):
        """
        Forward pass for training with extra memory-access logging.
        Loss behavior is unchanged from the full-attention base model.
        """
        original_logits, loss, stats = super().forward(idx_BxT, targets_BxT)

        # Sliding window semantics in this repo: window_size previous tokens plus self.
        window_plus_self = self.config.window_size + 1

        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        is_doc_start_BxT = is_doc_start_from_doc_idx(documents_idx_BxT)
        doc_pos_BxT = doc_relative_positions(is_doc_start_BxT).float()
        is_real_BxT = idx_BxT != self.config.pad_token_id

        access_BxT = torch.clamp(doc_pos_BxT + 1.0, max=float(window_plus_self))
        baseline_BxT = doc_pos_BxT + 1.0
        access_BxT = torch.where(is_real_BxT, access_BxT, torch.zeros_like(access_BxT))
        baseline_BxT = torch.where(is_real_BxT, baseline_BxT, torch.zeros_like(baseline_BxT))
        _, memory_access_rate = global_margin_clamped_excess(
            access_BxT,
            baseline_BxT,
            target=0.0,
            penalty="hinge",
        )

        denom = is_real_BxT.sum().clamp(min=1)
        stats["memory_access_count"] = (access_BxT.sum() / denom).detach()
        stats["memory_access_rate"] = memory_access_rate.detach()
        stats["memory_access_rate_num"] = access_BxT.sum().detach()
        stats["memory_access_rate_den"] = baseline_BxT.sum().detach()

        stats["memory_normal_access_count"] = (access_BxT.sum() / denom).detach()
        for layer_i in range(self.config.n_layer):
            stats[f"memory_access_rate_layer_{layer_i}"] = memory_access_rate.detach()

        return original_logits, loss, stats
