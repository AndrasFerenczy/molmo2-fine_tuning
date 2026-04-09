from __future__ import annotations

"""
Shared base class for beacon models.
"""

from typing import Optional, Tuple, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import gumbel_model.utils.gumbel_sigmoid as gumbel_sigmoid_utils
from gumbel_model.full_attention_model import IGNORE_INDEX, ModelConfig

from gumbel_model.utils.dmc_accumulation import (
    dmc_exact_accumulation as util_dmc_exact_accumulation,
    dmc_exact_accumulation_torch as util_dmc_exact_accumulation_torch,
)
from gumbel_model.utils.sampling import sample_next_token
from gumbel_model.utils.segmented_ops import (
    is_doc_start_from_doc_idx,
    segmented_cumsum,
    doc_relative_positions,
    masked_global_margin_clamped_excess,
    masked_per_document_count,
)

from gumbel_model.model import (
    RMSNorm,
    MLP,
    compute_left_padded_position_ids,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    precompute_freqs_cis,
    apply_rotary_emb,
    IGNORE_INDEX,
    validate_left_padded_tokens,
)

import math
import inspect

try:
    from gumbel_model.attention.triton_keybias_flash_attention import keybias_attention as triton_keybias_attention
except Exception:
    triton_keybias_attention = None

from dataclasses import dataclass


def dmc_partial_accumulation(
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    omega: Tensor,
    window: int = 12,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    DMC partial accumulation via windowed cumprod.

    For each position i, looks back at most `window` steps and computes:
        weight[j] = prod_{l=j+1}^{i} alpha_l   (for j in [i-W+1, ..., i])
        z_i = sum_j weight[j] * omega_j
        k_out_i = sum_j weight[j] * omega_j * k_j / z_i

    The cumprod of at most `window` values never overflows.
    Differentiable through alpha via cumprod gradients.

    Args:
        k: [B, H, T, D] original keys
        v: [B, H, T, D] original values
        alpha: [B, H, T] merge decisions (0=fresh entry, 1=merge with previous)
        omega: [B, H, T] new token weights
        window: max lookback for accumulation
        eps: clamping epsilon for division stability

    Returns:
        k_compressed: [B, H, T, D] accumulated keys
        v_compressed: [B, H, T, D] accumulated values
    """
    B, H, T, D = k.shape
    W = min(window, T)

    # Work in float32
    alpha_f = alpha.float()
    omega_f = omega.float()
    k_f = k.float()
    v_f = v.float()

    # Pad left by W-1 (alpha=0 kills connection to pre-padding positions)
    alpha_pad = F.pad(alpha_f, (W - 1, 0), value=0.0)   # [B, H, T+W-1]
    omega_pad = F.pad(omega_f, (W - 1, 0), value=0.0)   # [B, H, T+W-1]
    k_pad = F.pad(k_f, (0, 0, W - 1, 0), value=0.0)     # [B, H, T+W-1, D]
    v_pad = F.pad(v_f, (0, 0, W - 1, 0), value=0.0)     # [B, H, T+W-1, D]

    # Unfold into windows of size W along T dimension
    alpha_win = alpha_pad.unfold(-1, W, 1)                # [B, H, T, W]
    omega_win = omega_pad.unfold(-1, W, 1)                # [B, H, T, W]
    k_win = k_pad.unfold(2, W, 1).permute(0, 1, 2, 4, 3) # [B, H, T, W, D]
    v_win = v_pad.unfold(2, W, 1).permute(0, 1, 2, 4, 3) # [B, H, T, W, D]

    # Compute weights in log-space for better numerical stability.
    # alpha_win[..., :] = [alpha_{i-W+1}, ..., alpha_i]
    # weight for position j: prod_{l=j+1}^{i} alpha_l
    # weight for j=i (self): 1 (empty product)
    # We need alpha values from index 1 onward (skip alpha at window start).
    alpha_for_prod = alpha_win[..., 1:]                    # [B, H, T, W-1]
    # Preserve exact zeros from hard decisions (alpha=0 -> log weight = -inf).
    neg_inf = torch.full_like(alpha_for_prod, float("-inf"))
    log_alpha = torch.log(alpha_for_prod.clamp(min=eps))
    log_alpha = torch.where(alpha_for_prod > 0.0, log_alpha, neg_inf)
    log_alpha_rev = log_alpha.flip(-1)                     # [B, H, T, W-1]
    log_cumprod_rev = torch.cumsum(log_alpha_rev, dim=-1)  # [B, H, T, W-1]
    cumprod_fwd = torch.exp(log_cumprod_rev).flip(-1)      # [B, H, T, W-1]

    ones = torch.ones(*cumprod_fwd.shape[:-1], 1, device=k.device, dtype=cumprod_fwd.dtype)
    weights = torch.cat([cumprod_fwd, ones], dim=-1)       # [B, H, T, W]

    # Weighted sums
    w_omega = weights * omega_win                          # [B, H, T, W]
    z = w_omega.sum(dim=-1)                                # [B, H, T]

    w_omega_unsq = w_omega.unsqueeze(-1)                   # [B, H, T, W, 1]
    K = (w_omega_unsq * k_win).sum(dim=-2)                 # [B, H, T, D]
    V = (w_omega_unsq * v_win).sum(dim=-2)                 # [B, H, T, D]

    z_safe = z.unsqueeze(-1).clamp(min=eps)
    return (K / z_safe).to(k.dtype), (V / z_safe).to(v.dtype)


def dmc_exact_accumulation(
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    omega: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Exact full-history accumulation with Triton acceleration when available.
    """
    return util_dmc_exact_accumulation(k, v, alpha, omega, eps=eps)


def dmc_exact_accumulation_torch(
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    omega: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Torch exact full-history accumulation for SDPA-only inference paths."""
    return util_dmc_exact_accumulation_torch(k, v, alpha, omega, eps=eps)

class GumbelDMCAttention(nn.Module):
    def __init__(self, config: GumbelDMCConfig):
        super().__init__()
        assert config.hidden_size % config.n_head == 0
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.window_size = config.window_size
        self.decision_prob_eps = config.decision_prob_eps
        self.accumulation_mode = config.accumulation_mode
        self.warp_specialize = config.warp_specialize
        # Valid merge decisions mask: position 0 has no previous token to merge into.
        alpha_valid_mask = torch.ones(1, 1, config.block_size, dtype=torch.float32)
        alpha_valid_mask[:, :, 0] = 0.0
        self.register_buffer("alpha_valid_mask", alpha_valid_mask, persistent=False)
        # Learned additive bias applied to decision logits before gumbel-sigmoid.
        self.decision_head_bias = nn.Parameter(
            torch.full((self.n_head,), float(config.decision_head_bias_init)),
            requires_grad=False,
        )

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        documents_idx_BxT: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, t, c = x.size()

        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head)
        k = k.view(b, t, self.n_head, c // self.n_head)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        # Make merging decisions from pre-RoPE k to avoid position-dependent
        # noise from rotary embeddings corrupting the merge signal.
        # k is [B, T, H, D] here (pre-transpose, pre-RoPE).
        decision_logits = k[..., 0].transpose(1, 2) + self.decision_head_bias.view(1, self.n_head, 1).to(k.dtype)
        sample = gumbel_sigmoid_utils.gumbel_sigmoid(
            decision_logits,
            tau=1.0,
            stochastic=self.training,
        )  # [B, H, T]
        alphas_hard_BxHxT = sample.hard
        alphas_soft_BxHxT = sample.soft
        z_BxHxT = sample.pre_sigmoid
        # Keep inference behavior discrete while allowing smooth training dynamics.
        alphas_BxHxT = alphas_soft_BxHxT if self.training else alphas_hard_BxHxT
        # Force merge decision to 0 at each document start (including position 0)
        # out-of-place, so autograd can safely backprop through sigmoid outputs.
        if documents_idx_BxT is not None:
            is_doc_start_BxT = torch.ones((b, t), dtype=torch.bool, device=alphas_BxHxT.device)
            if t > 1:
                is_doc_start_BxT[:, 1:] = documents_idx_BxT[:, 1:] != documents_idx_BxT[:, :-1]
            valid_merge_Bx1xT = (~is_doc_start_BxT).unsqueeze(1).to(alphas_BxHxT.dtype)
            alphas_BxHxT = alphas_BxHxT * valid_merge_Bx1xT
        else:
            # Fallback for call sites without document ids: at least mask t=0.
            valid_merge_Bx1xT = self.alpha_valid_mask[:, :, :t]
            alphas_BxHxT = alphas_BxHxT * valid_merge_Bx1xT
        omegas_BxHxT = torch.sigmoid(q[..., 0].transpose(1, 2))  # [B, H, T]

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        # Key/value accumulation for compressed memory.
        if self.accumulation_mode == "windowed":
            k_for_attn, v_for_attn = dmc_partial_accumulation(
                k,
                v,
                alphas_BxHxT,
                omegas_BxHxT,
                window=self.window_size,
                eps=self.decision_prob_eps,
            )
        elif self.accumulation_mode == "exact":
            k_for_attn, v_for_attn = dmc_exact_accumulation(
                k,
                v,
                alphas_BxHxT,
                omegas_BxHxT,
                eps=self.decision_prob_eps,
            )
        else:
            raise ValueError(
                f"Unknown accumulation_mode={self.accumulation_mode!r}. "
                "Expected one of {'windowed', 'exact'}."
            )
        # dmc_partial_accumulation upcasts to float32 internally; cast back for Triton kernel
        k_for_attn = k_for_attn.to(torch.bfloat16)
        v_for_attn = v_for_attn.to(torch.bfloat16)

        # DMS-style masking:
        # - train: use logit-space retain probabilities (no probability clamp)
        # - eval: exact hard 0/-inf retain semantics
        if self.training:
            log_one_minus_alpha_BxHxT = F.logsigmoid(-z_BxHxT)
            # Force retain=1 (log=0) at invalid merge positions.
            log_one_minus_alpha_BxHxT = torch.where(
                valid_merge_Bx1xT > 0.5,
                log_one_minus_alpha_BxHxT,
                torch.zeros_like(log_one_minus_alpha_BxHxT),
            )
        else:
            alphas_hard_masked_BxHxT = alphas_hard_BxHxT * valid_merge_Bx1xT
            log_one_minus_alpha_BxHxT = torch.where(
                alphas_hard_masked_BxHxT > 0.5,
                torch.full_like(alphas_hard_masked_BxHxT, float("-inf")),
                torch.zeros_like(alphas_hard_masked_BxHxT),
            )

        # Per-key retain bias: shifted so key j uses decision alpha_{j+1}.
        # Last key gets bias=0 (no next token to merge into).
        key_bias_BxHxT = torch.zeros_like(alphas_BxHxT)
        key_bias_BxHxT[:, :, :-1] = log_one_minus_alpha_BxHxT[:, :, 1:]
        key_bias_BxHxT = key_bias_BxHxT.to(torch.bfloat16)

        if documents_idx_BxT is not None:
            documents_idx_BxT = documents_idx_BxT.contiguous()
        assert triton_keybias_attention is not None and q.is_cuda, (
            "Dense DMC forward requires CUDA + Triton key-bias attention."
        )
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        y = triton_keybias_attention(
            q, k_for_attn, v_for_attn, sm_scale,
            key_bias_BxHxT,
            key_bias_window=1,
            warp_specialize=self.warp_specialize,
            documents_idx_BxT=documents_idx_BxT,
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = y.to(self.c_proj.weight.dtype)
        y = self.resid_dropout(self.c_proj(y))
        return y, alphas_BxHxT, alphas_soft_BxHxT, alphas_hard_BxHxT



class GumbelDMCBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = GumbelDMCAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(self,
        x: Tensor,
        freqs_cis: Tensor,
        documents_idx_BxT: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        attn_output, alphas_BxHxT, alphas_soft_BxHxT, alphas_hard_BxHxT = self.attn(
            self.attention_norm(x),
            freqs_cis,
            documents_idx_BxT=documents_idx_BxT,
        )
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, alphas_BxHxT, alphas_soft_BxHxT, alphas_hard_BxHxT

@dataclass
class GumbelDMCConfig(ModelConfig):
    """
    Configuration for DMC models.
    """
    decision_prob_eps: float = 1e-6
    window_size: int = 12
    accumulation_mode: str = "windowed"
    warp_specialize: bool = True
    decision_head_bias_init: float = 0.0
    efficiency_loss_weight: float = 1.0
    target_memory_access_rate: float = 0.0
    efficiency_penalty: str = "hinge"  # one of {"hinge", "abs"}
    

class GumbelDMCModel(nn.Module):
    """
    DMC (Dynamic Memory Compression) model.
    Uses gumbel-sigmoid merge decisions + partial accumulation for k/v compression.
    Standard T-length sequences (no interleaved beacons).
    """

    def __init__(self, config: GumbelDMCConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_token_id is not None, "pad_token_id must be provided in config"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([GumbelDMCBlock(config) for _ in range(config.n_layer)]),
            output_norm = RMSNorm(config)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.config.hidden_size // self.config.n_head, self.config.block_size
            ),
            persistent=False,
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate_document_idx(self, idx_BxT: Tensor) -> Tensor:
        return generate_left_padded_document_idx(
            idx_BxT,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )

    def forward_hidden_states(
        self, idx_BxT: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass for training. Standard T-length sequence (no beacon interleaving).
        """
        device = idx_BxT.device
        b, t = idx_BxT.size()
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=True,
            context="forward inputs",
        )

        documents_idx_BxT = self.generate_document_idx(idx_BxT)

        position_ids_BxT = compute_left_padded_position_ids(is_real_BxT)

        assert t <= self.freqs_cis.shape[0], f"Cannot forward sequence of length {t}, block size is only {self.freqs_cis.shape[0]}"

        all_freqs_cis = self.freqs_cis.to(device)
        freqs_cis = all_freqs_cis[position_ids_BxT]

        tok_emb = self.transformer.wte(idx_BxT)
        x = self.transformer.drop(tok_emb)

        alphas_list = []
        alphas_soft_list = []
        alphas_hard_list = []
        for layer_idx, block in enumerate(self.transformer.h):
            x, alphas_BxHxT, alphas_soft_BxHxT, alphas_hard_BxHxT = block(
                x,
                freqs_cis,
                documents_idx_BxT=documents_idx_BxT,
            )
            alphas_list.append(alphas_BxHxT)
            alphas_soft_list.append(alphas_soft_BxHxT)
            alphas_hard_list.append(alphas_hard_BxHxT)
        alphas_LxBxHxT = torch.stack(alphas_list, dim=0)  # [L, B, H, T]
        alphas_soft_LxBxHxT = torch.stack(alphas_soft_list, dim=0)  # [L, B, H, T]
        alphas_hard_LxBxHxT = torch.stack(alphas_hard_list, dim=0)  # [L, B, H, T]

        x = self.transformer.output_norm(x)

        return x, alphas_LxBxHxT, alphas_soft_LxBxHxT, alphas_hard_LxBxHxT

    def forward(self, idx_BxT: Tensor, targets_BxT: Optional[Tensor] = None):
        """
        Forward pass. Returns logits, and optionally loss + stats if targets provided.
        """
        device = idx_BxT.device
        b, t = idx_BxT.size()
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)

        x_BxTxC, alphas_LxBxHxT, alphas_soft_LxBxHxT, alphas_hard_LxBxHxT = self.forward_hidden_states(idx_BxT)

        token_logits_BxTxV = self.lm_head(x_BxTxC)

        if targets_BxT is None:
            return token_logits_BxTxV

        # Compute loss, ignoring padded positions and EOS tokens
        targets_BxT = torch.where(
            (idx_BxT == self.config.eos_token_id) | (~is_real_BxT),
            IGNORE_INDEX,
            targets_BxT,
        )
        token_count = (targets_BxT != IGNORE_INDEX).sum()
        token_nll_sum = F.cross_entropy(
            token_logits_BxTxV.view(-1, token_logits_BxTxV.size(-1)),
            targets_BxT.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )

        token_loss = token_nll_sum / token_count.clamp(min=1)

        # Memory access stats: query i sees key j<i iff alpha_{j+1}=0, plus always sees itself.
        # visible_keys[i] = 1 + sum_{j=1}^{i} (1 - alpha_j)
        # Since alpha[0]=0 always, (1-alpha)[0]=1, so visible_keys[i] = cumsum(1-alpha)[i]
        alphas_float = alphas_LxBxHxT.float()  # [L, B, H, T] (phase-consistent: soft train, hard eval)
        alphas_soft_float = alphas_soft_LxBxHxT.float()
        alphas_hard_float = alphas_hard_LxBxHxT.float()
        alphas_soft_stats = alphas_soft_float.detach()
        alphas_hard_stats = alphas_hard_float.detach()
        decision_values_stats = alphas_soft_stats if self.training else alphas_hard_stats
        stats_mask_LxBxHxT = is_real_BxT.unsqueeze(0).unsqueeze(2).expand_as(alphas_float)
        stats_mask_f = stats_mask_LxBxHxT.to(alphas_soft_stats.dtype)
        stats_mask_count = stats_mask_f.sum().clamp(min=1)

        def _masked_mean(values: Tensor) -> Tensor:
            return (values * stats_mask_f).sum() / stats_mask_count

        decision_rate = _masked_mean(decision_values_stats)
        decision_rate_soft = _masked_mean(alphas_soft_stats)
        decision_rate_hard = _masked_mean(alphas_hard_stats)
        alpha_soft_distance = _masked_mean(torch.minimum(alphas_soft_stats, 1.0 - alphas_soft_stats))
        alpha_uncertainty = _masked_mean(4.0 * alphas_soft_stats * (1.0 - alphas_soft_stats))

        # Document-aware efficiency: reset cumsum at doc boundaries
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        is_doc_start_BxT = is_doc_start_from_doc_idx(documents_idx_BxT)
        # Expand to [L, B, H, T]
        is_doc_start_LxBxHxT = is_doc_start_BxT.unsqueeze(0).unsqueeze(2).expand_as(alphas_float)
        doc_baseline = (doc_relative_positions(is_doc_start_BxT).float() + 1.0).view(1, b, 1, t)
        def _memory_components(alphas_src: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            visible_keys = segmented_cumsum(1.0 - alphas_src, is_doc_start_LxBxHxT)  # [L, B, H, T]
            baseline = doc_baseline.expand_as(alphas_src)
            rate_per_position = visible_keys / baseline
            access_BxT = visible_keys.mean(dim=(0, 2))
            baseline_BxT = baseline.mean(dim=(0, 2))
            return visible_keys, baseline, rate_per_position, access_BxT, baseline_BxT

        def _aggregate(
            access_BxT: Tensor,
            baseline_BxT: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            return masked_global_margin_clamped_excess(
                access_BxT,
                baseline_BxT,
                is_real_BxT,
                self.config.target_memory_access_rate,
                penalty=self.config.efficiency_penalty,
            )

        visible_keys, baseline, _, access_BxT, baseline_BxT = _memory_components(alphas_float)
        _, _, _, access_BxT_soft, baseline_BxT_soft = _memory_components(alphas_soft_float)
        _, _, _, access_BxT_hard, baseline_BxT_hard = _memory_components(alphas_hard_float)

        weighted_excess, memory_access_rate, rate_num, rate_den = _aggregate(access_BxT, baseline_BxT)
        _, memory_access_rate_soft, _, _ = _aggregate(access_BxT_soft, baseline_BxT_soft)
        _, memory_access_rate_hard, _, _ = _aggregate(access_BxT_hard, baseline_BxT_hard)
        memory_access_count = _masked_mean(visible_keys)
        efficiency_term = self.config.efficiency_loss_weight * weighted_excess
        loss = token_loss + efficiency_term

        stats = {
            "token_nll_sum": token_nll_sum.detach(),
            "token_nll_count": token_count.detach(),
            "token_count": token_count.detach(),
            "decision_rate": decision_rate.detach(),
            "decision_rate_soft": decision_rate_soft.detach(),
            "decision_rate_hard": decision_rate_hard.detach(),
            "alpha_soft_distance": alpha_soft_distance.detach(),
            "alpha_uncertainty": alpha_uncertainty.detach(),
            "memory_access_count": memory_access_count.detach(),
            "memory_access_rate": memory_access_rate.detach(),
            "memory_access_rate_num": rate_num.detach(),
            "memory_access_rate_den": rate_den.detach(),
            "memory_access_rate_soft": memory_access_rate_soft.detach(),
            "memory_access_rate_hard": memory_access_rate_hard.detach(),
            "efficiency_loss_term": efficiency_term.detach(),
            "efficiency_excess": weighted_excess.detach(),
        }
        with torch.no_grad():
            doc_counts, doc_count_mask = masked_per_document_count(documents_idx_BxT, is_real_BxT)
            doc_lengths = doc_counts[doc_count_mask].float()
            if doc_lengths.numel() > 0:
                stats["document_length_mean"] = doc_lengths.mean()
                stats["document_length_std"] = doc_lengths.std(unbiased=False)
                stats["document_length_min"] = doc_lengths.min()
                stats["document_length_max"] = doc_lengths.max()
        with torch.no_grad():
            layer_mask_f = is_real_BxT.unsqueeze(1).expand(-1, self.config.n_head, -1).to(decision_values_stats.dtype)
            layer_mask_count = layer_mask_f.sum().clamp(min=1)
            decision_rate_per_layer = torch.zeros((self.config.n_layer,), device=device, dtype=decision_values_stats.dtype)
            for i in range(self.config.n_layer):
                decision_rate_per_layer[i] = (decision_values_stats[i] * layer_mask_f).sum() / layer_mask_count
            for i, rate in enumerate(decision_rate_per_layer):
                stats[f"decision_rate_layer_{i}"] = rate
            for layer_i in range(visible_keys.size(0)):
                layer_access_BxT = visible_keys[layer_i].mean(dim=1)  # [B, T] avg over H
                layer_baseline_BxT = baseline[layer_i].mean(dim=1)  # [B, T] avg over H
                _, layer_memory_access_rate, _, _ = masked_global_margin_clamped_excess(
                    layer_access_BxT,
                    layer_baseline_BxT,
                    is_real_BxT,
                    self.config.target_memory_access_rate,
                    penalty=self.config.efficiency_penalty,
                )
                stats[f"memory_access_rate_layer_{layer_i}"] = layer_memory_access_rate
            decision_bias_per_layer = torch.stack(
                [block.attn.decision_head_bias.float().mean() for block in self.transformer.h]
            )
            stats["decision_head_bias"] = decision_bias_per_layer.mean()
            for i, bias in enumerate(decision_bias_per_layer):
                stats[f"decision_head_bias_layer_{i}"] = bias

        return token_logits_BxTxV, loss, stats

    def _build_dmc_sdpa_attn_mask(
        self,
        key_bias_BxHxT: Tensor,
        documents_idx_BxT: Tensor,
    ) -> Tensor:
        b, n_head, t = key_bias_BxHxT.shape
        del b, n_head
        attn_mask = key_bias_BxHxT.unsqueeze(-2).expand(-1, -1, t, -1).clone()
        diag_idx = torch.arange(t, device=key_bias_BxHxT.device)
        attn_mask[:, :, diag_idx, diag_idx] = 0

        neg_inf = torch.full(
            (),
            float("-inf"),
            device=attn_mask.device,
            dtype=attn_mask.dtype,
        )
        causal_mask = torch.ones((t, t), device=attn_mask.device, dtype=torch.bool).tril()
        attn_mask = torch.where(causal_mask.view(1, 1, t, t), attn_mask, neg_inf)

        same_doc = (
            documents_idx_BxT.unsqueeze(-1) == documents_idx_BxT.unsqueeze(-2)
        ).unsqueeze(1)
        attn_mask = torch.where(same_doc, attn_mask, neg_inf)
        return attn_mask

    def _forward_generate_sdpa(self, idx_BxT: Tensor) -> Tensor:
        """
        Full-sequence SDPA forward used only by generation.

        This preserves generation on CPU/non-Triton paths while keeping dense
        training/eval forward Triton-only.
        """
        device = idx_BxT.device
        b, t = idx_BxT.size()
        if t == 0:
            return self.lm_head.weight.new_empty((b, 0, self.config.vocab_size))

        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context="generation forward inputs",
        )
        if not bool(is_real_BxT.all()):
            raise ValueError("DMC SDPA generation expects prompts without left padding after row extraction")

        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        position_ids_BxT = compute_left_padded_position_ids(is_real_BxT)
        if t > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is only {self.freqs_cis.shape[0]}"
            )

        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype
        all_freqs_cis = self.freqs_cis.to(device)
        freqs_cis = all_freqs_cis[position_ids_BxT]

        x = self.transformer.drop(self.transformer.wte(idx_BxT))
        for block in self.transformer.h:
            x_norm = block.attention_norm(x)
            q, k, v = block.attn.c_attn(x_norm).split(block.attn.hidden_size, dim=2)
            q = q.view(b, t, self.config.n_head, self.config.hidden_size // self.config.n_head)
            k = k.view(b, t, self.config.n_head, self.config.hidden_size // self.config.n_head)
            v = v.view(b, t, self.config.n_head, self.config.hidden_size // self.config.n_head).transpose(1, 2)

            decision_logits = (
                k[..., 0].transpose(1, 2)
                + block.attn.decision_head_bias.view(1, self.config.n_head, 1).to(k.dtype)
            )
            sample = gumbel_sigmoid_utils.gumbel_sigmoid(
                decision_logits,
                tau=1.0,
                stochastic=self.training,
            )
            alphas_hard_BxHxT = sample.hard
            alphas_soft_BxHxT = sample.soft
            z_BxHxT = sample.pre_sigmoid
            alphas_BxHxT = alphas_soft_BxHxT if self.training else alphas_hard_BxHxT

            is_doc_start_BxT = torch.ones((b, t), dtype=torch.bool, device=idx_BxT.device)
            if t > 1:
                is_doc_start_BxT[:, 1:] = documents_idx_BxT[:, 1:] != documents_idx_BxT[:, :-1]
            valid_merge_Bx1xT = (~is_doc_start_BxT).unsqueeze(1).to(alphas_BxHxT.dtype)
            alphas_BxHxT = alphas_BxHxT * valid_merge_Bx1xT
            omegas_BxHxT = torch.sigmoid(q[..., 0].transpose(1, 2))

            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
            q = q.transpose(1, 2).to(attn_dtype)
            k = k.transpose(1, 2).to(attn_dtype)
            v = v.to(attn_dtype)

            if block.attn.accumulation_mode == "windowed":
                k_for_attn, v_for_attn = dmc_partial_accumulation(
                    k,
                    v,
                    alphas_BxHxT,
                    omegas_BxHxT,
                    window=block.attn.window_size,
                    eps=block.attn.decision_prob_eps,
                )
            elif block.attn.accumulation_mode == "exact":
                k_for_attn, v_for_attn = dmc_exact_accumulation_torch(
                    k,
                    v,
                    alphas_BxHxT,
                    omegas_BxHxT,
                    eps=block.attn.decision_prob_eps,
                )
            else:
                raise ValueError(
                    f"Unknown accumulation_mode={block.attn.accumulation_mode!r}. "
                    "Expected one of {'windowed', 'exact'}."
                )
            k_for_attn = k_for_attn.to(attn_dtype)
            v_for_attn = v_for_attn.to(attn_dtype)

            if self.training:
                log_one_minus_alpha_BxHxT = F.logsigmoid(-z_BxHxT)
                log_one_minus_alpha_BxHxT = torch.where(
                    valid_merge_Bx1xT > 0.5,
                    log_one_minus_alpha_BxHxT,
                    torch.zeros_like(log_one_minus_alpha_BxHxT),
                )
            else:
                alphas_hard_masked_BxHxT = alphas_hard_BxHxT * valid_merge_Bx1xT
                log_one_minus_alpha_BxHxT = torch.where(
                    alphas_hard_masked_BxHxT > 0.5,
                    torch.full_like(alphas_hard_masked_BxHxT, float("-inf")),
                    torch.zeros_like(alphas_hard_masked_BxHxT),
                )

            key_bias_BxHxT = torch.zeros_like(alphas_BxHxT)
            key_bias_BxHxT[:, :, :-1] = log_one_minus_alpha_BxHxT[:, :, 1:]
            attn_mask = self._build_dmc_sdpa_attn_mask(
                key_bias_BxHxT.to(attn_dtype),
                documents_idx_BxT,
            )
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
            y = F.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=sm_scale,
            )

            y = y.transpose(1, 2).contiguous().view(b, t, self.config.hidden_size)
            y = y.to(block.attn.c_proj.weight.dtype)
            y = block.attn.resid_dropout(block.attn.c_proj(y))
            x = x + y
            x = x + block.mlp(block.mlp_norm(x))

        x = self.transformer.output_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_on_eos: bool = True,
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if max_new_tokens == 0:
            return idx_BxT.clone()

        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context="generation prompts",
        )

        max_real_prompt_tokens = int(is_real_BxT.sum(dim=1).max().item())
        total_real_tokens = max_real_prompt_tokens + int(max_new_tokens)
        if total_real_tokens > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot generate {max_new_tokens} new tokens from a prompt with "
                f"{max_real_prompt_tokens} real tokens when block size is {self.freqs_cis.shape[0]}"
            )

        output_rows = []
        for row_idx in range(idx_BxT.size(0)):
            row = idx_BxT[row_idx]
            prompt_T = row[is_real_BxT[row_idx]]
            generated_T = torch.full(
                (max_new_tokens,),
                self.config.pad_token_id,
                device=row.device,
                dtype=row.dtype,
            )
            sequence_BxT = prompt_T.unsqueeze(0)

            for step in range(max_new_tokens):
                logits_BxTxV = self._forward_generate_sdpa(sequence_BxT)
                next_token_B = sample_next_token(
                    logits_BxTxV[:, -1, : self.config.vocab_size],
                    torch.ones((1,), device=row.device, dtype=torch.bool),
                    pad_token_id=self.config.pad_token_id,
                    suppressed_token_ids=(self.config.pad_token_id,),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    forbidden_token_ids=forbidden_token_ids,
                ).to(row.dtype)
                generated_T[step] = next_token_B[0]
                if stop_on_eos and int(next_token_B[0].item()) == self.config.eos_token_id:
                    break
                sequence_BxT = torch.cat([sequence_BxT, next_token_B.view(1, 1)], dim=1)

            output_rows.append(torch.cat([row, generated_T], dim=0))

        return torch.stack(output_rows, dim=0)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
