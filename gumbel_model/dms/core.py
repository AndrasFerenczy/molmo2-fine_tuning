from __future__ import annotations

"""Core DMS model definition and train-time forward paths."""

import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import gumbel_model.utils.gumbel_sigmoid as gumbel_sigmoid_utils
from gumbel_model.full_attention_model import IGNORE_INDEX, ModelConfig
from gumbel_model.model import (
    MLP,
    RMSNorm,
    apply_rotary_emb,
    compute_left_padded_position_ids,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    precompute_freqs_cis,
    validate_left_padded_tokens,
)
from gumbel_model.utils.segmented_ops import (
    doc_relative_positions,
    is_doc_start_from_doc_idx,
    masked_global_margin_clamped_excess,
    masked_per_document_count,
    segmented_cumsum,
)

try:
    from gumbel_model.attention.triton_keybias_flash_attention import keybias_attention as triton_keybias_attention
except Exception:
    triton_keybias_attention = None


class GumbelDMSAttention(nn.Module):
    def __init__(self, config: GumbelDMSConfig):
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
        self.warp_specialize = config.warp_specialize
        self.gumbel_tau = config.gumbel_tau
        head_dim = config.hidden_size // config.n_head
        self.dms_head_weight = nn.Parameter(torch.empty(config.n_head, head_dim))
        nn.init.normal_(self.dms_head_weight, std=0.02)
        self.dms_head_bias = nn.Parameter(torch.zeros(config.n_head), requires_grad=False)

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

        x_BxTxHxCH = x.view(b, -1, self.n_head, c // self.n_head)
        decision_logits_BxTxH = torch.einsum('bthd,hd->bth', x_BxTxHxCH, self.dms_head_weight) + self.dms_head_bias
        sample = gumbel_sigmoid_utils.gumbel_sigmoid(
            decision_logits_BxTxH,
            tau=self.gumbel_tau,
            stochastic=self.training,
        )
        alphas_hard_BxTxH = sample.hard
        alphas_soft_BxTxH = sample.soft
        z_BxTxH = sample.pre_sigmoid
        alphas_hard_BxHxT = alphas_hard_BxTxH.permute(0, 2, 1).contiguous()
        alphas_soft_BxHxT = alphas_soft_BxTxH.permute(0, 2, 1).contiguous()
        alphas_BxHxT = alphas_soft_BxHxT if self.training else alphas_hard_BxHxT

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        k_for_attn = k
        v_for_attn = v

        if self.training:
            key_bias_BxHxT = F.logsigmoid(-z_BxTxH).permute(0, 2, 1).contiguous().to(torch.bfloat16)
        else:
            key_bias_BxHxT = torch.where(
                alphas_hard_BxHxT > 0.5,
                torch.tensor(float("-inf"), dtype=torch.bfloat16, device=alphas_BxHxT.device),
                torch.zeros_like(alphas_BxHxT, dtype=torch.bfloat16),
            )

        if documents_idx_BxT is not None:
            documents_idx_BxT = documents_idx_BxT.contiguous()
        assert triton_keybias_attention is not None and q.is_cuda, (
            "Dense DMS forward requires CUDA + Triton key-bias attention. "
            "Use forward_efficient() for non-CUDA or non-Triton evaluation."
        )
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        y = triton_keybias_attention(
            q, k_for_attn, v_for_attn, sm_scale,
            key_bias_BxHxT,
            # Keep semantics consistent with beacon models:
            # window_size means "number of previous tokens" (self always included),
            # so local unbiased span is window_size + 1 positions.
            key_bias_window=self.window_size + 1,
            warp_specialize=self.warp_specialize,
            documents_idx_BxT=documents_idx_BxT,
        )

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = y.to(self.c_proj.weight.dtype)
        y = self.resid_dropout(self.c_proj(y))
        return y, alphas_BxHxT, alphas_soft_BxHxT, alphas_hard_BxHxT


class GumbelDMSBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = GumbelDMSAttention(config)
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
class GumbelDMSConfig(ModelConfig):
    """
    Configuration for DMS models.
    """
    decision_prob_eps: float = 1e-6
    window_size: int = 12
    warp_specialize: bool = True
    decision_head_bias_init: float = 0.0
    efficiency_loss_weight: float = 1.0
    target_memory_access_rate: float = 0.0
    efficiency_penalty: str = "hinge"  # one of {"hinge", "abs"}
    gumbel_tau: float = 1.0  # Gumbel-Sigmoid temperature
    bimodal_loss_weight: float = 0.0


class GumbelDMSModelBase(nn.Module):
    def __init__(self, config: GumbelDMSConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_token_id is not None, "pad_token_id must be provided in config"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([GumbelDMSBlock(config) for _ in range(config.n_layer)]),
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

        # init dms head bias (after _init_weights which zeros all biases)
        if config.decision_head_bias_init != 0.0:
            with torch.no_grad():
                for block in self.transformer.h:
                    block.attn.dms_head_bias.fill_(config.decision_head_bias_init)

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

    def _compute_loss_and_stats(
        self,
        idx_BxT: Tensor,
        targets_BxT: Tensor,
        token_logits_BxTxV: Tensor,
        alphas_LxBxHxT: Tensor,
        alphas_soft_LxBxHxT: Tensor,
        alphas_hard_LxBxHxT: Tensor,
        stats_mask_BxT: Tensor,
        token_nll_sum_override: Optional[Tensor] = None,
        token_count_override: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        b, t = idx_BxT.size()

        # Compute loss, ignoring EOS targets
        if token_nll_sum_override is not None and token_count_override is not None:
            token_nll_sum = token_nll_sum_override
            token_count = token_count_override
        else:
            masked_targets_BxT = torch.where(
                (idx_BxT == self.config.eos_token_id) | (idx_BxT == self.config.pad_token_id),
                IGNORE_INDEX,
                targets_BxT,
            )
            token_count = (masked_targets_BxT != IGNORE_INDEX).sum()
            token_nll_sum = F.cross_entropy(
                token_logits_BxTxV.view(-1, token_logits_BxTxV.size(-1)),
                masked_targets_BxT.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        token_loss = token_nll_sum / token_count.clamp(min=1)

        # Memory access stats for DMS:
        # Query i always sees keys within the sliding window (regardless of alpha).
        # Keys outside the window are visible only if alpha_j = 0.
        # visible_keys[i] = min(doc_pos[i]+1, W+1) + sum_{j: doc_start..i-(W+1)} (1 - alpha_j)
        W = self.config.window_size
        alphas_float = alphas_LxBxHxT.float()  # [L, B, H, T] (phase-consistent: soft train, hard eval)
        alphas_soft_float = alphas_soft_LxBxHxT.float()
        alphas_hard_float = alphas_hard_LxBxHxT.float()
        alphas_soft_stats = alphas_soft_float.detach()
        alphas_hard_stats = alphas_hard_float.detach()
        decision_values_stats = alphas_soft_stats if self.training else alphas_hard_stats
        decision_rate = decision_values_stats.mean()
        decision_rate_soft = alphas_soft_stats.mean()
        decision_rate_hard = alphas_hard_stats.mean()
        alpha_variance = decision_values_stats.var(unbiased=False)
        alpha_variance_soft = alphas_soft_stats.var(unbiased=False)
        alpha_variance_hard = alphas_hard_stats.var(unbiased=False)
        alpha_soft_distance = torch.minimum(alphas_soft_stats, 1.0 - alphas_soft_stats).mean()
        alpha_uncertainty = (4.0 * alphas_soft_stats * (1.0 - alphas_soft_stats)).mean()

        # Document-aware: reset cumsum at doc boundaries
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        is_doc_start_BxT = is_doc_start_from_doc_idx(documents_idx_BxT)
        is_doc_start_LxBxHxT = is_doc_start_BxT.unsqueeze(0).unsqueeze(2).expand_as(alphas_float)

        # doc_pos[i] = 0-indexed position within document
        doc_pos = doc_relative_positions(is_doc_start_BxT).float()  # [B, T]
        doc_baseline = (doc_pos + 1.0).view(1, b, 1, t)

        def _memory_components(alphas_src: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            # C[i] = sum of (1 - alpha_j) from doc_start to position i
            C = segmented_cumsum(1.0 - alphas_src, is_doc_start_LxBxHxT)  # [L, B, H, T]
            doc_pos_exp = doc_pos.unsqueeze(0).unsqueeze(2).expand_as(alphas_src)  # [L, B, H, T]

            # Window contribution: always visible, capped by doc length
            window_keys = torch.clamp(doc_pos_exp + 1, max=W + 1)  # min(doc_pos+1, W+1)

            # Outside-window contribution: C[i - (W+1)] when doc_pos[i] >= W+1, else 0
            C_shifted = torch.zeros_like(C)
            if t > (W + 1):
                C_shifted[..., (W + 1):] = C[..., :-(W + 1)]
            C_shifted = C_shifted * (doc_pos_exp >= (W + 1))

            visible_keys = window_keys + C_shifted
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
                stats_mask_BxT,
                self.config.target_memory_access_rate,
                penalty=self.config.efficiency_penalty,
            )

        visible_keys, baseline, _, access_BxT, baseline_BxT = _memory_components(alphas_float)
        _, _, _, access_BxT_soft, baseline_BxT_soft = _memory_components(alphas_soft_float)
        _, _, _, access_BxT_hard, baseline_BxT_hard = _memory_components(alphas_hard_float)

        weighted_excess, memory_access_rate, memory_access_rate_num, memory_access_rate_den = _aggregate(
            access_BxT,
            baseline_BxT,
        )
        _, memory_access_rate_soft, _, _ = _aggregate(access_BxT_soft, baseline_BxT_soft)
        _, memory_access_rate_hard, _, _ = _aggregate(access_BxT_hard, baseline_BxT_hard)
        memory_access_count = visible_keys.mean()
        efficiency_term = self.config.efficiency_loss_weight * weighted_excess
        bimodal_penalty = (4.0 * alphas_soft_float * (1.0 - alphas_soft_float)).mean()
        bimodal_term = self.config.bimodal_loss_weight * bimodal_penalty
        loss = token_loss + efficiency_term + bimodal_term

        stats = {
            "token_nll_sum": token_nll_sum.detach(),
            "token_nll_count": token_count.detach(),
            "token_count": token_count.detach(),
            "decision_rate": decision_rate.detach(),
            "decision_rate_soft": decision_rate_soft.detach(),
            "decision_rate_hard": decision_rate_hard.detach(),
            "alpha_variance": alpha_variance.detach(),
            "alpha_variance_soft": alpha_variance_soft.detach(),
            "alpha_variance_hard": alpha_variance_hard.detach(),
            "alpha_soft_distance": alpha_soft_distance.detach(),
            "alpha_uncertainty": alpha_uncertainty.detach(),
            "memory_access_count": memory_access_count.detach(),
            "memory_access_rate": memory_access_rate.detach(),
            "memory_access_rate_num": memory_access_rate_num.detach(),
            "memory_access_rate_den": memory_access_rate_den.detach(),
            "memory_access_rate_soft": memory_access_rate_soft.detach(),
            "memory_access_rate_hard": memory_access_rate_hard.detach(),
            "efficiency_loss_term": efficiency_term.detach(),
            "efficiency_excess": weighted_excess.detach(),
            "bimodal_penalty": bimodal_penalty.detach(),
            "bimodal_loss_term": bimodal_term.detach(),
        }
        with torch.no_grad():
            doc_counts, doc_count_mask = masked_per_document_count(documents_idx_BxT, stats_mask_BxT)
            doc_lengths = doc_counts[doc_count_mask].float()
            if doc_lengths.numel() > 0:
                stats["document_length_mean"] = doc_lengths.mean()
                stats["document_length_std"] = doc_lengths.std(unbiased=False)
                stats["document_length_min"] = doc_lengths.min()
                stats["document_length_max"] = doc_lengths.max()
        with torch.no_grad():
            decision_rate_per_layer = decision_values_stats.mean(dim=(1, 2, 3))
            for i, rate in enumerate(decision_rate_per_layer):
                stats[f"decision_rate_layer_{i}"] = rate
            for layer_i in range(visible_keys.size(0)):
                layer_access_BxT = visible_keys[layer_i].mean(dim=1)
                layer_baseline_BxT = baseline[layer_i].mean(dim=1)
                _, layer_memory_access_rate, _, _ = masked_global_margin_clamped_excess(
                    layer_access_BxT,
                    layer_baseline_BxT,
                    stats_mask_BxT,
                    self.config.target_memory_access_rate,
                    penalty=self.config.efficiency_penalty,
                )
                stats[f"memory_access_rate_layer_{layer_i}"] = layer_memory_access_rate
            decision_bias_per_layer = torch.stack(
                [block.attn.dms_head_bias.float().mean() for block in self.transformer.h]
            )
            stats["decision_head_bias"] = decision_bias_per_layer.mean()
            for i, bias in enumerate(decision_bias_per_layer):
                stats[f"decision_head_bias_layer_{i}"] = bias
        return loss, stats

    def forward(
        self,
        idx_BxT: Tensor,
        targets_BxT: Optional[Tensor] = None,
    ):
        """
        Forward pass. Returns logits, and optionally loss + stats if targets provided.
        """
        x_BxTxC, alphas_LxBxHxT, alphas_soft_LxBxHxT, alphas_hard_LxBxHxT = self.forward_hidden_states(idx_BxT)
        token_logits_BxTxV = self.lm_head(x_BxTxC)
        if targets_BxT is None:
            return token_logits_BxTxV
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        loss, stats = self._compute_loss_and_stats(
            idx_BxT=idx_BxT,
            targets_BxT=targets_BxT,
            token_logits_BxTxV=token_logits_BxTxV,
            alphas_LxBxHxT=alphas_LxBxHxT,
            alphas_soft_LxBxHxT=alphas_soft_LxBxHxT,
            alphas_hard_LxBxHxT=alphas_hard_LxBxHxT,
            stats_mask_BxT=is_real_BxT,
        )
        return token_logits_BxTxV, loss, stats

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
