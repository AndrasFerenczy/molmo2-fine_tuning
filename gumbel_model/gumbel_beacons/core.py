from __future__ import annotations

"""Core Gumbel Beacons model definition and train-time forward paths."""

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
)

from . import ops as gumbel_ops

try:
    from gumbel_model.attention.triton_biased_flash_attention import gumbel_sliding_attention as triton_gumbel_sliding_attention
except Exception:
    triton_gumbel_sliding_attention = None


class GumbelBeaconsFlashAttention(nn.Module):
    def __init__(self, config: GumbelBeaconsConfig):
        super().__init__()
        self.enable_triton_attention = config.enable_triton_attention
        self.warp_specialize = config.warp_specialize
        assert config.hidden_size % config.n_head == 0
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        head_dim = config.hidden_size // config.n_head
        self.beacon_head_weight = nn.Parameter(torch.empty(config.n_head, head_dim))
        nn.init.normal_(self.beacon_head_weight, std=0.02)
        self.beacon_head_bias = nn.Parameter(torch.zeros(config.n_head), requires_grad=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.window_size = config.window_size
        self.apply_minimum_window_normals = config.apply_minimum_window_normals
        self.can_see_since_last_beacon = config.can_see_since_last_beacon
        self.gumbel_tau = config.gumbel_tau
        self.stochastic_eval_decisions = config.stochastic_eval_decisions

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        documents_idx_Bx2T: Optional[Tensor] = None,
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
        cache_beacon_log_alpha: Optional[Tensor] = None,
        write_positions: Optional[Tensor] = None,
        eval_uniform_BxTxH: Optional[Tensor] = None,
        capture_kv: bool = False,
    ) -> Tensor:
        b, two_t, c = x.size()

        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(b, two_t, self.n_head, c // self.n_head)
        k = k.view(b, two_t, self.n_head, c // self.n_head)
        v = v.view(b, two_t, self.n_head, c // self.n_head).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if capture_kv:
            self._captured_k = k.detach()
            self._captured_v = v.detach()

        # Make beacon decisions — use slice indexing to avoid torch.compile graph break
        x_beacon_BxTxC = x[:, 1::2, :]
        x_beacon_BxTxHxCH = x_beacon_BxTxC.view(b, -1, self.n_head, c // self.n_head)
        decision_logits_BxTxH = torch.einsum('bthd,hd->bth', x_beacon_BxTxHxCH, self.beacon_head_weight) + self.beacon_head_bias
        sample = gumbel_sigmoid_utils.gumbel_sigmoid(
            decision_logits_BxTxH,
            tau=self.gumbel_tau,
            stochastic=self.training or self.stochastic_eval_decisions,
            uniforms=eval_uniform_BxTxH if ((not self.training) and self.stochastic_eval_decisions) else None,
        )
        alphas_hard_BxTxH = sample.hard
        alphas_soft_BxTxH = sample.soft
        z_BxTxH = sample.pre_sigmoid
        # BxTxH -> BxHxT
        alphas_hard_BxHxT = alphas_hard_BxTxH.permute(0, 2, 1).contiguous()
        alphas_soft_BxHxT = alphas_soft_BxTxH.permute(0, 2, 1).contiguous()

        if self.training:
            log_alphas_BxHxT = F.logsigmoid(z_BxTxH).permute(0, 2, 1).contiguous()
            log_retain_BxHxT = F.logsigmoid(-z_BxTxH).permute(0, 2, 1).contiguous()
        else:
            neg_inf = torch.full_like(alphas_hard_BxHxT, float("-inf"))
            zero = torch.zeros_like(alphas_hard_BxHxT)
            log_alphas_BxHxT = torch.where(alphas_hard_BxHxT > 0.5, zero, neg_inf)
            log_retain_BxHxT = torch.where(alphas_hard_BxHxT > 0.5, neg_inf, zero)

        if cache_k is not None and cache_v is not None and cache_beacon_log_alpha is not None and write_positions is not None:
            # Write all positions to cache (uniform structure: always 2 slots)
            batch_idx = torch.arange(b, device=x.device)
            cache_k[batch_idx, :, write_positions[:, 0], :] = k[:, :, 0, :]
            cache_k[batch_idx, :, write_positions[:, 1], :] = k[:, :, 1, :]
            cache_v[batch_idx, :, write_positions[:, 0], :] = v[:, :, 0, :]
            cache_v[batch_idx, :, write_positions[:, 1], :] = v[:, :, 1, :]
            cache_beacon_log_alpha[batch_idx, :, write_positions[:, 1]] = log_alphas_BxHxT[:, :, -1].to(cache_beacon_log_alpha.dtype)
            k_for_attn = cache_k
            v_for_attn = cache_v
        else:
            k_for_attn = k
            v_for_attn = v

        if cache_k is None and cache_v is None:
            # Dense forward path for training/full-sequence eval: Triton-only.
            if self.can_see_since_last_beacon:
                use_exact_segment_mask = not self.training
                if use_exact_segment_mask:
                    log_prob_no_beacon_prefix_BxHxT = gumbel_ops.compute_segment_id_prefix_BxHxT(alphas_hard_BxHxT)
                else:
                    log_prob_no_beacon_prefix_BxHxT = gumbel_ops.compute_log_prob_no_beacon_prefix_BxHxT(
                        alphas_soft_BxHxT
                    )
            else:
                use_exact_segment_mask = False
                t = two_t // 2
                log_prob_no_beacon_prefix_BxHxT = torch.zeros(
                    b, self.n_head, t, device=q.device, dtype=q.dtype,
                )

            use_triton = (
                self.enable_triton_attention
                and
                triton_gumbel_sliding_attention is not None
                and self.dropout == 0.0
                and q.shape[-1] in {16, 32, 64, 128, 256}
            )
            assert use_triton, (
                "Triton attention is required for training but not available. "
                "Check enable_triton_attention, dropout, and head_dim."
            )
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
            y = triton_gumbel_sliding_attention(
                q,
                k_for_attn,
                v_for_attn,
                sm_scale,
                log_prob_no_beacon_prefix_BxHxT,
                log_alphas_BxHxT,
                self.window_size,
                self.apply_minimum_window_normals,
                has_prefix_bias=self.can_see_since_last_beacon,
                warp_specialize=self.warp_specialize,
                documents_idx_BxT=documents_idx_Bx2T,
                use_exact_segment_mask=use_exact_segment_mask,
            )
        else:
            # Cached decoding path: SDPA with appended beacon bias feature.
            q = F.pad(q, (0, 1), value=1.0)
            k_for_attn = F.pad(k_for_attn, (0, 1), value=1.0)

            if cache_beacon_log_alpha is not None and cache_beacon_log_alpha.size(-1) == k_for_attn.size(2):
                k_for_attn[:, :, :, -1] = k_for_attn[:, :, :, -1] + cache_beacon_log_alpha.to(k_for_attn.dtype)
            else:
                k_for_attn[:, :, 1::2, -1] = k_for_attn[:, :, 1::2, -1] + log_alphas_BxHxT.to(k_for_attn.dtype)

            y = F.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        y = y.transpose(1, 2).contiguous().view(b, two_t, c)
        y = y.to(self.c_proj.weight.dtype)
        y = self.resid_dropout(self.c_proj(y))
        return y, alphas_hard_BxHxT, alphas_soft_BxHxT


class GumbelBeaconsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = GumbelBeaconsFlashAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(self,
        x: Tensor,
        freqs_cis: Tensor,
        documents_idx_Bx2T: Optional[Tensor] = None,
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
        cache_beacon_log_alpha: Optional[Tensor] = None,
        write_positions: Optional[Tensor] = None,
        eval_uniform_BxTxH: Optional[Tensor] = None,
        capture_kv: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        attn_output, alphas_hard_BxHxT, alphas_soft_BxHxT = self.attn(
            self.attention_norm(x),
            freqs_cis,
            documents_idx_Bx2T=documents_idx_Bx2T,
            cache_k=cache_k,
            cache_v=cache_v,
            cache_beacon_log_alpha=cache_beacon_log_alpha,
            write_positions=write_positions,
            eval_uniform_BxTxH=eval_uniform_BxTxH,
            capture_kv=capture_kv,
        )
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, alphas_hard_BxHxT, alphas_soft_BxHxT


@dataclass
class GumbelBeaconsConfig(ModelConfig):
    """
    Configuration for Gumbel Beacons models.
    """
    beacon_token_id: int = 50257
    window_size: int = 64
    apply_minimum_window_normals: bool = False
    can_see_since_last_beacon: bool = False
    enable_triton_attention: bool = True
    warp_specialize: bool = True
    beacons_loss_weight: float = 1.0
    decision_head_bias_init: float = 0.0
    target_memory_access_rate: float = 0.0
    efficiency_penalty: str = "hinge"  # one of {"hinge", "abs"}
    bimodal_loss_weight: float = 0.0  # Penalty weight encouraging alpha_soft toward {0,1}
    gumbel_tau: float = 1.0  # Gumbel-Sigmoid temperature
    stochastic_eval_decisions: bool = True  # If true, eval hard decisions mirror train-time hard Gumbel sampling.


class GumbelBeaconsModelBase(nn.Module):
    def __init__(self, config: GumbelBeaconsConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_token_id is not None, "pad_token_id must be provided in config"
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([GumbelBeaconsBlock(config) for _ in range(config.n_layer)]),
            output_norm = RMSNorm(config)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying can trigger compile-time warnings about tied-weight tracing.
        # So far this has been harmless, but keep an eye on future compiler changes.
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        # init beacon head bias (after _init_weights which zeros all biases)
        if config.decision_head_bias_init != 0.0:
            with torch.no_grad():
                for block in self.transformer.h:
                    block.attn.beacon_head_bias.fill_(config.decision_head_bias_init)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate_document_idx(
        self,
        idx_BxT: Tensor,
    ) -> Tensor:
        """
        Generate document indices for each token based on EOS token positions.
        """
        return generate_left_padded_document_idx(
            idx_BxT,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )

    def _expand_real_and_document_idx(self, idx_BxT: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=True,
            context="gumbel-beacons inputs",
        )
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        documents_idx_Bx2T = documents_idx_BxT.repeat_interleave(2, dim=1)
        return is_real_BxT, documents_idx_BxT, documents_idx_Bx2T

    def forward_hidden_states(
        self,
        idx_Bx2T: Tensor,
        *,
        documents_idx_Bx2T: Tensor,
        eval_uniforms_LxBxTxH: Optional[Tensor] = None,
        capture_kv: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass for training when inputs already include beacons/padding.
        """
        device = idx_Bx2T.device
        b, two_t = idx_Bx2T.size()
        t = two_t // 2

        assert two_t % 2 == 0, "Doubled sequence length must be even"

        position_ids_Bx2T = torch.arange(t, device=device).repeat_interleave(2).unsqueeze(0).expand(b, -1)

        assert t <= self.freqs_cis.shape[0], f"Cannot forward sequence of length {two_t}, block size is only {self.freqs_cis.shape[0]}"

        all_freqs_cis = self.freqs_cis.to(device)
        freqs_cis = all_freqs_cis[position_ids_Bx2T]

        tok_emb = self.transformer.wte(idx_Bx2T)
        x = self.transformer.drop(tok_emb)

        if eval_uniforms_LxBxTxH is not None:
            expected_shape = (self.config.n_layer, b, t, self.config.n_head)
            if tuple(eval_uniforms_LxBxTxH.shape) != expected_shape:
                raise ValueError(
                    "eval_uniforms_LxBxTxH must have shape "
                    f"{expected_shape}, got {tuple(eval_uniforms_LxBxTxH.shape)}"
                )

        past_key_values_Lx2 = [None for _ in range(self.config.n_layer)]
        alphas_hard_LxBxHxT = torch.zeros(self.config.n_layer, b, self.config.n_head, t, device=device)
        alphas_soft_LxBxHxT = torch.zeros(self.config.n_layer, b, self.config.n_head, t, device=device)
        captured_kvs: list[tuple[Tensor, Tensor, Tensor, Tensor]] | None = [] if capture_kv else None
        for layer_idx, block in enumerate(self.transformer.h):
            eval_uniform_BxTxH = None
            if eval_uniforms_LxBxTxH is not None:
                eval_uniform_BxTxH = eval_uniforms_LxBxTxH[layer_idx]
            x, alphas_hard_BxHxT, alphas_soft_BxHxT = block(
                x,
                freqs_cis,
                documents_idx_Bx2T=documents_idx_Bx2T,
                eval_uniform_BxTxH=eval_uniform_BxTxH,
                capture_kv=capture_kv,
            )
            alphas_hard_LxBxHxT[layer_idx, :, :, :] = alphas_hard_BxHxT
            alphas_soft_LxBxHxT[layer_idx, :, :, :] = alphas_soft_BxHxT
            if capture_kv:
                k = block.attn._captured_k  # [B, H, 2T, D]
                v = block.attn._captured_v
                captured_kvs.append((
                    k[:, :, ::2].clone(),   # k_normal [B, H, T, D]
                    v[:, :, ::2].clone(),   # v_normal
                    k[:, :, 1::2].clone(),  # k_beacon
                    v[:, :, 1::2].clone(),  # v_beacon
                ))
                del block.attn._captured_k, block.attn._captured_v

        x = self.transformer.output_norm(x)

        if capture_kv:
            return x, alphas_hard_LxBxHxT, alphas_soft_LxBxHxT, captured_kvs
        return x, alphas_hard_LxBxHxT, alphas_soft_LxBxHxT

    def add_beacons(self, idx_BxT: Tensor) -> Tensor:
        b, t = idx_BxT.size()
        device = idx_BxT.device

        idx_Bx2T = idx_BxT.repeat_interleave(2, dim=1)

        # The code tells PyTorch to overwrite every odd index with the special BEACON token ID replacing the duplicates!
        idx_Bx2T[:, 1::2] = self.config.beacon_token_id

        return idx_Bx2T

    def forward(
        self,
        idx_BxT: Tensor,                                    # Input tensor (B: batch size (how many lines of words), T: length of a line -> B = y axis, T = x axis) 
        targets_BxT: Optional[Tensor] = None,               # Targets moved by one
        *,                                                  # * = Python will throw an error because the * prevents this! model(inputs, targets, my_uniforms_tensor), You must explicitly state what argument you are targeting model(inputs, targets, eval_uniforms_LxBxTxH=my_uniforms_tensor)
        eval_uniforms_LxBxTxH: Optional[Tensor] = None,     # Gumbel-distribution related matrix for comparision purposes during test (not that important)
                                                            
    ):
        """
        Forward pass on the original token sequence with beacon insertion handled internally.
        Returns token/decision logits aligned to normal tokens.
        If targets are provided, also returns loss and stats.
        """
        device = idx_BxT.device                             # checks where it is stored (memory or device)

        # 1. Prepare Document IDs for both the normal sequence and the doubled sequence     
        is_real_BxT, documents_idx_BxT, documents_idx_Bx2T = self._expand_real_and_document_idx(idx_BxT)
        # 2. Wedge a "Beacon" token between every single word, doubling the length (T -> 2T)
        idx_Bx2T = self.add_beacons(idx_BxT)

        # Forward pass for training when inputs already include beacons/padding
        # x_Bx2T: output of the transformer = the embeddings
        # alphas_hard_LxBxHxT: which beacon to keep which not (yes/no)
        # alphas_hard_LxBxHxT but with probabilities
        x_Bx2T, alphas_hard_LxBxHxT, alphas_soft_LxBxHxT = self.forward_hidden_states(
            idx_Bx2T,
            documents_idx_Bx2T=documents_idx_Bx2T,
            eval_uniforms_LxBxTxH=eval_uniforms_LxBxTxH,
        )

        # Batch size and sequence length
        b = idx_BxT.size(0)
        t = idx_BxT.size(1)

        # This line removes all the beacons and grabs only the original, normal text embeddings.
        x_BxT = x_Bx2T[:, ::2]
        
        # Passes it through a linear layer to tranform back the dimension of the embeddings to the dimension of the vocabulary (maps each "abstract space embeddings to each of the vocabularies")
        token_logits_BxTxV = self.lm_head(x_BxT)

        # If we use it for generation, it gives back the logits
        if targets_BxT is None:
            return token_logits_BxTxV

        # Not for! masking not see the future
        # This is used for cutting every prediction after the EOS token, and only having a look at the tokens predicted before the EOS. 
        # It would not be worth calculating/penalizing the model on how he predicted after the EOS token
        # torch.where(condition, x, y) which is a for loop dx_BxT != self.config.eos_token_id: creates broadcasting
        masked_targets_BxT = torch.where(
            is_real_BxT & (idx_BxT != self.config.eos_token_id),
            targets_BxT,
            torch.full_like(targets_BxT, IGNORE_INDEX),
        )


        token_count = (masked_targets_BxT != IGNORE_INDEX).sum()

        # Loss function: predictions: token_logits_BxTxV, targets: masked_targets_BxT, not to score on: ignore_index values,
        token_nll_sum = F.cross_entropy(
            token_logits_BxTxV.view(-1, token_logits_BxTxV.size(-1)),
            masked_targets_BxT.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        ).float()


        ######
    
        alphas_hard_float = alphas_hard_LxBxHxT.float()
        alphas_soft_float = alphas_soft_LxBxHxT.float()

        # Primary decision-rate metric:
        # - continuous-train mode: soft decisions (with stochastic noise in decision path)
        # - hard-train/eval: hard sampled decisions
        alphas_soft_stats = alphas_soft_float.detach()
        alphas_hard_stats = alphas_hard_float.detach()
        decision_values_stats = alphas_soft_stats if self.training else alphas_hard_stats
        stats_mask_BxT = is_real_BxT
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        stats_mask_LxBxHxT = stats_mask_BxT.unsqueeze(0).unsqueeze(2).expand(n_layer, -1, n_head, -1)
        stats_mask_f = stats_mask_LxBxHxT.to(alphas_soft_stats.dtype)
        stats_mask_count = stats_mask_f.sum().clamp(min=1)

        def _masked_mean(values: Tensor) -> Tensor:
            if not bool(stats_mask_BxT.any()):
                return values.new_zeros(())
            return (values * stats_mask_f).sum() / stats_mask_count

        def _masked_var(values: Tensor) -> Tensor:
            if not bool(stats_mask_BxT.any()):
                return values.new_zeros(())
            mean = _masked_mean(values)
            return (((values - mean) ** 2) * stats_mask_f).sum() / stats_mask_count

        decision_rate = _masked_mean(decision_values_stats)
        decision_rate_soft = _masked_mean(alphas_soft_stats)
        decision_rate_hard = _masked_mean(alphas_hard_stats)
        alpha_variance = _masked_var(decision_values_stats)
        alpha_variance_soft = _masked_var(alphas_soft_stats)
        alpha_variance_hard = _masked_var(alphas_hard_stats)
        alpha_soft_distance = _masked_mean(torch.minimum(alphas_soft_stats, 1.0 - alphas_soft_stats))
        alpha_uncertainty = _masked_mean(4.0 * alphas_soft_stats * (1.0 - alphas_soft_stats))

        # Document-aware efficiency: compute doc starts from original tokens
        is_doc_start_BxT = is_doc_start_from_doc_idx(documents_idx_BxT)
        L = alphas_hard_float.size(0)
        is_doc_start_LBxT = is_doc_start_BxT.unsqueeze(0).expand(L, -1, -1).reshape(L * b, t)

        accesses_soft_LxBxHxT, normal_accesses_soft_LxBxHxT, beacon_accesses_soft_LxBxHxT = gumbel_ops.compute_soft_memory_accesses_BxHxT(
            alphas_soft_float.view(-1, alphas_soft_float.size(2), alphas_soft_float.size(3)),
            window_size=self.config.window_size,
            can_see_since_last_beacon=self.config.can_see_since_last_beacon,
            apply_minimum_window_normals=self.config.apply_minimum_window_normals,
            is_doc_start_BxT=is_doc_start_LBxT,
        )
        accesses_soft_LxBxHxT = accesses_soft_LxBxHxT.view_as(alphas_soft_float)
        normal_accesses_soft_LxBxHxT = normal_accesses_soft_LxBxHxT.view_as(alphas_soft_float)
        beacon_accesses_soft_LxBxHxT = beacon_accesses_soft_LxBxHxT.view_as(alphas_soft_float)

        accesses_hard_LxBxHxT, normal_accesses_hard_LxBxHxT, beacon_accesses_hard_LxBxHxT = gumbel_ops.compute_hard_memory_accesses_BxHxT(
            alphas_hard_float.view(-1, alphas_hard_float.size(2), alphas_hard_float.size(3)),
            window_size=self.config.window_size,
            can_see_since_last_beacon=self.config.can_see_since_last_beacon,
            apply_minimum_window_normals=self.config.apply_minimum_window_normals,
            is_doc_start_BxT=is_doc_start_LBxT,
        )
        accesses_hard_LxBxHxT = accesses_hard_LxBxHxT.view_as(alphas_hard_float)
        normal_accesses_hard_LxBxHxT = normal_accesses_hard_LxBxHxT.view_as(alphas_hard_float)
        beacon_accesses_hard_LxBxHxT = beacon_accesses_hard_LxBxHxT.view_as(alphas_hard_float)

        if self.training:
            accesses_LxBxHxT = accesses_soft_LxBxHxT
            normal_accesses_LxBxHxT = normal_accesses_soft_LxBxHxT
            beacon_accesses_LxBxHxT = beacon_accesses_soft_LxBxHxT
        else:
            accesses_LxBxHxT = accesses_hard_LxBxHxT
            normal_accesses_LxBxHxT = normal_accesses_hard_LxBxHxT
            beacon_accesses_LxBxHxT = beacon_accesses_hard_LxBxHxT
        memory_access_count = _masked_mean(accesses_LxBxHxT)
        normal_access_count = _masked_mean(normal_accesses_LxBxHxT)
        beacon_access_count = _masked_mean(beacon_accesses_LxBxHxT)
        # Normalize by document-relative causal baseline: doc_pos+1 visible keys at each position.
        doc_baseline = (doc_relative_positions(is_doc_start_BxT).float() + 1.0).view(1, b, 1, t)
        normal_baseline_per_query = doc_baseline.expand_as(accesses_LxBxHxT)
        # Per-document clamped excess, length-weighted:
        #   1. rate_per_position = accesses / baseline              [L, B, H, T]
        #   2. rate_BxT = mean over layers and heads                [B, T]
        #   3. For each doc d: doc_rate[d] = mean(rate at positions in d)
        #   4. doc_excess[d] = clamp(doc_rate[d] - target, min=0)
        #   5. efficiency_term = sum(doc_excess[d] * len(d)) / sum(len(d))
        #   6. reported rate  = sum(doc_rate[d] * len(d)) / sum(len(d))
        
        ######
        # memory usage related loss 
        def _aggregate(accesses_src: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            rate_src = accesses_src / normal_baseline_per_query  # [L, B, H, T]
            access_BxT = accesses_src.mean(dim=(0, 2))  # [B, T]
            baseline_BxT = normal_baseline_per_query.mean(dim=(0, 2))  # [B, T]
            weighted_excess, memory_access_rate, rate_num, rate_den = masked_global_margin_clamped_excess(
                access_BxT,
                baseline_BxT,
                stats_mask_BxT,
                self.config.target_memory_access_rate,
                penalty=self.config.efficiency_penalty,
            )
            return weighted_excess, memory_access_rate, rate_src, rate_num, rate_den

        weighted_excess, memory_access_rate, rate_per_position, rate_num, rate_den = _aggregate(accesses_LxBxHxT)
        _, memory_access_rate_soft, _, _, _ = _aggregate(accesses_soft_LxBxHxT)
        _, memory_access_rate_hard, _, _, _ = _aggregate(accesses_hard_LxBxHxT)

        ######
        # bimodal but we are not using it 
        token_loss = token_nll_sum / token_count.clamp(min=1)
        efficiency_term = self.config.beacons_loss_weight * weighted_excess
        bimodal_penalty = _masked_mean(4.0 * alphas_soft_float * (1.0 - alphas_soft_float))
        bimodal_term = self.config.bimodal_loss_weight * bimodal_penalty

        loss = token_loss + efficiency_term + bimodal_term
        beacon_position_count = stats_mask_BxT.to(torch.long).sum() * n_layer * n_head

        stats = {
            "token_nll_sum": token_nll_sum.detach(),
            "token_nll_count": token_count.detach(),
            "token_count": token_count.detach(),
            "beacon_loss_count": beacon_position_count.detach(),
            "beacon_count": beacon_position_count.detach(),
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
            "memory_access_rate_num": rate_num.detach(),
            "memory_access_rate_den": rate_den.detach(),
            "memory_access_rate_soft": memory_access_rate_soft.detach(),
            "memory_access_rate_hard": memory_access_rate_hard.detach(),
            "memory_normal_access_count": normal_access_count.detach(),
            "memory_beacon_access_count": beacon_access_count.detach(),
            "efficiency_loss_term": efficiency_term.detach(),
            "efficiency_excess": weighted_excess.detach(),
            "bimodal_penalty": bimodal_penalty.detach(),
            "bimodal_loss_term": bimodal_term.detach(),
        }
        with torch.no_grad():
            layer_mask_f = stats_mask_BxT.unsqueeze(1).expand(-1, n_head, -1).to(decision_values_stats.dtype)
            layer_mask_count = layer_mask_f.sum().clamp(min=1)
            decision_rate_per_layer = torch.zeros((n_layer,), device=device, dtype=decision_values_stats.dtype)
            if bool(stats_mask_BxT.any()):
                for i in range(n_layer):
                    decision_rate_per_layer[i] = (decision_values_stats[i] * layer_mask_f).sum() / layer_mask_count
            for i, rate in enumerate(decision_rate_per_layer):
                stats[f"decision_rate_layer_{i}"] = rate
                layer_access_BxT = accesses_LxBxHxT[i].mean(dim=1)  # [B, T], avg over heads
                layer_baseline_BxT = normal_baseline_per_query[i].mean(dim=1)  # [B, T], avg over heads
                _, layer_memory_access_rate, _, _ = masked_global_margin_clamped_excess(
                    layer_access_BxT,
                    layer_baseline_BxT,
                    stats_mask_BxT,
                    self.config.target_memory_access_rate,
                    penalty=self.config.efficiency_penalty,
                )
                stats[f"memory_access_rate_layer_{i}"] = layer_memory_access_rate
            decision_bias_per_layer = torch.stack(
                [
                    block.attn.beacon_head_bias.float().mean()
                    for block in self.transformer.h
                ]
            )
            stats["decision_head_bias"] = decision_bias_per_layer.mean()
            for i, bias in enumerate(decision_bias_per_layer):
                stats[f"decision_head_bias_layer_{i}"] = bias
            doc_counts, doc_count_mask = masked_per_document_count(documents_idx_BxT, stats_mask_BxT)
            doc_lengths = doc_counts[doc_count_mask].float()
            if doc_lengths.numel() > 0:
                stats["document_length_mean"] = doc_lengths.mean()
                stats["document_length_std"] = doc_lengths.std(unbiased=False)
                stats["document_length_min"] = doc_lengths.min()
                stats["document_length_max"] = doc_lengths.max()

        return token_logits_BxTxV, loss, stats

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
