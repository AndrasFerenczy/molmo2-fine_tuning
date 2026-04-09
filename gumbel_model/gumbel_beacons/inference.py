from __future__ import annotations

"""Autoregressive evaluation and generation utilities for Gumbel Beacons models."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from modeling.models.model import apply_rotary_emb, infer_is_real_tokens, validate_left_padded_tokens
from modeling.models.utils.decode_attention import KVSegment, masked_kv_attention
import modeling.models.utils.gumbel_sigmoid as gumbel_sigmoid_utils
from modeling.models.utils.sampling import sample_next_token
from modeling.models.utils.segmented_ops import (
    doc_relative_positions,
    is_doc_start_from_doc_idx,
    masked_global_margin_clamped_excess,
    masked_per_document_count,
)

from . import ops as gumbel_ops
from .core import IGNORE_INDEX


@dataclass
class _KVMemory:
    """Per-layer key/value retained memory with variable-length per (batch, head)."""
    k: list[Tensor]    # [n_layer] of [B, H, max_T, D]
    v: list[Tensor]    # [n_layer] of [B, H, max_T, D]
    len: list[Tensor]  # [n_layer] of [B, H]


@dataclass
class _WindowMemory:
    """Per-layer sliding-window memory with segment tracking."""
    k: list[Tensor]       # [n_layer] of [B, H, W, D]
    v: list[Tensor]       # [n_layer] of [B, H, W, D]
    segment: list[Tensor] # [n_layer] of [B, H, W]
    len: list[Tensor]     # [n_layer] of [B]


@dataclass
class _GumbelDecodeState:
    normal_retained: _KVMemory
    normal_window: _WindowMemory
    beacon_retained: _KVMemory
    segment_id: list[Tensor]
    processed_tokens_B: Tensor
    next_doc_start_B: Tensor
    attn_dtype: torch.dtype
    normal_window_tokens: int
    head_dim: int


class GumbelBeaconsInferenceMixin:
    def _build_decode_state(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        device: torch.device,
        attn_dtype: torch.dtype,
    ) -> _GumbelDecodeState:
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_dim = self.config.hidden_size // self.config.n_head

        use_since_last = bool(self.config.can_see_since_last_beacon)
        use_window_normals = (not use_since_last) or bool(self.config.apply_minimum_window_normals)
        normal_window_tokens = self.config.window_size if use_window_normals else 0
        normal_window_storage = max(normal_window_tokens, 1)

        def _make_kv_memory(max_t: int) -> _KVMemory:
            return _KVMemory(
                k=[torch.zeros((batch_size, n_head, max_t, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
                v=[torch.zeros((batch_size, n_head, max_t, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
                len=[torch.zeros((batch_size, n_head), device=device, dtype=torch.long) for _ in range(n_layer)],
            )

        return _GumbelDecodeState(
            normal_retained=_make_kv_memory(max_tokens),
            normal_window=_WindowMemory(
                k=[torch.zeros((batch_size, n_head, normal_window_storage, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
                v=[torch.zeros((batch_size, n_head, normal_window_storage, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
                segment=[torch.zeros((batch_size, n_head, normal_window_storage), device=device, dtype=torch.long) for _ in range(n_layer)],
                len=[torch.zeros((batch_size,), device=device, dtype=torch.long) for _ in range(n_layer)],
            ),
            beacon_retained=_make_kv_memory(max_tokens),
            segment_id=[torch.zeros((batch_size, n_head), device=device, dtype=torch.long) for _ in range(n_layer)],
            processed_tokens_B=torch.zeros((batch_size,), device=device, dtype=torch.long),
            next_doc_start_B=torch.ones((batch_size,), device=device, dtype=torch.bool),
            attn_dtype=attn_dtype,
            normal_window_tokens=normal_window_tokens,
            head_dim=head_dim,
        )

    def _build_decode_state_from_forward(
        self,
        captured_kvs: list[tuple[Tensor, Tensor, Tensor, Tensor]],
        alphas_hard_LxBxHxT: Tensor,
        is_real_BxT: Tensor,
        idx_BxT: Tensor,
        *,
        total_real_tokens: int,
        attn_dtype: torch.dtype,
    ) -> _GumbelDecodeState:
        """Build decode state by replaying memory management with pre-computed KVs and alphas."""
        b, t = is_real_BxT.shape
        device = is_real_BxT.device
        n_layer = self.config.n_layer

        state = self._build_decode_state(
            batch_size=b,
            max_tokens=total_real_tokens,
            device=device,
            attn_dtype=attn_dtype,
        )

        for col in range(t):
            active_B = is_real_BxT[:, col]
            if not bool(active_B.any()):
                continue

            for layer_idx in range(n_layer):
                doc_start_B = state.next_doc_start_B & active_B
                self._reset_layer_memory(state, layer_idx, doc_start_B)

                k_normal_BxHxD = captured_kvs[layer_idx][0][:, :, col, :]
                v_normal_BxHxD = captured_kvs[layer_idx][1][:, :, col, :]
                k_beacon_BxHxD = captured_kvs[layer_idx][2][:, :, col, :]
                v_beacon_BxHxD = captured_kvs[layer_idx][3][:, :, col, :]
                alphas_hard_BxH = alphas_hard_LxBxHxT[layer_idx, :, :, col]

                active_indices = active_B.nonzero(as_tuple=False).flatten().tolist()
                for batch_idx in active_indices:
                    self._update_normal_memory(state, layer_idx, batch_idx, k_normal_BxHxD, v_normal_BxHxD)
                    self._update_beacon_memory(state, layer_idx, batch_idx, alphas_hard_BxH, k_beacon_BxHxD, v_beacon_BxHxD)

            state.processed_tokens_B[active_B] += 1
            state.next_doc_start_B = torch.where(
                active_B,
                idx_BxT[:, col] == self.config.eos_token_id,
                state.next_doc_start_B,
            )

        return state

    def _reset_layer_memory(
        self,
        state: _GumbelDecodeState,
        layer_idx: int,
        reset_mask_B: Tensor,
    ) -> None:
        if not bool(reset_mask_B.any()):
            return
        state.normal_retained.len[layer_idx][reset_mask_B] = 0
        state.normal_window.len[layer_idx][reset_mask_B] = 0
        state.beacon_retained.len[layer_idx][reset_mask_B] = 0
        state.segment_id[layer_idx][reset_mask_B] = 0

    def _append_retained(
        self,
        ret_k: Tensor,
        ret_v: Tensor,
        ret_len_BxH: Tensor,
        batch_idx: int,
        token_k_HxD: Tensor,
        token_v_HxD: Tensor,
        heads_to_keep: Tensor,
    ) -> None:
        if heads_to_keep.numel() == 0:
            return
        pos = ret_len_BxH[batch_idx, heads_to_keep]
        ret_k[batch_idx, heads_to_keep, pos, :] = token_k_HxD[heads_to_keep]
        ret_v[batch_idx, heads_to_keep, pos, :] = token_v_HxD[heads_to_keep]
        ret_len_BxH[batch_idx, heads_to_keep] = pos + 1

    def _update_normal_memory(
        self,
        state: _GumbelDecodeState,
        layer_idx: int,
        batch_idx: int,
        k_BxHxD: Tensor,
        v_BxHxD: Tensor,
    ) -> None:
        k_HxD = k_BxHxD[batch_idx]
        v_HxD = v_BxHxD[batch_idx]
        n_head = self.config.n_head
        device = k_HxD.device
        nw = state.normal_window
        w_len = int(nw.len[layer_idx][batch_idx].item())

        if state.normal_window_tokens == 0:
            if self.config.can_see_since_last_beacon:
                nr = state.normal_retained
                self._append_retained(
                    nr.k[layer_idx], nr.v[layer_idx], nr.len[layer_idx],
                    batch_idx, k_HxD, v_HxD,
                    heads_to_keep=torch.arange(n_head, device=device, dtype=torch.long),
                )
            return

        if w_len == state.normal_window_tokens:
            old_k_HxD = nw.k[layer_idx][batch_idx, :, 0, :].clone()
            old_v_HxD = nw.v[layer_idx][batch_idx, :, 0, :].clone()
            old_segment_H = nw.segment[layer_idx][batch_idx, :, 0].clone()
            if state.normal_window_tokens > 1:
                nw.k[layer_idx][batch_idx, :, :-1, :] = nw.k[layer_idx][batch_idx, :, 1:, :].clone()
                nw.v[layer_idx][batch_idx, :, :-1, :] = nw.v[layer_idx][batch_idx, :, 1:, :].clone()
                nw.segment[layer_idx][batch_idx, :, :-1] = nw.segment[layer_idx][batch_idx, :, 1:].clone()
            insert_idx = state.normal_window_tokens - 1
            if self.config.can_see_since_last_beacon and self.config.apply_minimum_window_normals:
                same_segment_heads = (
                    old_segment_H == state.segment_id[layer_idx][batch_idx]
                ).nonzero(as_tuple=False).flatten()
                nr = state.normal_retained
                self._append_retained(
                    nr.k[layer_idx], nr.v[layer_idx], nr.len[layer_idx],
                    batch_idx, old_k_HxD, old_v_HxD,
                    heads_to_keep=same_segment_heads,
                )
        else:
            insert_idx = w_len
            nw.len[layer_idx][batch_idx] = w_len + 1

        nw.k[layer_idx][batch_idx, :, insert_idx, :] = k_HxD
        nw.v[layer_idx][batch_idx, :, insert_idx, :] = v_HxD
        nw.segment[layer_idx][batch_idx, :, insert_idx] = state.segment_id[layer_idx][batch_idx]

    def _update_beacon_memory(
        self,
        state: _GumbelDecodeState,
        layer_idx: int,
        batch_idx: int,
        alphas_hard_BxH: Tensor,
        k_BxHxD: Tensor,
        v_BxHxD: Tensor,
    ) -> None:
        active_heads = (alphas_hard_BxH[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
        if active_heads.numel() == 0:
            return
        br = state.beacon_retained
        self._append_retained(
            br.k[layer_idx], br.v[layer_idx], br.len[layer_idx],
            batch_idx, k_BxHxD[batch_idx], v_BxHxD[batch_idx],
            heads_to_keep=active_heads,
        )
        if self.config.can_see_since_last_beacon:
            state.normal_retained.len[layer_idx][batch_idx, active_heads] = 0
            state.segment_id[layer_idx][batch_idx, active_heads] = (
                state.segment_id[layer_idx][batch_idx, active_heads] + 1
            )

    def _attend_from_decode_state(
        self,
        state: _GumbelDecodeState,
        layer_idx: int,
        q_BxHxD: Tensor,
        *,
        curr_normal_k_BxHxD: Optional[Tensor] = None,
        curr_normal_v_BxHxD: Optional[Tensor] = None,
        curr_beacon_k_BxHxD: Optional[Tensor] = None,
        curr_beacon_v_BxHxD: Optional[Tensor] = None,
        check_finite: bool = False,
    ) -> Tensor:
        nr = state.normal_retained
        nw = state.normal_window
        br = state.beacon_retained
        segments = [
            KVSegment(nr.k[layer_idx], nr.v[layer_idx], nr.len[layer_idx]),
        ]
        if state.normal_window_tokens > 0:
            segments.append(KVSegment(nw.k[layer_idx], nw.v[layer_idx], nw.len[layer_idx]))
        segments.append(KVSegment(br.k[layer_idx], br.v[layer_idx], br.len[layer_idx]))

        extra_kv: list[tuple[Tensor, Tensor]] = []
        if curr_normal_k_BxHxD is not None and curr_normal_v_BxHxD is not None:
            extra_kv.append((curr_normal_k_BxHxD, curr_normal_v_BxHxD))
        if curr_beacon_k_BxHxD is not None and curr_beacon_v_BxHxD is not None:
            extra_kv.append((curr_beacon_k_BxHxD, curr_beacon_v_BxHxD))

        return masked_kv_attention(
            q_BxHxD,
            segments,
            extra_kv=extra_kv or None,
            attn_dtype=state.attn_dtype,
            check_finite=check_finite,
            error_context=f" in decode state layer={layer_idx}",
        )

    def _decode_one_token_step(
        self,
        state: _GumbelDecodeState,
        token_B: Tensor,
        active_mask_B: Tensor,
        freqs_all: Tensor,
        *,
        eval_uniforms_LxBx1xH: Optional[Tensor] = None,
        check_finite: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b = token_B.size(0)
        device = token_B.device
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_dim = state.head_dim

        active_mask_B = active_mask_B.to(dtype=torch.bool, device=device)
        zero_logits = self.lm_head.weight.new_zeros((b, self.config.vocab_size))
        zero_alpha = torch.zeros((n_layer, b, n_head), device=device, dtype=torch.float32)
        if not bool(active_mask_B.any()):
            return zero_logits, zero_alpha, zero_alpha, zero_alpha

        if eval_uniforms_LxBx1xH is not None:
            expected_shape = (n_layer, b, 1, n_head)
            if tuple(eval_uniforms_LxBx1xH.shape) != expected_shape:
                raise ValueError(
                    "eval_uniforms_LxBx1xH must have shape "
                    f"{expected_shape}, got {tuple(eval_uniforms_LxBx1xH.shape)}"
                )

        if bool((state.processed_tokens_B[active_mask_B] >= self.freqs_cis.shape[0]).any()):
            raise ValueError(
                f"Cannot decode beyond block size {self.freqs_cis.shape[0]} in gumbel-beacons generation"
            )

        safe_token_B = torch.where(active_mask_B, token_B, torch.zeros_like(token_B))
        idx_pair = torch.stack(
            [
                safe_token_B,
                torch.full_like(safe_token_B, self.config.beacon_token_id),
            ],
            dim=1,
        )
        position_ids_B = torch.where(
            active_mask_B,
            state.processed_tokens_B,
            torch.zeros_like(state.processed_tokens_B),
        )
        freqs_pair = freqs_all[position_ids_B].unsqueeze(1).expand(-1, 2, -1)
        x = self.transformer.drop(self.transformer.wte(idx_pair))

        alphas_LxBxH = torch.zeros((n_layer, b, n_head), device=device, dtype=torch.float32)
        alphas_soft_LxBxH = torch.zeros_like(alphas_LxBxH)
        alphas_hard_LxBxH = torch.zeros_like(alphas_LxBxH)
        active_batch_indices = active_mask_B.nonzero(as_tuple=False).flatten().tolist()

        for layer_idx, block in enumerate(self.transformer.h):
            doc_start_normal_B = state.next_doc_start_B & active_mask_B
            self._reset_layer_memory(state, layer_idx, doc_start_normal_B)

            x_norm = block.attention_norm(x)
            q, k, v = block.attn.c_attn(x_norm).split(block.attn.hidden_size, dim=2)
            q = q.view(b, 2, n_head, head_dim)
            k = k.view(b, 2, n_head, head_dim)
            v = v.view(b, 2, n_head, head_dim)

            x_beacon_Bx1xHxD = x_norm[:, 1:2, :].view(b, 1, n_head, head_dim)
            decision_logits_Bx1xH = torch.einsum(
                "bthd,hd->bth",
                x_beacon_Bx1xHxD,
                block.attn.beacon_head_weight,
            ) + block.attn.beacon_head_bias
            eval_uniform_BxTxH = None
            if eval_uniforms_LxBx1xH is not None:
                eval_uniform_BxTxH = eval_uniforms_LxBx1xH[layer_idx]
            sample = gumbel_sigmoid_utils.gumbel_sigmoid(
                decision_logits_Bx1xH,
                tau=block.attn.gumbel_tau,
                stochastic=self.training or block.attn.stochastic_eval_decisions,
                uniforms=eval_uniform_BxTxH if ((not self.training) and block.attn.stochastic_eval_decisions) else None,
            )
            alphas_hard_Bx1xH = sample.hard
            alphas_soft_Bx1xH = sample.soft
            alphas_hard_BxH = alphas_hard_Bx1xH.squeeze(1).float()
            alphas_soft_BxH = alphas_soft_Bx1xH.squeeze(1).float()
            alphas_phase_BxH = alphas_soft_BxH if self.training else alphas_hard_BxH

            active_mask_BxH = active_mask_B.unsqueeze(1)
            alphas_hard_BxH = torch.where(active_mask_BxH, alphas_hard_BxH, torch.zeros_like(alphas_hard_BxH))
            alphas_soft_BxH = torch.where(active_mask_BxH, alphas_soft_BxH, torch.zeros_like(alphas_soft_BxH))
            alphas_phase_BxH = torch.where(active_mask_BxH, alphas_phase_BxH, torch.zeros_like(alphas_phase_BxH))
            alphas_LxBxH[layer_idx] = alphas_phase_BxH
            alphas_soft_LxBxH[layer_idx] = alphas_soft_BxH
            alphas_hard_LxBxH[layer_idx] = alphas_hard_BxH

            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_pair)
            q_Bx2xHxD = q.to(state.attn_dtype)
            k_Bx2xHxD = k.to(state.attn_dtype)
            v_Bx2xHxD = v.to(state.attn_dtype)

            q_normal_BxHxD = q_Bx2xHxD[:, 0]
            k_normal_BxHxD = k_Bx2xHxD[:, 0]
            v_normal_BxHxD = v_Bx2xHxD[:, 0]
            attn_out_normal_BxHxD = self._attend_from_decode_state(
                state,
                layer_idx,
                q_normal_BxHxD,
                curr_normal_k_BxHxD=k_normal_BxHxD,
                curr_normal_v_BxHxD=v_normal_BxHxD,
                check_finite=check_finite,
            )

            for batch_idx in active_batch_indices:
                self._update_normal_memory(state, layer_idx, batch_idx, k_normal_BxHxD, v_normal_BxHxD)

            q_beacon_BxHxD = q_Bx2xHxD[:, 1]
            k_beacon_BxHxD = k_Bx2xHxD[:, 1]
            v_beacon_BxHxD = v_Bx2xHxD[:, 1]
            attn_out_beacon_BxHxD = self._attend_from_decode_state(
                state,
                layer_idx,
                q_beacon_BxHxD,
                curr_beacon_k_BxHxD=k_beacon_BxHxD,
                curr_beacon_v_BxHxD=v_beacon_BxHxD,
                check_finite=check_finite,
            )

            for batch_idx in active_batch_indices:
                self._update_beacon_memory(state, layer_idx, batch_idx, alphas_hard_BxH, k_beacon_BxHxD, v_beacon_BxHxD)

            attn_out_pair = torch.stack(
                [attn_out_normal_BxHxD, attn_out_beacon_BxHxD],
                dim=1,
            ).reshape(b, 2, self.config.hidden_size).to(x.dtype)
            attn_out_pair = block.attn.resid_dropout(block.attn.c_proj(attn_out_pair))
            x = x + attn_out_pair
            x = x + block.mlp(block.mlp_norm(x))
            if check_finite and (not torch.isfinite(x).all()):
                raise RuntimeError(f"Non-finite hidden state in decode step layer={layer_idx}")

        x = self.transformer.output_norm(x)
        logits_pair_Bx2xV = self.lm_head(x)
        token_logits_BxV = logits_pair_Bx2xV[:, 0, :]
        token_logits_BxV = torch.where(active_mask_B.unsqueeze(1), token_logits_BxV, torch.zeros_like(token_logits_BxV))
        state.processed_tokens_B[active_mask_B] = state.processed_tokens_B[active_mask_B] + 1
        state.next_doc_start_B = torch.where(
            active_mask_B,
            safe_token_B == self.config.eos_token_id,
            state.next_doc_start_B,
        )
        return token_logits_BxV, alphas_LxBxH, alphas_soft_LxBxH, alphas_hard_LxBxH

    def _sample_next_token(
        self,
        logits_BxV: Tensor,
        active_mask_B: Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        return sample_next_token(
            logits_BxV,
            active_mask_B,
            pad_token_id=self.config.pad_token_id,
            suppressed_token_ids=(self.config.beacon_token_id, self.config.pad_token_id),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbidden_token_ids=forbidden_token_ids,
        )

    def forward_efficient(
        self,
        idx_BxT: Tensor,
        targets_BxT: Optional[Tensor] = None,
        *,
        return_intermediates: bool = False,
        progress: bool = False,
        progress_every: Optional[int] = None,
        progress_prefix: str = "forward_efficient",
        check_finite: bool = False,
        store_token_logits: bool = True,
        eval_uniforms_LxBxTxH: Optional[Tensor] = None,
    ):
        """
        Autoregressive gumbel-beacons forward with explicit memory eviction.
        Processes one normal/beacon pair per step and materializes only currently
        reachable memory instead of the full doubled sequence.
        """
        device = idx_BxT.device
        b, t = idx_BxT.size()
        if t == 0:
            empty_logits = self.lm_head.weight.new_empty((b, 0, self.config.vocab_size))
            if targets_BxT is None:
                if return_intermediates:
                    empty_alpha = torch.zeros(
                        (self.config.n_layer, b, self.config.n_head, 0),
                        device=device,
                        dtype=torch.float32,
                    )
                    return empty_logits, None, None, empty_alpha, empty_alpha, empty_alpha
                return empty_logits
            zero = empty_logits.new_zeros(())
            stats = {
                "token_nll_sum": zero.detach(),
                "token_nll_count": torch.tensor(0, device=device, dtype=torch.long),
                "token_count": torch.tensor(0, device=device, dtype=torch.long),
            }
            if return_intermediates:
                empty_alpha = torch.zeros(
                    (self.config.n_layer, b, self.config.n_head, 0),
                    device=device,
                    dtype=torch.float32,
                )
                return empty_logits, zero, stats, empty_alpha, empty_alpha, empty_alpha
            return empty_logits, zero, stats

        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=True,
            context="forward_efficient inputs",
        )
        max_real_tokens = int(is_real_BxT.sum(dim=1).max().item()) if t > 0 else 0
        if max_real_tokens > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot forward {max_real_tokens} real tokens, block size is only {self.freqs_cis.shape[0]}"
            )
        if progress and (progress_every is None or progress_every <= 0):
            progress_every = max(1, t // 20)

        n_layer = self.config.n_layer
        n_head = self.config.n_head
        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype
        freqs_all = self.freqs_cis.to(device)
        state = self._build_decode_state(
            batch_size=b,
            max_tokens=max_real_tokens,
            device=device,
            attn_dtype=attn_dtype,
        )

        collect_token_logits = bool(store_token_logits) or (targets_BxT is None) or bool(return_intermediates)
        token_logits_BxTxV = None
        if collect_token_logits:
            token_logits_BxTxV = self.lm_head.weight.new_zeros((b, t, self.config.vocab_size))

        alphas_LxBxHxT = None
        alphas_soft_LxBxHxT = None
        alphas_hard_LxBxHxT = None
        if targets_BxT is not None or return_intermediates:
            alphas_LxBxHxT = torch.zeros((n_layer, b, n_head, t), device=device, dtype=torch.float32)
            alphas_soft_LxBxHxT = torch.zeros_like(alphas_LxBxHxT)
            alphas_hard_LxBxHxT = torch.zeros_like(alphas_LxBxHxT)

        if targets_BxT is not None:
            token_nll_sum = torch.zeros((), device=device, dtype=torch.float32)
            token_count = torch.zeros((), device=device, dtype=torch.long)

        if eval_uniforms_LxBxTxH is not None:
            expected_shape = (n_layer, b, t, n_head)
            if tuple(eval_uniforms_LxBxTxH.shape) != expected_shape:
                raise ValueError(
                    "eval_uniforms_LxBxTxH must have shape "
                    f"{expected_shape}, got {tuple(eval_uniforms_LxBxTxH.shape)}"
                )
        stats_mask_BxT = is_real_BxT

        for step in range(t):
            if progress:
                done = step + 1
                if done == 1 or done == t or (done % progress_every == 0):
                    pct = 100.0 * done / max(t, 1)
                    print(f"[{progress_prefix}] {done}/{t} ({pct:.1f}%)", flush=True)

            logits_step_BxV, alphas_step_LxBxH, alphas_soft_step_LxBxH, alphas_hard_step_LxBxH = (
                self._decode_one_token_step(
                    state,
                    idx_BxT[:, step],
                    is_real_BxT[:, step],
                    freqs_all,
                    eval_uniforms_LxBx1xH=(
                        eval_uniforms_LxBxTxH[:, :, step : step + 1, :]
                        if eval_uniforms_LxBxTxH is not None
                        else None
                    ),
                    check_finite=check_finite,
                )
            )
            if collect_token_logits:
                token_logits_BxTxV[:, step, :] = logits_step_BxV
            if alphas_LxBxHxT is not None:
                alphas_LxBxHxT[:, :, :, step] = alphas_step_LxBxH
                alphas_soft_LxBxHxT[:, :, :, step] = alphas_soft_step_LxBxH
                alphas_hard_LxBxHxT[:, :, :, step] = alphas_hard_step_LxBxH

            if targets_BxT is not None:
                target_step_B = targets_BxT[:, step]
                target_step_B = torch.where(
                    is_real_BxT[:, step],
                    target_step_B,
                    torch.full_like(target_step_B, IGNORE_INDEX),
                )
                target_step_B = torch.where(
                    idx_BxT[:, step] == self.config.eos_token_id,
                    torch.full_like(target_step_B, IGNORE_INDEX),
                    target_step_B,
                )
                target_step_B = torch.where(
                    stats_mask_BxT[:, step],
                    target_step_B,
                    torch.full_like(target_step_B, IGNORE_INDEX),
                )
                token_nll_sum = token_nll_sum + F.cross_entropy(
                    logits_step_BxV,
                    target_step_B,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                ).float()
                token_count = token_count + (target_step_B != IGNORE_INDEX).sum()

        if collect_token_logits:
            assert token_logits_BxTxV is not None
        else:
            token_logits_BxTxV = self.lm_head.weight.new_empty((b, 0, self.config.vocab_size))

        if targets_BxT is None:
            if return_intermediates:
                assert alphas_LxBxHxT is not None
                assert alphas_soft_LxBxHxT is not None
                assert alphas_hard_LxBxHxT is not None
                return (
                    token_logits_BxTxV,
                    None,
                    None,
                    alphas_LxBxHxT,
                    alphas_soft_LxBxHxT,
                    alphas_hard_LxBxHxT,
                )
            return token_logits_BxTxV

        assert alphas_LxBxHxT is not None
        assert alphas_soft_LxBxHxT is not None
        assert alphas_hard_LxBxHxT is not None
        alphas_hard_float = alphas_hard_LxBxHxT.float()
        alphas_soft_float = alphas_soft_LxBxHxT.float()
        alphas_soft_stats = alphas_soft_float.detach()
        alphas_hard_stats = alphas_hard_float.detach()
        decision_values_stats = alphas_soft_stats if self.training else alphas_hard_stats

        stats_mask_LxBxHxT = stats_mask_BxT.unsqueeze(0).unsqueeze(2).expand(n_layer, -1, n_head, -1)
        stats_mask_f = stats_mask_LxBxHxT.to(alphas_soft_stats.dtype)
        stats_mask_count = stats_mask_f.sum()

        def _masked_mean(values: Tensor) -> Tensor:
            if not bool(stats_mask_BxT.any()):
                return values.new_zeros(())
            return (values * stats_mask_f).sum() / stats_mask_count.clamp(min=1)

        def _masked_var(values: Tensor) -> Tensor:
            if not bool(stats_mask_BxT.any()):
                return values.new_zeros(())
            mean = _masked_mean(values)
            return (((values - mean) ** 2) * stats_mask_f).sum() / stats_mask_count.clamp(min=1)

        decision_rate = _masked_mean(decision_values_stats)
        decision_rate_soft = _masked_mean(alphas_soft_stats)
        decision_rate_hard = _masked_mean(alphas_hard_stats)
        alpha_variance = _masked_var(decision_values_stats)
        alpha_variance_soft = _masked_var(alphas_soft_stats)
        alpha_variance_hard = _masked_var(alphas_hard_stats)
        alpha_soft_distance = _masked_mean(torch.minimum(alphas_soft_stats, 1.0 - alphas_soft_stats))
        alpha_uncertainty = _masked_mean(4.0 * alphas_soft_stats * (1.0 - alphas_soft_stats))

        documents_idx_BxT = self.generate_document_idx(idx_BxT)
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

        doc_baseline = (doc_relative_positions(is_doc_start_BxT).float() + 1.0).view(1, b, 1, t)
        normal_baseline_per_query = doc_baseline.expand_as(accesses_LxBxHxT)

        def _aggregate(accesses_src: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            rate_src = accesses_src / normal_baseline_per_query
            access_BxT = accesses_src.mean(dim=(0, 2))
            baseline_BxT = normal_baseline_per_query.mean(dim=(0, 2))
            weighted_excess, memory_access_rate, memory_access_rate_num, memory_access_rate_den = (
                masked_global_margin_clamped_excess(
                    access_BxT,
                    baseline_BxT,
                    stats_mask_BxT,
                    self.config.target_memory_access_rate,
                    penalty=self.config.efficiency_penalty,
                )
            )
            return weighted_excess, memory_access_rate, rate_src, memory_access_rate_num, memory_access_rate_den

        weighted_excess, memory_access_rate, rate_per_position, memory_access_rate_num, memory_access_rate_den = (
            _aggregate(accesses_LxBxHxT)
        )
        _, memory_access_rate_soft, _, _, _ = _aggregate(accesses_soft_LxBxHxT)
        _, memory_access_rate_hard, _, _, _ = _aggregate(accesses_hard_LxBxHxT)

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
            "memory_access_rate_num": memory_access_rate_num.detach(),
            "memory_access_rate_den": memory_access_rate_den.detach(),
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
                    decision_rate_per_layer[i] = (
                        decision_values_stats[i] * layer_mask_f
                    ).sum() / layer_mask_count
            for i, rate in enumerate(decision_rate_per_layer):
                stats[f"decision_rate_layer_{i}"] = rate
                layer_access_BxT = accesses_LxBxHxT[i].mean(dim=1)
                layer_baseline_BxT = normal_baseline_per_query[i].mean(dim=1)
                _, layer_memory_access_rate, _, _ = masked_global_margin_clamped_excess(
                    layer_access_BxT,
                    layer_baseline_BxT,
                    stats_mask_BxT,
                    self.config.target_memory_access_rate,
                    penalty=self.config.efficiency_penalty,
                )
                stats[f"memory_access_rate_layer_{i}"] = layer_memory_access_rate
            decision_bias_per_layer = torch.stack(
                [block.attn.beacon_head_bias.float().mean() for block in self.transformer.h]
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

        if return_intermediates:
            return (
                token_logits_BxTxV,
                loss,
                stats,
                alphas_LxBxHxT,
                alphas_soft_LxBxHxT,
                alphas_hard_LxBxHxT,
            )
        return token_logits_BxTxV, loss, stats

    @torch.no_grad()
    def generate(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        stop_on_eos: bool = True,
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if max_new_tokens == 0:
            return idx_BxT.clone()

        device = idx_BxT.device
        b, t = idx_BxT.size()
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

        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype
        freqs_all = self.freqs_cis.to(device)

        # Prefill generation state token-by-token via the SDPA decode path.
        state = self._build_decode_state(
            batch_size=b,
            max_tokens=total_real_tokens,
            device=device,
            attn_dtype=attn_dtype,
        )
        last_logits_BxV = self.lm_head.weight.new_zeros((b, self.config.vocab_size))
        for col in range(t):
            logits_step_BxV, _, _, _ = self._decode_one_token_step(
                state,
                idx_BxT[:, col],
                is_real_BxT[:, col],
                freqs_all,
            )
            active_col_mask_B = is_real_BxT[:, col].unsqueeze(1)
            last_logits_BxV = torch.where(active_col_mask_B, logits_step_BxV, last_logits_BxV)

        # Phase 3: Autoregressive decode
        generated_BxT = torch.full(
            (b, max_new_tokens),
            self.config.pad_token_id,
            device=device,
            dtype=idx_BxT.dtype,
        )
        finished_B = torch.zeros((b,), device=device, dtype=torch.bool)
        next_logits_BxV = last_logits_BxV

        for step in range(max_new_tokens):
            sample_mask_B = ~finished_B
            if not bool(sample_mask_B.any()):
                break

            next_token_B = self._sample_next_token(
                next_logits_BxV,
                sample_mask_B,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_token_ids=forbidden_token_ids,
            ).to(idx_BxT.dtype)
            generated_BxT[:, step] = torch.where(
                sample_mask_B,
                next_token_B,
                generated_BxT[:, step],
            )

            newly_finished_B = sample_mask_B & stop_on_eos & (next_token_B == self.config.eos_token_id)
            decode_active_B = sample_mask_B & ~newly_finished_B
            if bool(decode_active_B.any()):
                next_logits_BxV_step, _, _, _ = self._decode_one_token_step(
                    state,
                    next_token_B,
                    decode_active_B,
                    freqs_all,
                )
                next_logits_BxV = torch.where(
                    decode_active_B.unsqueeze(1),
                    next_logits_BxV_step,
                    next_logits_BxV,
                )
            finished_B = finished_B | newly_finished_B

        return torch.cat([idx_BxT, generated_BxT], dim=1)
