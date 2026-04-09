from __future__ import annotations

"""Autoregressive evaluation and generation utilities for DMS models."""

import math
import os
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

import gumbel_model.utils.gumbel_sigmoid as gumbel_sigmoid_utils
from gumbel_model.full_attention_model import IGNORE_INDEX
from gumbel_model.model import apply_rotary_emb, infer_is_real_tokens, validate_left_padded_tokens
from gumbel_model.utils.decode_attention import KVSegment, masked_kv_attention
from gumbel_model.utils.sampling import sample_next_token


class GumbelDMSInferenceMixin:
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
    ):
        """
        Autoregressive DMS forward that actually removes aged-out keys from memory.
        Uses hard per-head decisions to maintain retained-vs-removed memory.
        """
        device = idx_BxT.device
        b, t = idx_BxT.size()
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=True,
            context="forward_efficient inputs",
        )
        if t == 0:
            empty_logits = self.lm_head.weight.new_empty((b, 0, self.config.vocab_size))
            if targets_BxT is None:
                return empty_logits
            zero = empty_logits.new_zeros(())
            return empty_logits, zero, {
                "token_nll_sum": zero.detach(),
                "token_nll_count": torch.tensor(0, device=device, dtype=torch.long),
                "token_count": torch.tensor(0, device=device, dtype=torch.long),
            }

        if not bool(is_real_BxT.all()):
            x_BxTxC, alphas_LxBxHxT, alphas_soft_LxBxHxT, alphas_hard_LxBxHxT = self.forward_hidden_states(idx_BxT)
            token_logits_BxTxV = self.lm_head(x_BxTxC)
            if targets_BxT is None:
                if return_intermediates:
                    return (
                        token_logits_BxTxV,
                        None,
                        None,
                        alphas_LxBxHxT,
                        alphas_soft_LxBxHxT,
                        alphas_hard_LxBxHxT,
                    )
                return token_logits_BxTxV

            masked_targets_BxT = torch.where(
                is_real_BxT & (idx_BxT != self.config.eos_token_id),
                targets_BxT,
                torch.full_like(targets_BxT, IGNORE_INDEX),
            )
            token_count = (masked_targets_BxT != IGNORE_INDEX).sum()
            token_nll_sum = F.cross_entropy(
                token_logits_BxTxV.view(-1, token_logits_BxTxV.size(-1)),
                masked_targets_BxT.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            ).float()
            loss, stats = self._compute_loss_and_stats(
                idx_BxT=idx_BxT,
                targets_BxT=targets_BxT,
                token_logits_BxTxV=token_logits_BxTxV,
                alphas_LxBxHxT=alphas_LxBxHxT,
                alphas_soft_LxBxHxT=alphas_soft_LxBxHxT,
                alphas_hard_LxBxHxT=alphas_hard_LxBxHxT,
                token_nll_sum_override=token_nll_sum,
                token_count_override=token_count,
                stats_mask_BxT=is_real_BxT,
            )
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

        if t > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is only {self.freqs_cis.shape[0]}"
            )
        if not progress:
            env_progress = os.environ.get("DMS_FORWARD_EFFICIENT_PROGRESS", "").strip().lower()
            progress = env_progress in {"1", "true", "yes", "on"}
        if progress_every is None:
            env_every = os.environ.get("DMS_FORWARD_EFFICIENT_PROGRESS_EVERY", "").strip()
            progress_every = int(env_every) if env_every else 0
        if progress and (progress_every is None or progress_every <= 0):
            progress_every = max(1, t // 20)

        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_dim = self.config.hidden_size // self.config.n_head
        window_tokens = self.config.window_size
        window_storage = max(window_tokens, 1)
        sm_scale = 1.0 / math.sqrt(head_dim)

        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype
        freqs_all = self.freqs_cis.to(device)
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        is_doc_start_BxT = torch.zeros((b, t), device=device, dtype=torch.bool)
        is_doc_start_BxT[:, 0] = True
        if t > 1:
            is_doc_start_BxT[:, 1:] = documents_idx_BxT[:, 1:] != documents_idx_BxT[:, :-1]

        retained_k = [
            torch.zeros((b, n_head, t, head_dim), device=device, dtype=attn_dtype)
            for _ in range(n_layer)
        ]
        retained_v = [
            torch.zeros((b, n_head, t, head_dim), device=device, dtype=attn_dtype)
            for _ in range(n_layer)
        ]
        retained_len = [
            torch.zeros((b, n_head), device=device, dtype=torch.long)
            for _ in range(n_layer)
        ]
        window_k = [
            torch.zeros((b, n_head, window_storage, head_dim), device=device, dtype=attn_dtype)
            for _ in range(n_layer)
        ]
        window_v = [
            torch.zeros((b, n_head, window_storage, head_dim), device=device, dtype=attn_dtype)
            for _ in range(n_layer)
        ]
        window_alpha = [
            torch.zeros((b, n_head, window_storage), device=device, dtype=torch.float32)
            for _ in range(n_layer)
        ]
        window_len = [
            torch.zeros((b,), device=device, dtype=torch.long)
            for _ in range(n_layer)
        ]

        alphas_layers = [[] for _ in range(n_layer)]
        alphas_soft_layers = [[] for _ in range(n_layer)]
        alphas_hard_layers = [[] for _ in range(n_layer)]
        logits_steps = []
        if targets_BxT is not None:
            token_nll_sum = torch.zeros((), device=device, dtype=torch.float32)
            token_count = torch.zeros((), device=device, dtype=torch.long)

        for step in range(t):
            if progress:
                done = step + 1
                if done == 1 or done == t or (done % progress_every == 0):
                    pct = 100.0 * done / max(t, 1)
                    print(f"[{progress_prefix}] {done}/{t} ({pct:.1f}%)", flush=True)
            step_doc_start_B = is_doc_start_BxT[:, step]
            idx_step = idx_BxT[:, step:step + 1]
            x = self.transformer.drop(self.transformer.wte(idx_step))
            freqs_step = freqs_all[step].view(1, 1, -1).expand(b, 1, -1)

            for layer_idx, block in enumerate(self.transformer.h):
                if step_doc_start_B.any():
                    retained_len[layer_idx][step_doc_start_B] = 0
                    window_len[layer_idx][step_doc_start_B] = 0

                x_norm = block.attention_norm(x)
                q, k, v = block.attn.c_attn(x_norm).split(block.attn.hidden_size, dim=2)
                q = q.view(b, 1, n_head, head_dim)
                k = k.view(b, 1, n_head, head_dim)
                v = v.view(b, 1, n_head, head_dim)

                decision_logits_BxTxH = torch.einsum('bthd,hd->bth', x_norm.view(b, 1, n_head, head_dim), block.attn.dms_head_weight) + block.attn.dms_head_bias
                sample = gumbel_sigmoid_utils.gumbel_sigmoid(
                    decision_logits_BxTxH,
                    tau=block.attn.gumbel_tau,
                    stochastic=self.training,
                )
                alphas_hard_BxTxH = sample.hard
                alphas_soft_BxTxH = sample.soft
                alphas_hard_BxH = alphas_hard_BxTxH.squeeze(1).float()
                alphas_soft_BxH = alphas_soft_BxTxH.squeeze(1).float()
                alphas_phase_BxH = alphas_soft_BxH if self.training else alphas_hard_BxH
                alphas_layers[layer_idx].append(alphas_phase_BxH)
                alphas_soft_layers[layer_idx].append(alphas_soft_BxH)
                alphas_hard_layers[layer_idx].append(alphas_hard_BxH)

                q, k = apply_rotary_emb(q, k, freqs_cis=freqs_step)
                q_BxHxD = q[:, 0].to(attn_dtype)
                k_BxHxD = k[:, 0].to(attn_dtype)
                v_BxHxD = v[:, 0].to(attn_dtype)
                if check_finite and (
                    (not torch.isfinite(q_BxHxD).all())
                    or (not torch.isfinite(k_BxHxD).all())
                    or (not torch.isfinite(v_BxHxD).all())
                ):
                    raise RuntimeError(
                        f"Non-finite QKV at step={step}, layer={layer_idx}, "
                        f"max_retained={int(retained_len[layer_idx].max().item())}, "
                        f"max_window={int(window_len[layer_idx].max().item())}"
                    )

                window_len_B = window_len[layer_idx]
                attn_out_BxHxD = masked_kv_attention(
                    q_BxHxD,
                    [
                        KVSegment(retained_k[layer_idx], retained_v[layer_idx], retained_len[layer_idx]),
                        KVSegment(window_k[layer_idx], window_v[layer_idx], window_len_B),
                    ],
                    extra_kv=[(k_BxHxD, v_BxHxD)],
                    attn_dtype=attn_dtype,
                    check_finite=check_finite,
                    error_context=f" at step={step}, layer={layer_idx}",
                )

                # Update retained/window memory state after producing attention output.
                for batch_idx in range(b):
                    k_curr_HxD = k_BxHxD[batch_idx]
                    v_curr_HxD = v_BxHxD[batch_idx]
                    w_len = int(window_len_B[batch_idx].item())
                    if window_tokens == 0:
                        keep_heads = (alphas_hard_BxH[batch_idx] <= 0.5).nonzero(as_tuple=False).flatten()
                        for head_idx in keep_heads.tolist():
                            pos = int(retained_len[layer_idx][batch_idx, head_idx].item())
                            retained_k[layer_idx][batch_idx, head_idx, pos, :] = k_curr_HxD[head_idx]
                            retained_v[layer_idx][batch_idx, head_idx, pos, :] = v_curr_HxD[head_idx]
                            retained_len[layer_idx][batch_idx, head_idx] = pos + 1
                    else:
                        if w_len == window_tokens:
                            old_k_HxD = window_k[layer_idx][batch_idx, :, 0, :].clone()
                            old_v_HxD = window_v[layer_idx][batch_idx, :, 0, :].clone()
                            # Clone before in-place shift below; otherwise this view is overwritten
                            # and keep/drop decisions use the wrong token's alpha.
                            old_alpha_H = window_alpha[layer_idx][batch_idx, :, 0].clone()
                            if window_tokens > 1:
                                window_k[layer_idx][batch_idx, :, :-1, :] = (
                                    window_k[layer_idx][batch_idx, :, 1:, :].clone()
                                )
                                window_v[layer_idx][batch_idx, :, :-1, :] = (
                                    window_v[layer_idx][batch_idx, :, 1:, :].clone()
                                )
                                window_alpha[layer_idx][batch_idx, :, :-1] = (
                                    window_alpha[layer_idx][batch_idx, :, 1:].clone()
                                )
                            insert_idx = window_tokens - 1
                            keep_heads = (old_alpha_H <= 0.5).nonzero(as_tuple=False).flatten()
                            for head_idx in keep_heads.tolist():
                                pos = int(retained_len[layer_idx][batch_idx, head_idx].item())
                                retained_k[layer_idx][batch_idx, head_idx, pos, :] = old_k_HxD[head_idx]
                                retained_v[layer_idx][batch_idx, head_idx, pos, :] = old_v_HxD[head_idx]
                                retained_len[layer_idx][batch_idx, head_idx] = pos + 1
                        else:
                            insert_idx = w_len
                            window_len[layer_idx][batch_idx] = w_len + 1

                        window_k[layer_idx][batch_idx, :, insert_idx, :] = k_curr_HxD
                        window_v[layer_idx][batch_idx, :, insert_idx, :] = v_curr_HxD
                        window_alpha[layer_idx][batch_idx, :, insert_idx] = alphas_hard_BxH[batch_idx]

                attn_out = attn_out_BxHxD.view(b, 1, self.config.hidden_size).to(x.dtype)
                attn_out = block.attn.resid_dropout(block.attn.c_proj(attn_out))
                x = x + attn_out
                x = x + block.mlp(block.mlp_norm(x))
                if check_finite and not torch.isfinite(x).all():
                    raise RuntimeError(
                        f"Non-finite hidden state at step={step}, layer={layer_idx}"
                    )

            x = self.transformer.output_norm(x)
            logits_step_Bx1xV = self.lm_head(x)
            logits_steps.append(logits_step_Bx1xV)
            if targets_BxT is not None:
                target_step_B = targets_BxT[:, step]
                target_step_B = torch.where(
                    idx_BxT[:, step] == self.config.eos_token_id,
                    torch.full_like(target_step_B, IGNORE_INDEX),
                    target_step_B,
                )
                target_step_B = torch.where(
                    is_real_BxT[:, step],
                    target_step_B,
                    torch.full_like(target_step_B, IGNORE_INDEX),
                )
                token_nll_sum = token_nll_sum + F.cross_entropy(
                    logits_step_Bx1xV.squeeze(1),
                    target_step_B,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                ).float()
                token_count = token_count + (target_step_B != IGNORE_INDEX).sum()

        token_logits_BxTxV = torch.cat(logits_steps, dim=1)
        alphas_LxBxHxT = torch.stack([torch.stack(layer_vals, dim=2) for layer_vals in alphas_layers], dim=0)
        alphas_soft_LxBxHxT = torch.stack([torch.stack(layer_vals, dim=2) for layer_vals in alphas_soft_layers], dim=0)
        alphas_hard_LxBxHxT = torch.stack([torch.stack(layer_vals, dim=2) for layer_vals in alphas_hard_layers], dim=0)

        if targets_BxT is None:
            if return_intermediates:
                return (
                    token_logits_BxTxV,
                    None,
                    None,
                    alphas_LxBxHxT,
                    alphas_soft_LxBxHxT,
                    alphas_hard_LxBxHxT,
                )
            return token_logits_BxTxV

        loss, stats = self._compute_loss_and_stats(
            idx_BxT=idx_BxT,
            targets_BxT=targets_BxT,
            token_logits_BxTxV=token_logits_BxTxV,
            alphas_LxBxHxT=alphas_LxBxHxT,
            alphas_soft_LxBxHxT=alphas_soft_LxBxHxT,
            alphas_hard_LxBxHxT=alphas_hard_LxBxHxT,
            token_nll_sum_override=token_nll_sum,
            token_count_override=token_count,
            stats_mask_BxT=is_real_BxT,
        )
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

    def _build_dms_decode_state(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        device: torch.device,
    ) -> dict:
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_dim = self.config.hidden_size // self.config.n_head
        window_tokens = self.config.window_size
        window_storage = max(window_tokens, 1)
        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype

        return {
            "retained_k": [torch.zeros((batch_size, n_head, max_tokens, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
            "retained_v": [torch.zeros((batch_size, n_head, max_tokens, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
            "retained_len": [torch.zeros((batch_size, n_head), device=device, dtype=torch.long) for _ in range(n_layer)],
            "window_k": [torch.zeros((batch_size, n_head, window_storage, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
            "window_v": [torch.zeros((batch_size, n_head, window_storage, head_dim), device=device, dtype=attn_dtype) for _ in range(n_layer)],
            "window_alpha": [torch.zeros((batch_size, n_head, window_storage), device=device, dtype=torch.float32) for _ in range(n_layer)],
            "window_len": [torch.zeros((batch_size,), device=device, dtype=torch.long) for _ in range(n_layer)],
            "processed_tokens_B": torch.zeros((batch_size,), device=device, dtype=torch.long),
            "attn_dtype": attn_dtype,
            "window_tokens": window_tokens,
        }

    def _dms_decode_one_token_step(
        self,
        state: dict,
        token_B: Tensor,
        active_mask_B: Tensor,
        is_doc_start_B: Tensor,
        freqs_all: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Process one token [B] through all layers. Returns (logits_BxV, alphas_hard_BxH, alphas_soft_BxH)."""
        b = token_B.size(0)
        device = token_B.device
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_dim = self.config.hidden_size // self.config.n_head
        window_tokens = state["window_tokens"]
        attn_dtype = state["attn_dtype"]

        safe_token_B = torch.where(active_mask_B, token_B, torch.zeros_like(token_B))
        idx_step = safe_token_B.unsqueeze(1)  # [B, 1]
        x = self.transformer.drop(self.transformer.wte(idx_step))

        position_ids_B = torch.where(
            active_mask_B,
            state["processed_tokens_B"],
            torch.zeros_like(state["processed_tokens_B"]),
        )
        freqs_step = freqs_all[position_ids_B].unsqueeze(1)  # [B, 1, D_rope]

        step_doc_start_B = is_doc_start_B & active_mask_B
        active_indices = active_mask_B.nonzero(as_tuple=False).flatten().tolist()

        alphas_hard_layers = []
        alphas_soft_layers = []

        for layer_idx, block in enumerate(self.transformer.h):
            if step_doc_start_B.any():
                state["retained_len"][layer_idx][step_doc_start_B] = 0
                state["window_len"][layer_idx][step_doc_start_B] = 0

            x_norm = block.attention_norm(x)
            q, k, v = block.attn.c_attn(x_norm).split(block.attn.hidden_size, dim=2)
            q = q.view(b, 1, n_head, head_dim)
            k = k.view(b, 1, n_head, head_dim)
            v = v.view(b, 1, n_head, head_dim)

            decision_logits_BxTxH = torch.einsum(
                'bthd,hd->bth',
                x_norm.view(b, 1, n_head, head_dim),
                block.attn.dms_head_weight,
            ) + block.attn.dms_head_bias
            sample = gumbel_sigmoid_utils.gumbel_sigmoid(
                decision_logits_BxTxH,
                tau=block.attn.gumbel_tau,
                stochastic=self.training,
            )
            alphas_hard_BxTxH = sample.hard
            alphas_soft_BxTxH = sample.soft
            alphas_hard_BxH = alphas_hard_BxTxH.squeeze(1).float()
            alphas_soft_BxH = alphas_soft_BxTxH.squeeze(1).float()
            alphas_hard_layers.append(alphas_hard_BxH)
            alphas_soft_layers.append(alphas_soft_BxH)

            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_step)
            q_BxHxD = q[:, 0].to(attn_dtype)
            k_BxHxD = k[:, 0].to(attn_dtype)
            v_BxHxD = v[:, 0].to(attn_dtype)

            attn_out_BxHxD = masked_kv_attention(
                q_BxHxD,
                [
                    KVSegment(state["retained_k"][layer_idx], state["retained_v"][layer_idx], state["retained_len"][layer_idx]),
                    KVSegment(state["window_k"][layer_idx], state["window_v"][layer_idx], state["window_len"][layer_idx]),
                ],
                extra_kv=[(k_BxHxD, v_BxHxD)],
                attn_dtype=attn_dtype,
            )

            # Update memory state for active batch elements
            for batch_idx in active_indices:
                k_curr_HxD = k_BxHxD[batch_idx]
                v_curr_HxD = v_BxHxD[batch_idx]
                w_len = int(state["window_len"][layer_idx][batch_idx].item())
                if window_tokens == 0:
                    keep_heads = (alphas_hard_BxH[batch_idx] <= 0.5).nonzero(as_tuple=False).flatten()
                    for head_idx in keep_heads.tolist():
                        pos = int(state["retained_len"][layer_idx][batch_idx, head_idx].item())
                        state["retained_k"][layer_idx][batch_idx, head_idx, pos, :] = k_curr_HxD[head_idx]
                        state["retained_v"][layer_idx][batch_idx, head_idx, pos, :] = v_curr_HxD[head_idx]
                        state["retained_len"][layer_idx][batch_idx, head_idx] = pos + 1
                else:
                    if w_len == window_tokens:
                        old_k_HxD = state["window_k"][layer_idx][batch_idx, :, 0, :].clone()
                        old_v_HxD = state["window_v"][layer_idx][batch_idx, :, 0, :].clone()
                        old_alpha_H = state["window_alpha"][layer_idx][batch_idx, :, 0].clone()
                        if window_tokens > 1:
                            state["window_k"][layer_idx][batch_idx, :, :-1, :] = state["window_k"][layer_idx][batch_idx, :, 1:, :].clone()
                            state["window_v"][layer_idx][batch_idx, :, :-1, :] = state["window_v"][layer_idx][batch_idx, :, 1:, :].clone()
                            state["window_alpha"][layer_idx][batch_idx, :, :-1] = state["window_alpha"][layer_idx][batch_idx, :, 1:].clone()
                        insert_idx = window_tokens - 1
                        keep_heads = (old_alpha_H <= 0.5).nonzero(as_tuple=False).flatten()
                        for head_idx in keep_heads.tolist():
                            pos = int(state["retained_len"][layer_idx][batch_idx, head_idx].item())
                            state["retained_k"][layer_idx][batch_idx, head_idx, pos, :] = old_k_HxD[head_idx]
                            state["retained_v"][layer_idx][batch_idx, head_idx, pos, :] = old_v_HxD[head_idx]
                            state["retained_len"][layer_idx][batch_idx, head_idx] = pos + 1
                    else:
                        insert_idx = w_len
                        state["window_len"][layer_idx][batch_idx] = w_len + 1

                    state["window_k"][layer_idx][batch_idx, :, insert_idx, :] = k_curr_HxD
                    state["window_v"][layer_idx][batch_idx, :, insert_idx, :] = v_curr_HxD
                    state["window_alpha"][layer_idx][batch_idx, :, insert_idx] = alphas_hard_BxH[batch_idx]

            attn_out = attn_out_BxHxD.view(b, 1, self.config.hidden_size).to(x.dtype)
            attn_out = block.attn.resid_dropout(block.attn.c_proj(attn_out))
            x = x + attn_out
            x = x + block.mlp(block.mlp_norm(x))

        x = self.transformer.output_norm(x)
        logits_BxV = self.lm_head(x).squeeze(1)

        state["processed_tokens_B"][active_mask_B] += 1

        return logits_BxV, torch.stack(alphas_hard_layers), torch.stack(alphas_soft_layers)

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

        state = self._build_dms_decode_state(
            batch_size=b,
            max_tokens=total_real_tokens,
            device=device,
        )
        freqs_all = self.freqs_cis.to(device)

        # Compute doc starts for prompt
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        is_doc_start_BxT = torch.zeros((b, t), device=device, dtype=torch.bool)
        is_doc_start_BxT[:, 0] = True
        if t > 1:
            is_doc_start_BxT[:, 1:] = documents_idx_BxT[:, 1:] != documents_idx_BxT[:, :-1]

        last_logits_BxV = self.lm_head.weight.new_zeros((b, self.config.vocab_size))

        # Prefill: process prompt tokens column by column
        for col in range(t):
            active_B = is_real_BxT[:, col]
            if not bool(active_B.any()):
                continue
            logits_BxV, _, _ = self._dms_decode_one_token_step(
                state, idx_BxT[:, col], active_B, is_doc_start_BxT[:, col], freqs_all,
            )
            last_logits_BxV = torch.where(active_B.unsqueeze(1), logits_BxV, last_logits_BxV)

        # Decode
        generated_BxT = torch.full(
            (b, max_new_tokens),
            self.config.pad_token_id,
            device=device,
            dtype=idx_BxT.dtype,
        )
        finished_B = torch.zeros((b,), device=device, dtype=torch.bool)
        next_logits_BxV = last_logits_BxV
        # Track doc starts for generated tokens
        next_is_doc_start_B = (idx_BxT[:, -1] == self.config.eos_token_id)

        for step in range(max_new_tokens):
            sample_mask_B = ~finished_B
            if not bool(sample_mask_B.any()):
                break

            next_token_B = sample_next_token(
                next_logits_BxV,
                sample_mask_B,
                pad_token_id=self.config.pad_token_id,
                suppressed_token_ids=(self.config.pad_token_id,),
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
                next_logits_step, _, _ = self._dms_decode_one_token_step(
                    state, next_token_B, decode_active_B, next_is_doc_start_B, freqs_all,
                )
                next_logits_BxV = torch.where(
                    decode_active_B.unsqueeze(1),
                    next_logits_step,
                    next_logits_BxV,
                )
            next_is_doc_start_B = (next_token_B == self.config.eos_token_id)
            finished_B = finished_B | newly_finished_B

        return torch.cat([idx_BxT, generated_BxT], dim=1)
