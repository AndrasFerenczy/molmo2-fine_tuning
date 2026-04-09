from __future__ import annotations

"""Shared beacon decision and memory-access helpers used by training and inference."""

import math
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from gumbel_model.utils.segmented_ops import (
    doc_relative_positions,
    segmented_cumsum,
    segmented_logcumsumexp,
)


def compute_log_prob_no_beacon_prefix_BxHxT(alphas_BxHxT: Tensor) -> Tensor:
    """
    O(T) prefix sum for log-probability of no beacon between key and query positions.
    Returns shape [B, H, T] where prefix[i] = sum_{j=0}^{i-1} log(1-alpha_j).
    For query q and key k: log P(no beacon between k and q) = prefix[q] - prefix[k].

    Use with soft decisions during training. With hard 0/1 decisions, this can contain -inf;
    eval should use compute_segment_id_prefix_BxHxT for exact hard masking.
    """
    log_1_minus_alphas_BxHxT = torch.log1p(-alphas_BxHxT)
    b, h, t = alphas_BxHxT.size()
    prefix_log_BxHxT = torch.zeros(b, h, t, device=alphas_BxHxT.device, dtype=alphas_BxHxT.dtype)
    if t > 1:
        prefix_log_BxHxT[:, :, 1:] = torch.cumsum(log_1_minus_alphas_BxHxT[:, :, :-1], dim=-1)
    return prefix_log_BxHxT


def compute_segment_id_prefix_BxHxT(alphas_BxHxT: Tensor) -> Tensor:
    """
    Eval-time alternative to compute_log_prob_no_beacon_prefix_BxHxT.
    prefix[t] = number of hard beacons strictly before position t (0, 1, 2, ...).
    Used with USE_EXACT_SEGMENT_MASK=True in the Triton kernel, which computes:
      normal_bias = 0 if prefix[q] == prefix[k] else -inf
    This gives exact hard masking with no -inf arithmetic and no NaN.
    Integer values ≤ 127 are represented exactly in bfloat16; typical sequences
    have far fewer beacons per layer than that.
    """
    hard_alpha = (alphas_BxHxT > 0.5).to(alphas_BxHxT.dtype)
    b, h, t = alphas_BxHxT.size()
    prefix_BxHxT = torch.zeros(b, h, t, device=alphas_BxHxT.device, dtype=alphas_BxHxT.dtype)
    if t > 1:
        prefix_BxHxT[:, :, 1:] = hard_alpha[:, :, :-1].cumsum(dim=-1)
    return prefix_BxHxT


def build_gumbel_sliding_attn_bias_BxHx2Tx2T(
    prefix_log_BxHxT: Tensor,
    beacon_log_alpha_BxHxT: Tensor,
    window_size: int,
    zero_normal_bias_in_window: bool,
) -> Tensor:
    """
    Build additive attention bias for doubled sequences (normal/beacon interleaved).
    - Normal keys: prefix[q] - prefix[k].
      If zero_normal_bias_in_window is True, force this to 0 in the sliding window.
    - Beacon keys: log(alpha_k), independent of query.
    """
    b, h, t = prefix_log_BxHxT.shape
    two_t = t * 2
    device = prefix_log_BxHxT.device

    prefix_2t_BxHx2T = prefix_log_BxHxT.repeat_interleave(2, dim=2)
    normal_bias = prefix_2t_BxHx2T.unsqueeze(-1) - prefix_2t_BxHx2T.unsqueeze(-2)

    if zero_normal_bias_in_window:
        q_pos_2t = torch.arange(two_t, device=device).view(1, 1, two_t, 1)
        k_pos_2t = torch.arange(two_t, device=device).view(1, 1, 1, two_t)
        rel = (q_pos_2t - k_pos_2t) // 2
        in_window = (rel >= 0) & (rel <= window_size)
        normal_bias = torch.where(in_window, torch.zeros((), dtype=normal_bias.dtype, device=device), normal_bias)

    beacon_bias_2t_BxHx2T = torch.zeros((b, h, two_t), device=device, dtype=prefix_log_BxHxT.dtype)
    beacon_bias_2t_BxHx2T[:, :, 1::2] = beacon_log_alpha_BxHxT

    is_beacon_key = (torch.arange(two_t, device=device) % 2 == 1).view(1, 1, 1, two_t)
    bias = torch.where(
        is_beacon_key,
        beacon_bias_2t_BxHx2T.unsqueeze(-2),
        normal_bias,
    )
    # Beacons always self-attend: zero beacon bias on diagonal.
    diag_mask = torch.eye(two_t, device=device, dtype=torch.bool).view(1, 1, two_t, two_t)
    bias = torch.where(diag_mask & is_beacon_key, torch.zeros((), dtype=bias.dtype, device=device), bias)
    return bias


def compute_expected_memory_accesses_BxHxT(
    alphas_BxHxT: Tensor,
    window_size: int,
    can_see_since_last_beacon: bool,
    apply_minimum_window_normals: bool,
    is_doc_start_BxT: Tensor | None = None,
    log_retain_BxHxT: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute expected visible key count for each normal-token query.

    When is_doc_start_BxT is provided, all cumulative computations reset at
    document boundaries, giving a tighter efficiency estimate.

    Args:
        log_retain_BxHxT: log(1 - alpha) pre-computed in a numerically stable
            way (e.g. via F.logsigmoid(-z)).  Required when
            can_see_since_last_beacon is True.

    Returns:
    - total_accesses_BxHxT
    - normal_accesses_BxHxT
    - beacon_accesses_BxHxT
    """
    if alphas_BxHxT.dim() != 3:
        raise ValueError(f"Expected alphas_BxHxT to be rank-3, got {alphas_BxHxT.shape}")
    b, h, t = alphas_BxHxT.shape
    if t == 0:
        empty = alphas_BxHxT
        return empty, empty, empty

    dtype = alphas_BxHxT.dtype
    device = alphas_BxHxT.device

    # Prepare doc-aware flag expanded to [B, H, T]
    have_doc_info = is_doc_start_BxT is not None
    if have_doc_info:
        is_doc_start_BxHxT = is_doc_start_BxT.unsqueeze(1).expand(b, h, t)

    # --- Beacon accesses: cumsum of alphas from doc_start to i-1 ---
    if t > 1:
        # Shifted alphas: we want sum of alphas[0..i-1]
        shifted = torch.cat(
            (torch.zeros((b, h, 1), device=device, dtype=dtype), alphas_BxHxT[:, :, :-1]),
            dim=-1,
        )
        if have_doc_info:
            # Zero out shifted values at doc boundaries — no previous beacons in new doc
            shifted = torch.where(is_doc_start_BxHxT, torch.zeros_like(shifted), shifted)
            beacon_accesses_BxHxT = segmented_cumsum(shifted, is_doc_start_BxHxT)
        else:
            beacon_accesses_BxHxT = torch.cumsum(shifted, dim=-1)
    else:
        beacon_accesses_BxHxT = torch.zeros_like(alphas_BxHxT)

    # --- Normal accesses ---
    if not can_see_since_last_beacon:
        if have_doc_info:
            doc_pos = doc_relative_positions(is_doc_start_BxT).unsqueeze(1).expand(b, h, t).to(dtype)
            normal_accesses_BxHxT = (doc_pos + 1.0).clamp(max=float(window_size + 1))
        else:
            positions = torch.arange(t, device=device, dtype=dtype).view(1, 1, t)
            normal_accesses_BxHxT = (positions + 1.0).clamp(max=float(window_size + 1)).expand(b, h, -1).contiguous()
        total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
        return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT

    # --- Since-last-beacon visibility ---
    assert log_retain_BxHxT is not None, (
        "log_retain_BxHxT is required when can_see_since_last_beacon=True"
    )
    log_retain = log_retain_BxHxT

    if have_doc_info:
        # Segmented prefix sum of log_retain, where prefix[i] = sum from doc_start to i-1
        # segmented_cumsum gives sum from doc_start to i (inclusive), so subtract log_retain[i]
        seg_log_retain_cs = segmented_cumsum(log_retain, is_doc_start_BxHxT)
        seg_log_retain_prefix = seg_log_retain_cs - log_retain  # sum from doc_start to i-1

        # neg prefix for logcumsumexp
        neg_seg_prefix = -seg_log_retain_prefix
        seg_cum_neg_log = segmented_logcumsumexp(neg_seg_prefix, is_doc_start_BxHxT)
        # Clamp the log-space value before exp to prevent inf (strict_normal is bounded
        # by T in theory, but numerical accumulation can exceed that).
        log_strict_normal = seg_log_retain_prefix + seg_cum_neg_log
        log_strict_normal = log_strict_normal.clamp(max=math.log(float(t) + 1.0))
        strict_normal_BxHxT = torch.exp(log_strict_normal)
    else:
        log_retain_prefix = torch.zeros((b, h, t + 1), device=device, dtype=dtype)
        log_retain_prefix[:, :, 1:] = torch.cumsum(log_retain, dim=-1)
        neg_log_retain_prefix = -log_retain_prefix[:, :, :t]
        cum_neg_log = torch.logcumsumexp(neg_log_retain_prefix, dim=-1)
        log_strict_normal = log_retain_prefix[:, :, :t] + cum_neg_log
        log_strict_normal = log_strict_normal.clamp(max=math.log(float(t) + 1.0))
        strict_normal_BxHxT = torch.exp(log_strict_normal)

    if not apply_minimum_window_normals:
        normal_accesses_BxHxT = strict_normal_BxHxT
        total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
        return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT

    # --- Minimum-window + since-last-beacon ---
    if have_doc_info:
        doc_pos = doc_relative_positions(is_doc_start_BxT).unsqueeze(1).expand(b, h, t).to(dtype)
        base_window_BxHxT = (doc_pos + 1.0).clamp(max=float(window_size + 1))
    else:
        positions = torch.arange(t, device=device, dtype=dtype).view(1, 1, t)
        base_window_BxHxT = (positions + 1.0).clamp(max=float(window_size + 1))
    normal_accesses_BxHxT = base_window_BxHxT.expand(b, h, -1).contiguous()

    if window_size >= t:
        total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
        return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT

    if have_doc_info:
        # Element-wise: for positions where doc_pos > window_size, compute outside accesses.
        # Reuse exclusive prefix: seg_log_retain_prefix[i] = sum_{doc_start..i-1} log_retain[m]
        # (already computed above for strict_normal).
        # log_prod_window[i] = seg_log_retain_prefix[i] - seg_log_retain_prefix[i-W]
        #                    = sum_{i-W..i-1} log_retain[m]
        doc_pos_long = doc_pos.long()
        beyond_window = doc_pos_long > window_size  # [B, H, T]

        abs_idx = torch.arange(t, device=device, dtype=torch.long).view(1, 1, t).expand(b, h, t)
        shift_w = (abs_idx - window_size).clamp(min=0)
        shift_wm1 = (abs_idx - window_size + 1).clamp(min=0) if window_size > 0 else shift_w

        # At positions where beyond_window=False, shift_w/shift_wm1 may cross
        # document boundaries in seg_log_retain_prefix, producing meaningless
        # values that become inf after exp.  Although the forward is protected
        # by torch.where, the backward computes 0*inf=NaN.  Fix: gather from
        # abs_idx (self) at those positions so the prefix difference is 0.
        safe_shift_w = torch.where(beyond_window, shift_w, abs_idx)
        safe_shift_wm1 = torch.where(beyond_window, shift_wm1, abs_idx)

        prefix_at_query = seg_log_retain_prefix  # [B, H, T]
        prefix_at_shift_w = torch.gather(seg_log_retain_prefix, -1, safe_shift_w)
        log_prod_window = prefix_at_query - prefix_at_shift_w
        prod_window = torch.exp(log_prod_window)

        if window_size == 0:
            prod_window_minus_one = torch.ones_like(prod_window)
        else:
            prefix_at_shift_wm1 = torch.gather(seg_log_retain_prefix, -1, safe_shift_wm1)
            log_prod_window_minus_one = prefix_at_query - prefix_at_shift_wm1
            prod_window_minus_one = torch.exp(log_prod_window_minus_one)

        strict_vals = torch.gather(strict_normal_BxHxT, -1, safe_shift_w)
        outside_accesses = prod_window * strict_vals - prod_window_minus_one
        normal_accesses_BxHxT = normal_accesses_BxHxT + torch.where(
            beyond_window, outside_accesses, torch.zeros_like(outside_accesses)
        )
    else:
        retain_log_prefix_BxHxTp1 = torch.zeros((b, h, t + 1), device=device, dtype=dtype)
        retain_log_prefix_BxHxTp1[:, :, 1:] = torch.cumsum(log_retain, dim=-1)

        all_idx = torch.arange(t, device=device)
        mask = all_idx > window_size
        if mask.any():
            query_idx = all_idx[mask]
            strict_idx = query_idx - window_size

            log_prod_window = (
                retain_log_prefix_BxHxTp1[:, :, query_idx]
                - retain_log_prefix_BxHxTp1[:, :, query_idx - window_size]
            )
            prod_window = torch.exp(log_prod_window)

            if window_size == 0:
                prod_window_minus_one = torch.ones_like(prod_window)
            else:
                log_prod_window_minus_one = (
                    retain_log_prefix_BxHxTp1[:, :, query_idx]
                    - retain_log_prefix_BxHxTp1[:, :, query_idx - window_size + 1]
                )
                prod_window_minus_one = torch.exp(log_prod_window_minus_one)

            outside_accesses = prod_window * strict_normal_BxHxT[:, :, strict_idx] - prod_window_minus_one
            outside_BxHxT = torch.zeros((b, h, t), device=device, dtype=dtype)
            scatter_idx = query_idx.view(1, 1, -1).expand(b, h, -1)
            outside_BxHxT = outside_BxHxT.scatter(-1, scatter_idx, outside_accesses)
            normal_accesses_BxHxT = normal_accesses_BxHxT + outside_BxHxT

    total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
    return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT


class _StrictNormalSTE(torch.autograd.Function):
    """
    Differentiable "since-last-beacon" normal-access count.

    Forward (identical to segmented_cumsum of ones with reset at beacons):
        strict_normal[i] = 1 + retain_shifted[i] * strict_normal[i-1]
    where retain_shifted[i] = 1 - alpha[i-1] (shifted so position i resets
    when the beacon at i-1 fires).

    Backward provides STE-compatible gradients:
        dL/d(retain_shifted[k]) = strict_normal[k-1]
                                  * reverse_segment_sum(dL/d(strict_normal), k)
    which correctly captures "firing a beacon at k reduces downstream
    strict_normal values by strict_normal[k]".
    """

    @staticmethod
    def forward(ctx, retain_shifted: Tensor, is_reset: Tensor) -> Tensor:
        # is_reset: True where retain_shifted should act as 0 (doc start or beacon).
        # Forward uses segmented_cumsum of ones for speed.
        ones = torch.ones_like(retain_shifted)
        strict_normal = segmented_cumsum(ones, is_reset)
        ctx.save_for_backward(retain_shifted, strict_normal, is_reset)
        return strict_normal

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None]:
        retain_shifted, strict_normal, is_reset = ctx.saved_tensors
        T = retain_shifted.shape[-1]
        if T <= 1:
            return torch.zeros_like(retain_shifted), None

        # Reverse segmented cumsum of grad_output within segments.
        # Segment ends in forward correspond to segment starts in reverse.
        is_segment_end = torch.zeros_like(is_reset)
        is_segment_end[..., -1] = True
        is_segment_end[..., :-1] = is_reset[..., 1:]
        suffix_sum = segmented_cumsum(
            grad_output.flip(-1), is_segment_end.flip(-1)
        ).flip(-1)

        # dL/d(retain_shifted[k]) = strict_normal[k-1] * suffix_sum[k]
        shifted_sn = torch.zeros_like(strict_normal)
        shifted_sn[..., 1:] = strict_normal[..., :-1]
        grad_retain = shifted_sn * suffix_sum
        return grad_retain, None


def _strict_normal_ste(
    alphas_BxHxT: Tensor,
    is_doc_start_BxHxT: Tensor | None,
) -> Tensor:
    """
    Compute since-last-beacon strict_normal with STE gradients through alpha.
    Works for both train (STE hard alphas) and eval (hard alphas).
    """
    b, h, t = alphas_BxHxT.shape
    dtype = alphas_BxHxT.dtype
    device = alphas_BxHxT.device

    retain = 1.0 - alphas_BxHxT  # STE: forward 0/1, backward sigmoid'
    retain_shifted = torch.cat(
        [torch.ones(b, h, 1, device=device, dtype=dtype), retain[:, :, :-1]],
        dim=-1,
    )
    # Reset at doc boundaries (force retain=0 → strict_normal resets to 1).
    if is_doc_start_BxHxT is not None:
        retain_shifted = torch.where(
            is_doc_start_BxHxT,
            torch.zeros_like(retain_shifted),
            retain_shifted,
        )
    # is_reset marks positions where a new segment starts
    # (beacon fired at previous position, or doc boundary).
    is_reset = retain_shifted < 0.5
    if is_doc_start_BxHxT is None:
        is_reset[:, :, 0] = True
    return _StrictNormalSTE.apply(retain_shifted, is_reset)


def _soft_strict_normal(
    alphas_BxHxT: Tensor,
    is_doc_start_BxHxT: Tensor | None,
) -> Tensor:
    """
    Compute since-last-beacon strict_normal with fully differentiable soft alphas.

    Recurrence: strict_normal[i] = 1 + (1 - alpha[i-1]) * strict_normal[i-1],
    with resets to 1 at document boundaries.

    Implemented via a parallel log-space scan:
      strict_i = exp(prefix_i) * cumsum_j<=i exp(-prefix_j),
    where prefix_i = sum_{m < i} log(1 - alpha_m) (segmented by documents).
    This avoids the O(T) Python loop and is significantly faster on long sequences.
    """
    b, h, t = alphas_BxHxT.shape
    if t == 0:
        return alphas_BxHxT
    out_dtype = alphas_BxHxT.dtype

    # Use float64 for the log-space scan to avoid catastrophic cancellation when
    # long runs push prefixes to very large magnitudes (e.g., |prefix| ~ 1e5).
    # In float32 this can saturate strict_normal to its clamp bound spuriously.
    # We cast back to the original dtype at the end.
    # Clamp retain away from 0 to keep log finite and numerically stable.
    retain = torch.clamp(1.0 - alphas_BxHxT, min=1e-12).double()
    log_retain = torch.log(retain)

    if is_doc_start_BxHxT is not None:
        seg_log_retain_cs = segmented_cumsum(log_retain, is_doc_start_BxHxT)
        # Exclusive segmented prefix: sum_{m < i} log_retain[m]
        seg_log_prefix = seg_log_retain_cs - log_retain
        seg_cum_neg_log = segmented_logcumsumexp(-seg_log_prefix, is_doc_start_BxHxT)
        log_strict_normal = seg_log_prefix + seg_cum_neg_log
    else:
        # Exclusive prefix: prefix[i] = sum_{m < i} log_retain[m]
        log_prefix = torch.zeros_like(log_retain)
        if t > 1:
            log_prefix[:, :, 1:] = torch.cumsum(log_retain[:, :, :-1], dim=-1)
        cum_neg_log = torch.logcumsumexp(-log_prefix, dim=-1)
        log_strict_normal = log_prefix + cum_neg_log

    # strict_normal is theoretically bounded by document length (<= t).
    log_strict_normal = log_strict_normal.clamp(max=math.log(float(t) + 1.0))
    strict_normal = torch.exp(log_strict_normal).to(out_dtype)
    return strict_normal


def compute_soft_memory_accesses_BxHxT(
    alphas_BxHxT: Tensor,
    window_size: int,
    can_see_since_last_beacon: bool,
    apply_minimum_window_normals: bool,
    is_doc_start_BxT: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Soft (fully differentiable) memory-access metric for training.

    Uses continuous sigmoid alphas — no hard thresholding, no STE.
    Smooth gradients flow directly through sigmoid to beacon logits.
    """
    if alphas_BxHxT.dim() != 3:
        raise ValueError(f"Expected rank-3, got {alphas_BxHxT.shape}")
    b, h, t = alphas_BxHxT.shape
    if t == 0:
        empty = alphas_BxHxT
        return empty, empty, empty

    dtype = alphas_BxHxT.dtype
    device = alphas_BxHxT.device

    have_doc_info = is_doc_start_BxT is not None
    if have_doc_info:
        is_doc_start_BxHxT = is_doc_start_BxT.unsqueeze(1).expand(b, h, t)

    # --- Beacon accesses: cumsum of soft alphas ---
    if t > 1:
        shifted = torch.cat(
            (torch.zeros((b, h, 1), device=device, dtype=dtype), alphas_BxHxT[:, :, :-1]),
            dim=-1,
        )
        if have_doc_info:
            shifted = torch.where(is_doc_start_BxHxT, torch.zeros_like(shifted), shifted)
            beacon_accesses_BxHxT = segmented_cumsum(shifted, is_doc_start_BxHxT)
        else:
            beacon_accesses_BxHxT = torch.cumsum(shifted, dim=-1)
    else:
        beacon_accesses_BxHxT = torch.zeros_like(alphas_BxHxT)

    # --- Normal accesses ---
    if not can_see_since_last_beacon:
        if have_doc_info:
            doc_pos = doc_relative_positions(is_doc_start_BxT).unsqueeze(1).expand(b, h, t).to(dtype)
            normal_accesses_BxHxT = (doc_pos + 1.0).clamp(max=float(window_size + 1))
        else:
            positions = torch.arange(t, device=device, dtype=dtype).view(1, 1, t)
            normal_accesses_BxHxT = (positions + 1.0).clamp(max=float(window_size + 1)).expand(b, h, -1).contiguous()
        total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
        return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT

    # --- Since-last: differentiable recurrence via log-space parallel scan ---
    strict_normal_BxHxT = _soft_strict_normal(
        alphas_BxHxT,
        is_doc_start_BxHxT if have_doc_info else None,
    )

    if not apply_minimum_window_normals:
        normal_accesses_BxHxT = strict_normal_BxHxT
    else:
        if have_doc_info:
            doc_pos = doc_relative_positions(is_doc_start_BxT).unsqueeze(1).expand(b, h, t).to(dtype)
            base_window_BxHxT = (doc_pos + 1.0).clamp(max=float(window_size + 1))
        else:
            positions = torch.arange(t, device=device, dtype=dtype).view(1, 1, t)
            base_window_BxHxT = (positions + 1.0).clamp(max=float(window_size + 1))
        normal_accesses_BxHxT = torch.maximum(
            base_window_BxHxT.expand(b, h, -1),
            strict_normal_BxHxT,
        )

    total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
    return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT


def compute_hard_memory_accesses_BxHxT(
    alphas_BxHxT: Tensor,
    window_size: int,
    can_see_since_last_beacon: bool,
    apply_minimum_window_normals: bool,
    is_doc_start_BxT: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Hard memory-access metric usable for both training and eval.

    Expects alphas that are already 0/1 in forward (STE during training,
    hard at eval).  Beacon-accesses use cumsum (differentiable
    through STE).  Since-last normal-accesses use a custom autograd Function
    that gives exact hard counts in forward and STE gradients in backward.

    Returns:
    - total_accesses_BxHxT
    - normal_accesses_BxHxT
    - beacon_accesses_BxHxT
    """
    if alphas_BxHxT.dim() != 3:
        raise ValueError(f"Expected alphas_BxHxT to be rank-3, got {alphas_BxHxT.shape}")
    b, h, t = alphas_BxHxT.shape
    if t == 0:
        empty = alphas_BxHxT
        return empty, empty, empty

    dtype = alphas_BxHxT.dtype
    device = alphas_BxHxT.device

    # Prepare doc-aware flag expanded to [B, H, T]
    have_doc_info = is_doc_start_BxT is not None
    if have_doc_info:
        is_doc_start_BxHxT = is_doc_start_BxT.unsqueeze(1).expand(b, h, t)

    # --- Beacon accesses: cumsum of alphas from doc_start to i-1 ---
    if t > 1:
        shifted = torch.cat(
            (torch.zeros((b, h, 1), device=device, dtype=dtype), alphas_BxHxT[:, :, :-1]),
            dim=-1,
        )
        if have_doc_info:
            shifted = torch.where(is_doc_start_BxHxT, torch.zeros_like(shifted), shifted)
            beacon_accesses_BxHxT = segmented_cumsum(shifted, is_doc_start_BxHxT)
        else:
            beacon_accesses_BxHxT = torch.cumsum(shifted, dim=-1)
    else:
        beacon_accesses_BxHxT = torch.zeros_like(alphas_BxHxT)

    # --- Normal accesses ---
    if not can_see_since_last_beacon:
        if have_doc_info:
            doc_pos = doc_relative_positions(is_doc_start_BxT).unsqueeze(1).expand(b, h, t).to(dtype)
            normal_accesses_BxHxT = (doc_pos + 1.0).clamp(max=float(window_size + 1))
        else:
            positions = torch.arange(t, device=device, dtype=dtype).view(1, 1, t)
            normal_accesses_BxHxT = (positions + 1.0).clamp(max=float(window_size + 1)).expand(b, h, -1).contiguous()
        total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
        return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT

    # --- Since-last-beacon: differentiable recurrence via custom backward ---
    strict_normal_BxHxT = _strict_normal_ste(
        alphas_BxHxT,
        is_doc_start_BxHxT if have_doc_info else None,
    )

    if not apply_minimum_window_normals:
        normal_accesses_BxHxT = strict_normal_BxHxT
    else:
        if have_doc_info:
            doc_pos = doc_relative_positions(is_doc_start_BxT).unsqueeze(1).expand(b, h, t).to(dtype)
            base_window_BxHxT = (doc_pos + 1.0).clamp(max=float(window_size + 1))
        else:
            positions = torch.arange(t, device=device, dtype=dtype).view(1, 1, t)
            base_window_BxHxT = (positions + 1.0).clamp(max=float(window_size + 1))
        normal_accesses_BxHxT = torch.maximum(
            base_window_BxHxT.expand(b, h, -1),
            strict_normal_BxHxT,
        )

    total_accesses_BxHxT = normal_accesses_BxHxT + beacon_accesses_BxHxT
    return total_accesses_BxHxT, normal_accesses_BxHxT, beacon_accesses_BxHxT


def attn_scores_mod_for_soft_beacons_factory(
    alphas_BxHxT: Tensor,
    apply_minimum_window_normals: bool,
    window_size: int,
    can_see_since_last_beacon: bool,
) -> callable:
    """
    Factory for attention scores modifier for soft beacons.
    apply_minimum_window_normals and can_see_since_last_beacon cannot be both True. If both
    """
    b, h, t = alphas_BxHxT.size()

    # alpha=1 (beacon exists): log(1) = 0 — visible.
    # alpha=0 (no beacon): -inf — truly invisible, no finite-floor leak.
    log_alphas_BxHxT = torch.where(
        alphas_BxHxT > 0.5,
        torch.zeros_like(alphas_BxHxT),
        torch.tensor(float("-inf"), dtype=alphas_BxHxT.dtype, device=alphas_BxHxT.device),
    )
    # Work around torch-inductor FlexAttention limitation in backward:
    # indexing the same grad-requiring tensor multiple times inside score_mod
    # is not supported. Use cloned tensors for the second index path.
    log_alphas_for_k_BxHxT = log_alphas_BxHxT.clone()

    # Only compute prefix sums when needed (O(T) instead of O(T^2))
    if can_see_since_last_beacon:
        prefix_log_BxHxT = compute_log_prob_no_beacon_prefix_BxHxT(alphas_BxHxT)
        prefix_log_for_k_BxHxT = prefix_log_BxHxT.clone()

    def attn_scores_mod_for_soft_beacons(
        score: Tensor,
        batch: Tensor,
        head: Tensor,
        q_idx: Tensor,
        k_idx: Tensor
    ) -> Tensor:
        """
        Modifier for attention scores to handle soft beacons.
        score is a scalar tensor in flex_attention's API.
        """
        # Document masking: mask cross-document attention

        k_idx_in_T = k_idx // 2
        q_idx_in_T = q_idx // 2

        is_beacon = (k_idx % 2 == 1)

        # Beacon key: add log alpha (if alpha~=0, this adds -inf effectively removing the beacon)
        beacon_score = score + log_alphas_for_k_BxHxT[batch, head, k_idx_in_T]

        # Normal token key: depends on config
        if can_see_since_last_beacon:
            # log_prob_no_beacon(k, q) = prefix[q] - prefix[k]
            gated_score = score + prefix_log_BxHxT[batch, head, q_idx_in_T] - prefix_log_for_k_BxHxT[batch, head, k_idx_in_T]
            if apply_minimum_window_normals:
                k_out_of_sliding_window = (q_idx - k_idx) // 2 > window_size
                normal_score = torch.where(k_out_of_sliding_window, gated_score, score)
            else:
                normal_score = gated_score
        else:
            k_out_of_sliding_window = (q_idx - k_idx) // 2 > window_size
            normal_score = torch.where(k_out_of_sliding_window, float('-inf'), score)

        result = torch.where(is_beacon, beacon_score, normal_score)
        return result

    return attn_scores_mod_for_soft_beacons
