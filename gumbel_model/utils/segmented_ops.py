"""
Segmented (document-aware) primitives for efficiency computation.

All ops work on the last dimension and reset at document boundaries.
"""

from __future__ import annotations

import torch
from torch import Tensor


def is_doc_start_from_doc_idx(documents_idx: Tensor) -> Tensor:
    """
    Convert document index tensor to boolean doc-start indicator.

    Position 0 is always a doc start. Position i is a doc start if
    documents_idx[..., i] != documents_idx[..., i-1].

    Args:
        documents_idx: [..., T] integer document indices

    Returns:
        [..., T] bool tensor, True at document starts
    """
    is_start = torch.zeros_like(documents_idx, dtype=torch.bool)
    is_start[..., 0] = True
    if documents_idx.shape[-1] > 1:
        is_start[..., 1:] = documents_idx[..., 1:] != documents_idx[..., :-1]
    return is_start


def _last_doc_start_index(is_doc_start: Tensor) -> Tensor:
    """
    For each position, find the index of the most recent doc start (inclusive).

    Uses cummax on masked indices: doc-start positions get their own index,
    non-doc-start positions get -inf so cummax propagates the last start.

    Args:
        is_doc_start: [..., T] bool tensor

    Returns:
        [..., T] long tensor of indices into the last dimension
    """
    T = is_doc_start.shape[-1]
    device = is_doc_start.device
    idx = torch.arange(T, device=device, dtype=torch.long)
    # Broadcast idx to match is_doc_start shape
    shape = [1] * (is_doc_start.dim() - 1) + [T]
    idx = idx.view(shape).expand_as(is_doc_start)
    # At non-doc-start positions, set to -1 so cummax ignores them
    masked_idx = torch.where(is_doc_start, idx, torch.full_like(idx, -1))
    last_start, _ = masked_idx.cummax(dim=-1)
    return last_start


def segmented_cumsum(x: Tensor, is_doc_start: Tensor) -> Tensor:
    """
    Cumulative sum along the last dimension that resets at each document boundary.

    Algorithm:
    1. Compute global cumsum
    2. At each doc start, record the cumsum value just before (i.e., global_cs[i] - x[i])
    3. Use cummax on doc-start indices to find each position's doc-start
    4. Gather the reset offset from that index
    5. Subtract the offset

    Args:
        x: [..., T] values to cumsum
        is_doc_start: [..., T] bool tensor, True at document starts

    Returns:
        [..., T] segmented cumsum
    """
    global_cs = torch.cumsum(x, dim=-1)
    # offset_at_start[i] = global_cs[i] - x[i] at doc starts, else 0
    # This is the cumsum value from before the current doc segment.
    offset_at_start = torch.where(is_doc_start, global_cs - x, torch.zeros_like(global_cs))
    # For each position, find the index of the last doc start
    last_start = _last_doc_start_index(is_doc_start)
    # Gather the offset from the last doc start
    offset = torch.gather(offset_at_start, -1, last_start)
    return global_cs - offset


def doc_relative_positions(is_doc_start: Tensor) -> Tensor:
    """
    Return 0-indexed position within each document.

    Position i gets value `i - last_doc_start_index(i)`.

    Args:
        is_doc_start: [..., T] bool tensor

    Returns:
        [..., T] long tensor of relative positions (0 at each doc start)
    """
    T = is_doc_start.shape[-1]
    device = is_doc_start.device
    idx = torch.arange(T, device=device, dtype=torch.long)
    shape = [1] * (is_doc_start.dim() - 1) + [T]
    idx = idx.view(shape).expand_as(is_doc_start)
    last_start = _last_doc_start_index(is_doc_start)
    return idx - last_start


def segmented_logcumsumexp(x: Tensor, is_doc_start: Tensor) -> Tensor:
    """
    Log-cumulative-sum-exp along the last dimension, resetting at doc boundaries.

    Uses a vectorized separator-offset trick in float64 with a dynamic per-row
    separator magnitude derived from the detached value range. This suppresses
    cross-document leakage while avoiding Python loops and in-place recurrence
    updates that can fail under torch.compile/AOT autograd.

    Args:
        x: [..., T] values
        is_doc_start: [..., T] bool tensor

    Returns:
        [..., T] segmented logcumsumexp (same dtype as input)
    """
    if x.shape != is_doc_start.shape:
        raise ValueError(f"x and is_doc_start must have same shape, got {x.shape} vs {is_doc_start.shape}")

    t = x.shape[-1]
    if t == 0:
        return x

    # Vectorized separator trick (no Python loop).
    # We choose a per-row separator magnitude from detached value range so that
    # contributions from previous documents are suppressed to ~exp(-50) or less.
    # Float64 arithmetic keeps separator add/subtract numerically stable.
    orig_dtype = x.dtype
    x_f64 = x.double()

    doc_id = torch.cumsum(is_doc_start.long(), dim=-1) - 1  # [..., T]
    max_doc_id = doc_id[..., -1:]  # [..., 1]

    x_max = x_f64.detach().amax(dim=-1, keepdim=True)
    x_min = x_f64.detach().amin(dim=-1, keepdim=True)
    range_width = (x_max - x_min).clamp_min(1.0)
    big_neg = -(range_width + 50.0)

    separator = (max_doc_id - doc_id).double() * big_neg
    shifted = x_f64 + separator
    global_lcse = torch.logcumsumexp(shifted, dim=-1)
    result = global_lcse - separator
    return result.to(orig_dtype)


def per_document_mean(values_BxT: Tensor, documents_idx_BxT: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute the mean of values within each document.

    Uses scatter_add to sum values per document, then divides by document length.

    Args:
        values_BxT: [B, T] values to average (must be float and require grad if
                    you want gradients to flow through)
        documents_idx_BxT: [B, T] integer document indices (0-indexed per batch element)

    Returns:
        doc_means: [N_docs] mean value per document (flattened across batch)
        doc_mask: [N_docs] bool, True for documents that actually exist
    """
    B, T = values_BxT.shape
    device = values_BxT.device

    # Compile-friendly upper bound on docs per sequence (avoid Tensor.item() graph break).
    # document_idx is monotonic and starts at 0, so true max docs <= T.
    max_doc_id = T
    batch_offset = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * max_doc_id
    flat_doc_idx = (batch_offset + documents_idx_BxT).reshape(-1)  # [B*T]

    num_slots = B * max_doc_id
    flat_values = values_BxT.reshape(-1)  # [B*T]

    doc_sum = torch.zeros(num_slots, device=device, dtype=flat_values.dtype)
    doc_sum.scatter_add_(0, flat_doc_idx, flat_values)

    doc_count = torch.zeros(num_slots, device=device, dtype=torch.long)
    doc_count.scatter_add_(0, flat_doc_idx, torch.ones(B * T, device=device, dtype=torch.long))

    doc_mask = doc_count > 0
    doc_means = doc_sum / doc_count.clamp(min=1).to(flat_values.dtype)

    return doc_means, doc_mask


def per_document_clamped_excess(
    rate_BxT: Tensor,
    documents_idx_BxT: Tensor,
    target: float,
    penalty: str = "hinge",
) -> tuple[Tensor, Tensor]:
    """
    Compute length-weighted efficiency loss with per-document clamping.

    Algorithm:
        1. For each document d, compute its mean rate:
              doc_rate[d] = sum(rate[positions in d]) / len(d)
        2. Clamp excess per document:
              doc_excess[d] = max(doc_rate[d] - target, 0)
        3. Return length-weighted average of excesses:
              loss = sum(doc_excess[d] * len(d)) / sum(len(d))
           and the length-weighted mean rate (for reporting):
              rate = sum(doc_rate[d] * len(d)) / sum(len(d))

    This ensures:
        - Each document is clamped independently (no cross-doc subsidizing)
        - Long documents contribute proportionally more (no short-doc inflation)

    Args:
        rate_BxT: [B, T] per-position rate values
        documents_idx_BxT: [B, T] integer document indices (0-indexed per batch element)
        target: target memory access rate
        penalty: one of {"hinge", "abs"}

    Returns:
        weighted_excess: scalar, length-weighted mean of clamped per-doc excesses
        weighted_rate: scalar, length-weighted mean of per-doc rates (for reporting)
    """
    B, T = rate_BxT.shape
    device = rate_BxT.device

    # Compile-friendly upper bound on docs per sequence (avoid Tensor.item() graph break).
    max_doc_id = T
    batch_offset = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * max_doc_id
    flat_doc_idx = (batch_offset + documents_idx_BxT).reshape(-1)  # [B*T]

    num_slots = B * max_doc_id
    flat_rate = rate_BxT.reshape(-1)

    doc_sum = torch.zeros(num_slots, device=device, dtype=flat_rate.dtype)
    doc_sum.scatter_add_(0, flat_doc_idx, flat_rate)

    doc_count = torch.zeros(num_slots, device=device, dtype=torch.long)
    doc_count.scatter_add_(0, flat_doc_idx, torch.ones(B * T, device=device, dtype=torch.long))

    doc_mask = doc_count > 0
    doc_len = doc_count[doc_mask].float()
    doc_rate = doc_sum[doc_mask] / doc_len

    delta = doc_rate - target
    if penalty == "hinge":
        doc_excess = torch.clamp(delta, min=0.0)
    elif penalty == "abs":
        doc_excess = torch.abs(delta)
    else:
        raise ValueError(f"Unknown penalty={penalty!r}. Expected one of {{'hinge', 'abs'}}.")

    total_len = doc_len.sum()
    weighted_excess = (doc_excess * doc_len).sum() / total_len
    weighted_rate = (doc_rate * doc_len).sum() / total_len

    return weighted_excess, weighted_rate


def masked_per_document_clamped_excess(
    rate_BxT: Tensor,
    documents_idx_BxT: Tensor,
    mask_BxT: Tensor,
    target: float,
    penalty: str = "hinge",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Masked variant of per_document_clamped_excess.

    Positions where mask_BxT is False are excluded entirely from both the
    document means and the final weighted aggregation.
    """
    B, T = rate_BxT.shape
    device = rate_BxT.device
    mask_BxT = mask_BxT.to(dtype=torch.bool, device=device)

    if not mask_BxT.any():
        zero = rate_BxT.new_zeros(())
        return zero, zero, zero, zero

    max_doc_id = T
    batch_offset = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * max_doc_id
    flat_doc_idx = (batch_offset + documents_idx_BxT).reshape(-1)

    num_slots = B * max_doc_id
    flat_rate = rate_BxT.reshape(-1)
    flat_mask = mask_BxT.reshape(-1)
    flat_mask_f = flat_mask.to(flat_rate.dtype)

    doc_sum = torch.zeros(num_slots, device=device, dtype=flat_rate.dtype)
    doc_sum.scatter_add_(0, flat_doc_idx, flat_rate * flat_mask_f)

    doc_count = torch.zeros(num_slots, device=device, dtype=torch.long)
    doc_count.scatter_add_(0, flat_doc_idx, flat_mask.to(torch.long))

    doc_mask = doc_count > 0
    doc_len = doc_count[doc_mask].float()
    doc_rate = doc_sum[doc_mask] / doc_len.clamp(min=1)

    delta = doc_rate - target
    if penalty == "hinge":
        doc_excess = torch.clamp(delta, min=0.0)
    elif penalty == "abs":
        doc_excess = torch.abs(delta)
    else:
        raise ValueError(f"Unknown penalty={penalty!r}. Expected one of {{'hinge', 'abs'}}.")

    weighted_rate_den = doc_len.sum()
    weighted_rate_num = (doc_rate * doc_len).sum()
    weighted_excess = (doc_excess * doc_len).sum() / weighted_rate_den.clamp(min=1)
    weighted_rate = weighted_rate_num / weighted_rate_den.clamp(min=1)

    return weighted_excess, weighted_rate, weighted_rate_num, weighted_rate_den


def global_clamped_excess(
    rate_BxT: Tensor,
    target: float,
    penalty: str = "hinge",
) -> tuple[Tensor, Tensor]:
    """
    Compute global clamped excess, ignoring document boundaries.

    Args:
        rate_BxT: [B, T] per-position rate values
        target: target memory access rate
        penalty: one of {"hinge", "abs"}

    Returns:
        excess: scalar, clamp(mean(rate_BxT) - target, min=0)
        rate: scalar, mean(rate_BxT)
    """
    rate = rate_BxT.mean()
    delta = rate - target
    if penalty == "hinge":
        excess = torch.clamp(delta, min=0.0)
    elif penalty == "abs":
        excess = torch.abs(delta)
    else:
        raise ValueError(f"Unknown penalty={penalty!r}. Expected one of {{'hinge', 'abs'}}.")
    return excess, rate


def masked_global_clamped_excess(
    rate_BxT: Tensor,
    mask_BxT: Tensor,
    target: float,
    penalty: str = "hinge",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Masked variant of global_clamped_excess.
    """
    mask_BxT = mask_BxT.to(dtype=torch.bool, device=rate_BxT.device)
    if not mask_BxT.any():
        zero = rate_BxT.new_zeros(())
        return zero, zero, zero, zero

    mask_f = mask_BxT.to(rate_BxT.dtype)
    rate_num = (rate_BxT * mask_f).sum()
    rate_den = mask_f.sum()
    rate = rate_num / rate_den.clamp(min=1)
    delta = rate - target
    if penalty == "hinge":
        excess = torch.clamp(delta, min=0.0)
    elif penalty == "abs":
        excess = torch.abs(delta)
    else:
        raise ValueError(f"Unknown penalty={penalty!r}. Expected one of {{'hinge', 'abs'}}.")
    return excess, rate, rate_num, rate_den


def global_margin_clamped_excess(
    access_BxT: Tensor,
    baseline_BxT: Tensor,
    target: float,
    penalty: str = "hinge",
) -> tuple[Tensor, Tensor]:
    """
    Compute global margin-style excess from sums, avoiding per-position ratios.

    excess = clamp(access_total - target * baseline_total, min=0) / baseline_total
    rate = access_total / baseline_total

    Args:
        access_BxT: [B, T] expected accesses per position
        baseline_BxT: [B, T] baseline accesses per position
        target: target memory access rate
        penalty: one of {"hinge", "abs"}

    Returns:
        excess: scalar normalized clamped excess
        rate: scalar global ratio of sums
    """
    access_total = access_BxT.sum()
    baseline_total = baseline_BxT.sum().clamp(min=1e-12)
    rate = access_total / baseline_total
    delta = access_total - target * baseline_total
    if penalty == "hinge":
        excess = torch.clamp(delta, min=0.0) / baseline_total
    elif penalty == "abs":
        excess = torch.abs(delta) / baseline_total
    else:
        raise ValueError(f"Unknown penalty={penalty!r}. Expected one of {{'hinge', 'abs'}}.")
    return excess, rate


def masked_global_margin_clamped_excess(
    access_BxT: Tensor,
    baseline_BxT: Tensor,
    mask_BxT: Tensor,
    target: float,
    penalty: str = "hinge",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Masked variant of global_margin_clamped_excess.
    """
    mask_BxT = mask_BxT.to(dtype=torch.bool, device=access_BxT.device)
    if not mask_BxT.any():
        zero = access_BxT.new_zeros(())
        return zero, zero, zero, zero

    mask_f = mask_BxT.to(access_BxT.dtype)
    access_total = (access_BxT * mask_f).sum()
    baseline_total = (baseline_BxT * mask_f).sum().clamp(min=1e-12)
    rate = access_total / baseline_total
    delta = access_total - target * baseline_total
    if penalty == "hinge":
        excess = torch.clamp(delta, min=0.0) / baseline_total
    elif penalty == "abs":
        excess = torch.abs(delta) / baseline_total
    else:
        raise ValueError(f"Unknown penalty={penalty!r}. Expected one of {{'hinge', 'abs'}}.")
    return excess, rate, access_total, baseline_total


def per_document_count(documents_idx_BxT: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute token count for each document.

    Args:
        documents_idx_BxT: [B, T] integer document indices (0-indexed per batch element)

    Returns:
        doc_count: [N_docs] token count per document (flattened across batch)
        doc_mask: [N_docs] bool, True for documents that actually exist
    """
    B, T = documents_idx_BxT.shape
    device = documents_idx_BxT.device

    # Compile-friendly upper bound on docs per sequence (avoid Tensor.item() graph break).
    max_doc_id = T
    batch_offset = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * max_doc_id
    flat_doc_idx = (batch_offset + documents_idx_BxT).reshape(-1)  # [B*T]

    num_slots = B * max_doc_id
    doc_count = torch.zeros(num_slots, device=device, dtype=torch.long)
    doc_count.scatter_add_(0, flat_doc_idx, torch.ones(B * T, device=device, dtype=torch.long))
    doc_mask = doc_count > 0
    return doc_count, doc_mask


def masked_per_document_count(
    documents_idx_BxT: Tensor,
    mask_BxT: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Compute token counts per document after excluding masked-out positions.
    """
    B, T = documents_idx_BxT.shape
    device = documents_idx_BxT.device
    mask_BxT = mask_BxT.to(dtype=torch.bool, device=device)

    max_doc_id = T
    batch_offset = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1) * max_doc_id
    flat_doc_idx = (batch_offset + documents_idx_BxT).reshape(-1)

    num_slots = B * max_doc_id
    doc_count = torch.zeros(num_slots, device=device, dtype=torch.long)
    doc_count.scatter_add_(0, flat_doc_idx, mask_BxT.reshape(-1).to(torch.long))
    doc_mask = doc_count > 0
    return doc_count, doc_mask
