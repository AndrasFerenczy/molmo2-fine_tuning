"""Shared batched masked SDPA for autoregressive decode with variable-length KV segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class KVSegment:
    """A contiguous block of keys/values with per-element validity lengths.

    k, v: [B, H, max_T, D]
    lengths: [B, H] (per-head) or [B] (shared across heads, broadcast).
    """
    k: Tensor
    v: Tensor
    lengths: Tensor


def masked_kv_attention(
    q_BxHxD: Tensor,
    segments: list[KVSegment],
    *,
    extra_kv: list[tuple[Tensor, Tensor]] | None = None,
    attn_dtype: torch.dtype,
    check_finite: bool = False,
    error_context: str = "",
) -> Tensor:
    """Run scaled dot-product attention over variable-length KV segments.

    Assembles retained/window/current KV segments into a single padded tensor
    with a validity mask, then runs F.scaled_dot_product_attention.

    Args:
        q_BxHxD: query tensor [B, H, D].
        segments: list of KVSegment, each contributing a block of keys/values.
        extra_kv: optional list of (k_BxHxD, v_BxHxD) single-token entries
                  that are always valid (e.g. current token being attended to).
        attn_dtype: dtype for the attention computation.
        check_finite: if True, raise on invalid masks or non-finite values.
        error_context: string appended to error messages for debugging.

    Returns:
        out_BxHxD: attention output [B, H, D] in float32.
    """
    b, n_head, head_dim = q_BxHxD.shape
    device = q_BxHxD.device
    n_extra = len(extra_kv) if extra_kv else 0

    seg_maxlens = []
    for seg in segments:
        if seg.lengths.numel() == 0:
            seg_maxlens.append(0)
        else:
            seg_maxlens.append(int(seg.lengths.max().item()))
    total_len = sum(seg_maxlens) + n_extra

    if total_len == 0:
        return q_BxHxD.new_zeros((b, n_head, head_dim), dtype=torch.float32)

    keys = torch.zeros((b, n_head, total_len, head_dim), device=device, dtype=attn_dtype)
    values = torch.zeros_like(keys)
    mask = torch.zeros((b, n_head, total_len), device=device, dtype=torch.bool)

    col = 0
    for seg, max_len in zip(segments, seg_maxlens):
        if max_len == 0:
            continue
        keys[:, :, col:col + max_len, :] = seg.k[:, :, :max_len, :]
        values[:, :, col:col + max_len, :] = seg.v[:, :, :max_len, :]
        pos = torch.arange(max_len, device=device)
        if seg.lengths.dim() == 1:
            # [B] lengths — shared across heads, broadcast
            valid = pos.view(1, max_len) < seg.lengths.unsqueeze(-1)  # [B, max_len]
            mask[:, :, col:col + max_len] = valid.unsqueeze(1)
        else:
            # [B, H] lengths — per-head
            valid = pos.view(1, 1, max_len) < seg.lengths.unsqueeze(-1)  # [B, H, max_len]
            mask[:, :, col:col + max_len] = valid
        col += max_len

    if extra_kv:
        for extra_k, extra_v in extra_kv:
            keys[:, :, col, :] = extra_k
            values[:, :, col, :] = extra_v
            mask[:, :, col] = True
            col += 1

    bh = b * n_head
    q_flat = q_BxHxD.reshape(bh, 1, head_dim)
    k_flat = keys.reshape(bh, total_len, head_dim)
    v_flat = values.reshape(bh, total_len, head_dim)
    mask_flat = mask.reshape(bh, 1, total_len)

    if check_finite:
        if (mask_flat.sum(dim=-1) <= 0).any():
            raise RuntimeError(f"Invalid mask with no valid keys{error_context}")
        if not (torch.isfinite(q_flat).all() and torch.isfinite(k_flat).all() and torch.isfinite(v_flat).all()):
            raise RuntimeError(f"Non-finite QKV{error_context}")

    out = F.scaled_dot_product_attention(
        q_flat, k_flat, v_flat,
        attn_mask=mask_flat,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(head_dim),
    )
    out_BxHxD = out.reshape(b, n_head, head_dim).float()

    if check_finite and not torch.isfinite(out_BxHxD).all():
        raise RuntimeError(f"Non-finite attention output{error_context}")

    return out_BxHxD
