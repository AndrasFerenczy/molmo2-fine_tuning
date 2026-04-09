from __future__ import annotations

from . import core, inference, ops
from .core import (
    GumbelBeaconsBlock,
    GumbelBeaconsConfig,
    GumbelBeaconsFlashAttention,
    GumbelBeaconsModelBase,
    IGNORE_INDEX,
    ModelConfig,
    triton_gumbel_sliding_attention,
)
from .inference import GumbelBeaconsInferenceMixin
from .ops import (
    _soft_strict_normal,
    _strict_normal_ste,
    attn_scores_mod_for_soft_beacons_factory,
    build_gumbel_sliding_attn_bias_BxHx2Tx2T,
    compute_expected_memory_accesses_BxHxT,
    compute_hard_memory_accesses_BxHxT,
    compute_log_prob_no_beacon_prefix_BxHxT,
    compute_segment_id_prefix_BxHxT,
    compute_soft_memory_accesses_BxHxT,
)


class GumbelBeaconsModel(GumbelBeaconsInferenceMixin, GumbelBeaconsModelBase):
    pass


__all__ = [
    "core",
    "inference",
    "ops",
    "GumbelBeaconsBlock",
    "GumbelBeaconsConfig",
    "GumbelBeaconsFlashAttention",
    "GumbelBeaconsInferenceMixin",
    "GumbelBeaconsModel",
    "GumbelBeaconsModelBase",
    "IGNORE_INDEX",
    "ModelConfig",
    "_soft_strict_normal",
    "_strict_normal_ste",
    "attn_scores_mod_for_soft_beacons_factory",
    "build_gumbel_sliding_attn_bias_BxHx2Tx2T",
    "compute_expected_memory_accesses_BxHxT",
    "compute_hard_memory_accesses_BxHxT",
    "compute_log_prob_no_beacon_prefix_BxHxT",
    "compute_segment_id_prefix_BxHxT",
    "compute_soft_memory_accesses_BxHxT",
    "triton_gumbel_sliding_attention",
]
