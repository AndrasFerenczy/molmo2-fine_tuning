from __future__ import annotations

from . import core, inference
from .core import (
    GumbelDMSAttention,
    GumbelDMSBlock,
    GumbelDMSConfig,
    GumbelDMSModelBase,
    IGNORE_INDEX,
    ModelConfig,
    triton_keybias_attention,
)
from .inference import GumbelDMSInferenceMixin


class GumbelDMSModel(GumbelDMSInferenceMixin, GumbelDMSModelBase):
    pass


__all__ = [
    "core",
    "inference",
    "GumbelDMSAttention",
    "GumbelDMSBlock",
    "GumbelDMSConfig",
    "GumbelDMSInferenceMixin",
    "GumbelDMSModel",
    "GumbelDMSModelBase",
    "IGNORE_INDEX",
    "ModelConfig",
    "triton_keybias_attention",
]
