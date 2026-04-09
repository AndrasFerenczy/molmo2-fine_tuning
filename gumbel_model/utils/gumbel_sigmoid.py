from typing import NamedTuple

import torch
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function_unary


class GumbelSigmoidSample(NamedTuple):
    hard: Tensor
    soft: Tensor
    pre_sigmoid: Tensor


def _clamp_uniform_open_interval(uniforms: Tensor, eps: float) -> Tensor:
    lower = float(eps)
    upper = torch.nextafter(
        torch.ones((), device=uniforms.device, dtype=uniforms.dtype),
        torch.zeros((), device=uniforms.device, dtype=uniforms.dtype),
    )
    uniforms = uniforms.clamp(min=lower)
    return torch.minimum(uniforms, upper)


def _prepare_uniforms(
    logits: Tensor,
    uniforms: Tensor | None,
    eps: float,
) -> Tensor:
    if uniforms is None:
        uniforms = torch.rand(logits.shape, device=logits.device, dtype=torch.float32)
    else:
        if uniforms.shape != logits.shape:
            raise ValueError(
                "uniforms must match logits shape "
                f"{tuple(logits.shape)}, got {tuple(uniforms.shape)}"
            )
        uniforms = uniforms.to(device=logits.device, dtype=torch.float32)
    return _clamp_uniform_open_interval(uniforms, eps)


def gumbel_sigmoid(
    logits: Tensor,
    tau: float = 1.0,
    *,
    stochastic: bool = True,
    eps: float = 1e-10,
    uniforms: Tensor | None = None,
) -> GumbelSigmoidSample:
    r"""
    Sample binary decisions from the Gumbel-Sigmoid distribution.

    Args:
      logits: unnormalized log probabilities (usually from a linear layer)
      tau: strictly positive scalar temperature

    Returns:
      GumbelSigmoidSample with:
        hard: straight-through hard 0/1 sample
        soft: sigmoid(pre_sigmoid)
        pre_sigmoid: sampled or deterministic logit before sigmoid
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(
            gumbel_sigmoid,
            (logits,),
            logits,
            tau=tau,
            stochastic=stochastic,
            eps=eps,
            uniforms=uniforms,
        )

    tau = float(tau)
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if uniforms is not None and not stochastic:
        raise ValueError("uniforms can only be provided when stochastic=True")

    if stochastic:
        # Gumbel-Sigmoid uses the difference of two Gumbels,
        # which is equivalent to sampling from a Logistic distribution.
        # Logic: G1 - G2 + logits
        # Compute in float32 to prevent bfloat16 saturation: torch.rand_like on
        # bfloat16 tensors can produce values that round to exactly 1.0 (~0.4%),
        # causing log(0) = -inf noise and y_soft = 1.0 exactly, which breaks
        # any downstream log(1 - y_soft) computation without capping.
        uniforms_f32 = _prepare_uniforms(logits, uniforms, eps)
        logistic_noise = torch.log(uniforms_f32) - torch.log1p(-uniforms_f32)
        pre_sigmoid = (logits.float() + logistic_noise) / tau
    else:
        pre_sigmoid = logits.float() / tau

    y_soft = torch.sigmoid(pre_sigmoid)
    y_hard = (y_soft > 0.5).float()
    hard = y_hard - y_soft.detach() + y_soft
    return GumbelSigmoidSample(hard=hard, soft=y_soft, pre_sigmoid=pre_sigmoid)
