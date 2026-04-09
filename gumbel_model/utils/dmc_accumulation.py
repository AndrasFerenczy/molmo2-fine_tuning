from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _affine_prefix_scan(a: Tensor, b: Tensor) -> Tensor:
    if a.dim() != 3:
        raise ValueError(f"Expected a to have shape [B, H, T], got {a.shape}")
    if b.shape[:3] != a.shape:
        raise ValueError(f"Expected b leading shape {a.shape}, got {b.shape}")

    a_scan = a.contiguous().clone()
    b_scan = b.contiguous().clone()
    t = a.size(-1)
    offset = 1
    while offset < t:
        prev_a = a_scan.clone()
        prev_b = b_scan.clone()
        mult = prev_a[:, :, offset:]
        for _ in range(prev_b.dim() - prev_a.dim()):
            mult = mult.unsqueeze(-1)
        a_scan[:, :, offset:] = prev_a[:, :, offset:] * prev_a[:, :, :-offset]
        b_scan[:, :, offset:] = prev_b[:, :, offset:] + mult * prev_b[:, :, :-offset]
        offset <<= 1

    return b_scan


def dmc_exact_accumulation_torch(
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    omega: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    alpha_f = alpha.float().contiguous()
    omega_f = omega.float().contiguous()
    k_f = k.float().contiguous()
    v_f = v.float().contiguous()

    weighted_k = omega_f.unsqueeze(-1) * k_f
    weighted_v = omega_f.unsqueeze(-1) * v_f

    K = _affine_prefix_scan(alpha_f, weighted_k)
    V = _affine_prefix_scan(alpha_f, weighted_v)
    z = _affine_prefix_scan(alpha_f, omega_f)

    z_safe = z.unsqueeze(-1).clamp(min=eps)
    k_out = (K / z_safe).to(k.dtype).contiguous()
    v_out = (V / z_safe).to(v.dtype).contiguous()
    return k_out, v_out


if triton is not None:
    @triton.jit
    def _dmc_exact_fwd_kernel(
        k_ptr, v_ptr, alpha_ptr, omega_ptr,
        yk_ptr, yv_ptr, K_ptr, V_ptr, z_ptr,
        BH, T, D, eps,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_db = tl.program_id(1)
        if pid_bh >= BH:
            return

        offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        base_bt = pid_bh * T

        K_prev = tl.zeros([BLOCK_D], dtype=tl.float32)
        V_prev = tl.zeros([BLOCK_D], dtype=tl.float32)
        z_prev = tl.zeros([], dtype=tl.float32)

        for t in tl.range(0, T):
            idx_bt = base_bt + t
            idx_btd = idx_bt * D + offs_d

            a = tl.load(alpha_ptr + idx_bt).to(tl.float32)
            w = tl.load(omega_ptr + idx_bt).to(tl.float32)
            k = tl.load(k_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)

            K_cur = w * k + a * K_prev
            V_cur = w * v + a * V_prev
            z_cur = w + a * z_prev
            inv_z = 1.0 / tl.maximum(z_cur, eps)

            tl.store(K_ptr + idx_btd, K_cur, mask=mask_d)
            tl.store(V_ptr + idx_btd, V_cur, mask=mask_d)
            tl.store(yk_ptr + idx_btd, K_cur * inv_z, mask=mask_d)
            tl.store(yv_ptr + idx_btd, V_cur * inv_z, mask=mask_d)
            if pid_db == 0:
                tl.store(z_ptr + idx_bt, z_cur)

            K_prev = K_cur
            V_prev = V_cur
            z_prev = z_cur

    @triton.jit
    def _dmc_exact_bwd_gz_kernel(
        gyk_ptr, gyv_ptr, K_ptr, V_ptr, z_ptr, alpha_ptr, gz_ptr,
        BH, T, D, eps,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        if pid_bh >= BH:
            return
        base_bt = pid_bh * T
        gz_next = tl.zeros([], dtype=tl.float32)
        for t in tl.range(0, T):
            tr = T - 1 - t
            idx_bt = base_bt + tr
            z_cur = tl.load(z_ptr + idx_bt).to(tl.float32)
            inv_z = 1.0 / tl.maximum(z_cur, eps)

            dot = tl.zeros([], dtype=tl.float32)
            for d0 in tl.range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < D
                idx_btd = idx_bt * D + offs_d

                gyk = tl.load(gyk_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
                gyv = tl.load(gyv_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
                K = tl.load(K_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
                V = tl.load(V_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
                dot += tl.sum(gyk * K + gyv * V, axis=0)

            s = -dot * inv_z * inv_z
            gz_cur = gz_next + s
            tl.store(gz_ptr + idx_bt, gz_cur)

            a = tl.load(alpha_ptr + idx_bt).to(tl.float32)
            gz_next = a * gz_cur

    @triton.jit
    def _dmc_exact_bwd_vec_kernel(
        k_ptr, v_ptr, alpha_ptr, omega_ptr, K_ptr, V_ptr, z_ptr, gz_ptr, gyk_ptr, gyv_ptr,
        gk_ptr, gv_ptr, ga_ptr, gw_ptr,
        BH, T, D, eps,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_db = tl.program_id(1)
        if pid_bh >= BH:
            return
        base_bt = pid_bh * T
        offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        gK_next = tl.zeros([BLOCK_D], dtype=tl.float32)
        gV_next = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t in tl.range(0, T):
            tr = T - 1 - t
            idx_bt = base_bt + tr
            idx_btd = idx_bt * D + offs_d

            z_cur = tl.load(z_ptr + idx_bt).to(tl.float32)
            inv_z = 1.0 / tl.maximum(z_cur, eps)

            gyk = tl.load(gyk_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
            gyv = tl.load(gyv_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
            gK_cur = gK_next + gyk * inv_z
            gV_cur = gV_next + gyv * inv_z

            w = tl.load(omega_ptr + idx_bt).to(tl.float32)
            a = tl.load(alpha_ptr + idx_bt).to(tl.float32)
            k = tl.load(k_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + idx_btd, mask=mask_d, other=0.0).to(tl.float32)

            gk = w * gK_cur
            gv = w * gV_cur
            tl.store(gk_ptr + idx_btd, gk, mask=mask_d)
            tl.store(gv_ptr + idx_btd, gv, mask=mask_d)

            gw_partial = tl.sum(gK_cur * k + gV_cur * v, axis=0)

            K_prev = tl.zeros([BLOCK_D], dtype=tl.float32)
            V_prev = tl.zeros([BLOCK_D], dtype=tl.float32)
            z_prev = tl.zeros([], dtype=tl.float32)
            if tr > 0:
                idx_prev_bt = idx_bt - 1
                idx_prev_btd = idx_prev_bt * D + offs_d
                K_prev = tl.load(K_ptr + idx_prev_btd, mask=mask_d, other=0.0).to(tl.float32)
                V_prev = tl.load(V_ptr + idx_prev_btd, mask=mask_d, other=0.0).to(tl.float32)
                z_prev = tl.load(z_ptr + idx_prev_bt).to(tl.float32)

            ga_partial = tl.sum(gK_cur * K_prev + gV_cur * V_prev, axis=0)
            tl.atomic_add(gw_ptr + idx_bt, gw_partial)
            tl.atomic_add(ga_ptr + idx_bt, ga_partial)

            if pid_db == 0:
                gz_cur = tl.load(gz_ptr + idx_bt).to(tl.float32)
                tl.atomic_add(gw_ptr + idx_bt, gz_cur)
                tl.atomic_add(ga_ptr + idx_bt, gz_cur * z_prev)

            gK_next = a * gK_cur
            gV_next = a * gV_cur


class _DMCExactAccumulationTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k: Tensor, v: Tensor, alpha: Tensor, omega: Tensor, eps: float):
        if triton is None:
            raise RuntimeError("Triton is not available for exact DMC accumulation")
        if not (k.is_cuda and v.is_cuda and alpha.is_cuda and omega.is_cuda):
            raise ValueError("Triton exact accumulation requires CUDA tensors")

        k_f = k.float().contiguous()
        v_f = v.float().contiguous()
        alpha_f = alpha.float().contiguous()
        omega_f = omega.float().contiguous()
        b, h, t, d = k_f.shape
        bh = b * h

        yk_f = torch.empty_like(k_f)
        yv_f = torch.empty_like(v_f)
        K_f = torch.empty_like(k_f)
        V_f = torch.empty_like(v_f)
        z_f = torch.empty((b, h, t), device=k.device, dtype=torch.float32)

        block_d = 128
        grid = (bh, triton.cdiv(d, block_d))
        _dmc_exact_fwd_kernel[grid](
            k_f.view(bh, t, d),
            v_f.view(bh, t, d),
            alpha_f.view(bh, t),
            omega_f.view(bh, t),
            yk_f.view(bh, t, d),
            yv_f.view(bh, t, d),
            K_f.view(bh, t, d),
            V_f.view(bh, t, d),
            z_f.view(bh, t),
            bh, t, d, float(eps),
            BLOCK_D=block_d,
        )

        ctx.eps = float(eps)
        ctx.block_d = block_d
        ctx.save_for_backward(k_f, v_f, alpha_f, omega_f, K_f, V_f, z_f)
        return yk_f, yv_f

    @staticmethod
    def backward(ctx, grad_yk: Tensor, grad_yv: Tensor):
        k_f, v_f, alpha_f, omega_f, K_f, V_f, z_f = ctx.saved_tensors
        eps = ctx.eps
        block_d = ctx.block_d
        b, h, t, d = k_f.shape
        bh = b * h
        gyk_f = grad_yk.float().contiguous()
        gyv_f = grad_yv.float().contiguous()

        gz_f = torch.empty((b, h, t), device=k_f.device, dtype=torch.float32)
        ga_f = torch.zeros((b, h, t), device=k_f.device, dtype=torch.float32)
        gw_f = torch.zeros((b, h, t), device=k_f.device, dtype=torch.float32)
        gk_f = torch.empty_like(k_f)
        gv_f = torch.empty_like(v_f)

        _dmc_exact_bwd_gz_kernel[(bh,)](
            gyk_f.view(bh, t, d),
            gyv_f.view(bh, t, d),
            K_f.view(bh, t, d),
            V_f.view(bh, t, d),
            z_f.view(bh, t),
            alpha_f.view(bh, t),
            gz_f.view(bh, t),
            bh, t, d, float(eps),
            BLOCK_D=block_d,
        )

        _dmc_exact_bwd_vec_kernel[(bh, triton.cdiv(d, block_d))](
            k_f.view(bh, t, d),
            v_f.view(bh, t, d),
            alpha_f.view(bh, t),
            omega_f.view(bh, t),
            K_f.view(bh, t, d),
            V_f.view(bh, t, d),
            z_f.view(bh, t),
            gz_f.view(bh, t),
            gyk_f.view(bh, t, d),
            gyv_f.view(bh, t, d),
            gk_f.view(bh, t, d),
            gv_f.view(bh, t, d),
            ga_f.view(bh, t),
            gw_f.view(bh, t),
            bh, t, d, float(eps),
            BLOCK_D=block_d,
        )

        return gk_f, gv_f, ga_f, gw_f, None


def dmc_exact_accumulation(
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    omega: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    use_triton = (
        triton is not None
        and k.is_cuda and v.is_cuda and alpha.is_cuda and omega.is_cuda
        and k.dim() == 4 and v.dim() == 4 and alpha.dim() == 3 and omega.dim() == 3
        and k.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and v.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and alpha.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and omega.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if use_triton:
        yk_f, yv_f = _DMCExactAccumulationTritonFn.apply(k, v, alpha, omega, float(eps))
        # Preserve higher precision; caller can downcast when desired.
        return yk_f.contiguous(), yv_f.contiguous()
    return dmc_exact_accumulation_torch(k, v, alpha, omega, eps=eps)
