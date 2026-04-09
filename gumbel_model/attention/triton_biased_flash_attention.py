"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

try:
    import pytest
except Exception:
    class _PytestStub:
        @staticmethod
        def parametrize(*args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator
    pytest = _PytestStub()
import torch
import torch.nn.functional as F
import os

import triton
import triton.language as tl
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:
    TensorDescriptor = None

try:
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
except Exception:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_hip():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


def is_cuda():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except Exception:
        return False


def supports_host_descriptor():
    return TensorDescriptor is not None and is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    off_z, off_h, bias_ptr, stride_bz, stride_bh, stride_bm, stride_bn, HAS_BIAS: tl.constexpr,  #
                    document_ptr, stride_doz, stride_dot, HAS_DOCUMENT_MASK: tl.constexpr,  #
                    prefix_ptr, beacon_ptr, stride_pz, stride_ph, stride_pt, stride_az, stride_ah, stride_at,
                    window_size, ZERO_NORMAL_BIAS_IN_WINDOW: tl.constexpr, HAS_GUMBEL_BIAS: tl.constexpr,  #
                    HAS_PREFIX_BIAS: tl.constexpr, USE_EXACT_SEGMENT_MASK: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # Hoist loop-invariant prefix_q load for gumbel bias path.
    if HAS_GUMBEL_BIAS:
        q_tok = offs_m // 2
        if HAS_PREFIX_BIAS:
            prefix_q = tl.load(prefix_ptr + off_z * stride_pz + off_h * stride_ph + q_tok * stride_pt)
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        qk = qk * qk_scale
        offs_n_abs = start_n + offs_n
        if HAS_DOCUMENT_MASK:
            docs_q_ptrs = document_ptr + off_z * stride_doz + offs_m * stride_dot
            docs_k_ptrs = document_ptr + off_z * stride_doz + offs_n_abs * stride_dot
            docs_q = tl.load(docs_q_ptrs)
            docs_k = tl.load(docs_k_ptrs)
            same_doc = docs_q[:, None] == docs_k[None, :]
            qk = qk + tl.where(same_doc, 0.0, -1.0e6)
        if HAS_BIAS:
            bias_ptrs = (
                bias_ptr
                + off_z * stride_bz
                + off_h * stride_bh
                + offs_m[:, None] * stride_bm
                + offs_n_abs[None, :] * stride_bn
            )
            qk += tl.load(bias_ptrs) * 1.44269504
        elif HAS_GUMBEL_BIAS:
            k_tok = offs_n_abs // 2

            if HAS_PREFIX_BIAS:
                prefix_k_ptrs = prefix_ptr + off_z * stride_pz + off_h * stride_ph + k_tok * stride_pt
                prefix_k = tl.load(prefix_k_ptrs)
                if USE_EXACT_SEGMENT_MASK:
                    # Eval path: prefix stores integer segment_id (0, 1, 2, ...).
                    # Same segment → 0 (full visibility). Different segment → -inf (hard mask).
                    # No subtraction of -inf values: equality on exact integers is NaN-free.
                    normal_bias = tl.where(prefix_q[:, None] == prefix_k[None, :], 0.0, float('-inf'))
                else:
                    normal_bias = prefix_q[:, None] - prefix_k[None, :]

                if ZERO_NORMAL_BIAS_IN_WINDOW:
                    rel = q_tok[:, None] - k_tok[None, :]
                    in_window = (rel >= 0) & (rel <= window_size)
                    normal_bias = tl.where(in_window, 0.0, normal_bias)
            else:
                # No since-last-beacon: hard sliding window for normal keys.
                # Without this, normal keys outside the window are NOT masked,
                # making attention effectively full-causal on normal tokens.
                rel = q_tok[:, None] - k_tok[None, :]
                out_of_window = rel > window_size
                normal_bias = tl.where(out_of_window, -1.0e6, 0.0)

            beacon_k_ptrs = beacon_ptr + off_z * stride_az + off_h * stride_ah + k_tok * stride_at
            beacon_bias = tl.load(beacon_k_ptrs)
            k_is_beacon = (offs_n_abs % 2) == 1
            gumbel_bias = tl.where(k_is_beacon[None, :], beacon_bias[None, :], normal_bias)
            # Beacons always self-attend: zero beacon bias on diagonal.
            is_self_beacon = (offs_m[:, None] == offs_n_abs[None, :]) & k_is_beacon[None, :]
            gumbel_bias = tl.where(is_self_beacon, 0.0, gumbel_bias)
            qk += gumbel_bias * 1.44269504
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if TensorDescriptor is None or not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]

    # Filter out configs where BLOCK_M > N_CTX
    # Filter out configs where BLOCK_M < BLOCK_N when causal is True
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, bias_ptr,  #
              stride_bz, stride_bh, stride_bm, stride_bn, N_CTX,  #
              document_ptr, stride_doz, stride_dot,  #
              prefix_ptr, beacon_ptr, stride_pz, stride_ph, stride_pt, stride_az, stride_ah, stride_at,  #
              window_size,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              HAS_BIAS: tl.constexpr,  #
              HAS_DOCUMENT_MASK: tl.constexpr,  #
              HAS_GUMBEL_BIAS: tl.constexpr,  #
              ZERO_NORMAL_BIAS_IN_WINDOW: tl.constexpr,  #
              HAS_PREFIX_BIAS: tl.constexpr,  #
              USE_EXACT_SEGMENT_MASK: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              DTYPE_IS_BF16: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else (tl.bfloat16 if DTYPE_IS_BF16 else tl.float16)
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    # Large finite negative instead of -inf: avoids -inf - -inf = NaN in alpha = exp2(m_i - m_ij)
    # when an entire tile is masked (e.g. all beacon biases = -inf). See triton_keybias_flash_attention.py.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        off_z, off_h, bias_ptr, stride_bz, stride_bh, stride_bm, stride_bn, HAS_BIAS,  #
                                        document_ptr, stride_doz, stride_dot, HAS_DOCUMENT_MASK,  #
                                        prefix_ptr, beacon_ptr, stride_pz, stride_ph, stride_pt, stride_az, stride_ah, stride_at,
                                        window_size, ZERO_NORMAL_BIAS_IN_WINDOW, HAS_GUMBEL_BIAS,  #
                                        HAS_PREFIX_BIAS, USE_EXACT_SEGMENT_MASK,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        off_z, off_h, bias_ptr, stride_bz, stride_bh, stride_bm, stride_bn, HAS_BIAS,  #
                                        document_ptr, stride_doz, stride_dot, HAS_DOCUMENT_MASK,  #
                                        prefix_ptr, beacon_ptr, stride_pz, stride_ph, stride_pt, stride_az, stride_ah, stride_at,
                                        window_size, ZERO_NORMAL_BIAS_IN_WINDOW, HAS_GUMBEL_BIAS,  #
                                        HAS_PREFIX_BIAS, USE_EXACT_SEGMENT_MASK,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   DPREFIX, DBEACON, PREFIX, BEACON,
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   stride_dprefix_t, stride_dbeacon_t, stride_prefix_t, stride_beacon_t,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   window_size, MASK: tl.constexpr, ZERO_NORMAL_BIAS_IN_WINDOW: tl.constexpr, HAS_GUMBEL_BIAS: tl.constexpr,
                   HAS_PREFIX_BIAS: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        if HAS_GUMBEL_BIAS:
            k_tok = offs_n // 2
            q_tok = offs_m // 2
            if HAS_PREFIX_BIAS:
                prefix_k = tl.load(PREFIX + k_tok * stride_prefix_t)
                prefix_q = tl.load(PREFIX + q_tok * stride_prefix_t)
                normal_bias = prefix_q[None, :] - prefix_k[:, None]
                if ZERO_NORMAL_BIAS_IN_WINDOW:
                    rel = q_tok[None, :] - k_tok[:, None]
                    in_window = (rel >= 0) & (rel <= window_size)
                    normal_bias = tl.where(in_window, 0.0, normal_bias)
            else:
                rel = q_tok[None, :] - k_tok[:, None]
                out_of_window = rel > window_size
                normal_bias = tl.where(out_of_window, -1.0e6, 0.0)
            beacon_bias = tl.load(BEACON + k_tok * stride_beacon_t)
            k_is_beacon = (offs_n % 2) == 1
            bias = tl.where(k_is_beacon[:, None], beacon_bias[:, None], normal_bias)
            qkT += bias * 1.44269504
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(do.dtype)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))
        if HAS_GUMBEL_BIAS:
            k_is_beacon = (offs_n % 2) == 1
            k_tok = offs_n // 2

            sum_q_per_k = tl.sum(dsT.to(tl.float32), axis=1)
            sum_q_beacon = tl.where(k_is_beacon, sum_q_per_k, 0.0)
            tl.atomic_add(DBEACON + k_tok * stride_dbeacon_t, sum_q_beacon)

            if HAS_PREFIX_BIAS:
                sum_q_normal = tl.where(k_is_beacon, 0.0, sum_q_per_k)
                tl.atomic_add(DPREFIX + k_tok * stride_dprefix_t, -sum_q_normal)

                sum_k_normal_per_q = tl.sum(
                    tl.where(k_is_beacon[:, None], 0.0, dsT.to(tl.float32)),
                    axis=0,
                )
                q_tok = offs_m // 2
                tl.atomic_add(DPREFIX + q_tok * stride_dprefix_t, sum_k_normal_per_q)
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D, PREFIX, BEACON,
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 stride_prefix_t, stride_beacon_t,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 window_size, MASK: tl.constexpr, ZERO_NORMAL_BIAS_IN_WINDOW: tl.constexpr, HAS_GUMBEL_BIAS: tl.constexpr,
                 HAS_PREFIX_BIAS: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        if HAS_GUMBEL_BIAS:
            q_tok = offs_m // 2
            k_tok = offs_n // 2
            if HAS_PREFIX_BIAS:
                prefix_q = tl.load(PREFIX + q_tok * stride_prefix_t)
                prefix_k = tl.load(PREFIX + k_tok * stride_prefix_t)
                normal_bias = prefix_q[:, None] - prefix_k[None, :]
                if ZERO_NORMAL_BIAS_IN_WINDOW:
                    rel = q_tok[:, None] - k_tok[None, :]
                    in_window = (rel >= 0) & (rel <= window_size)
                    normal_bias = tl.where(in_window, 0.0, normal_bias)
            else:
                rel = q_tok[:, None] - k_tok[None, :]
                out_of_window = rel > window_size
                normal_bias = tl.where(out_of_window, -1.0e6, 0.0)
            beacon_bias = tl.load(BEACON + k_tok * stride_beacon_t)
            k_is_beacon = (offs_n % 2) == 1
            bias = tl.where(k_is_beacon[None, :], beacon_bias[None, :], normal_bias)
            qk += bias * 1.44269504
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(kT.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D, DPREFIX, DBEACON, PREFIX, BEACON,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              stride_dprefix_z, stride_dprefix_h, stride_dprefix_t,  #
              stride_dbeacon_z, stride_dbeacon_h, stride_dbeacon_t,  #
              stride_prefix_z, stride_prefix_h, stride_prefix_t,  #
              stride_beacon_z, stride_beacon_h, stride_beacon_t,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              CAUSAL: tl.constexpr,  #
              WINDOW_SIZE: tl.constexpr,  #
              ZERO_NORMAL_BIAS_IN_WINDOW: tl.constexpr,  #
              HAS_GUMBEL_BIAS: tl.constexpr,  #
              HAS_PREFIX_BIAS: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    dprefix_adj = (stride_dprefix_h * (bhid % H) + stride_dprefix_z * (bhid // H)).to(tl.int64)
    dbeacon_adj = (stride_dbeacon_h * (bhid % H) + stride_dbeacon_z * (bhid // H)).to(tl.int64)
    prefix_adj = (stride_prefix_h * (bhid % H) + stride_prefix_z * (bhid // H)).to(tl.int64)
    beacon_adj = (stride_beacon_h * (bhid % H) + stride_beacon_z * (bhid // H)).to(tl.int64)
    DPREFIX += dprefix_adj
    DBEACON += dbeacon_adj
    PREFIX += prefix_adj
    BEACON += beacon_adj

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv,  #
                                Q, k, v, sm_scale,  #
                                DO,  #
                                M, D, DPREFIX, DBEACON, PREFIX, BEACON,  #
                                stride_tok, stride_d,  #
                                stride_dprefix_t, stride_dbeacon_t, stride_prefix_t, stride_beacon_t,  #
                                H, N_CTX,  #
                                MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                                start_n, start_m, num_steps,  #
                                WINDOW_SIZE,  #
                                MASK=True,  #
                                ZERO_NORMAL_BIAS_IN_WINDOW=ZERO_NORMAL_BIAS_IN_WINDOW,  #
                                HAS_GUMBEL_BIAS=HAS_GUMBEL_BIAS,  #
                                HAS_PREFIX_BIAS=HAS_PREFIX_BIAS,  #
                                )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        DPREFIX, DBEACON, PREFIX, BEACON,  #
        stride_tok, stride_d,  #
        stride_dprefix_t, stride_dbeacon_t, stride_prefix_t, stride_beacon_t,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        WINDOW_SIZE,  #
        MASK=False,  #
        ZERO_NORMAL_BIAS_IN_WINDOW=ZERO_NORMAL_BIAS_IN_WINDOW,  #
        HAS_GUMBEL_BIAS=HAS_GUMBEL_BIAS,  #
        HAS_PREFIX_BIAS=HAS_PREFIX_BIAS,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V,  #
                          do, m, D, PREFIX, BEACON,  #
                          stride_tok, stride_d,  #
                          stride_prefix_t, stride_beacon_t,  #
                          H, N_CTX,  #
                          BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                          start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                          WINDOW_SIZE,  #
                          MASK=True,  #
                          ZERO_NORMAL_BIAS_IN_WINDOW=ZERO_NORMAL_BIAS_IN_WINDOW,  #
                          HAS_GUMBEL_BIAS=HAS_GUMBEL_BIAS,  #
                          HAS_PREFIX_BIAS=HAS_PREFIX_BIAS,  #
                          )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D, PREFIX, BEACON,  #
                      stride_tok, stride_d,  #
                      stride_prefix_t, stride_beacon_t,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, start_n, num_steps,  #
                      WINDOW_SIZE,  #
                      MASK=False,  #
                      ZERO_NORMAL_BIAS_IN_WINDOW=ZERO_NORMAL_BIAS_IN_WINDOW,  #
                      HAS_GUMBEL_BIAS=HAS_GUMBEL_BIAS,  #
                      HAS_PREFIX_BIAS=HAS_PREFIX_BIAS,  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True, bias=None):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        has_bias = bias is not None
        if has_bias:
            assert bias.dim() == 4, "bias must have shape [B, H, Q, K]"
            assert bias.shape[0] == q.shape[0] and bias.shape[1] == q.shape[1], "bias batch/head mismatch"
            assert bias.shape[2] == q.shape[2] and bias.shape[3] == k.shape[2], "bias sequence mismatch"
            assert bias.is_cuda and bias.is_contiguous(), "bias must be CUDA and contiguous"
            assert bias.dtype in (torch.float16, torch.bfloat16, torch.float32), "unsupported bias dtype"
            bias_ptr = bias
            stride_bz, stride_bh, stride_bm, stride_bn = bias.stride()
        else:
            # Dummy pointer/strides when bias is disabled in-kernel.
            bias_ptr = q
            stride_bz = stride_bh = stride_bm = stride_bn = 0
        # Dummy document pointer/strides for generic attention path.
        document_ptr = q
        stride_doz = stride_dot = 0
        # Dummy gumbel-specific inputs for generic attention path.
        prefix_ptr = q
        beacon_ptr = q
        stride_pz = stride_ph = stride_pt = 0
        stride_az = stride_ah = stride_at = 0
        window_size = 0
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o, bias_ptr,  #
            stride_bz, stride_bh, stride_bm, stride_bn,  #
            N_CTX=q.shape[2],  #
            document_ptr=document_ptr, stride_doz=stride_doz, stride_dot=stride_dot,  #
            prefix_ptr=prefix_ptr, beacon_ptr=beacon_ptr,
            stride_pz=stride_pz, stride_ph=stride_ph, stride_pt=stride_pt,
            stride_az=stride_az, stride_ah=stride_ah, stride_at=stride_at,
            window_size=window_size,  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            HAS_BIAS=has_bias,  #
            HAS_DOCUMENT_MASK=False,  #
            HAS_GUMBEL_BIAS=False,  #
            ZERO_NORMAL_BIAS_IN_WINDOW=False,  #
            HAS_PREFIX_BIAS=False,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            DTYPE_IS_BF16=q.dtype == torch.bfloat16,  #
            **extra_kern_args)

        if has_bias:
            ctx.save_for_backward(q, k, v, o, M, bias)
        else:
            ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        ctx.has_bias = has_bias
        return o

    @staticmethod
    def backward(ctx, do):
        if ctx.has_bias:
            q, k, v, _, _, bias = ctx.saved_tensors
            # Bias-aware backward path (supports dBias): recompute attention probabilities
            # in PyTorch autograd math to produce dq/dk/dv/dbias consistently.
            qf = q.float()
            kf = k.float()
            vf = v.float()
            dof = do.float()
            bf = bias.float()

            scores = torch.matmul(qf, kf.transpose(2, 3)) * ctx.sm_scale + bf
            if ctx.causal:
                q_len = q.shape[2]
                k_len = k.shape[2]
                causal_mask = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~causal_mask.view(1, 1, q_len, k_len), float("-inf"))

            p = torch.softmax(scores, dim=-1)

            dv = torch.matmul(p.transpose(2, 3), dof)
            dp = torch.matmul(dof, vf.transpose(2, 3))
            ds = (dp - (dp * p).sum(dim=-1, keepdim=True)) * p

            dq = torch.matmul(ds, kf) * ctx.sm_scale
            dk = torch.matmul(ds.transpose(2, 3), qf) * ctx.sm_scale
            dbias = ds

            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, dbias.to(bias.dtype)

        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        dummy_vec = torch.empty((BATCH, N_HEAD, max(N_CTX // 2, 1)), device=q.device, dtype=torch.float32)
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta, dummy_vec, dummy_vec, dummy_vec, dummy_vec,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            dummy_vec.stride(0), dummy_vec.stride(1), dummy_vec.stride(2),  #
            dummy_vec.stride(0), dummy_vec.stride(1), dummy_vec.stride(2),  #
            dummy_vec.stride(0), dummy_vec.stride(1), dummy_vec.stride(2),  #
            dummy_vec.stride(0), dummy_vec.stride(1), dummy_vec.stride(2),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,  #
            WINDOW_SIZE=0,  #
            ZERO_NORMAL_BIAS_IN_WINDOW=False,  #
            HAS_GUMBEL_BIAS=False,  #
            HAS_PREFIX_BIAS=False,  #
        )

        return dq, dk, dv, None, None, None, None


def attention(q, k, v, causal, sm_scale, warp_specialize=True):
    return _attention.apply(q, k, v, causal, sm_scale, warp_specialize, None)


@triton.jit
def _gumbel_bwd_dkdv_kernel(
    Q, K, V, sm_scale,
    DO,
    DK, DV,
    M, D,
    PREFIX, BEACON,
    DOCUMENT_IDX,
    DPREFIX_Q_PARTIAL, DPREFIX_K_PARTIAL, DBEACON_K_PARTIAL,
    stride_z, stride_h, stride_tok, stride_d,
    stride_prefix_z, stride_prefix_h, stride_prefix_t,
    stride_beacon_z, stride_beacon_h, stride_beacon_t,
    stride_doc_z, stride_doc_t,
    stride_partial_z, stride_partial_h, stride_partial_kblk, stride_partial_t,
    H, N_CTX,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    ZERO_NORMAL_BIAS_IN_WINDOW: tl.constexpr,
    HAS_DOCUMENT_MASK: tl.constexpr,
    HAS_PREFIX_BIAS: tl.constexpr,
    USE_EXACT_SEGMENT_MASK: tl.constexpr,
):
    """Kernel A+C: compute dK, dV, and bias gradients (dprefix/dbeacon).

    Fuses the former Kernel C (bias grad recomputation) into the dK/dV kernel
    to eliminate a redundant O(N^2) pass over Q*K blocks.

    Step-by-step:
    1) Recompute local attention scores (QK + gumbel bias) for each q/k block.
    2) Reconstruct probabilities p using saved row logsumexp (M).
    3) Build ds = d(score) from upstream do and saved softmax stats (D).
    4) Accumulate dV from p^T @ do and dK from ds^T @ Q.
    5) Accumulate bias gradients: dbeacon (beacon keys) and dprefix (normal keys).
    """
    pid_k = tl.program_id(0)
    bhid = tl.program_id(1)

    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    PREFIX += (stride_prefix_h * (bhid % H) + stride_prefix_z * (bhid // H)).to(tl.int64)
    BEACON += (stride_beacon_h * (bhid % H) + stride_beacon_z * (bhid // H)).to(tl.int64)
    DOCUMENT_IDX += (stride_doc_z * (bhid // H)).to(tl.int64)
    partial_adj_h = (stride_partial_h * (bhid % H) + stride_partial_z * (bhid // H) + pid_k * stride_partial_kblk).to(tl.int64)
    DPREFIX_Q_PARTIAL += partial_adj_h
    DPREFIX_K_PARTIAL += partial_adj_h
    DBEACON_K_PARTIAL += partial_adj_h

    offs_d = tl.arange(0, HEAD_DIM)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Load K, V block once.
    k = tl.load(K + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d)
    v = tl.load(V + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d)

    k_tok = offs_k // 2
    k_is_beacon = (offs_k % 2) == 1
    docs_k = tl.load(DOCUMENT_IDX + offs_k * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_K], dtype=tl.int32)
    if HAS_PREFIX_BIAS:
        prefix_k = tl.load(PREFIX + k_tok * stride_prefix_t)
    beacon_bias_k = tl.load(BEACON + k_tok * stride_beacon_t)

    dk = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dbeacon_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    dprefix_k_acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Step 1-5 over q-blocks (causal: only q >= k for this k-block).
    q_start = pid_k * BLOCK_K
    num_q_blocks = (N_CTX - q_start) // BLOCK_Q
    for q_blk_idx in range(num_q_blocks):
        offs_q = q_start + q_blk_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q = tl.load(Q + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
        do = tl.load(DO + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
        m_q = tl.load(M + offs_q)
        Di = tl.load(D + offs_q)

        # Step 1: recompute scores with gumbel bias.
        qk = tl.dot(q, tl.trans(k))
        q_tok = offs_q // 2
        docs_q = tl.load(DOCUMENT_IDX + offs_q * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_Q], dtype=tl.int32)
        if HAS_PREFIX_BIAS:
            prefix_q = tl.load(PREFIX + q_tok * stride_prefix_t)
            if USE_EXACT_SEGMENT_MASK:
                # Keep backward score reconstruction consistent with exact-segment
                # forward mode and avoid overflow from large positive prefix diffs.
                normal_bias = tl.where(prefix_q[:, None] == prefix_k[None, :], 0.0, -1.0e6)
            else:
                normal_bias = prefix_q[:, None] - prefix_k[None, :]
            if ZERO_NORMAL_BIAS_IN_WINDOW:
                rel = q_tok[:, None] - k_tok[None, :]
                in_window = (rel >= 0) & (rel <= WINDOW_SIZE)
                normal_bias = tl.where(in_window, 0.0, normal_bias)
        else:
            rel = q_tok[:, None] - k_tok[None, :]
            out_of_window = rel > WINDOW_SIZE
            normal_bias = tl.where(out_of_window, -1.0e6, 0.0)
        bias = tl.where(k_is_beacon[None, :], beacon_bias_k[None, :], normal_bias)
        # Beacons always self-attend: zero beacon bias on diagonal.
        is_self_beacon = (offs_q[:, None] == offs_k[None, :]) & k_is_beacon[None, :]
        bias = tl.where(is_self_beacon, 0.0, bias)
        qk = qk * (sm_scale * 1.44269504) + bias * 1.44269504

        # Step 2: reconstruct probabilities from saved M.
        p = tl.math.exp2(qk - m_q[:, None])
        causal_mask = (offs_q[:, None] >= offs_k[None, :])
        if HAS_DOCUMENT_MASK:
            doc_mask = docs_q[:, None] == docs_k[None, :]
            p = tl.where(causal_mask & doc_mask, p, 0.0)
        else:
            p = tl.where(causal_mask, p, 0.0)

        p16 = p.to(do.dtype)
        dv += tl.dot(tl.trans(p16), do)

        # Step 3: softmax backward core (dp -> ds).
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds16 = ds.to(q.dtype)
        # Step 4: accumulate dK contributions.
        dk += tl.dot(tl.trans(ds16), q)

        # Step 5: accumulate bias gradients (fused from former Kernel C).
        # Beacon gradient: sum ds over q-axis for beacon-key columns.
        # Exclude self-attention diagonal (beacon bias is 0 there).
        ds_no_self = tl.where(is_self_beacon, 0.0, ds)
        sum_q_per_k_all = tl.sum(ds_no_self, axis=0)
        dbeacon_acc += tl.where(k_is_beacon, sum_q_per_k_all, 0.0)
        if HAS_PREFIX_BIAS:
            # Normal (prefix) gradient: zero out beacon keys and in-window entries.
            ds_normal = tl.where(k_is_beacon[None, :], 0.0, ds)
            if ZERO_NORMAL_BIAS_IN_WINDOW:
                ds_normal = tl.where(in_window, 0.0, ds_normal)
            # Prefix key-side: negative contribution.
            sum_q_normal_per_k = tl.sum(ds_normal, axis=0)
            dprefix_k_acc -= sum_q_normal_per_k
            # Prefix query-side: positive contribution (atomic within k-block partial).
            sum_k_per_q = tl.sum(ds_normal, axis=1)
            q_tok_out = offs_q // 2
            tl.atomic_add(DPREFIX_Q_PARTIAL + q_tok_out * stride_partial_t, sum_k_per_q)

    # Write dK and dV.
    dk_ptrs = DK + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d
    tl.store(dk_ptrs, dk * sm_scale)
    dv_ptrs = DV + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write k-side bias gradient partials.
    if HAS_PREFIX_BIAS:
        tl.atomic_add(
            DPREFIX_K_PARTIAL + k_tok * stride_partial_t,
            tl.where(k_is_beacon, 0.0, dprefix_k_acc),
        )
    tl.atomic_add(
        DBEACON_K_PARTIAL + k_tok * stride_partial_t,
        tl.where(k_is_beacon, dbeacon_acc, 0.0),
    )


@triton.jit
def _gumbel_bwd_dq_kernel(
    Q, K, V, sm_scale,
    DO,
    DQ,
    M, D,
    PREFIX, BEACON,
    DOCUMENT_IDX,
    stride_z, stride_h, stride_tok, stride_d,
    stride_prefix_z, stride_prefix_h, stride_prefix_t,
    stride_beacon_z, stride_beacon_h, stride_beacon_t,
    stride_doc_z, stride_doc_t,
    H, N_CTX,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    ZERO_NORMAL_BIAS_IN_WINDOW: tl.constexpr,
    HAS_DOCUMENT_MASK: tl.constexpr,
    HAS_PREFIX_BIAS: tl.constexpr,
    USE_EXACT_SEGMENT_MASK: tl.constexpr,
):
    """Kernel B: compute dQ with gumbel bias recomputation.

    Step-by-step:
    1) Fix a q-block and sweep eligible k-blocks.
    2) Recompute scores and probabilities p for each q/k block pair.
    3) Build ds = d(score) from do and saved softmax stats.
    4) Accumulate dQ from ds @ K^T.
    """
    LN2: tl.constexpr = 0.6931471824645996

    pid_q = tl.program_id(0)
    bhid = tl.program_id(1)

    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    M += off_chz
    D += off_chz
    PREFIX += (stride_prefix_h * (bhid % H) + stride_prefix_z * (bhid // H)).to(tl.int64)
    BEACON += (stride_beacon_h * (bhid % H) + stride_beacon_z * (bhid // H)).to(tl.int64)
    DOCUMENT_IDX += (stride_doc_z * (bhid // H)).to(tl.int64)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)

    q = tl.load(Q + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
    do = tl.load(DO + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
    m_q = tl.load(M + offs_q)
    Di = tl.load(D + offs_q)

    q_tok = offs_q // 2
    docs_q = tl.load(DOCUMENT_IDX + offs_q * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_Q], dtype=tl.int32)
    if HAS_PREFIX_BIAS:
        prefix_q = tl.load(PREFIX + q_tok * stride_prefix_t)

    dq = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # Step 1-4 over k-blocks (causal: only k <= q for this q-block).
    num_k_blocks = (pid_q * BLOCK_Q + BLOCK_Q) // BLOCK_K
    for k_blk_idx in range(num_k_blocks):
        offs_k = k_blk_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        kT = tl.load(K + offs_k[None, :] * stride_tok + offs_d[:, None] * stride_d)
        vT = tl.load(V + offs_k[None, :] * stride_tok + offs_d[:, None] * stride_d)

        k_tok = offs_k // 2
        k_is_beacon = (offs_k % 2) == 1
        docs_k = tl.load(DOCUMENT_IDX + offs_k * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_K], dtype=tl.int32)
        if HAS_PREFIX_BIAS:
            prefix_k = tl.load(PREFIX + k_tok * stride_prefix_t)
        beacon_bias_k = tl.load(BEACON + k_tok * stride_beacon_t)

        # Step 1: recompute block scores with gumbel bias.
        qk = tl.dot(q, kT)
        if HAS_PREFIX_BIAS:
            if USE_EXACT_SEGMENT_MASK:
                # Match exact-segment forward mode and prevent exp overflow.
                normal_bias = tl.where(prefix_q[:, None] == prefix_k[None, :], 0.0, -1.0e6)
            else:
                normal_bias = prefix_q[:, None] - prefix_k[None, :]
            if ZERO_NORMAL_BIAS_IN_WINDOW:
                rel = q_tok[:, None] - k_tok[None, :]
                in_window = (rel >= 0) & (rel <= WINDOW_SIZE)
                normal_bias = tl.where(in_window, 0.0, normal_bias)
        else:
            rel = q_tok[:, None] - k_tok[None, :]
            out_of_window = rel > WINDOW_SIZE
            normal_bias = tl.where(out_of_window, -1.0e6, 0.0)
        bias = tl.where(k_is_beacon[None, :], beacon_bias_k[None, :], normal_bias)
        # Beacons always self-attend: zero beacon bias on diagonal.
        is_self_beacon = (offs_q[:, None] == offs_k[None, :]) & k_is_beacon[None, :]
        bias = tl.where(is_self_beacon, 0.0, bias)
        qk = qk * (sm_scale * 1.44269504) + bias * 1.44269504

        # Step 2: reconstruct p using saved row logsumexp.
        p = tl.math.exp2(qk - m_q[:, None])
        causal_mask = (offs_q[:, None] >= offs_k[None, :])
        if HAS_DOCUMENT_MASK:
            doc_mask = docs_q[:, None] == docs_k[None, :]
            p = tl.where(causal_mask & doc_mask, p, 0.0)
        else:
            p = tl.where(causal_mask, p, 0.0)

        # Step 3: softmax backward core.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds16 = ds.to(kT.dtype)
        # Step 4: accumulate dQ contribution.
        dq += tl.dot(ds16, tl.trans(kT))

    dq_ptrs = DQ + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d
    dq *= sm_scale
    tl.store(dq_ptrs, dq)


class _gumbel_sliding_attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        prefix_log_BxHxT,
        beacon_log_alpha_BxHxT,
        window_size,
        zero_normal_bias_in_window,
        has_prefix_bias=True,
        warp_specialize=True,
        documents_idx_BxT=None,
        use_exact_segment_mask=False,
    ):
        # Forward overview:
        # 1) Validate shapes for doubled 2T layout and optional document indices.
        # 2) Launch Triton fused attention forward that recomputes gumbel bias on the fly.
        # 3) Save tensors required for exact backward recomputation (q/k/v/o/M/prefix/beacon[/docs]).
        # The kernel assumes contiguous [B*H*2T, D] layout; ensure inputs are contiguous.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert q.shape[2] % 2 == 0, "Expected doubled sequence length (2T)."
        assert prefix_log_BxHxT.shape[:2] == q.shape[:2]
        assert beacon_log_alpha_BxHxT.shape[:2] == q.shape[:2]
        assert prefix_log_BxHxT.shape[2] * 2 == q.shape[2]
        assert beacon_log_alpha_BxHxT.shape[2] * 2 == q.shape[2]
        has_document_mask = documents_idx_BxT is not None
        if has_document_mask:
            assert documents_idx_BxT.shape[0] == q.shape[0]
            assert documents_idx_BxT.shape[1] == q.shape[2]
            assert documents_idx_BxT.is_cuda and documents_idx_BxT.is_contiguous()

        # --- Pad sequence dimension to multiple of BLOCK_M_MAX to avoid OOB in Triton kernel ---
        BLOCK_M_MAX = 128
        N_CTX_ORIG = q.shape[2]
        pad_amount = (-N_CTX_ORIG) % BLOCK_M_MAX  # 0 if already aligned
        if pad_amount > 0:
            q = F.pad(q, (0, 0, 0, pad_amount))  # pad seq dim (dim=2)
            k = F.pad(k, (0, 0, 0, pad_amount))
            v = F.pad(v, (0, 0, 0, pad_amount))
            pad_t = pad_amount // 2  # T = N_CTX // 2; pad_amount is always even
            prefix_log_BxHxT = F.pad(prefix_log_BxHxT, (0, pad_t))
            beacon_log_alpha_BxHxT = F.pad(beacon_log_alpha_BxHxT, (0, pad_t))
            if has_document_mask:
                pad_val = documents_idx_BxT.max() + 1
                documents_idx_BxT = F.pad(documents_idx_BxT, (0, pad_amount), value=pad_val.item())

        o = torch.empty_like(q)
        stage = 3  # causal
        extra_kern_args = {}
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80

        # We use gumbel-derived bias, so dense bias is disabled via dummy pointer/strides.
        bias_ptr = q
        stride_bz = stride_bh = stride_bm = stride_bn = 0
        if has_document_mask:
            document_ptr = documents_idx_BxT
            stride_doz, stride_dot = documents_idx_BxT.stride()
        else:
            document_ptr = q
            stride_doz = stride_dot = 0
        stride_pz, stride_ph, stride_pt = prefix_log_BxHxT.stride()
        stride_az, stride_ah, stride_at = beacon_log_alpha_BxHxT.stride()

        _attn_fwd[grid](
            sm_scale, M,
            q.shape[0], q.shape[1],
            desc_q, desc_k, desc_v, desc_o, bias_ptr,
            stride_bz, stride_bh, stride_bm, stride_bn,
            N_CTX=q.shape[2],
            document_ptr=document_ptr, stride_doz=stride_doz, stride_dot=stride_dot,
            prefix_ptr=prefix_log_BxHxT, beacon_ptr=beacon_log_alpha_BxHxT,
            stride_pz=stride_pz, stride_ph=stride_ph, stride_pt=stride_pt,
            stride_az=stride_az, stride_ah=stride_ah, stride_at=stride_at,
            window_size=int(window_size),
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            HAS_BIAS=False,
            HAS_DOCUMENT_MASK=has_document_mask,
            HAS_GUMBEL_BIAS=True,
            ZERO_NORMAL_BIAS_IN_WINDOW=bool(zero_normal_bias_in_window),
            HAS_PREFIX_BIAS=bool(has_prefix_bias),
            USE_EXACT_SEGMENT_MASK=bool(use_exact_segment_mask),
            STAGE=stage,
            warp_specialize=warp_specialize,
            IS_HOPPER=is_hopper(),
            DTYPE_IS_BF16=q.dtype == torch.bfloat16,
            **extra_kern_args,
        )

        # --- Slice output back to original sequence length if we padded ---
        if pad_amount > 0:
            o = o[:, :, :N_CTX_ORIG, :]

        if has_document_mask:
            ctx.save_for_backward(q, k, v, o, M, prefix_log_BxHxT, beacon_log_alpha_BxHxT, documents_idx_BxT)
        else:
            ctx.save_for_backward(q, k, v, o, M, prefix_log_BxHxT, beacon_log_alpha_BxHxT)
        ctx.sm_scale = sm_scale
        ctx.zero_normal_bias_in_window = bool(zero_normal_bias_in_window)
        ctx.has_prefix_bias = bool(has_prefix_bias)
        ctx.window_size = int(window_size)
        ctx.has_document_mask = has_document_mask
        # Backward uses this to keep score reconstruction aligned with forward
        # and to avoid overflow when prefix encodes segment IDs.
        ctx.use_exact_segment_mask = bool(use_exact_segment_mask)
        return o

    @staticmethod
    def backward(ctx, do):
        # Backward overview:
        # 1) Preprocess to compute per-row delta used by softmax backward.
        # 2) Kernel A computes dK/dV + bias gradients (dPrefix/dBeacon) in one fused pass.
        # 3) Kernel B computes dQ.
        # 4) Reduce bias gradient partials over k-blocks.
        if ctx.has_document_mask:
            q, k, v, o, M, prefix_log_BxHxT, beacon_log_alpha_BxHxT, documents_idx_BxT = ctx.saved_tensors
        else:
            q, k, v, o, M, prefix_log_BxHxT, beacon_log_alpha_BxHxT = ctx.saved_tensors
            documents_idx_BxT = q
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()
        do = do.contiguous()
        if ctx.has_document_mask:
            documents_idx_BxT = documents_idx_BxT.contiguous()

        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape
        T = N_CTX // 2

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dprefix = torch.zeros((BATCH, N_HEAD, T), device=q.device, dtype=torch.float32)
        dbeacon = torch.zeros((BATCH, N_HEAD, T), device=q.device, dtype=torch.float32)

        PRE_BLOCK = 128
        NUM_WARPS = 4
        NUM_STAGES = 2
        BLOCK_Q_DKDV = 64  # Inner loop block for dK/dV kernel (must be <= BLOCK_K for causal alignment)
        BLOCK_Q_DQ = 128   # Grid block for dQ kernel
        BLOCK_K = 64

        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,
            delta,
            BATCH, N_HEAD, N_CTX,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM,
        )

        # Step 2: dK / dV + bias gradients (fused).
        num_k_blocks = N_CTX // BLOCK_K
        dprefix_q_partials = torch.zeros((BATCH, N_HEAD, num_k_blocks, T), device=q.device, dtype=torch.float32)
        dprefix_k_partials = torch.zeros((BATCH, N_HEAD, num_k_blocks, T), device=q.device, dtype=torch.float32)
        dbeacon_k_partials = torch.zeros((BATCH, N_HEAD, num_k_blocks, T), device=q.device, dtype=torch.float32)

        grid_dkdv = (num_k_blocks, BATCH * N_HEAD)
        _gumbel_bwd_dkdv_kernel[grid_dkdv](
            q, k, v, ctx.sm_scale,
            do,
            dk, dv,
            M, delta,
            prefix_log_BxHxT, beacon_log_alpha_BxHxT, documents_idx_BxT,
            dprefix_q_partials, dprefix_k_partials, dbeacon_k_partials,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            prefix_log_BxHxT.stride(0), prefix_log_BxHxT.stride(1), prefix_log_BxHxT.stride(2),
            beacon_log_alpha_BxHxT.stride(0), beacon_log_alpha_BxHxT.stride(1), beacon_log_alpha_BxHxT.stride(2),
            documents_idx_BxT.stride(0), documents_idx_BxT.stride(1) if ctx.has_document_mask else 0,
            dprefix_q_partials.stride(0), dprefix_q_partials.stride(1), dprefix_q_partials.stride(2), dprefix_q_partials.stride(3),
            N_HEAD, N_CTX,
            BLOCK_Q=BLOCK_Q_DKDV, BLOCK_K=BLOCK_K,
            HEAD_DIM=HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            WINDOW_SIZE=ctx.window_size,
            ZERO_NORMAL_BIAS_IN_WINDOW=ctx.zero_normal_bias_in_window,
            HAS_DOCUMENT_MASK=ctx.has_document_mask,
            HAS_PREFIX_BIAS=ctx.has_prefix_bias,
            USE_EXACT_SEGMENT_MASK=ctx.use_exact_segment_mask,
        )

        # Step 3: dQ.
        grid_dq = (N_CTX // BLOCK_Q_DQ, BATCH * N_HEAD)
        _gumbel_bwd_dq_kernel[grid_dq](
            q, k, v, ctx.sm_scale,
            do,
            dq,
            M, delta,
            prefix_log_BxHxT, beacon_log_alpha_BxHxT, documents_idx_BxT,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            prefix_log_BxHxT.stride(0), prefix_log_BxHxT.stride(1), prefix_log_BxHxT.stride(2),
            beacon_log_alpha_BxHxT.stride(0), beacon_log_alpha_BxHxT.stride(1), beacon_log_alpha_BxHxT.stride(2),
            documents_idx_BxT.stride(0), documents_idx_BxT.stride(1) if ctx.has_document_mask else 0,
            N_HEAD, N_CTX,
            BLOCK_Q=BLOCK_Q_DQ, BLOCK_K=BLOCK_K,
            HEAD_DIM=HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            WINDOW_SIZE=ctx.window_size,
            ZERO_NORMAL_BIAS_IN_WINDOW=ctx.zero_normal_bias_in_window,
            HAS_DOCUMENT_MASK=ctx.has_document_mask,
            HAS_PREFIX_BIAS=ctx.has_prefix_bias,
            USE_EXACT_SEGMENT_MASK=ctx.use_exact_segment_mask,
        )

        # Reduce bias gradient partials over k-blocks.
        dprefix = dprefix_q_partials.sum(dim=2) + dprefix_k_partials.sum(dim=2)
        dbeacon = dbeacon_k_partials.sum(dim=2)

        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            None,
            dprefix.to(prefix_log_BxHxT.dtype),
            dbeacon.to(beacon_log_alpha_BxHxT.dtype),
            None,
            None,
            None,
            None,
            None,
            None,  # use_exact_segment_mask
        )


def gumbel_sliding_attention(
    q,
    k,
    v,
    sm_scale,
    prefix_log_BxHxT,
    beacon_log_alpha_BxHxT,
    window_size,
    zero_normal_bias_in_window,
    has_prefix_bias=True,
    warp_specialize=True,
    documents_idx_BxT=None,
    use_exact_segment_mask=False,
):
    """
    Triton attention path specialized for gumbel sliding beacons.
    Bias is derived from:
    - normal keys: prefix[q] - prefix[k] (when has_prefix_bias=True, use_exact_segment_mask=False)
      or: 0 if same segment else -inf (when use_exact_segment_mask=True, for eval exact masking)
    - beacon keys: beacon_log_alpha[k]
    - optional zeroing for normal keys inside sliding window
    When has_prefix_bias=False, normal keys get zero bias (no prefix loads).
    When use_exact_segment_mask=True, prefix_log_BxHxT must contain integer segment IDs
    (0, 1, 2, ...) instead of log-probabilities.
    """
    return _gumbel_sliding_attention.apply(
        q,
        k,
        v,
        sm_scale,
        prefix_log_BxHxT,
        beacon_log_alpha_BxHxT,
        window_size,
        zero_normal_bias_in_window,
        has_prefix_bias,
        warp_specialize,
        documents_idx_BxT,
        use_exact_segment_mask,
    )

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')


@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [128, 1024, (2 if is_hip() else 4) * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("warp_specialize", [False, True] if is_blackwell() else [False])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []))
def test_op(Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, dtype=torch.float16):
    if mode == "fwd" and "fp16" in provider:
        pytest.skip("Avoid running the forward computation twice.")
    if mode == "bwd" and "fp8" in provider:
        pytest.skip("Backward pass with FP8 is not supported.")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    if mode == "bwd":
        dout = torch.randn_like(q)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, causal, sm_scale, warp_specialize).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS = 4, 32
# vary seq length for fixed head and batch=4
configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            # Enable warpspec for causal fwd on Hopper
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False, True] if enable_ws else [False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2**i for i in range(10, 15)],
                        line_arg="provider",
                        line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                        (["flash"] if HAS_FLASH else []),
                        line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                        (["Flash-2"] if HAS_FLASH else []),
                        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                        ylabel="TFLOPS",
                        plot_name=
                        f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                        args={
                            "H": N_HEADS,
                            "BATCH": BATCH,
                            "HEAD_DIM": HEAD_DIM,
                            "mode": mode,
                            "causal": causal,
                            "warp_specialize": warp_specialize,
                        },
                    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
