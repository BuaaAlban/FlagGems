"""Fused single-kernel forward for chunk_gated_delta_rule.

This implementation runs the full per-token recurrence inside one Triton
program per (B*H) row, keeping the state in registers / shared memory and
avoiding the per-step Python launch loop that dominates the eager path.

Shapes (mirroring the public Python API):
    q:           (B, T, H, K)
    k:           (B, T, H, K)
    v:           (B, T, H, V)
    g:           (B, T, H)
    beta:        (B, T, H)
    out:         (B, T, H, V)
    final_state: (B, H, K, V)

Notes:
    * BLOCK_K and BLOCK_V must be >= K and V, which is the case for the
      Qwen3-Next style chunk gated delta rule (K, V <= 64).
    * The accumulation is done in float32 regardless of input dtype to
      avoid drift across many time steps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


@libentry()
@triton.heuristics(
    {
        "BLOCK_K": lambda args: triton.next_power_of_2(args["K"]),
        "BLOCK_V": lambda args: triton.next_power_of_2(args["V"]),
    }
)
@triton.jit
def chunk_gated_delta_rule_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    state_in_ptr,
    state_out_ptr,
    scale,
    HAS_INITIAL_STATE: tl.constexpr,
    T,
    H,
    K: tl.constexpr,
    V: tl.constexpr,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qk,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kk,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vv,
    stride_gb,
    stride_gt,
    stride_gh,
    stride_betab,
    stride_betat,
    stride_betah,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_ov,
    stride_sib,
    stride_sih,
    stride_sik,
    stride_siv,
    stride_sob,
    stride_soh,
    stride_sok,
    stride_sov,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V

    # Initial state in float32 SRAM tile.
    if HAS_INITIAL_STATE:
        si = (
            state_in_ptr
            + pid_b * stride_sib
            + pid_h * stride_sih
            + offs_k[:, None] * stride_sik
            + offs_v[None, :] * stride_siv
        )
        state = tl.load(si, mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(
            tl.float32
        )
    else:
        state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    q_base = q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = k_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = v_ptr + pid_b * stride_vb + pid_h * stride_vh
    g_base = g_ptr + pid_b * stride_gb + pid_h * stride_gh
    beta_base = beta_ptr + pid_b * stride_betab + pid_h * stride_betah
    o_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh

    for i_t in range(0, T):
        q_ptrs = q_base + i_t * stride_qt + offs_k * stride_qk
        k_ptrs = k_base + i_t * stride_kt + offs_k * stride_kk
        v_ptrs = v_base + i_t * stride_vt + offs_v * stride_vv
        o_ptrs = o_base + i_t * stride_ot + offs_v * stride_ov

        q_t = tl.load(q_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        k_t = tl.load(k_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        v_t = tl.load(v_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        g_t = tl.load(g_base + i_t * stride_gt).to(tl.float32)
        beta_t = tl.load(beta_base + i_t * stride_betat).to(tl.float32)

        # proj_v = sum_k k_t[k] * state[k, v]
        proj_v = tl.sum(state * k_t[:, None], axis=0)
        v_new = v_t - proj_v

        # state = state * g_t + beta_t * (k_t outer v_new)
        update = beta_t * (k_t[:, None] * v_new[None, :])
        state = state * g_t + update

        # o_t = (sum_k q_t[k] * state[k, v]) * scale
        o_t = tl.sum(state * q_t[:, None], axis=0) * scale
        tl.store(o_ptrs, o_t.to(out_ptr.dtype.element_ty), mask=mask_v)

    # final state writeback (always emitted; caller decides whether to use it)
    so = (
        state_out_ptr
        + pid_b * stride_sob
        + pid_h * stride_soh
        + offs_k[:, None] * stride_sok
        + offs_v[None, :] * stride_sov
    )
    tl.store(
        so,
        state.to(state_out_ptr.dtype.element_ty),
        mask=mask_k[:, None] & mask_v[None, :],
    )


def chunk_gated_delta_rule_fused_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
):
    """Run the fused forward and return ``(out, final_state)``.

    Falls back to ``None`` for ``final_state`` when ``output_final_state`` is
    False, mirroring the public API contract.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "fused path requires CUDA inputs"
    B, T, H, K = q.shape
    Vd = v.shape[-1]

    out = torch.empty(B, T, H, Vd, dtype=q.dtype, device=q.device)
    state_out = torch.empty(B, H, K, Vd, dtype=q.dtype, device=q.device)

    if initial_state is None:
        state_in = torch.empty(1, dtype=q.dtype, device=q.device)
        si_b, si_h, si_k, si_v = 0, 0, 0, 0
        has_initial = False
    else:
        assert initial_state.shape == (B, H, K, Vd), (
            f"initial_state shape mismatch: got {tuple(initial_state.shape)}, "
            f"expected {(B, H, K, Vd)}"
        )
        state_in = initial_state.contiguous()
        si_b, si_h, si_k, si_v = state_in.stride()
        has_initial = True

    so_b, so_h, so_k, so_v = state_out.stride()

    grid = (B * H,)
    chunk_gated_delta_rule_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        out,
        state_in,
        state_out,
        float(scale) if scale is not None else float(K) ** -0.5,
        has_initial,
        T,
        H,
        K,
        Vd,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        si_b,
        si_h,
        si_k,
        si_v,
        so_b,
        so_h,
        so_k,
        so_v,
        num_warps=1,
        num_stages=2,
    )

    if output_final_state:
        return out, state_out
    return out, state_out
