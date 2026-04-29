from flag_gems.fused.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flag_gems.fused.FLA.chunk_o import chunk_fwd_o
from flag_gems.fused.FLA.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)
from flag_gems.fused.FLA.wy_fast import recompute_w_u_fwd


def prepare_fla(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_size=64,
    **kwargs,
):
    g_out, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g_out,
        cu_seqlens=cu_seqlens,
    )
    return g_out, A, w, u


def recurrence_fla(
    g, k, w, u, initial_state=None, output_final_state=False, cu_seqlens=None, **kwargs
):
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return h, v_new, final_state


def output_fla(q, k, v_new, h, g, scale, cu_seqlens=None, **kwargs):
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return o


def fwd_with_fla(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_size=64,
):
    g_out, A, w, u = prepare_fla(
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    h, v_new, final_state = recurrence_fla(
        g_out,
        k,
        w,
        u,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = output_fla(
        q,
        k,
        v_new,
        h,
        g_out,
        scale,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state
