from flag_gems.fused.chunk_gated_delta_rule.utils import eager_chunk_gated_delta_rule


def chunk_gated_delta_rule_fwd(
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
    out, final_state = eager_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return out, final_state
