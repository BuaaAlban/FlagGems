import torch

from flag_gems.fused.chunk_gated_delta_rule.triton_fwd import (
    chunk_gated_delta_rule_fused_fwd,
)
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
    if cu_seqlens is not None:
        pass

    use_fused = (
        torch.cuda.is_available()
        and q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and g.is_cuda
        and beta.is_cuda
    )

    if use_fused:
        out, final_state = chunk_gated_delta_rule_fused_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    else:
        out, final_state = eager_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )

    return out, final_state
