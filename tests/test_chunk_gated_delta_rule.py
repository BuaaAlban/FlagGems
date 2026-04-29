import torch

from flag_gems.fused.chunk_gated_delta_rule import (
    chunk_gated_delta_rule_fwd as pkg_chunk_gated_delta_rule_fwd,
)


def _eager_chunk_gated_delta_rule(
    q, k, v, g, beta, scale, initial_state=None, output_final_state=False, **kwargs
):
    b, t, h, kdim = q.shape
    vdim = v.shape[-1]
    device = q.device
    dtype = q.dtype

    state = (
        initial_state.clone().to(dtype=dtype, device=device)
        if initial_state is not None
        else torch.zeros(b, h, kdim, vdim, dtype=dtype, device=device)
    )
    out = torch.empty(b, t, h, vdim, dtype=dtype, device=device)

    for i_t in range(t):
        q_t = q[:, i_t, :, :]
        k_t = k[:, i_t, :, :]
        v_t = v[:, i_t, :, :]
        beta_t = beta[:, i_t, :, None, None]
        g_t = g[:, i_t, :, None]

        proj = torch.einsum("bhk,bhkd->bhd", k_t, state)
        v_new = v_t - proj
        state = state * g_t.unsqueeze(-1) + beta_t * torch.einsum(
            "bhk,bhd->bhkd", k_t, v_new
        )
        out[:, i_t, :, :] = torch.einsum("bhk,bhkd->bhd", q_t, state)

    final_state = state if output_final_state else state
    return out, final_state


def test_import_chunk_gated_delta_rule_symbol():
    assert callable(pkg_chunk_gated_delta_rule_fwd)


def test_eager_reference_smoke_shape():
    b, t, h, kdim, vdim = 1, 3, 2, 4, 4
    dtype = torch.float32
    device = "cpu"

    q = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    k = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    v = torch.randn(b, t, h, vdim, dtype=dtype, device=device)
    g = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))

    out, final_state = _eager_chunk_gated_delta_rule(
        q, k, v, g, beta, scale=None, output_final_state=True
    )

    assert out.shape == (b, t, h, vdim)
    assert final_state.shape == (b, h, kdim, vdim)
