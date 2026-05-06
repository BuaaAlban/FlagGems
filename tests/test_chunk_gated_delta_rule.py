import pytest
import torch

import flag_gems
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
    s = scale if scale is not None else kdim**-0.5

    state = (
        initial_state.to(dtype=dtype, device=device)
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
        out[:, i_t, :, :] = torch.einsum("bhk,bhkd->bhd", q_t, state) * s

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


@pytest.mark.parametrize("dtype", [torch.float32])
def test_chunk_gated_delta_rule_forward_matches_eager_small(dtype):
    b, t, h, kdim, vdim, chunk_size = 1, 4, 2, 4, 4, 4
    device = "cpu"

    q = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    k = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    v = torch.randn(b, t, h, vdim, dtype=dtype, device=device)
    g = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))

    ref_out, ref_final = _eager_chunk_gated_delta_rule(
        q, k, v, g, beta, scale=None, output_final_state=True
    )
    out, final_state = flag_gems.chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale=None, output_final_state=True, chunk_size=chunk_size
    )

    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(final_state, ref_final)


@pytest.mark.parametrize(
    "b,t,h,kdim,vdim,chunk_size",
    [
        (1, 4, 2, 4, 4, 4),
        (2, 7, 2, 8, 8, 4),
        (1, 64, 2, 16, 16, 32),
        (1, 65, 2, 16, 16, 32),
        (1, 33, 4, 16, 16, 16),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_chunk_gated_delta_rule_forward_parametrized(
    b, t, h, kdim, vdim, chunk_size, dtype
):
    device = "cpu"

    q = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    k = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    v = torch.randn(b, t, h, vdim, dtype=dtype, device=device)
    g = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))

    ref_out, ref_final = _eager_chunk_gated_delta_rule(
        q, k, v, g, beta, scale=None, output_final_state=True
    )
    out, final_state = flag_gems.chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale=None, output_final_state=True, chunk_size=chunk_size
    )

    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(final_state, ref_final)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused Triton path"
)
@pytest.mark.parametrize(
    "b,t,h,kdim,vdim,chunk_size",
    [
        (1, 16, 2, 8, 8, 16),
        (1, 64, 2, 16, 16, 64),
        (1, 128, 2, 32, 32, 64),
        (2, 128, 4, 32, 32, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_cuda_matches_eager(b, t, h, kdim, vdim, chunk_size, dtype):
    """Stable-input cross-check between fused CUDA path and pure-PyTorch eager."""
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(b, t, h, kdim, dtype=dtype, device=device) * 0.5
    k = torch.randn(b, t, h, kdim, dtype=dtype, device=device) * 0.5
    v = torch.randn(b, t, h, vdim, dtype=dtype, device=device) * 0.5
    g = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device)) * 0.5
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device)) * 0.5
    scale = kdim**-0.5

    ref_out, ref_final = _eager_chunk_gated_delta_rule(
        q, k, v, g, beta, scale, output_final_state=True, chunk_size=chunk_size
    )
    out, final_state = flag_gems.chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(final_state, ref_final, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FLA kernels"
)
def test_fla_stage_wrappers_match_old_chain():
    from flag_gems.fused.chunk_gated_delta_rule.forward_helpers import fwd_with_fla

    pytest.importorskip("flag_gems.fused.FLA.chunk")

    dtype = torch.bfloat16
    device = "cuda"
    b, t, h, hv, kdim, vdim, chunk_size = 1, 8, 2, 2, 4, 4, 4

    q = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    k = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    v = torch.randn(b, t, hv, vdim, dtype=dtype, device=device)
    g = torch.randn(b, t, hv, dtype=dtype, device=device)
    beta = torch.sigmoid(torch.randn(b, t, hv, dtype=dtype, device=device))
    initial_state = torch.zeros(b, hv, kdim, vdim, dtype=dtype, device=device)

    try:
        o_new, final_new = fwd_with_fla(
            q,
            k,
            v,
            g,
            beta,
            scale=None,
            initial_state=initial_state.clone(),
            output_final_state=True,
            chunk_size=chunk_size,
        )
    except RuntimeError as exc:
        if "no allocator was set" in str(exc):
            pytest.xfail(
                reason="Current environment lacks Triton allocator for FLA autotuning kernels"
            )
        raise

    (
        g_ref,
        o_ref,
        A_ref,
        final_ref,
        w_ref,
        h_ref,
        v_new_ref,
    ) = flag_gems.fused.FLA.chunk.chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=initial_state.clone(),
        output_final_state=True,
    )

    torch.testing.assert_close(o_new, o_ref)
    torch.testing.assert_close(final_new, final_ref)


@pytest.mark.skip(
    reason="chunk_gated_delta_rule backward is not implemented in current MR"
)
@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_chunk_gated_delta_rule_backward_matches_eager_small(has_initial_state, dtype):
    pytest.skip("Backward is not implemented in current MR.")
