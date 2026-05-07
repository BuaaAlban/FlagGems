import dataclasses
from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


@dataclasses.dataclass
class _TorchCudaBuffer:
    ptr: int

    def data_ptr(self) -> int:
        return self.ptr


def _install_triton_cuda_allocator():
    if not torch.cuda.is_available():
        return

    from triton.runtime._allocation import NullAllocator, set_allocator

    def allocator(size, alignment, stream):  # noqa: ARG001
        mem = torch.cuda.caching_allocator_alloc(size)
        return _TorchCudaBuffer(ptr=mem)

    set_allocator(allocator)

    def _reset():
        try:
            set_allocator(NullAllocator())
        except Exception:  # noqa: BLE001
            pass

    import atexit

    atexit.register(_reset)


_install_triton_cuda_allocator()


CHUNK_GATED_DELTA_RULE_SHAPES = [
    (1, 64, 2, 16, 16, 64),
    (1, 128, 2, 32, 32, 64),
    (2, 128, 4, 32, 32, 64),
    (1, 256, 4, 64, 64, 64),
]


def _torch_reference_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,  # noqa: ARG001
    chunk_size=64,  # noqa: ARG001
):
    b, t, h, kdim = q.shape
    vdim = v.shape[-1]
    state = (
        initial_state.clone().to(dtype=q.dtype, device=q.device)
        if initial_state is not None
        else torch.zeros(b, h, kdim, vdim, dtype=q.dtype, device=q.device)
    )
    out = torch.empty(b, t, h, vdim, dtype=q.dtype, device=q.device)
    s = scale if scale is not None else kdim**-0.5

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


def chunk_gated_delta_rule_input_fn(shape, cur_dtype, device):
    b, t, h, kdim, vdim, chunk_size = shape
    q = torch.randn(b, t, h, kdim, dtype=cur_dtype, device=device)
    k = torch.randn(b, t, h, kdim, dtype=cur_dtype, device=device)
    v = torch.randn(b, t, h, vdim, dtype=cur_dtype, device=device)
    g = torch.sigmoid(torch.randn(b, t, h, dtype=cur_dtype, device=device)) * 0.5
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=cur_dtype, device=device)) * 0.5

    yield q, k, v, g, beta, {
        "scale": kdim**-0.5,
        "initial_state": None,
        "output_final_state": True,
        "chunk_size": chunk_size,
    }

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        initial_state = torch.zeros(b, h, kdim, vdim, dtype=cur_dtype, device=device)
        yield q, k, v, g, beta, {
            "scale": kdim**-0.5,
            "initial_state": initial_state,
            "output_final_state": True,
            "chunk_size": chunk_size,
        }


class ChunkGatedDeltaRuleBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPE_DESC = "B, T, H, K, V, BT"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in CHUNK_GATED_DELTA_RULE_SHAPES:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def set_more_shapes(self):
        return []


@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        input_fn=chunk_gated_delta_rule_input_fn,
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=_torch_reference_chunk_gated_delta_rule,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.chunk_gated_delta_rule_fwd)
    bench.run()
