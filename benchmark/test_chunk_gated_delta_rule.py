import dataclasses

import pytest
import torch

import flag_gems
from benchmark.attri_util import BenchLevel
from benchmark.performance_utils import Config, GenericBenchmark


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


def _chunk_shapes():
    return [
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
    device = q.device
    dtype = q.dtype
    s = scale if scale is not None else kdim**-0.5

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
        out[:, i_t, :, :] = torch.einsum("bhk,bhkd->bhd", q_t, state) * s

    final_state = state if output_final_state else state
    return out, final_state


def chunk_gated_delta_rule_input_fn(shape, dtype, device):
    b, t, h, kdim, vdim, chunk_size = shape
    q = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    k = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    v = torch.randn(b, t, h, vdim, dtype=dtype, device=device)
    g = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))

    yield (
        q,
        k,
        v,
        g,
        beta,
        kdim**-0.5,
        None,
        True,
        chunk_size,
    )

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        initial_state = torch.zeros(b, h, kdim, vdim, dtype=dtype, device=device)
        yield (
            q,
            k,
            v,
            g,
            beta,
            kdim**-0.5,
            initial_state,
            True,
            chunk_size,
        )


class ChunkGatedDeltaRuleBenchmark(GenericBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in _chunk_shapes():
            yield from self.input_fn(shape, cur_dtype, self.device)


def _torch_baseline(*args, **kwargs):
    out, final_state = _torch_reference_chunk_gated_delta_rule(
        *args,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "scale",
                "initial_state",
                "output_final_state",
                "cu_seqlens",
                "chunk_size",
            }
        },
    )
    return out, final_state


def _gems_chunk_gated_delta_rule(*args, **kwargs):
    out, final_state = flag_gems.chunk_gated_delta_rule_fwd(
        *args,
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "scale",
                "initial_state",
                "output_final_state",
                "cu_seqlens",
                "chunk_size",
            }
        },
    )
    return out, final_state


@pytest.mark.chunk_gated_delta_rule
def test_perf_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=_torch_baseline,
        input_fn=chunk_gated_delta_rule_input_fn,
        dtypes=[torch.float32],
    )
    bench.set_gems(_gems_chunk_gated_delta_rule)
    bench.run()
