import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES, BenchLevel
from benchmark.performance_utils import Config, GenericBenchmark


def _chunk_shapes():
    return [
        (1, 16, 2, 16, 16, 16),
        (1, 64, 2, 16, 16, 32),
        (2, 128, 4, 32, 32, 32),
        (1, 256, 4, 64, 64, 64),
    ]


def chunk_gated_delta_rule_input_fn(shape, dtype, device):
    b, t, h, kdim, vdim, chunk_size = shape
    q = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    k = torch.randn(b, t, h, kdim, dtype=dtype, device=device)
    v = torch.randn(b, t, h, vdim, dtype=dtype, device=device)
    g = torch.randn(b, t, h, dtype=dtype, device=device)
    beta = torch.sigmoid(torch.randn(b, t, h, dtype=dtype, device=device))

    yield (
        (q,),
        {
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": kdim**-0.5,
            "initial_state": None,
            "output_final_state": True,
            "chunk_size": chunk_size,
        },
    )

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        initial_state = torch.zeros(b, h, kdim, vdim, dtype=dtype, device=device)
        yield (
            (q,),
            {
                "k": k,
                "v": v,
                "g": g,
                "beta": beta,
                "scale": kdim**-0.5,
                "initial_state": initial_state,
                "output_final_state": True,
                "chunk_size": chunk_size,
            },
        )


class ChunkGatedDeltaRuleBenchmark(GenericBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in _chunk_shapes():
            yield from self.input_fn(shape, cur_dtype, self.device)


def _torch_chunk_gated_delta_rule(q, **kwargs):
    return flag_gems.fused.FLA.chunk.chunk_gated_delta_rule_fwd(
        q,
        kwargs["k"],
        kwargs["v"],
        kwargs["g"],
        kwargs["beta"],
        kwargs["scale"],
        kwargs["initial_state"],
        kwargs["output_final_state"],
    )


@pytest.mark.chunk_gated_delta_rule
def test_perf_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=_torch_chunk_gated_delta_rule,
        input_fn=chunk_gated_delta_rule_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.chunk_gated_delta_rule_fwd)

    try:
        bench.run()
    except BaseException as exc:  # noqa: BLE001
        if "no allocator was set" in str(exc):
            pytest.xfail(
                reason="Current environment lacks Triton allocator for CUDA benchmark path"
            )
        raise
