import pytest
import torch

from benchmark.performance_utils import GenericBenchmark


def svd_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp,


SVD_SHAPES = [
    # Rank-1 (analytic fast path)
    (4096, 1),
    (1, 4096),
    (1024, 1),
    (1, 1024),
    # 2x2 (analytic)
    (4096, 2, 2),
    (256, 2, 2),
    # Small square
    (3, 3),
    (8, 8),
    (16, 16),
    # Small non-square
    (16, 8),
    (8, 16),
    # Rank-2 (analytic)
    (4096, 2, 16),
    (4096, 16, 2),
    (1024, 2, 32),
    (1024, 32, 2),
    # Batched small (Jacobi)
    (10, 3, 3),
    (100, 8, 8),
    (1000, 8, 8),
    (50, 16, 16),
    (200, 16, 16),
    # Medium square
    (32, 32),
    (64, 64),
    (128, 128),
    # Medium non-square
    (64, 32),
    (32, 64),
    (128, 64),
    (64, 128),
    # Batched medium (Gram+eigh wins for high batch)
    (10, 32, 32),
    (50, 32, 32),
    (10, 64, 32),
    (16, 128, 128),
    (16, 64, 128),
    (16, 128, 64),
    # Large (Gram+eigh)
    (256, 256),
    (512, 512),
    (1024, 1024),
    (1024, 256),
    (256, 1024),
    (2, 256, 256),
]


class SVDBenchmark(GenericBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = SVD_SHAPES

    def set_more_shapes(self):
        return SVD_SHAPES


@pytest.mark.svd
def test_perf_svd():
    bench = SVDBenchmark(
        op_name="svd",
        torch_op=torch.svd,
        input_fn=svd_input_fn,
        dtypes=[torch.float32],
    )
    bench.run()
