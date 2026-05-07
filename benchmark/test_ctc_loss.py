from typing import Generator

import pytest
import torch

from . import base, consts

CTC_DEFAULT_SHAPES = [
    (64, 4, 32, 16),
    (256, 16, 64, 48),
    (512, 32, 64, 48),
    (1024, 32, 128, 96),
]


def ctc_loss_input_fn(shape, cur_dtype, device):
    T, N, C, S = shape
    blank = 0
    # torch.nn.functional.ctc_loss only supports float32 log_probs, so we
    # always generate float32 inputs here regardless of `cur_dtype`.
    raw = torch.randn(T, N, C, dtype=torch.float32, device=device)
    log_probs = torch.nn.functional.log_softmax(raw, dim=-1)
    targets = torch.randint(0, C - 1, (N, S), dtype=torch.long, device=device)
    targets = targets + (targets >= blank).to(targets.dtype)
    target_lengths = torch.full((N,), S, dtype=torch.long, device=device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

    yield log_probs, targets, input_lengths, target_lengths, {
        "blank": blank,
        "reduction": "mean",
        "zero_infinity": False,
    }

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield log_probs, targets, input_lengths, target_lengths, {
            "blank": blank,
            "reduction": "sum",
            "zero_infinity": False,
        }
        yield log_probs, targets, input_lengths, target_lengths, {
            "blank": blank,
            "reduction": "none",
            "zero_infinity": False,
        }
        # Concatenated targets layout
        concat_targets = torch.cat(
            [targets[i, : int(target_lengths[i].item())] for i in range(N)], dim=0
        )
        yield log_probs, concat_targets, input_lengths, target_lengths, {
            "blank": blank,
            "reduction": "mean",
            "zero_infinity": False,
        }


class CTCLossBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in CTC_DEFAULT_SHAPES:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def set_more_shapes(self):
        return []


@pytest.mark.ctc_loss
def test_ctc_loss():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=torch.nn.functional.ctc_loss,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.ctc_loss
def test_ctc_loss_backward():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=torch.nn.functional.ctc_loss,
        dtypes=[torch.float32],
        is_backward=True,
    )
    bench.run()
