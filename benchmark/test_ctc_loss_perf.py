from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import BenchLevel
from benchmark.performance_utils import Config, GenericBenchmark

CTC_DEFAULT_SHAPES = [
    (64, 4, 32, 16),
    (256, 16, 64, 48),
    (512, 32, 64, 48),
    (1024, 32, 128, 96),
]


def _make_targets(N, S, C, device, blank=0):
    target_lengths = torch.full((N,), S, dtype=torch.long, device=device)
    targets = torch.randint(0, C - 1, (N, S), dtype=torch.long, device=device)
    targets = targets + (targets >= blank).to(targets.dtype)
    return targets, target_lengths


def _reference_ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    *args,
    **kwargs,
):
    """Reference op: PyTorch's ctc_loss expects float32 for fp16/bf16 inputs."""
    ref_log_probs = log_probs if log_probs.dtype == torch.float32 else log_probs.float()
    out = torch.nn.functional.ctc_loss(
        ref_log_probs,
        targets,
        input_lengths,
        target_lengths,
        *args,
        **kwargs,
    )
    return out.to(log_probs.dtype) if out.dtype != log_probs.dtype else out


def ctc_loss_input_fn(shape, cur_dtype, device):
    T, N, C, S = shape
    blank = 0
    raw = torch.randn(T, N, C, dtype=torch.float32, device=device)
    log_probs = torch.nn.functional.log_softmax(raw, dim=-1).to(cur_dtype)
    targets, target_lengths = _make_targets(N, S, C, device, blank=blank)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=device)

    yield log_probs, targets, input_lengths, target_lengths, {
        "blank": blank,
        "reduction": "mean",
        "zero_infinity": False,
    }

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
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


CTC_BENCH_DTYPES = [torch.float32, torch.float16]
if flag_gems.runtime.device.support_bf16:
    CTC_BENCH_DTYPES.append(torch.bfloat16)


class CTCLossBenchmark(GenericBenchmark):
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in CTC_DEFAULT_SHAPES:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def set_more_shapes(self):
        return []


@pytest.mark.ctc_loss
def test_perf_ctc_loss():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=_reference_ctc_loss,
        dtypes=CTC_BENCH_DTYPES,
    )
    bench.run()


@pytest.mark.ctc_loss
def test_perf_ctc_loss_backward():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=_reference_ctc_loss,
        dtypes=CTC_BENCH_DTYPES,
        is_backward=True,
    )
    bench.run()
