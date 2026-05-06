import pytest
import torch
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference
from .conftest import QUICK_MODE


def _make_targets(N, S, C, device, blank=0, varying=False):
    if varying:
        target_lengths = torch.randint(1, S + 1, (N,), device=device, dtype=torch.long)
    else:
        target_lengths = torch.full((N,), S, device=device, dtype=torch.long)
    raw = torch.randint(0, C - 1, (N, S), device=device, dtype=torch.long)
    targets = raw + (raw >= blank).to(raw.dtype)
    return targets, target_lengths


def _reference_ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """Move inputs to reference device, upcast fp16/bf16 to fp32 for PyTorch."""
    ref_log_probs = to_reference(log_probs.detach(), False)
    if ref_log_probs.dtype != torch.float32:
        ref_log_probs = ref_log_probs.float()
    ref_log_probs.requires_grad_(log_probs.requires_grad)
    ref_targets = to_reference(targets) if torch.is_tensor(targets) else targets
    # PyTorch requires input_lengths and target_lengths to both be tensors or both lists.
    # Normalize to tensors for the reference path.
    if isinstance(input_lengths, (list, tuple)):
        ref_il = torch.tensor(input_lengths, dtype=torch.long)
    else:
        ref_il = to_reference(input_lengths)
    if isinstance(target_lengths, (list, tuple)):
        ref_tl = torch.tensor(target_lengths, dtype=torch.long)
    else:
        ref_tl = to_reference(target_lengths)
    ref_out = F.ctc_loss(
        ref_log_probs,
        ref_targets,
        ref_il,
        ref_tl,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    return ref_log_probs, ref_out.to(log_probs.dtype)


def _assert_forward_backward(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    *,
    dtype,
    blank=0,
    reduction="mean",
    zero_infinity=False,
    check_backward=True,
    equal_nan=True,
):
    ref_log_probs, ref_out = _reference_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    res_out = flag_gems.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )

    T = log_probs.shape[0]
    if torch.is_tensor(target_lengths):
        max_target = max(1, int(target_lengths.max().item()))
    else:
        max_target = max(1, max(target_lengths))
    reduce_dim = T * (max_target + 1)
    gems_assert_close(
        res_out, ref_out, dtype, equal_nan=equal_nan, reduce_dim=reduce_dim
    )

    if not check_backward:
        return

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, False).to(ref_out.dtype)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_log_probs, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, log_probs, out_grad)
    gems_assert_close(
        res_in_grad,
        ref_in_grad,
        dtype,
        equal_nan=equal_nan,
        reduce_dim=reduce_dim,
    )


# ============================================================================
# Forward + backward over the parameter matrix
# ============================================================================


CTC_DTYPES = [torch.float32]
if not QUICK_MODE:
    CTC_DTYPES = [torch.float32, torch.float16]
    if flag_gems.runtime.device.support_bf16:
        CTC_DTYPES.append(torch.bfloat16)

CTC_REDUCTIONS = ["none", "mean", "sum"] if not QUICK_MODE else ["mean", "sum"]


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("reduction", CTC_REDUCTIONS)
def test_accuracy_ctc_loss(dtype, reduction):
    torch.manual_seed(42)
    T, N, C, S = (8, 2, 8, 4) if QUICK_MODE else (16, 4, 12, 6)
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, blank=0)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=0,
        reduction=reduction,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("reduction", CTC_REDUCTIONS)
def test_accuracy_ctc_loss_concatenated_targets(dtype, reduction):
    torch.manual_seed(42)
    T, N, C, S = (8, 2, 8, 4) if QUICK_MODE else (16, 4, 12, 6)
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()

    target_lengths = torch.randint(
        1, S + 1, (N,), device=flag_gems.device, dtype=torch.long
    )
    pieces = [
        torch.randint(
            1,
            C,
            (int(target_lengths[i].item()),),
            device=flag_gems.device,
            dtype=torch.long,
        )
        for i in range(N)
    ]
    targets = torch.cat(pieces)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=0,
        reduction=reduction,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("reduction", CTC_REDUCTIONS)
def test_accuracy_ctc_loss_intlist_lengths(dtype, reduction):
    torch.manual_seed(42)
    T, N, C, S = 12, 3, 8, 4
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()
    targets, target_lengths_t = _make_targets(N, S, C, flag_gems.device)
    input_lengths = [T] * N
    target_lengths = target_lengths_t.tolist()

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=0,
        reduction=reduction,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_accuracy_ctc_loss_varying_input_lengths(dtype):
    torch.manual_seed(42)
    T, N, C, S = 20, 4, 12, 6
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, varying=True)
    input_lengths = torch.tensor(
        [20, 15, 18, 12], dtype=torch.long, device=flag_gems.device
    )

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=0,
        reduction="mean",
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
@pytest.mark.parametrize("blank", [0, 3])
def test_accuracy_ctc_loss_nonzero_blank(dtype, blank):
    torch.manual_seed(42)
    T, N, C, S = 12, 3, 8, 4
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device, blank=blank)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=blank,
        reduction="none",
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_accuracy_ctc_loss_zero_infinity(dtype):
    torch.manual_seed(42)
    T, N, C = 2, 1, 5
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()
    targets = torch.tensor([[1, 1, 1]], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([T], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([3], dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_accuracy_ctc_loss_unbatched(dtype):
    torch.manual_seed(42)
    T, C, S = 9, 6, 4
    blank = 1
    raw = torch.randn((T, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()
    raw_t = torch.randint(0, C - 1, (S,), device=flag_gems.device, dtype=torch.long)
    targets = raw_t + (raw_t >= blank).to(raw_t.dtype)
    input_lengths = torch.tensor(T, dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor(S, dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=blank,
        reduction="none",
    )


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", CTC_DTYPES)
def test_accuracy_ctc_loss_repeated_labels(dtype):
    torch.manual_seed(42)
    T, N, C = 5, 1, 4
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).to(dtype).detach().requires_grad_()
    targets = torch.tensor([[1, 1]], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([T], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([2], dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=dtype,
        blank=0,
        reduction="none",
    )


@pytest.mark.ctc_loss
def test_accuracy_ctc_loss_empty_target():
    torch.manual_seed(42)
    T, N, C = 12, 2, 7
    raw = torch.randn((T, N, C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1).detach().requires_grad_()
    targets = torch.tensor(
        [[1, 2, 3, 0], [0, 0, 0, 0]], dtype=torch.long, device=flag_gems.device
    )
    input_lengths = torch.tensor([T, T], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([3, 0], dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=torch.float32,
        blank=0,
        reduction="none",
    )


@pytest.mark.ctc_loss
def test_accuracy_ctc_loss_noncontiguous():
    torch.manual_seed(42)
    T, N, C, S = 16, 3, 12, 6
    # Build a contiguous (T, N, 2C) and slice to half the C dim → non-contiguous
    raw = torch.randn((T, N, 2 * C), dtype=torch.float32, device=flag_gems.device)
    log_probs = F.log_softmax(raw, dim=-1)[:, :, :C]
    log_probs = log_probs.detach().requires_grad_()
    assert not log_probs.is_contiguous()
    targets, target_lengths = _make_targets(N, S, C, flag_gems.device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=flag_gems.device)

    _assert_forward_backward(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        dtype=torch.float32,
        blank=0,
        reduction="mean",
    )


@pytest.mark.ctc_loss
def test_accuracy_ctc_loss_invalid_blank():
    log_probs = F.log_softmax(
        torch.randn((6, 2, 5), dtype=torch.float32, device=flag_gems.device), dim=-1
    )
    targets = torch.tensor([[1, 2], [2, 3]], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([6, 6], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([2, 2], dtype=torch.long, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=-1)
    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=5)


@pytest.mark.ctc_loss
def test_accuracy_ctc_loss_invalid_reduction():
    log_probs = F.log_softmax(
        torch.randn((6, 1, 5), dtype=torch.float32, device=flag_gems.device), dim=-1
    )
    targets = torch.tensor([[1, 2]], dtype=torch.long, device=flag_gems.device)
    lengths = torch.tensor([6], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([2], dtype=torch.long, device=flag_gems.device)

    with pytest.raises(ValueError):
        flag_gems.ctc_loss(
            log_probs, targets, lengths, target_lengths, reduction="batchmean"
        )


@pytest.mark.ctc_loss
def test_accuracy_ctc_loss_invalid_concatenated_targets():
    log_probs = F.log_softmax(
        torch.randn((6, 2, 5), dtype=torch.float32, device=flag_gems.device), dim=-1
    )
    targets = torch.tensor([1, 2, 3], dtype=torch.long, device=flag_gems.device)
    input_lengths = torch.tensor([6, 6], dtype=torch.long, device=flag_gems.device)
    target_lengths = torch.tensor([2, 2], dtype=torch.long, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.ctc_loss(log_probs, targets, input_lengths, target_lengths)
