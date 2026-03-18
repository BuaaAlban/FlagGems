import logging
import math
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def smooth_l1_loss_none_func(x, y, beta):
    diff = x.to(tl.float32) - y.to(tl.float32)
    ad = tl.abs(diff)
    # When beta == 0, use L1 loss
    loss = tl.where(
        beta == 0.0,
        ad,
        tl.where(ad < beta, 0.5 * diff * diff / beta, ad - 0.5 * beta),
    )
    return loss


@libentry()
@triton.jit
def smooth_l1_loss_kernel_1(
    inp,
    target,
    mid,
    M,
    beta,
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    inp_val = tl.load(inp + offset, mask=mask, other=0).to(tl.float32)
    target_val = tl.load(target + offset, mask=mask, other=0).to(tl.float32)

    diff = inp_val - target_val
    ad = tl.abs(diff)
    loss = tl.where(
        beta == 0.0,
        ad,
        tl.where(ad < beta, 0.5 * diff * diff / beta, ad - 0.5 * beta),
    )

    # Reduction.MEAN.value: 1, Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(loss) / M
    else:
        sum_val = tl.sum(loss)

    tl.store(mid + pid, sum_val)


@libentry()
@triton.jit
def smooth_l1_loss_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


def smooth_l1_loss(inp, target, reduction=Reduction.MEAN.value, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")

    if reduction == Reduction.NONE.value:
        return smooth_l1_loss_none_func(inp, target, beta)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    if M == 0:
        # For empty tensor, mean reduction returns NaN, sum returns 0
        if reduction == Reduction.MEAN.value:
            return torch.tensor(float("nan"), dtype=dtype, device=inp.device)
        else:
            return torch.tensor(0.0, dtype=dtype, device=inp.device)

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        smooth_l1_loss_kernel_1[(mid_size, 1, 1)](
            inp, target, mid, M, beta, block_size, reduction
        )
        smooth_l1_loss_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)

    return out
