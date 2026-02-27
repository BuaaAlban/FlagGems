import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def conv_transpose2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """Determines the output size of a transposed 2D convolution operation.

    Args:
        in_size: Input spatial size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        output_padding: Output padding.
        dilation: Dilation.

    Returns:
        Output size of transposed 2D convolution.
    """
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv_transpose2d_forward"),
    key=[
        "in_n",
        "in_c",
        "input_height",
        "input_width",
        "out_c",
        "out_height",
        "out_width",
        "weight_height",
        "weight_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    in_c,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_h_stride,
    input_w_stride,
    weight_inc_stride,
    weight_outc_stride,
    weight_h_stride,
    weight_w_stride,
    output_n_stride,
    output_c_stride,
    output_h_stride,
    output_w_stride,
    in_c_per_group: tl.constexpr,
    out_c_per_group: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Triton kernel for transposed 2D convolution forward pass.

    For each output position (oh, ow), we iterate over kernel positions (kh, kw)
    and input channels (ci). The mapping from output to input is:
        ih = (oh + padding_h - kh * dilation_h) / stride_h
        iw = (ow + padding_w - kw * dilation_w) / stride_w
    Only valid when the numerator is non-negative, divisible by stride,
    and the resulting ih/iw is in [0, input_height/input_width).

    Weight shape: (in_channels, out_channels/groups, kH, kW)
    """
    pid_ni_ho_wo = tle.program_id(0)
    pid_co = tle.program_id(1)
    pid_group = tle.program_id(2)

    # Decompose flattened index into (n, oh, ow)
    ni_ho_wo_offset = pid_ni_ho_wo * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    ni_ho_offset = ni_ho_wo_offset // out_width
    n_idx = ni_ho_offset // out_height
    oh_idx = ni_ho_offset % out_height
    ow_idx = ni_ho_wo_offset % out_width

    # Output channel offset within this group
    co_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    # Initialize accumulator in float32 for numerical stability
    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)

    # Loop over input channels (blocked) and kernel spatial positions
    BLOCK_CI_COUNT = (in_c_per_group + BLOCK_CI - 1) // BLOCK_CI
    for hwc in range(weight_height * weight_width * BLOCK_CI_COUNT):
        c_block = (hwc % BLOCK_CI_COUNT) * BLOCK_CI
        hw = hwc // BLOCK_CI_COUNT
        kh = hw // weight_width
        kw = hw % weight_width

        # For conv_transpose2d: ih = (oh + padding - kh * dilation) / stride
        # Valid only if divisible by stride and in range [0, input_height)
        oh_shifted_h = oh_idx + padding_height - kh * dilation_height
        ow_shifted_w = ow_idx + padding_width - kw * dilation_width

        # Check divisibility by stride
        ih_valid = oh_shifted_h % stride_height == 0
        iw_valid = ow_shifted_w % stride_width == 0

        ih = oh_shifted_h // stride_height
        iw = ow_shifted_w // stride_width

        ci_offset = c_block + tl.arange(0, BLOCK_CI)

        # Input pointer: input[n, group * in_c_per_group + ci, ih, iw]
        curr_input_pointer = (
            input_pointer
            + (input_n_stride * n_idx)[:, None]
            + (input_c_stride * (pid_group * in_c_per_group + ci_offset))[None, :]
            + (input_h_stride * ih)[:, None]
            + (input_w_stride * iw)[:, None]
        )

        input_mask = (
            (n_idx < in_n)[:, None]
            & (ci_offset < in_c_per_group)[None, :]
            & ih_valid[:, None]
            & iw_valid[:, None]
            & (ih >= 0)[:, None]
            & (ih < input_height)[:, None]
            & (iw >= 0)[:, None]
            & (iw < input_width)[:, None]
        )

        # Weight pointer: weight[group * in_c_per_group + ci, co, kh, kw]
        curr_weight_pointer = (
            weight_pointer
            + (weight_inc_stride * (pid_group * in_c_per_group + ci_offset))[:, None]
            + (weight_outc_stride * co_offset)[None, :]
            + weight_h_stride * kh
            + weight_w_stride * kw
        )

        weight_mask = (ci_offset < in_c_per_group)[:, None] & (
            co_offset < out_c_per_group
        )[None, :]

        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)

        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    # Add bias: bias[group * out_c_per_group + co]
    bias_ptr = bias_pointer + pid_group * out_c_per_group + co_offset
    bias_mask = co_offset < out_c_per_group
    bias_val = tl.load(bias_ptr, mask=bias_mask, other=0.0).to(tl.float32)
    accum += bias_val[None, :]

    # Store output: output[n, group * out_c_per_group + co, oh, ow]
    output_ptr = (
        output_pointer
        + (output_n_stride * n_idx)[:, None]
        + (output_c_stride * (pid_group * out_c_per_group + co_offset))[None, :]
        + (output_h_stride * oh_idx)[:, None]
        + (output_w_stride * ow_idx)[:, None]
    )
    output_mask = (
        (n_idx < in_n)[:, None]
        & (co_offset < out_c_per_group)[None, :]
        & (oh_idx < out_height)[:, None]
        & (ow_idx < out_width)[:, None]
    )

    tl.store(output_ptr, accum, mask=output_mask)


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    """Transposed 2D convolution (a.k.a. fractionally-strided convolution).

    Args:
        input: Input tensor of shape (N, C_in, H_in, W_in).
        weight: Weight tensor of shape (C_in, C_out/groups, kH, kW).
        bias: Optional bias tensor of shape (C_out,).
        stride: Stride of the convolution.
        padding: Padding added to both sides of the input.
        output_padding: Additional size added to one side of the output.
        groups: Number of blocked connections from input to output channels.
        dilation: Spacing between kernel elements.

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out).
    """
    logger.debug("GEMS CONV_TRANSPOSE2D")
    assert input.ndim == 4, f"Input must be 4D, got shape {input.shape}"
    assert weight.ndim == 4, f"Weight must be 4D, got shape {weight.shape}"

    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (list, tuple)):
        padding_h, padding_w = padding
    else:
        padding_h = padding_w = padding

    if isinstance(output_padding, (list, tuple)):
        output_padding_h, output_padding_w = output_padding
    else:
        output_padding_h = output_padding_w = output_padding

    if isinstance(dilation, (list, tuple)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    in_n, in_c, input_height, input_width = input.shape
    # Weight shape for conv_transpose2d: (in_channels, out_channels/groups, kH, kW)
    weight_in_c, out_c_per_group, kH, kW = weight.shape
    assert (
        in_c == weight_in_c
    ), f"Input channels ({in_c}) must match weight in_channels ({weight_in_c})"

    out_c = out_c_per_group * groups
    in_c_per_group = in_c // groups

    out_height = conv_transpose2d_output_size(
        input_height, kH, stride_h, padding_h, output_padding_h, dilation_h
    )
    out_width = conv_transpose2d_output_size(
        input_width, kW, stride_w, padding_w, output_padding_w, dilation_w
    )

    output = torch.empty(
        (in_n, out_c, out_height, out_width),
        device=input.device,
        dtype=input.dtype,
    )

    if bias is None:
        bias_tensor = torch.zeros(out_c, device=input.device, dtype=input.dtype)
    else:
        bias_tensor = bias

    grid = lambda META: (
        triton.cdiv(in_n * out_height * out_width, META["BLOCK_NI_HO_WO"]),
        triton.cdiv(out_c_per_group, META["BLOCK_CO"]),
        groups,
    )

    with torch_device_fn.device(input.device):
        conv_transpose2d_forward_kernel[grid](
            input,
            weight,
            output,
            bias_tensor,
            in_n,
            in_c,
            input_height,
            input_width,
            out_c,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
            *output.stride(),
            in_c_per_group,
            out_c_per_group,
            kH,
            kW,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups=groups,
        )

    return output
