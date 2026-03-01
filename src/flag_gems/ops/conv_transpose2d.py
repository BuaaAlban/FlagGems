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
    n_subgrids: tl.constexpr,
    max_H_sub,
    max_W_sub,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """Triton kernel for transposed 2D convolution forward pass.

    Uses sub-grid tiling to eliminate wasted iterations for stride > 1.
    Output positions are divided into stride_h * stride_w sub-grids based on
    (oh % stride_h, ow % stride_w). Within each sub-grid, only the valid
    (kh, kw) kernel positions are iterated, avoiding the wasted iterations
    where the divisibility check fails (which is ~75% for stride=2).

    For stride=1, n_subgrids=1 and this degenerates to the standard approach.

    Weight shape: (in_channels, out_channels/groups, kH, kW)
    """
    pid_raw = tle.program_id(0)
    pid_co = tle.program_id(1)
    pid_group = tle.program_id(2)

    # Determine sub-grid from program id
    pid_subgrid = pid_raw % n_subgrids
    pid_spatial = pid_raw // n_subgrids

    sub_r = pid_subgrid // stride_width
    sub_s = pid_subgrid % stride_width

    # Compute sub-grid spatial dimensions
    H_sub = (out_height - sub_r + stride_height - 1) // stride_height
    W_sub = (out_width - sub_s + stride_width - 1) // stride_width

    # Decompose spatial tile into (n, oh_sub_idx, ow_sub_idx)
    spatial_offset = pid_spatial * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    n_hw = H_sub * W_sub
    n_idx = spatial_offset // n_hw
    hw_idx = spatial_offset % n_hw
    oh_sub_idx = hw_idx // W_sub
    ow_sub_idx = hw_idx % W_sub

    # Map sub-grid indices to actual output positions
    oh = sub_r + oh_sub_idx * stride_height
    ow = sub_s + ow_sub_idx * stride_width

    # Output channel offset within this group
    co_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    # Initialize accumulator in float32 for numerical stability
    accum = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)

    BLOCK_CI_COUNT: tl.constexpr = (in_c_per_group + BLOCK_CI - 1) // BLOCK_CI

    # Loop over kernel positions - only valid ones for this sub-grid.
    # The outer loops over kh, kw are unrolled (constexpr bounds).
    # The validity check uses sub_r/sub_s which are uniform across all
    # threads in the block, so invalid iterations are skipped entirely.
    for kh in range(weight_height):
        # Check if kh is valid for this sub-grid (divisibility check only;
        # ih_base can be negative, handled by mask bounds check below)
        kh_shifted = sub_r + padding_height - kh * dilation_height
        if kh_shifted % stride_height == 0:
            ih_base = kh_shifted // stride_height

            for kw in range(weight_width):
                kw_shifted = sub_s + padding_width - kw * dilation_width
                if kw_shifted % stride_width == 0:
                    iw_base = kw_shifted // stride_width

                    # Input positions: ih = oh_sub_idx + ih_base
                    ih = oh_sub_idx + ih_base
                    iw = ow_sub_idx + iw_base

                    for ci_block in range(BLOCK_CI_COUNT):
                        ci_offset = ci_block * BLOCK_CI + tl.arange(0, BLOCK_CI)

                        # Load input[n, group*ci_per_g + ci, ih, iw]
                        curr_input_pointer = (
                            input_pointer
                            + (input_n_stride * n_idx)[:, None]
                            + (
                                input_c_stride
                                * (pid_group * in_c_per_group + ci_offset)
                            )[None, :]
                            + (input_h_stride * ih)[:, None]
                            + (input_w_stride * iw)[:, None]
                        )

                        input_mask = (
                            (n_idx < in_n)[:, None]
                            & (ci_offset < in_c_per_group)[None, :]
                            & (ih >= 0)[:, None]
                            & (ih < input_height)[:, None]
                            & (iw >= 0)[:, None]
                            & (iw < input_width)[:, None]
                        )

                        # Load weight[group*ci_per_g + ci, co, kh, kw]
                        curr_weight_pointer = (
                            weight_pointer
                            + (
                                weight_inc_stride
                                * (pid_group * in_c_per_group + ci_offset)
                            )[:, None]
                            + (weight_outc_stride * co_offset)[None, :]
                            + weight_h_stride * kh
                            + weight_w_stride * kw
                        )

                        weight_mask = (ci_offset < in_c_per_group)[:, None] & (
                            co_offset < out_c_per_group
                        )[None, :]

                        input_block = tl.load(
                            curr_input_pointer,
                            mask=input_mask,
                            other=0.0,
                        )
                        weight_block = tl.load(
                            curr_weight_pointer,
                            mask=weight_mask,
                            other=0.0,
                        )

                        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    # Add bias: bias[group * out_c_per_group + co]
    bias_ptr = bias_pointer + pid_group * out_c_per_group + co_offset
    bias_mask = co_offset < out_c_per_group
    bias_val = tl.load(bias_ptr, mask=bias_mask, other=0.0).to(tl.float32)
    accum += bias_val[None, :]

    # Store output[n, group*oc_per_g + co, oh, ow]
    output_ptr = (
        output_pointer
        + (output_n_stride * n_idx)[:, None]
        + (output_c_stride * (pid_group * out_c_per_group + co_offset))[None, :]
        + (output_h_stride * oh)[:, None]
        + (output_w_stride * ow)[:, None]
    )
    output_mask = (
        (n_idx < in_n)[:, None]
        & (co_offset < out_c_per_group)[None, :]
        & (oh_sub_idx < H_sub)[:, None]
        & (ow_sub_idx < W_sub)[:, None]
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

    # Sub-grid tiling: divide output into stride_h * stride_w sub-grids
    n_subgrids = stride_h * stride_w
    max_H_sub = (out_height + stride_h - 1) // stride_h
    max_W_sub = (out_width + stride_w - 1) // stride_w
    max_sub_spatial = in_n * max_H_sub * max_W_sub

    grid = lambda META: (
        n_subgrids * triton.cdiv(max_sub_spatial, META["BLOCK_NI_HO_WO"]),
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
            n_subgrids=n_subgrids,
            max_H_sub=max_H_sub,
            max_W_sub=max_W_sub,
        )

    return output
