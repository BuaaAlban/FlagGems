import os

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark


class Conv1DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 512, 64, 3, 1, 0, 1),
            (64, 48, 1024, 128, 5, 2, 2, 1),
            (16, 24, 2048, 96, 7, 1, 3, 2),
            (8, 8, 8192, 16, 11, 4, 5, 1),
            (4, 4, 16384, 4, 15, 2, 7, 1),
            (32, 64, 512, 64, 3, 1, "valid", 1),
            (64, 48, 1024, 128, 5, 2, "valid", 1),
            (16, 24, 2048, 96, 7, 1, "same", 2),
            (8, 8, 8192, 16, 11, 1, "same", 1),
        ]


@pytest.mark.conv1d
def test_perf_conv1d():
    def conv1d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_l,
            out_c,
            kernel,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_l)
        weight_shape = (out_c, input_c // groups, kernel)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = Conv1DBenchmark(
        input_fn=conv1d_input_fn,
        op_name="conv1d",
        torch_op=torch.nn.functional.conv1d,
        dtypes=[
            torch.float16,
            torch.float32,
        ],  # Exclude bfloat16 due to cuDNN limitations
    )
    bench.set_gems(flag_gems.conv1d)
    bench.run()


class Conv2DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
            (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
            (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
            (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
            (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
            (16, 32, 12, 12, 24, 3, 3, 2, "valid", 1),
            (32, 64, 128, 128, 32, 3, 3, 1, "valid", 1),
            (16, 32, 24, 24, 24, 3, 3, 1, "same", 2),
            (32, 64, 210, 210, 16, 5, 5, 1, "same", 1),
        ]


@pytest.mark.conv2d
def test_perf_conv2d():
    def conv2d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_h,
            input_w,
            out_c,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_h, input_w)
        weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    if flag_gems.vendor_name == "hygon":
        os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"] = "0"
    torch.backends.cudnn.allow_tf32 = False
    bench = Conv2DBenchmark(
        input_fn=conv2d_input_fn,
        op_name="conv2d",
        torch_op=torch.nn.functional.conv2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv2d)
    bench.run()
    if flag_gems.vendor_name == "hygon":
        del os.environ["TRITON_HIP_USE_NEW_STREAM_PIPELINE"]


class ConvTranspose2DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return [
            # (batch, in_c, in_h, in_w, out_c, kh, kw, stride, padding, groups)
            (32, 64, 16, 16, 32, 3, 3, 1, 1, 1),
            (32, 64, 16, 16, 32, 3, 3, 2, 1, 1),
            (16, 32, 32, 32, 64, 3, 3, 2, 1, 1),
            (16, 32, 8, 8, 24, 5, 5, 2, 2, 1),
            (8, 128, 4, 4, 64, 4, 4, 2, 1, 1),
            (4, 256, 8, 8, 128, 3, 3, 2, 1, 1),
        ]


@pytest.mark.conv_transpose2d
def test_perf_conv_transpose2d():
    def conv_transpose2d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_h,
            input_w,
            out_c,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_h, input_w)
        # weight shape for conv_transpose2d: (in_channels, out_channels/groups, kH, kW)
        weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
        inp = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": inp,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        input_fn=conv_transpose2d_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_transpose2d)
    bench.run()


class Conv3DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.conv3d
def test_perf_conv3d():
    def conv3d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_d,
            input_h,
            input_w,
            out_c,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_d, input_h, input_w)
        weight_shape = (out_c, input_c // groups, kernel_d, kernel_h, kernel_w)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = Conv3DBenchmark(
        input_fn=conv3d_input_fn,
        op_name="conv3d",
        torch_op=torch.nn.functional.conv3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv3d)
    bench.run()
