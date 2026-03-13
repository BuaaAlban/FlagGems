import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Interpolation mode constants (matching PyTorch aten)
MODE_BILINEAR = 0
MODE_NEAREST = 1
MODE_BICUBIC = 2

# Padding mode constants
PAD_ZEROS = 0
PAD_BORDER = 1
PAD_REFLECTION = 2


# ============================================================
# Helper: coordinate transform (unnormalize grid value)
# ============================================================
@triton.jit
def _unnormalize(coord, size, align_corners: tl.constexpr):
    if align_corners:
        return ((coord + 1.0) * 0.5) * (size - 1)
    else:
        return ((coord + 1.0) * size - 1.0) * 0.5


@triton.jit
def _reflect_coord(coord, twice_low, twice_high):
    # Matches PyTorch's reflect_coordinates exactly:
    # min = twice_low / 2, span = (twice_high - twice_low) / 2
    # coord = abs(coord - min)
    # extra = fmod(coord, span), flips = floor(coord / span)
    # result = (extra + min) if flips even, else (span - extra + min)
    min_val = twice_low * 0.5
    span = (twice_high - twice_low) * 0.5
    coord = tl.abs(coord - min_val)
    # fmod: extra = coord - floor(coord / span) * span
    div = coord / span
    flips = tl.floor(div).to(tl.int32)
    extra = coord - flips.to(tl.float32) * span
    is_even = (flips % 2) == 0
    coord = tl.where(is_even, extra + min_val, span - extra + min_val)
    return coord


@triton.jit
def _compute_source_index(
    coord,
    size,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
):
    coord = _unnormalize(coord, size, align_corners)
    # padding_mode: 0=zeros, 1=border, 2=reflection
    if padding_mode == 1:  # border
        coord = tl.minimum(tl.maximum(coord, 0.0), size - 1.0)
    elif padding_mode == 2:  # reflection
        if align_corners:
            coord = _reflect_coord(coord, 0.0, 2.0 * (size - 1))
        else:
            coord = _reflect_coord(coord, -1.0, 2.0 * size - 1.0)
        coord = tl.minimum(tl.maximum(coord, 0.0), size - 1.0)
    # For zeros (0), no clamping needed — we use masking at load time
    return coord


@triton.jit
def _nearbyint(x):
    """Round to nearest integer with ties going to nearest even (banker's rounding).

    Matches std::nearbyint / IEEE 754 default rounding, which PyTorch uses
    for nearest-mode grid_sampler.
    """
    rounded = tl.floor(x + 0.5)
    # When fractional part is exactly 0.5, round to even
    frac = x - tl.floor(x)
    is_half = frac == 0.5
    # If rounded is odd, subtract 1 to get even
    rounded_int = rounded.to(tl.int64)
    is_odd = (rounded_int % 2) != 0
    rounded = tl.where(is_half & is_odd, rounded - 1.0, rounded)
    return rounded


@triton.jit
def _in_bounds(ix, iy, IH, IW):
    return (ix >= 0) & (ix < IW) & (iy >= 0) & (iy < IH)


@triton.jit
def _in_bounds_3d(ix, iy, iz, ID, IH, IW):
    return (ix >= 0) & (ix < IW) & (iy >= 0) & (iy < IH) & (iz >= 0) & (iz < ID)


@triton.jit
def _cubic_interp_weight(t):
    # Cubic convolution with alpha = -0.75 (matching PyTorch)
    A = -0.75
    abs_t = tl.abs(t)
    abs_t2 = abs_t * abs_t
    abs_t3 = abs_t2 * abs_t
    w = tl.where(
        abs_t <= 1.0,
        (A + 2.0) * abs_t3 - (A + 3.0) * abs_t2 + 1.0,
        tl.where(
            abs_t < 2.0,
            A * abs_t3 - 5.0 * A * abs_t2 + 8.0 * A * abs_t - 4.0 * A,
            0.0,
        ),
    )
    return w


@triton.jit
def _clip_or_reflect_1d(
    ix, size, padding_mode: tl.constexpr, align_corners: tl.constexpr
):
    # Per-point clip/reflect for bicubic neighbor coordinates
    if padding_mode == 1:  # border
        return tl.minimum(tl.maximum(ix.to(tl.float32), 0.0), size - 1.0).to(tl.int64)
    elif padding_mode == 2:  # reflection
        ix_f = ix.to(tl.float32)
        if align_corners:
            ix_f = _reflect_coord(ix_f, 0.0, 2.0 * (size - 1))
        else:
            ix_f = _reflect_coord(ix_f, -1.0, 2.0 * size - 1.0)
        return tl.minimum(tl.maximum(ix_f, 0.0), size - 1.0).to(tl.int64)
    else:
        return ix  # zeros: keep original, check bounds at load time


# ============================================================
# 4D Kernels: grid_sampler_2d
# ============================================================


@libentry()
@triton.jit
def grid_sampler_2d_nearest_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    inp_sN,
    inp_sC,
    inp_sH,
    inp_sW,
    grid_sN,
    grid_sH,
    grid_sW,
    grid_sC,
    out_sN,
    out_sC,
    out_sH,
    out_sW,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n = tle.program_id(1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < OH * OW
    ow = idx % OW
    oh = idx // OW

    # Load grid coordinates
    grid_offset = n * grid_sN + oh * grid_sH + ow * grid_sW
    gx = tl.load(grid_ptr + grid_offset + 0 * grid_sC, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1 * grid_sC, mask=mask, other=0.0)

    # Cast to float32 for coordinate computation and handle NaN as -1
    gx = tl.where(gx != gx, -1.0, gx).to(tl.float32)
    gy = tl.where(gy != gy, -1.0, gy).to(tl.float32)

    ix = _compute_source_index(gx, IW, padding_mode, align_corners)
    iy = _compute_source_index(gy, IH, padding_mode, align_corners)

    # Nearest: round to nearest integer (banker's rounding, matching PyTorch)
    ix_nearest = _nearbyint(ix).to(tl.int64)
    iy_nearest = _nearbyint(iy).to(tl.int64)

    in_bound = _in_bounds(ix_nearest, iy_nearest, IH, IW)
    safe_ix = tl.where(in_bound, ix_nearest, 0)
    safe_iy = tl.where(in_bound, iy_nearest, 0)

    inp_offset = n * inp_sN + safe_iy * inp_sH + safe_ix * inp_sW

    for c in range(0, C):
        val = tl.load(
            input_ptr + inp_offset + c * inp_sC, mask=mask & in_bound, other=0.0
        )
        out_offset = n * out_sN + c * out_sC + oh * out_sH + ow * out_sW
        tl.store(output_ptr + out_offset, val, mask=mask)


@libentry()
@triton.jit
def grid_sampler_2d_bilinear_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    inp_sN,
    inp_sC,
    inp_sH,
    inp_sW,
    grid_sN,
    grid_sH,
    grid_sW,
    grid_sC,
    out_sN,
    out_sC,
    out_sH,
    out_sW,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n = tle.program_id(1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < OH * OW
    ow = idx % OW
    oh = idx // OW

    grid_offset = n * grid_sN + oh * grid_sH + ow * grid_sW
    gx = tl.load(grid_ptr + grid_offset + 0 * grid_sC, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1 * grid_sC, mask=mask, other=0.0)

    gx = tl.where(gx != gx, -1.0, gx).to(tl.float32)
    gy = tl.where(gy != gy, -1.0, gy).to(tl.float32)

    ix = _compute_source_index(gx, IW, padding_mode, align_corners)
    iy = _compute_source_index(gy, IH, padding_mode, align_corners)

    # Bilinear interpolation: 4 corners
    ix0 = tl.floor(ix).to(tl.int64)
    iy0 = tl.floor(iy).to(tl.int64)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # Fractional weights
    wx = ix - ix0.to(tl.float32)
    wy = iy - iy0.to(tl.float32)

    # Bound checks for zero-padding
    if padding_mode == 0:  # zeros
        m00 = _in_bounds(ix0, iy0, IH, IW)
        m01 = _in_bounds(ix1, iy0, IH, IW)
        m10 = _in_bounds(ix0, iy1, IH, IW)
        m11 = _in_bounds(ix1, iy1, IH, IW)
    else:
        m00 = mask
        m01 = mask
        m10 = mask
        m11 = mask

    # Clamp indices for safe load (border/reflection already in range)
    safe_ix0 = tl.minimum(tl.maximum(ix0, 0), IW - 1)
    safe_iy0 = tl.minimum(tl.maximum(iy0, 0), IH - 1)
    safe_ix1 = tl.minimum(tl.maximum(ix1, 0), IW - 1)
    safe_iy1 = tl.minimum(tl.maximum(iy1, 0), IH - 1)

    base = n * inp_sN
    off00 = base + safe_iy0 * inp_sH + safe_ix0 * inp_sW
    off01 = base + safe_iy0 * inp_sH + safe_ix1 * inp_sW
    off10 = base + safe_iy1 * inp_sH + safe_ix0 * inp_sW
    off11 = base + safe_iy1 * inp_sH + safe_ix1 * inp_sW

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    for c in range(0, C):
        c_off = c * inp_sC
        v00 = tl.load(input_ptr + off00 + c_off, mask=mask & m00, other=0.0).to(
            tl.float32
        )
        v01 = tl.load(input_ptr + off01 + c_off, mask=mask & m01, other=0.0).to(
            tl.float32
        )
        v10 = tl.load(input_ptr + off10 + c_off, mask=mask & m10, other=0.0).to(
            tl.float32
        )
        v11 = tl.load(input_ptr + off11 + c_off, mask=mask & m11, other=0.0).to(
            tl.float32
        )

        result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
        out_offset = n * out_sN + c * out_sC + oh * out_sH + ow * out_sW
        tl.store(output_ptr + out_offset, result, mask=mask)


@libentry()
@triton.jit
def grid_sampler_2d_bicubic_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    N,
    C,
    IH,
    IW,
    OH,
    OW,
    inp_sN,
    inp_sC,
    inp_sH,
    inp_sW,
    grid_sN,
    grid_sH,
    grid_sW,
    grid_sC,
    out_sN,
    out_sC,
    out_sH,
    out_sW,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n = tle.program_id(1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < OH * OW
    ow = idx % OW
    oh = idx // OW

    grid_offset = n * grid_sN + oh * grid_sH + ow * grid_sW
    gx = tl.load(grid_ptr + grid_offset + 0 * grid_sC, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1 * grid_sC, mask=mask, other=0.0)

    gx = tl.where(gx != gx, -1.0, gx).to(tl.float32)
    gy = tl.where(gy != gy, -1.0, gy).to(tl.float32)

    # For bicubic: unnormalize only (no center clamping).
    # Each neighbor point is individually clipped/reflected.
    ix = _unnormalize(gx, IW, align_corners)
    iy = _unnormalize(gy, IH, align_corners)

    ix_floor = tl.floor(ix).to(tl.int64)
    iy_floor = tl.floor(iy).to(tl.int64)

    tx = ix - ix_floor.to(tl.float32)
    ty = iy - iy_floor.to(tl.float32)

    # Compute cubic weights for x and y
    wx_m1 = _cubic_interp_weight(tx + 1.0)
    wx_0 = _cubic_interp_weight(tx)
    wx_1 = _cubic_interp_weight(tx - 1.0)
    wx_2 = _cubic_interp_weight(tx - 2.0)

    wy_m1 = _cubic_interp_weight(ty + 1.0)
    wy_0 = _cubic_interp_weight(ty)
    wy_1 = _cubic_interp_weight(ty - 1.0)
    wy_2 = _cubic_interp_weight(ty - 2.0)

    base = n * inp_sN

    for c in range(0, C):
        c_off = c * inp_sC
        result = tl.zeros_like(tx)

        # 4 rows (dy = -1, 0, 1, 2)
        for dy in range(-1, 3):
            iy_val = iy_floor + dy
            safe_iy = _clip_or_reflect_1d(iy_val, IH, padding_mode, align_corners)
            if padding_mode == 0:  # zeros
                y_in = (iy_val >= 0) & (iy_val < IH)
            else:
                y_in = mask

            # Accumulate across 4 columns
            row_val = tl.zeros_like(tx)
            for dx in range(-1, 3):
                ix_val = ix_floor + dx
                safe_ix = _clip_or_reflect_1d(ix_val, IW, padding_mode, align_corners)
                if padding_mode == 0:  # zeros
                    in_bound = y_in & (ix_val >= 0) & (ix_val < IW)
                else:
                    in_bound = y_in
                off = base + c_off + safe_iy * inp_sH + safe_ix * inp_sW
                val = tl.load(input_ptr + off, mask=mask & in_bound, other=0.0).to(
                    tl.float32
                )
                if dx == -1:
                    row_val += val * wx_m1
                elif dx == 0:
                    row_val += val * wx_0
                elif dx == 1:
                    row_val += val * wx_1
                else:
                    row_val += val * wx_2

            if dy == -1:
                result += row_val * wy_m1
            elif dy == 0:
                result += row_val * wy_0
            elif dy == 1:
                result += row_val * wy_1
            else:
                result += row_val * wy_2

        out_offset = n * out_sN + c * out_sC + oh * out_sH + ow * out_sW
        tl.store(output_ptr + out_offset, result, mask=mask)


# ============================================================
# 5D Kernels: grid_sampler_3d
# ============================================================


@libentry()
@triton.jit
def grid_sampler_3d_nearest_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    N,
    C,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    inp_sN,
    inp_sC,
    inp_sD,
    inp_sH,
    inp_sW,
    grid_sN,
    grid_sD,
    grid_sH,
    grid_sW,
    grid_sC,
    out_sN,
    out_sC,
    out_sD,
    out_sH,
    out_sW,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n = tle.program_id(1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = OD * OH * OW
    mask = idx < total
    ow = idx % OW
    oh = (idx // OW) % OH
    od = idx // (OW * OH)

    grid_offset = n * grid_sN + od * grid_sD + oh * grid_sH + ow * grid_sW
    gx = tl.load(grid_ptr + grid_offset + 0 * grid_sC, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1 * grid_sC, mask=mask, other=0.0)
    gz = tl.load(grid_ptr + grid_offset + 2 * grid_sC, mask=mask, other=0.0)

    gx = tl.where(gx != gx, -1.0, gx).to(tl.float32)
    gy = tl.where(gy != gy, -1.0, gy).to(tl.float32)
    gz = tl.where(gz != gz, -1.0, gz).to(tl.float32)

    ix = _compute_source_index(gx, IW, padding_mode, align_corners)
    iy = _compute_source_index(gy, IH, padding_mode, align_corners)
    iz = _compute_source_index(gz, ID, padding_mode, align_corners)

    ix_nearest = _nearbyint(ix).to(tl.int64)
    iy_nearest = _nearbyint(iy).to(tl.int64)
    iz_nearest = _nearbyint(iz).to(tl.int64)

    in_bound = _in_bounds_3d(ix_nearest, iy_nearest, iz_nearest, ID, IH, IW)
    safe_ix = tl.where(in_bound, ix_nearest, 0)
    safe_iy = tl.where(in_bound, iy_nearest, 0)
    safe_iz = tl.where(in_bound, iz_nearest, 0)

    inp_offset = n * inp_sN + safe_iz * inp_sD + safe_iy * inp_sH + safe_ix * inp_sW

    for c in range(0, C):
        val = tl.load(
            input_ptr + inp_offset + c * inp_sC, mask=mask & in_bound, other=0.0
        )
        out_offset = n * out_sN + c * out_sC + od * out_sD + oh * out_sH + ow * out_sW
        tl.store(output_ptr + out_offset, val, mask=mask)


@libentry()
@triton.jit
def grid_sampler_3d_trilinear_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    N,
    C,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    inp_sN,
    inp_sC,
    inp_sD,
    inp_sH,
    inp_sW,
    grid_sN,
    grid_sD,
    grid_sH,
    grid_sW,
    grid_sC,
    out_sN,
    out_sC,
    out_sD,
    out_sH,
    out_sW,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    n = tle.program_id(1)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = OD * OH * OW
    mask = idx < total
    ow = idx % OW
    oh = (idx // OW) % OH
    od = idx // (OW * OH)

    grid_offset = n * grid_sN + od * grid_sD + oh * grid_sH + ow * grid_sW
    gx = tl.load(grid_ptr + grid_offset + 0 * grid_sC, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1 * grid_sC, mask=mask, other=0.0)
    gz = tl.load(grid_ptr + grid_offset + 2 * grid_sC, mask=mask, other=0.0)

    gx = tl.where(gx != gx, -1.0, gx).to(tl.float32)
    gy = tl.where(gy != gy, -1.0, gy).to(tl.float32)
    gz = tl.where(gz != gz, -1.0, gz).to(tl.float32)

    ix = _compute_source_index(gx, IW, padding_mode, align_corners)
    iy = _compute_source_index(gy, IH, padding_mode, align_corners)
    iz = _compute_source_index(gz, ID, padding_mode, align_corners)

    # Trilinear: 8 corners
    ix0 = tl.floor(ix).to(tl.int64)
    iy0 = tl.floor(iy).to(tl.int64)
    iz0 = tl.floor(iz).to(tl.int64)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    wx = ix - ix0.to(tl.float32)
    wy = iy - iy0.to(tl.float32)
    wz = iz - iz0.to(tl.float32)

    # 8 weight combos
    w000 = (1.0 - wx) * (1.0 - wy) * (1.0 - wz)
    w001 = wx * (1.0 - wy) * (1.0 - wz)
    w010 = (1.0 - wx) * wy * (1.0 - wz)
    w011 = wx * wy * (1.0 - wz)
    w100 = (1.0 - wx) * (1.0 - wy) * wz
    w101 = wx * (1.0 - wy) * wz
    w110 = (1.0 - wx) * wy * wz
    w111 = wx * wy * wz

    # Bound masks for zero-padding
    if padding_mode == 0:  # zeros
        m000 = _in_bounds_3d(ix0, iy0, iz0, ID, IH, IW)
        m001 = _in_bounds_3d(ix1, iy0, iz0, ID, IH, IW)
        m010 = _in_bounds_3d(ix0, iy1, iz0, ID, IH, IW)
        m011 = _in_bounds_3d(ix1, iy1, iz0, ID, IH, IW)
        m100 = _in_bounds_3d(ix0, iy0, iz1, ID, IH, IW)
        m101 = _in_bounds_3d(ix1, iy0, iz1, ID, IH, IW)
        m110 = _in_bounds_3d(ix0, iy1, iz1, ID, IH, IW)
        m111 = _in_bounds_3d(ix1, iy1, iz1, ID, IH, IW)
    else:
        m000 = mask
        m001 = mask
        m010 = mask
        m011 = mask
        m100 = mask
        m101 = mask
        m110 = mask
        m111 = mask

    # Safe indices
    s_ix0 = tl.minimum(tl.maximum(ix0, 0), IW - 1)
    s_iy0 = tl.minimum(tl.maximum(iy0, 0), IH - 1)
    s_iz0 = tl.minimum(tl.maximum(iz0, 0), ID - 1)
    s_ix1 = tl.minimum(tl.maximum(ix1, 0), IW - 1)
    s_iy1 = tl.minimum(tl.maximum(iy1, 0), IH - 1)
    s_iz1 = tl.minimum(tl.maximum(iz1, 0), ID - 1)

    base = n * inp_sN

    off000 = base + s_iz0 * inp_sD + s_iy0 * inp_sH + s_ix0 * inp_sW
    off001 = base + s_iz0 * inp_sD + s_iy0 * inp_sH + s_ix1 * inp_sW
    off010 = base + s_iz0 * inp_sD + s_iy1 * inp_sH + s_ix0 * inp_sW
    off011 = base + s_iz0 * inp_sD + s_iy1 * inp_sH + s_ix1 * inp_sW
    off100 = base + s_iz1 * inp_sD + s_iy0 * inp_sH + s_ix0 * inp_sW
    off101 = base + s_iz1 * inp_sD + s_iy0 * inp_sH + s_ix1 * inp_sW
    off110 = base + s_iz1 * inp_sD + s_iy1 * inp_sH + s_ix0 * inp_sW
    off111 = base + s_iz1 * inp_sD + s_iy1 * inp_sH + s_ix1 * inp_sW

    for c in range(0, C):
        c_off = c * inp_sC
        v000 = tl.load(input_ptr + off000 + c_off, mask=mask & m000, other=0.0).to(
            tl.float32
        )
        v001 = tl.load(input_ptr + off001 + c_off, mask=mask & m001, other=0.0).to(
            tl.float32
        )
        v010 = tl.load(input_ptr + off010 + c_off, mask=mask & m010, other=0.0).to(
            tl.float32
        )
        v011 = tl.load(input_ptr + off011 + c_off, mask=mask & m011, other=0.0).to(
            tl.float32
        )
        v100 = tl.load(input_ptr + off100 + c_off, mask=mask & m100, other=0.0).to(
            tl.float32
        )
        v101 = tl.load(input_ptr + off101 + c_off, mask=mask & m101, other=0.0).to(
            tl.float32
        )
        v110 = tl.load(input_ptr + off110 + c_off, mask=mask & m110, other=0.0).to(
            tl.float32
        )
        v111 = tl.load(input_ptr + off111 + c_off, mask=mask & m111, other=0.0).to(
            tl.float32
        )

        result = (
            w000 * v000
            + w001 * v001
            + w010 * v010
            + w011 * v011
            + w100 * v100
            + w101 * v101
            + w110 * v110
            + w111 * v111
        )
        out_offset = n * out_sN + c * out_sC + od * out_sD + oh * out_sH + ow * out_sW
        tl.store(output_ptr + out_offset, result, mask=mask)


# ============================================================
# Python wrappers
# ============================================================

BLOCK_SIZE = 256


def grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners):
    logger.debug("GEMS GRID_SAMPLER_2D")
    assert input.ndim == 4, "input must be 4D (N, C, H, W)"
    assert grid.ndim == 4, "grid must be 4D (N, H_out, W_out, 2)"
    assert grid.shape[-1] == 2, "grid last dim must be 2"
    assert input.shape[0] == grid.shape[0], "batch size mismatch"

    N, C, IH, IW = input.shape
    _, OH, OW, _ = grid.shape

    # Ensure contiguous for stride-based access
    input = input.contiguous()
    grid = grid.contiguous()

    output = torch.empty((N, C, OH, OW), device=input.device, dtype=input.dtype)

    if N == 0 or C == 0 or OH == 0 or OW == 0:
        return output

    total = OH * OW
    grid_fn = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]), N)

    # Select kernel based on interpolation mode
    if interpolation_mode == MODE_NEAREST:
        kernel = grid_sampler_2d_nearest_kernel
    elif interpolation_mode == MODE_BILINEAR:
        kernel = grid_sampler_2d_bilinear_kernel
    elif interpolation_mode == MODE_BICUBIC:
        kernel = grid_sampler_2d_bicubic_kernel
    else:
        raise ValueError(f"Unsupported interpolation mode: {interpolation_mode}")

    with torch_device_fn.device(input.device):
        kernel[grid_fn](
            output,
            input,
            grid,
            N,
            C,
            IH,
            IW,
            OH,
            OW,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            grid.stride(0),
            grid.stride(1),
            grid.stride(2),
            grid.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            padding_mode=padding_mode,
            align_corners=align_corners,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output


def grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners):
    logger.debug("GEMS GRID_SAMPLER_3D")
    assert input.ndim == 5, "input must be 5D (N, C, D, H, W)"
    assert grid.ndim == 5, "grid must be 5D (N, D_out, H_out, W_out, 3)"
    assert grid.shape[-1] == 3, "grid last dim must be 3"
    assert input.shape[0] == grid.shape[0], "batch size mismatch"

    N, C, ID, IH, IW = input.shape
    _, OD, OH, OW, _ = grid.shape

    input = input.contiguous()
    grid = grid.contiguous()

    output = torch.empty((N, C, OD, OH, OW), device=input.device, dtype=input.dtype)

    if N == 0 or C == 0 or OD == 0 or OH == 0 or OW == 0:
        return output

    total = OD * OH * OW
    grid_fn = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]), N)

    if interpolation_mode == MODE_NEAREST:
        kernel = grid_sampler_3d_nearest_kernel
    elif interpolation_mode == MODE_BILINEAR:
        # For 5D input, "bilinear" mode actually uses trilinear
        kernel = grid_sampler_3d_trilinear_kernel
    else:
        raise ValueError(
            f"Unsupported interpolation mode for 5D: {interpolation_mode}. "
            "Only bilinear(trilinear) and nearest are supported."
        )

    with torch_device_fn.device(input.device):
        kernel[grid_fn](
            output,
            input,
            grid,
            N,
            C,
            ID,
            IH,
            IW,
            OD,
            OH,
            OW,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            input.stride(4),
            grid.stride(0),
            grid.stride(1),
            grid.stride(2),
            grid.stride(3),
            grid.stride(4),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            output.stride(4),
            padding_mode=padding_mode,
            align_corners=align_corners,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return output
