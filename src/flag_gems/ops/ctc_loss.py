import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ============================================================================
# CTC Loss - Connectionist Temporal Classification
#
# Reference: Graves et al., "Connectionist Temporal Classification:
# Labelling Unsegmented Sequence Data with Recurrent Neural Networks", 2006
#
# Algorithm:
# 1. Expand target labels by inserting blanks: [b, l1, b, l2, b, ..., lS, b]
# 2. Forward pass (alpha): DP computing log probability of all valid paths
# 3. NLL = -log_sum_exp(alpha[T-1][S'-1], alpha[T-1][S'-2])
# 4. Backward pass (beta): reverse DP computed on-the-fly during backward
# 5. Gradient: grad[t,c] = prob[t,c] - exp(ab_sum - log_probs[t,c] + nll)
#
# All DP computation (alpha, beta, NLL, gradient) is implemented as Triton
# kernels for GPU acceleration.
# ============================================================================

NEG_INF = float("-inf")


@triton.jit
def _tl_log_sum_exp(a, b):
    """Stable log(exp(a) + exp(b)) in Triton."""
    max_v = tl.maximum(a, b)
    result = max_v + tl.log(tl.exp(a - max_v) + tl.exp(b - max_v))
    neg_inf = float("-inf")
    both_neg = (a == neg_inf) & (b == neg_inf)
    return tl.where(both_neg, neg_inf, result)


@libentry()
@triton.jit
def ctc_alpha_kernel(
    # Inputs
    log_probs_ptr,  # (T, N, C) float32
    expanded_labels_ptr,  # (N, S_prime_max) int64
    input_lengths_ptr,  # (N,) int64
    target_lengths_ptr,  # (N,) int64
    # Outputs
    log_alpha_ptr,  # (N, T, S_prime_max) float32
    nll_ptr,  # (N,) float32
    # Dimensions
    T_max,
    N,
    C,
    S_prime_max: tl.constexpr,
    blank: tl.constexpr,
    # Strides
    lp_stride_t,
    lp_stride_n,
    lp_stride_c,
    alpha_stride_n,
    alpha_stride_t,
    alpha_stride_s,
    el_stride_n,
    el_stride_s,
    # Block size
    BLOCK_S: tl.constexpr,
):
    """Compute forward (alpha) DP table and NLL for one batch element.

    Grid: (N,) - one program per batch element.
    Sequential loop over T timesteps, parallel across S' label positions.
    """
    batch_idx = tle.program_id(0)
    if batch_idx >= N:
        return

    neg_inf = float("-inf")

    T_i = tl.load(input_lengths_ptr + batch_idx).to(tl.int64)
    tgt_len = tl.load(target_lengths_ptr + batch_idx).to(tl.int64)
    S_prime_i = 2 * tgt_len + 1

    # Base pointers for this batch
    alpha_base = log_alpha_ptr + batch_idx * alpha_stride_n
    el_base = expanded_labels_ptr + batch_idx * el_stride_n
    lp_base = log_probs_ptr + batch_idx * lp_stride_n

    # Load expanded labels for this batch: labels[0..S_prime_max-1]
    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S_prime_max

    labels = tl.load(el_base + s_offsets * el_stride_s, mask=s_mask, other=0).to(
        tl.int64
    )

    # Initialize all alpha to -inf for all timesteps
    # We'll write valid values below; invalid positions stay -inf

    # ---- Timestep t=0 initialization ----
    # alpha[0, s] = log_probs[0, labels[s]] for s in {0, 1} (if valid)
    # alpha[0, s] = -inf for s >= 2

    # Load log_probs[0, batch_idx, labels[s]] for all s
    # log_probs layout: (T, N, C), so lp[0, batch_idx, c] = lp_base + 0 * lp_stride_t + c * lp_stride_c
    emit_ptrs = lp_base + 0 * lp_stride_t + labels * lp_stride_c
    emit_vals = tl.load(emit_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    # alpha[0, 0] = emit[0] (blank)
    # alpha[0, 1] = emit[1] if tgt_len > 0 else -inf
    # alpha[0, s>=2] = -inf
    alpha_cur = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
    alpha_cur = tl.where(s_offsets == 0, emit_vals, alpha_cur)
    alpha_cur = tl.where((s_offsets == 1) & (tgt_len > 0), emit_vals, alpha_cur)

    # Mask out positions beyond S_prime_i
    valid_s = s_offsets < S_prime_i
    alpha_cur = tl.where(valid_s, alpha_cur, neg_inf)

    # Store alpha[batch, 0, :]
    alpha_store_ptrs = alpha_base + 0 * alpha_stride_t + s_offsets * alpha_stride_s
    tl.store(alpha_store_ptrs, alpha_cur, mask=s_mask)

    # ---- Forward DP: t = 1 to T_i - 1 ----
    for t in range(1, T_max):
        # prev = alpha[t-1, :]  (already in alpha_cur from last iteration)
        prev = alpha_cur

        # Transition 1: stay at s → prev[s]
        trans = prev

        # Transition 2: from s-1 → prev[s-1]
        # Shift prev right by 1: prev_m1[s] = prev[s-1] for s>=1, -inf for s=0
        # Use tl.where to handle the shift
        prev_m1 = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
        # For s >= 1: prev_m1[s] = prev[s-1]
        # We can use a shifted load: read prev at s_offsets - 1
        shifted_mask = (s_offsets >= 1) & (s_offsets < S_prime_max)
        shifted_ptrs = (
            alpha_base + (t - 1) * alpha_stride_t + (s_offsets - 1) * alpha_stride_s
        )
        prev_m1_loaded = tl.load(shifted_ptrs, mask=shifted_mask, other=neg_inf).to(
            tl.float32
        )
        prev_m1 = tl.where(s_offsets >= 1, prev_m1_loaded, neg_inf)

        trans = _tl_log_sum_exp(trans, prev_m1)

        # Transition 3: skip blank from s-2 → prev[s-2]
        # Only if: s >= 2 AND labels[s] != blank AND labels[s] != labels[s-2]
        prev_m2 = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
        shifted2_mask = (s_offsets >= 2) & (s_offsets < S_prime_max)
        shifted2_ptrs = (
            alpha_base + (t - 1) * alpha_stride_t + (s_offsets - 2) * alpha_stride_s
        )
        prev_m2_loaded = tl.load(shifted2_ptrs, mask=shifted2_mask, other=neg_inf).to(
            tl.float32
        )

        # Load labels[s-2] for comparison
        labels_m2_ptrs = el_base + (s_offsets - 2) * el_stride_s
        labels_m2 = tl.load(labels_m2_ptrs, mask=shifted2_mask, other=blank).to(
            tl.int64
        )

        can_skip = (s_offsets >= 2) & (labels != blank) & (labels != labels_m2)
        prev_m2 = tl.where(can_skip, prev_m2_loaded, neg_inf)
        trans = _tl_log_sum_exp(trans, prev_m2)

        # Emission: log_probs[t, batch_idx, labels[s]]
        emit_ptrs_t = lp_base + t * lp_stride_t + labels * lp_stride_c
        emit_t = tl.load(emit_ptrs_t, mask=s_mask, other=0.0).to(tl.float32)

        alpha_new = trans + emit_t

        # Mask invalid: s >= S_prime_i or t >= T_i
        valid = (s_offsets < S_prime_i) & (t < T_i)
        alpha_cur = tl.where(valid, alpha_new, neg_inf)

        # Store alpha[batch, t, :]
        alpha_store_ptrs_t = (
            alpha_base + t * alpha_stride_t + s_offsets * alpha_stride_s
        )
        tl.store(alpha_store_ptrs_t, alpha_cur, mask=s_mask)

    # ---- Compute NLL from alpha ----
    # NLL = -log_sum_exp(alpha[T_i-1, S_prime_i-1], alpha[T_i-1, S_prime_i-2])
    # When tgt_len == 0 (S_prime_i == 1), only alpha[T_i-1, 0] is valid,
    # which naturally equals sum(log_probs[0:T_i, blank]) from DP.
    if tgt_len > 0:
        a_last = tl.load(
            alpha_base + (T_i - 1) * alpha_stride_t + (S_prime_i - 1) * alpha_stride_s
        ).to(tl.float32)
        a_second = tl.load(
            alpha_base + (T_i - 1) * alpha_stride_t + (S_prime_i - 2) * alpha_stride_s
        ).to(tl.float32)
        nll_val = -_tl_log_sum_exp(a_last, a_second)
    else:
        a_only = tl.load(
            alpha_base + (T_i - 1) * alpha_stride_t + 0 * alpha_stride_s
        ).to(tl.float32)
        nll_val = -a_only

    tl.store(nll_ptr + batch_idx, nll_val)


@libentry()
@triton.jit
def ctc_beta_kernel(
    # Inputs
    log_probs_ptr,  # (T, N, C) float32
    expanded_labels_ptr,  # (N, S_prime_max) int64
    input_lengths_ptr,  # (N,) int64
    target_lengths_ptr,  # (N,) int64
    # Outputs
    log_beta_ptr,  # (N, T, S_prime_max) float32
    # Dimensions
    T_max,
    N,
    C,
    S_prime_max: tl.constexpr,
    blank: tl.constexpr,
    # Strides
    lp_stride_t,
    lp_stride_n,
    lp_stride_c,
    beta_stride_n,
    beta_stride_t,
    beta_stride_s,
    el_stride_n,
    el_stride_s,
    # Block size
    BLOCK_S: tl.constexpr,
):
    """Compute backward (beta) DP table for one batch element.

    Grid: (N,) - one program per batch element.
    Sequential reverse loop over T timesteps, parallel across S' positions.
    """
    batch_idx = tle.program_id(0)
    if batch_idx >= N:
        return

    neg_inf = float("-inf")

    T_i = tl.load(input_lengths_ptr + batch_idx).to(tl.int64)
    tgt_len = tl.load(target_lengths_ptr + batch_idx).to(tl.int64)
    S_prime_i = 2 * tgt_len + 1

    # Base pointers
    beta_base = log_beta_ptr + batch_idx * beta_stride_n
    el_base = expanded_labels_ptr + batch_idx * el_stride_n
    lp_base = log_probs_ptr + batch_idx * lp_stride_n

    s_offsets = tl.arange(0, BLOCK_S)
    s_mask = s_offsets < S_prime_max

    # Load labels
    labels = tl.load(el_base + s_offsets * el_stride_s, mask=s_mask, other=0).to(
        tl.int64
    )

    # ---- Initialize beta[T_i-1] ----
    # beta[T_i-1, S_prime_i-1] = log_probs[T_i-1, labels[S_prime_i-1]]
    # beta[T_i-1, S_prime_i-2] = log_probs[T_i-1, labels[S_prime_i-2]]
    # Everything else = -inf

    # Load emission at T_i-1
    emit_ptrs_last = lp_base + (T_i - 1) * lp_stride_t + labels * lp_stride_c
    emit_last = tl.load(emit_ptrs_last, mask=s_mask, other=0.0).to(tl.float32)

    beta_cur = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
    beta_cur = tl.where(
        (s_offsets == S_prime_i - 1) & (S_prime_i > 0), emit_last, beta_cur
    )
    beta_cur = tl.where(
        (s_offsets == S_prime_i - 2) & (S_prime_i > 1), emit_last, beta_cur
    )

    # Store beta[batch, T_i-1, :]
    beta_store_ptrs_last = (
        beta_base + (T_i - 1) * beta_stride_t + s_offsets * beta_stride_s
    )
    tl.store(beta_store_ptrs_last, beta_cur, mask=s_mask)

    # Initialize all other timesteps to -inf (for timesteps t >= T_i)
    # We do this by iterating backward and only writing valid values

    # ---- Backward DP: t = T_i - 2 down to 0 ----
    # We iterate from T_max-2 down to 0 but only compute for t < T_i
    for t_offset in range(1, T_max):
        t = T_i - 1 - t_offset  # actual timestep

        if t >= 0:
            # next_beta = beta[t+1, :] (stored in beta_cur from last iter)
            next_beta = beta_cur

            # Transition 1: stay at s → next_beta[s]
            trans = next_beta

            # Transition 2: from s+1 → next_beta[s+1]
            next_p1 = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
            shifted_mask = (s_offsets + 1 < S_prime_max) & s_mask
            shifted_ptrs = (
                beta_base + (t + 1) * beta_stride_t + (s_offsets + 1) * beta_stride_s
            )
            next_p1_loaded = tl.load(shifted_ptrs, mask=shifted_mask, other=neg_inf).to(
                tl.float32
            )
            next_p1 = tl.where(s_offsets + 1 < S_prime_max, next_p1_loaded, neg_inf)
            trans = _tl_log_sum_exp(trans, next_p1)

            # Transition 3: skip blank from s+2 → next_beta[s+2]
            # Only if: s+2 < S_prime_i AND labels[s] != blank
            #          AND labels[s] != labels[s+2]
            next_p2 = tl.full([BLOCK_S], value=neg_inf, dtype=tl.float32)
            shifted2_mask = (s_offsets + 2 < S_prime_max) & s_mask
            shifted2_ptrs = (
                beta_base + (t + 1) * beta_stride_t + (s_offsets + 2) * beta_stride_s
            )
            next_p2_loaded = tl.load(
                shifted2_ptrs, mask=shifted2_mask, other=neg_inf
            ).to(tl.float32)

            # Load labels[s+2] for comparison
            labels_p2_ptrs = el_base + (s_offsets + 2) * el_stride_s
            labels_p2 = tl.load(labels_p2_ptrs, mask=shifted2_mask, other=blank).to(
                tl.int64
            )

            can_skip = (
                (s_offsets + 2 < S_prime_i) & (labels != blank) & (labels != labels_p2)
            )
            next_p2 = tl.where(can_skip, next_p2_loaded, neg_inf)
            trans = _tl_log_sum_exp(trans, next_p2)

            # Emission: log_probs[t, batch_idx, labels[s]]
            emit_ptrs_t = lp_base + t * lp_stride_t + labels * lp_stride_c
            emit_t = tl.load(emit_ptrs_t, mask=s_mask, other=0.0).to(tl.float32)

            beta_new = trans + emit_t

            # Mask invalid positions
            valid = (s_offsets < S_prime_i) & (t < T_i)
            beta_cur = tl.where(valid, beta_new, neg_inf)

            # Store beta[batch, t, :]
            beta_store_ptrs_t = (
                beta_base + t * beta_stride_t + s_offsets * beta_stride_s
            )
            tl.store(beta_store_ptrs_t, beta_cur, mask=s_mask)


@libentry()
@triton.jit
def ctc_grad_kernel(
    # Inputs
    log_probs_ptr,  # (T, N, C)
    alpha_ptr,  # (N, T, S')
    beta_ptr,  # (N, T, S')
    nll_ptr,  # (N,)
    expanded_labels_ptr,  # (N, S')
    grad_out_ptr,  # (N,)
    # Outputs
    grad_ptr,  # (T, N, C)
    # Dimensions
    T_max: tl.constexpr,
    N,
    C: tl.constexpr,
    S_prime_max: tl.constexpr,
    # Strides
    lp_stride_t,
    lp_stride_n,
    lp_stride_c,
    alpha_stride_n,
    alpha_stride_t,
    alpha_stride_s,
    el_stride_n,
    el_stride_s,
    grad_stride_t,
    grad_stride_n,
    grad_stride_c,
    # Per-sample
    S_prime_ptr,  # (N,) actual S' per sample
    input_lengths_ptr,  # (N,) actual T per sample
    # Block sizes
    BLOCK_C: tl.constexpr,
):
    """Compute CTC gradient for one (batch, time) pair.

    Grid: (N, T_max) - one program per (batch, time) pair.
    """
    batch_idx = tle.program_id(0)
    t = tle.program_id(1)

    if batch_idx >= N:
        return

    T_i = tl.load(input_lengths_ptr + batch_idx).to(tl.int64)

    # Zero out gradients for padding timesteps
    if t >= T_i:
        for c_start in tl.static_range(0, C, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            c_mask = c_offsets < C
            out_ptrs = (
                grad_ptr
                + t * grad_stride_t
                + batch_idx * grad_stride_n
                + c_offsets * grad_stride_c
            )
            tl.store(out_ptrs, tl.zeros([BLOCK_C], dtype=tl.float32), mask=c_mask)
        return

    S_prime_i = tl.load(S_prime_ptr + batch_idx).to(tl.int64)
    nll = tl.load(nll_ptr + batch_idx).to(tl.float32)
    grad_scale = tl.load(grad_out_ptr + batch_idx).to(tl.float32)

    # For each class c, accumulate log_sum_exp of (alpha[t,s] + beta[t,s])
    # for all s where expanded_label[s] == c
    for c_start in tl.static_range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        # Load log_probs[t, batch_idx, c]
        lp_ptrs = (
            log_probs_ptr
            + t * lp_stride_t
            + batch_idx * lp_stride_n
            + c_offsets * lp_stride_c
        )
        lp_val = tl.load(lp_ptrs, mask=c_mask, other=0.0).to(tl.float32)

        # Initialize ab_sum to -inf for each class
        ab_sum = tl.full([BLOCK_C], value=float("-inf"), dtype=tl.float32)

        # Iterate over label positions
        for s in range(S_prime_max):
            if s < S_prime_i:
                # Load alpha[batch, t, s] and beta[batch, t, s]
                a_val = tl.load(
                    alpha_ptr
                    + batch_idx * alpha_stride_n
                    + t * alpha_stride_t
                    + s * alpha_stride_s
                ).to(tl.float32)
                b_val = tl.load(
                    beta_ptr
                    + batch_idx * alpha_stride_n
                    + t * alpha_stride_t
                    + s * alpha_stride_s
                ).to(tl.float32)
                ab_single = a_val + b_val

                # Get expanded label for position s
                lbl = tl.load(
                    expanded_labels_ptr + batch_idx * el_stride_n + s * el_stride_s
                ).to(tl.int64)

                # Update ab_sum for matching classes
                match_mask = (c_offsets == lbl) & c_mask
                # log_sum_exp update
                max_v = tl.maximum(ab_sum, ab_single)
                new_sum = max_v + tl.log(
                    tl.exp(ab_sum - max_v) + tl.exp(ab_single - max_v)
                )
                # Handle -inf case
                both_neg_inf = (ab_sum == float("-inf")) & (ab_single == float("-inf"))
                new_sum = tl.where(both_neg_inf, float("-inf"), new_sum)
                ab_sum = tl.where(match_mask, new_sum, ab_sum)

        # Compute gradient
        prob_val = tl.exp(lp_val)
        occupancy = tl.exp(ab_sum - lp_val + nll)
        occupancy = tl.where(ab_sum > float("-inf"), occupancy, 0.0)
        grad_val = (prob_val - occupancy) * grad_scale
        grad_val = tl.where(c_mask, grad_val, 0.0)

        out_ptrs = (
            grad_ptr
            + t * grad_stride_t
            + batch_idx * grad_stride_n
            + c_offsets * grad_stride_c
        )
        tl.store(out_ptrs, grad_val, mask=c_mask)


def _build_expanded_labels(targets, target_lengths, blank, N, S_max):
    """Build expanded label sequences with blanks inserted.

    Returns: (N, 2*S_max+1) tensor of expanded labels
    """
    S_prime_max = 2 * S_max + 1
    device = targets.device

    # Create expanded labels: [blank, l1, blank, l2, ..., blank, lS, blank]
    expanded = torch.full((N, S_prime_max), blank, dtype=torch.long, device=device)

    if S_max > 0:
        if targets.dim() == 2:
            # Vectorized: place targets at odd positions where valid
            idx = torch.arange(S_max, device=device).unsqueeze(0)  # (1, S_max)
            valid = idx < target_lengths.unsqueeze(1)  # (N, S_max)
            expanded[:, 1::2] = torch.where(
                valid, targets[:, :S_max], expanded[:, 1::2]
            )
        else:
            # 1D concatenated targets: build per-sample offsets and scatter
            offsets = torch.zeros(N + 1, dtype=torch.long, device=device)
            offsets[1:] = target_lengths.cumsum(0)
            for n in range(N):
                tl_n = target_lengths[n].item()
                if tl_n > 0:
                    expanded[n, 1 : 2 * tl_n : 2] = targets[
                        offsets[n] : offsets[n] + tl_n
                    ]

    return expanded


def _ctc_loss_forward(
    log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False
):
    """CTC loss forward pass - computes NLL and alpha DP table.

    Returns:
        Tuple of (neg_log_likelihood, log_alpha)
        - neg_log_likelihood: (N,) negative log-likelihood per sample
        - log_alpha: (N, T, 2*S_max+1) forward DP table (compatible with
          PyTorch native backward for cross-context safety)
    """
    logger.debug("GEMS CTC_LOSS FWD")

    assert log_probs.dim() == 3, "log_probs must be (T, N, C)"
    T, N, C = log_probs.shape

    log_probs = log_probs.contiguous()
    targets = targets.contiguous()
    input_lengths = input_lengths.to(torch.long).contiguous()
    target_lengths = target_lengths.to(torch.long).contiguous()

    # Use tensor shape to avoid GPU->CPU sync from .item()
    if targets.dim() == 2:
        S_max = targets.shape[1]
    else:
        S_max = target_lengths.max().item() if target_lengths.numel() > 0 else 0
    S_prime_max = 2 * S_max + 1

    expanded_labels = _build_expanded_labels(targets, target_lengths, blank, N, S_max)

    if log_probs.dtype == torch.float32 and log_probs.is_contiguous():
        lp = log_probs
    else:
        lp = log_probs.float().contiguous()

    log_alpha = torch.full(
        (N, T, S_prime_max), NEG_INF, dtype=torch.float32, device=log_probs.device
    )
    neg_log_likelihood = torch.empty(N, dtype=torch.float32, device=log_probs.device)

    BLOCK_S = triton.next_power_of_2(S_prime_max) if S_prime_max > 0 else 1

    grid_alpha = (N,)
    with torch_device_fn.device(log_probs.device):
        ctc_alpha_kernel[grid_alpha](
            lp,
            expanded_labels,
            input_lengths,
            target_lengths,
            log_alpha,
            neg_log_likelihood,
            T,
            N,
            C,
            S_prime_max,
            blank,
            lp.stride(0),
            lp.stride(1),
            lp.stride(2),
            log_alpha.stride(0),
            log_alpha.stride(1),
            log_alpha.stride(2),
            expanded_labels.stride(0),
            expanded_labels.stride(1),
            BLOCK_S=BLOCK_S,
        )

    return neg_log_likelihood, log_alpha


def _ctc_loss_backward(
    grad,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
    zero_infinity=False,
):
    """CTC loss backward pass - computes gradient from alpha (saved) and beta.

    Runs the beta kernel on-the-fly, then computes the full gradient using
    both alpha and beta DP tables.
    """
    logger.debug("GEMS CTC_LOSS BWD")

    T, N, C = log_probs.shape

    lp_detached = log_probs.detach()
    if lp_detached.dtype == torch.float32 and lp_detached.is_contiguous():
        log_probs_f = lp_detached
    else:
        log_probs_f = lp_detached.float().contiguous()
    targets = targets.contiguous()
    if isinstance(input_lengths, (list, tuple)):
        input_lengths = torch.tensor(
            input_lengths, dtype=torch.long, device=log_probs.device
        )
    if isinstance(target_lengths, (list, tuple)):
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long, device=log_probs.device
        )
    input_lengths = input_lengths.to(torch.long).contiguous()
    target_lengths = target_lengths.to(torch.long).contiguous()

    # Use tensor shape to avoid GPU->CPU sync from .item()
    if targets.dim() == 2:
        S_max = targets.shape[1]
    else:
        S_max = target_lengths.max().item() if target_lengths.numel() > 0 else 0
    S_prime_max = 2 * S_max + 1

    expanded_labels = _build_expanded_labels(targets, target_lengths, blank, N, S_max)

    BLOCK_S = triton.next_power_of_2(S_prime_max) if S_prime_max > 0 else 1

    # ---- Beta kernel ----
    log_beta = torch.full(
        (N, T, S_prime_max), NEG_INF, dtype=torch.float32, device=log_probs.device
    )
    grid_beta = (N,)
    with torch_device_fn.device(log_probs.device):
        ctc_beta_kernel[grid_beta](
            log_probs_f,
            expanded_labels,
            input_lengths,
            target_lengths,
            log_beta,
            T,
            N,
            C,
            S_prime_max,
            blank,
            log_probs_f.stride(0),
            log_probs_f.stride(1),
            log_probs_f.stride(2),
            log_beta.stride(0),
            log_beta.stride(1),
            log_beta.stride(2),
            expanded_labels.stride(0),
            expanded_labels.stride(1),
            BLOCK_S=BLOCK_S,
        )

    # ---- Gradient kernel ----
    grad_log_probs = torch.zeros(T, N, C, dtype=torch.float32, device=log_probs.device)
    S_prime_per_sample = (2 * target_lengths + 1).to(torch.long)
    BLOCK_C = triton.next_power_of_2(C)

    # Detach and copy all autograd-tracked tensors before passing to Triton.
    # When called via autograd dispatch, saved tensors may have autograd
    # metadata that causes incorrect kernel behavior if not fully detached.
    log_alpha = log_alpha.detach().clone()
    nll_detached = neg_log_likelihood.detach().clone()
    grad_detached = grad.detach().clone()

    grid_grad = (N, T)
    with torch_device_fn.device(log_probs.device):
        ctc_grad_kernel[grid_grad](
            log_probs_f,
            log_alpha,
            log_beta,
            nll_detached,
            expanded_labels,
            grad_detached,
            grad_log_probs,
            T,
            N,
            C,
            S_prime_max,
            log_probs_f.stride(0),
            log_probs_f.stride(1),
            log_probs_f.stride(2),
            log_alpha.stride(0),
            log_alpha.stride(1),
            log_alpha.stride(2),
            expanded_labels.stride(0),
            expanded_labels.stride(1),
            grad_log_probs.stride(0),
            grad_log_probs.stride(1),
            grad_log_probs.stride(2),
            S_prime_per_sample,
            input_lengths,
            BLOCK_C=BLOCK_C,
        )

    # Handle zero_infinity
    if zero_infinity:
        nll_d = neg_log_likelihood.detach()
        inf_mask = torch.isinf(nll_d) | (nll_d != nll_d)
        if inf_mask.any():
            zero_mask = inf_mask.unsqueeze(0).unsqueeze(2).expand_as(grad_log_probs)
            grad_log_probs = torch.where(
                zero_mask, torch.zeros_like(grad_log_probs), grad_log_probs
            )

    return grad_log_probs.to(log_probs.dtype)


def _ctc_loss_impl(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    zero_infinity=False,
):
    """Compute _ctc_loss (dispatched from aten::_ctc_loss / _ctc_loss.Tensor).

    Returns (neg_log_likelihood, log_alpha) matching native PyTorch's return
    format so that backward works correctly regardless of dispatch context.
    """
    logger.debug("GEMS _CTC_LOSS")

    if isinstance(input_lengths, (list, tuple)):
        input_lengths = torch.tensor(
            input_lengths, dtype=torch.long, device=log_probs.device
        )
    if isinstance(target_lengths, (list, tuple)):
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long, device=log_probs.device
        )

    return _ctc_loss_forward(
        log_probs, targets, input_lengths, target_lengths, blank, zero_infinity
    )


def _ctc_loss_backward_impl(
    grad,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
    zero_infinity=False,
):
    """Compute _ctc_loss_backward (dispatched from aten::_ctc_loss_backward).

    Runs beta kernel on-the-fly and computes gradient using alpha + beta.
    """
    logger.debug("GEMS _CTC_LOSS_BACKWARD")

    return _ctc_loss_backward(
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank,
        zero_infinity,
    )


def _ensure_tensor_lengths(lengths, device):
    """Convert list/tuple lengths to a device tensor."""
    if isinstance(lengths, (list, tuple)):
        return torch.tensor(lengths, dtype=torch.long, device=device)
    return lengths.to(device=device, dtype=torch.long)


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction=1,
    zero_infinity=False,
):
    """CTC loss with reduction (dispatched from aten::ctc_loss).

    Args:
        log_probs: (T, N, C) log-softmax output
        targets: (N, S) or 1D concatenated target labels
        input_lengths: (N,) actual input sequence lengths
        target_lengths: (N,) actual target sequence lengths
        blank: blank label index (default: 0)
        reduction: 0=none, 1=mean, 2=sum (default: 1)
        zero_infinity: replace inf losses with zero (default: False)

    Returns:
        Scalar or (N,) tensor depending on reduction mode
    """
    logger.debug("GEMS CTC_LOSS")

    # Normalize string reduction to int
    if isinstance(reduction, str):
        reduction = {"none": 0, "mean": 1, "sum": 2}[reduction]

    device = log_probs.device
    orig_dtype = log_probs.dtype

    # Promote to float32 for numerical stability
    if orig_dtype in (torch.float16, torch.bfloat16):
        log_probs = log_probs.float()

    input_lengths = _ensure_tensor_lengths(input_lengths, device)
    target_lengths = _ensure_tensor_lengths(target_lengths, device)

    # Core CTC forward
    neg_log_likelihood, _ = _ctc_loss_forward(
        log_probs, targets, input_lengths, target_lengths, blank, zero_infinity
    )

    # Handle zero_infinity before reduction
    if zero_infinity:
        neg_log_likelihood = torch.where(
            torch.isinf(neg_log_likelihood),
            neg_log_likelihood.new_zeros(()),
            neg_log_likelihood,
        )

    # Apply reduction
    if reduction == 0:
        result = neg_log_likelihood
    elif reduction == 2:
        result = neg_log_likelihood.sum()
    else:
        # Mean: divide by target lengths, then average across batch
        tl_float = target_lengths.to(dtype=neg_log_likelihood.dtype).clamp(min=1)
        result = (neg_log_likelihood / tl_float).mean()

    return result.to(orig_dtype) if result.dtype != orig_dtype else result
