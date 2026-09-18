#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Numba kernels for the hot loops of lbl_compute.

Every kernel here reproduces, bit for bit, the arithmetic of the numpy /
bottleneck code it replaces (same operations, same summation order, same
rounding):

- np.sum on float64 is numpy's pairwise summation (8 accumulators, blocks of
  128), reproduced in _np_sum.
- bottleneck's nansum is a sequential loop; bottleneck's nanstd is built by
  clang with FMA contraction on arm64 (asum += ai * ai is a fused
  multiply-add), reproduced with the llvm.fma intrinsic.
- np.nanpercentile (method 'linear') is reproduced in _np_quantile_sorted
  (numpy's _get_indexes / _get_gamma / _lerp).

All kernels use error_model='numpy' so that a division by zero gives inf/nan
as it does in numpy, instead of raising.

Created on 2026-09-18
"""
import numpy as np
from numba import njit, types
from numba.extending import intrinsic
import llvmlite.ir as llvm_ir

# numpy's pairwise summation block size (numpy/_core/src/umath/loops_utils.h)
PW_BLOCKSIZE = 128


# =============================================================================
# Exact replicas of numpy / bottleneck reductions
# =============================================================================
@intrinsic
def _fma(typingctx, a, b, c):
    """a * b + c with a single rounding (llvm.fma.f64)."""
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        fnty = llvm_ir.FunctionType(llvm_ir.DoubleType(),
                                    [llvm_ir.DoubleType()] * 3)
        fn = builder.module.declare_intrinsic('llvm.fma',
                                              [llvm_ir.DoubleType()], fnty)
        return builder.call(fn, args)

    return sig, codegen


@njit(cache=True, error_model='numpy')
def _block_sum(a, start, n):
    """numpy's pairwise_sum_DOUBLE on a[start:start + n] for n <= 128"""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[start + i]
        return res
    r0 = a[start + 0]
    r1 = a[start + 1]
    r2 = a[start + 2]
    r3 = a[start + 3]
    r4 = a[start + 4]
    r5 = a[start + 5]
    r6 = a[start + 6]
    r7 = a[start + 7]
    i = 8
    nlim = n - (n % 8)
    while i < nlim:
        r0 += a[start + i + 0]
        r1 += a[start + i + 1]
        r2 += a[start + i + 2]
        r3 += a[start + i + 3]
        r4 += a[start + i + 4]
        r5 += a[start + i + 5]
        r6 += a[start + i + 6]
        r7 += a[start + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[start + i]
        i += 1
    return res


@njit(cache=True, error_model='numpy')
def _pairwise_sum(a, start, n):
    """
    numpy's pairwise_sum_DOUBLE on a[start:start + n]: blocks of <= 128
    values, larger ranges split in two (first half rounded down to a
    multiple of 8) and summed as left + right.

    Written with an explicit stack: numba segfaults when it loads a cached
    recursive function.
    """
    if n <= PW_BLOCKSIZE:
        return _block_sum(a, start, n)
    # node stack (start, length, visited flag) and value stack
    nstack = 128
    node_start = np.empty(nstack, dtype=np.int64)
    node_len = np.empty(nstack, dtype=np.int64)
    node_seen = np.zeros(nstack, dtype=np.bool_)
    values = np.empty(nstack)
    top = 0
    vtop = 0
    node_start[0] = start
    node_len[0] = n
    node_seen[0] = False
    top = 1
    while top > 0:
        top -= 1
        nstart = node_start[top]
        nlen = node_len[top]
        if nlen <= PW_BLOCKSIZE:
            values[vtop] = _block_sum(a, nstart, nlen)
            vtop += 1
        elif node_seen[top]:
            # both children are on the value stack: left then right
            right = values[vtop - 1]
            left = values[vtop - 2]
            vtop -= 2
            values[vtop] = left + right
            vtop += 1
        else:
            n2 = nlen // 2
            n2 -= n2 % 8
            # revisit this node once its children are summed
            node_seen[top] = True
            top += 1
            # right child (processed second)
            node_start[top] = nstart + n2
            node_len[top] = nlen - n2
            node_seen[top] = False
            top += 1
            # left child (processed first)
            node_start[top] = nstart
            node_len[top] = n2
            node_seen[top] = False
            top += 1
    return values[0]


@njit(cache=True, error_model='numpy')
def _np_sum(a):
    """np.sum of a 1D float64 array"""
    return 0.0 + _pairwise_sum(a, 0, a.shape[0])


@njit(cache=True, error_model='numpy')
def _bn_nansum(a):
    """bottleneck.nansum of a 1D float64 array"""
    asum = 0.0
    for i in range(a.shape[0]):
        ai = a[i]
        if ai == ai:
            asum += ai
    return asum


@njit(cache=True, error_model='numpy')
def _bn_nanstd(a):
    """bottleneck.nanstd (ddof=0) of a 1D float64 array"""
    asum = 0.0
    count = 0
    for i in range(a.shape[0]):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        amean = asum / count
        asum = 0.0
        for i in range(a.shape[0]):
            ai = a[i]
            if ai == ai:
                ai -= amean
                asum = _fma(ai, ai, asum)
        return np.sqrt(asum / count)
    return np.nan


@njit(cache=True, error_model='numpy')
def _np_quantile_sorted(s, n, q):
    """
    numpy 'linear' quantile q (in [0, 1]) of the n first values of s, which
    must be sorted and free of NaN (numpy _quantile/_get_indexes/_lerp)
    """
    v = (n - 1) * q
    if v >= n - 1:
        prev = n - 1
        nxt = n - 1
        gamma = v - (-1.0)
    elif v < 0:
        prev = 0
        nxt = 0
        gamma = v - 0.0
    else:
        prev = int(np.floor(v))
        nxt = prev + 1
        gamma = v - prev
    a = s[prev]
    b = s[nxt]
    diff_b_a = b - a
    if gamma >= 0.5:
        return b - diff_b_a * (1 - gamma)
    return a + diff_b_a * gamma


@njit(cache=True, error_model='numpy')
def estimate_sigma(tmp, q_hi, q_lo):
    """
    lbl.core.math.estimate_sigma (sigma=1): half the distance between the
    q_lo and q_hi np.nanpercentile values (q given as fractions, computed as
    numpy does, p / 100.)
    """
    n_fin = 0
    n = 0
    buf = np.empty(tmp.shape[0])
    for i in range(tmp.shape[0]):
        x = tmp[i]
        if np.isfinite(x):
            n_fin += 1
        if not np.isnan(x):
            buf[n] = x
            n += 1
    if n_fin == 0:
        return np.nan
    srt = np.sort(buf[:n])
    upper = _np_quantile_sorted(srt, n, q_hi)
    lower = _np_quantile_sorted(srt, n, q_lo)
    return (upper - lower) / 2.0


# =============================================================================
# compute_rv kernels
# =============================================================================
@njit(cache=True, error_model='numpy')
def noise_model_windows(residuals, npoints, q_hi, q_lo):
    """
    Inner loop of lbl.science.general.estimate_noise_model for one order:
    robust sigma of the residuals in boxes of npoints pixels every
    npoints // 4 pixels (boxes with <= 50% valid pixels, and zero sigmas,
    are set to NaN)

    :return: box centers (pixels) and sigma in each box
    """
    npix = residuals.shape[0]
    indices = np.arange(0, npix, npoints // 4)
    sigma = np.zeros(indices.shape[0])
    for it in range(indices.shape[0]):
        istart = indices[it] - npoints // 2
        iend = indices[it] + npoints // 2
        if istart < 0:
            istart = 0
        if iend > npix:
            iend = npix
        tmp = residuals[istart:iend]
        nvalid = 0
        for i in range(tmp.shape[0]):
            if np.isfinite(tmp[i]):
                nvalid += 1
        frac_valid = nvalid / npoints
        if frac_valid > 0.5:
            sigma[it] = estimate_sigma(tmp, q_hi, q_lo)
    for it in range(indices.shape[0]):
        if sigma[it] == 0:
            sigma[it] = np.nan
    return indices, sigma


@njit(cache=True, error_model='numpy')
def _bouchy(vector, diff_vector, mean_rms):
    """bouchy_equation_line with a scalar mean_rms"""
    n = vector.shape[0]
    tmp = np.empty(n)
    for i in range(n):
        rms_pix = mean_rms / vector[i]
        tmp[i] = 1 / (rms_pix * rms_pix)
    rms_value = 1 / np.sqrt(_np_sum(tmp))
    for i in range(n):
        tmp[i] = diff_vector[i] * vector[i]
    part1 = _np_sum(tmp)
    for i in range(n):
        tmp[i] = vector[i] * vector[i]
    value = part1 / _np_sum(tmp)
    return value, rms_value


@njit(cache=True, error_model='numpy')
def _bouchy_arr(vector, diff_vector, mean_rms):
    """bouchy_equation_line with a per-pixel mean_rms array"""
    n = vector.shape[0]
    tmp = np.empty(n)
    for i in range(n):
        rms_pix = mean_rms[i] / vector[i]
        tmp[i] = 1 / (rms_pix * rms_pix)
    rms_value = 1 / np.sqrt(_np_sum(tmp))
    for i in range(n):
        tmp[i] = diff_vector[i] * vector[i]
    part1 = _np_sum(tmp)
    for i in range(n):
        tmp[i] = vector[i] * vector[i]
    value = part1 / _np_sum(tmp)
    return value, rms_value


@njit(cache=True, error_model='numpy')
def line_loop(iteration, flag_last_iter, orders, wave_start, wave_end,
              x_start_all, x_end_all, mask_keep, nwavegrid, sci_data, rms,
              model, dmodel, d2model, d3model, blaze, b_ratio, norm,
              proj_models, min_line_width,
              dv, sdv, d0v, sd0v, d2v, sd2v, d3v, sd3v, frac_line_valid,
              meanxpix, meanblaze, rmsratio, npixline, chi2, proj, sproj,
              passed_bounds):
    """
    The 'loop through all lines' block of lbl.science.general.compute_rv,
    with identical arithmetic. Arrays after min_line_width are updated in
    place.

    x_start_all / x_end_all are the floor()ed pixel positions of the line
    edges (from the wave -> pixel spline of the order), as int64.
    proj_models has shape (n_resproj, n_orders, n_pixels) (only read on the
    last iteration); proj / sproj have shape (n_resproj, n_lines).
    """
    npix = nwavegrid.shape[1]
    nproj = proj_models.shape[0]
    for line_it in range(orders.shape[0]):
        order_num = orders[line_it]
        # skip lines flagged as bad (in all but the second iteration)
        if (iteration != 1) and not mask_keep[line_it]:
            continue
        ww_ord = nwavegrid[order_num]
        x_start = x_start_all[line_it]
        x_end = x_end_all[line_it]
        # boundary conditions
        if (x_end - x_start) < min_line_width:
            mask_keep[line_it] = False
            continue
        if x_start < 0:
            mask_keep[line_it] = False
            continue
        if x_end > npix - 2:
            mask_keep[line_it] = False
            continue
        passed_bounds[line_it] = True
        wave_s = wave_start[line_it]
        wave_e = wave_end[line_it]
        # weights at the edge of the domain
        nseg = x_end - x_start + 1
        weight_mask = np.ones(nseg)
        if ww_ord[x_start] < wave_s:
            refdiff = ww_ord[x_start + 1] - wave_s
            wavediff = ww_ord[x_start + 1] - ww_ord[x_start]
            weight_mask[0] = 1 - refdiff / wavediff
        if ww_ord[x_end + 1] > wave_e:
            refdiff = wave_e - ww_ord[x_end]
            wavediff = ww_ord[x_end] - ww_ord[x_end - 1]
            weight_mask[nseg - 1] = 1 - (refdiff / wavediff)
        # mean xpix (bottleneck nansums) and mean blaze
        tmp = np.empty(nseg)
        for i in range(nseg):
            tmp[i] = weight_mask[i] * (x_start + i)
        meanxpix[line_it] = _bn_nansum(tmp) / _bn_nansum(weight_mask)
        meanblaze[line_it] = blaze[order_num, (x_start + x_end) // 2]
        # segments
        sci_seg = sci_data[order_num, x_start:x_end + 1]
        model_seg = model[order_num, x_start:x_end + 1]
        d_seg = np.empty(nseg)
        for i in range(nseg):
            d_seg[i] = dmodel[order_num, x_start + i] * weight_mask[i]
        # fraction of the line that is finite
        nfinite = 0
        for i in range(nseg):
            if np.isfinite(sci_seg[i]):
                nfinite += 1
        frac_line_valid[line_it] = nfinite / nseg
        # sum of the weights
        sum_weight_mask = _np_sum(weight_mask)
        # reject segments with a NaN or a <= 0 value in the model
        bad_model = False
        for i in range(nseg):
            if np.isnan(model_seg[i]):
                bad_model = True
                break
        if bad_model:
            continue
        for i in range(nseg):
            if model_seg[i] <= 0:
                bad_model = True
                break
        if bad_model:
            continue
        diff_seg = np.empty(nseg)
        for i in range(nseg):
            diff_seg[i] = (sci_seg[i] - model_seg[i]) * weight_mask[i]
        for i in range(nseg):
            tmp[i] = rms[order_num, x_start + i] * weight_mask[i]
        sum_rms = _np_sum(tmp)
        if not np.isfinite(sum_rms):
            continue
        mean_rms = sum_rms / sum_weight_mask
        # 1st derivative
        dv[line_it], sdv[line_it] = _bouchy(d_seg, diff_seg, mean_rms)
        if flag_last_iter:
            # 0th derivative
            nfin_model = 0
            for i in range(nseg):
                if np.isfinite(model_seg[i]):
                    nfin_model += 1
            if nfin_model >= 2:
                # np.nanmean (model_seg has no NaN here)
                v1 = _np_sum(model_seg) / nseg
                seg0 = np.empty(nseg)
                for i in range(nseg):
                    seg0[i] = model_seg[i] - v1
                d0v[line_it], sd0v[line_it] = _bouchy(seg0, diff_seg,
                                                      mean_rms)
            # 2nd derivative
            seg2 = np.empty(nseg)
            for i in range(nseg):
                seg2[i] = d2model[order_num, x_start + i] * weight_mask[i]
            d2v[line_it], sd2v[line_it] = _bouchy(seg2, diff_seg, mean_rms)
            # 3rd derivative
            seg3 = np.empty(nseg)
            for i in range(nseg):
                seg3[i] = d3model[order_num, x_start + i] * weight_mask[i]
            d3v[line_it], sd3v[line_it] = _bouchy(seg3, diff_seg, mean_rms)
            # residual projection tables
            #   note: as in compute_rv, frac_diff_seg is diff_seg itself, so
            #   diff_seg is divided in place once per table
            for ikey in range(nproj):
                pd_seg = np.empty(nseg)
                bn_seg = np.empty(nseg)
                frac_mean_rms = np.empty(nseg)
                for i in range(nseg):
                    pd_seg[i] = (proj_models[ikey, order_num, x_start + i] *
                                 weight_mask[i])
                    bn_seg[i] = (b_ratio[order_num, x_start + i] *
                                 norm[order_num, x_start + i])
                for i in range(nseg):
                    diff_seg[i] /= bn_seg[i]
                    frac_mean_rms[i] = mean_rms / bn_seg[i]
                pd_key, psd_key = _bouchy_arr(pd_seg, diff_seg, frac_mean_rms)
                if np.isfinite(pd_key) and np.isfinite(psd_key):
                    proj[ikey, line_it] = pd_key
                    sproj[ikey, line_it] = psd_key
            # line statistics for the reference table
            rmsratio[line_it] = _bn_nanstd(diff_seg) / mean_rms
            npixline[line_it] = nseg
            for i in range(nseg):
                x = diff_seg[i] / mean_rms
                tmp[i] = x * x
            chi2[line_it] = _bn_nansum(tmp)
