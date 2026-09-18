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
    return _estimate_sigma_buf(tmp, q_hi, q_lo, np.empty(tmp.shape[0]))


@njit(cache=True, error_model='numpy')
def _estimate_sigma_buf(tmp, q_hi, q_lo, buf):
    """estimate_sigma with a work buffer (len(buf) >= len(tmp))"""
    n_fin = 0
    n = 0
    for i in range(tmp.shape[0]):
        x = tmp[i]
        if np.isfinite(x):
            n_fin += 1
        if not np.isnan(x):
            buf[n] = x
            n += 1
    if n_fin == 0:
        return np.nan
    upper, lower = _two_quantiles(buf, n, q_hi, q_lo)
    return (upper - lower) / 2.0


@njit(cache=True, error_model='numpy')
def _quantile_ranks(n, q):
    """numpy 'linear' quantile q of n values: ranks prev, next and gamma"""
    v = (n - 1) * q
    if v >= n - 1:
        return n - 1, n - 1, v - (-1.0)
    elif v < 0:
        return 0, 0, v - 0.0
    prev = int(np.floor(v))
    return prev, prev + 1, v - prev


@njit(cache=True, error_model='numpy')
def _lerp(a, b, gamma):
    """numpy's _lerp"""
    diff_b_a = b - a
    if gamma >= 0.5:
        return b - diff_b_a * (1 - gamma)
    return a + diff_b_a * gamma


@njit(cache=True, error_model='numpy')
def _two_quantiles(buf, n, q1, q2):
    """
    numpy 'linear' quantiles q1 and q2 of buf[:n] (no NaN), by selection:
    the values of the ranks involved are found with quickselect (buf[:n]
    is reordered). Same values as from a sorted array.
    """
    p1, n1, g1 = _quantile_ranks(n, q1)
    p2, n2, g2 = _quantile_ranks(n, q2)
    # the (up to 4) ranks needed, in increasing order, each selected in the
    #   part of buf that is above the previous one
    ranks = np.empty(4, dtype=np.int64)
    ranks[0] = p1
    ranks[1] = n1
    ranks[2] = p2
    ranks[3] = n2
    ranks.sort()
    vals = np.empty(4)
    lo = 0
    last = -1
    for ir in range(4):
        rank = ranks[ir]
        if rank == last:
            vals[ir] = vals[ir - 1]
            continue
        if rank == last + 1 and last >= 0:
            # smallest value above the previous rank
            best = buf[lo]
            for i in range(lo + 1, n):
                if buf[i] < best:
                    best = buf[i]
            vals[ir] = best
            # keep the invariant: put it at position rank
            for i in range(lo, n):
                if buf[i] == best:
                    buf[i] = buf[lo]
                    buf[lo] = best
                    break
        else:
            vals[ir] = _select_range(buf, lo, n - 1, rank)
        last = rank
        lo = rank + 1
    # values at the ranks of each quantile
    v1p = vals[0]
    v1n = vals[0]
    v2p = vals[0]
    v2n = vals[0]
    for ir in range(4):
        if ranks[ir] == p1:
            v1p = vals[ir]
        if ranks[ir] == n1:
            v1n = vals[ir]
        if ranks[ir] == p2:
            v2p = vals[ir]
        if ranks[ir] == n2:
            v2n = vals[ir]
    return _lerp(v1p, v1n, g1), _lerp(v2p, v2n, g2)


# =============================================================================
# compute_rv kernels
# =============================================================================
@njit(cache=True, error_model='numpy')
def noise_model_windows(residuals, npoints, q_hi, q_lo, nstep=4):
    """
    Inner loop of lbl.science.general.estimate_noise_model for one order:
    robust sigma of the residuals in boxes of npoints pixels every
    npoints // nstep pixels (boxes with <= 50% valid pixels, and zero
    sigmas, are set to NaN). nstep=4 is the original sampling.

    :return: box centers (pixels) and sigma in each box
    """
    npix = residuals.shape[0]
    indices = np.arange(0, npix, max(npoints // nstep, 1))
    sigma = np.zeros(indices.shape[0])
    buf = np.empty(npoints + 1)
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
            if buf.shape[0] < tmp.shape[0]:
                buf = np.empty(tmp.shape[0])
            sigma[it] = _estimate_sigma_buf(tmp, q_hi, q_lo, buf)
    for it in range(indices.shape[0]):
        if sigma[it] == 0:
            sigma[it] = np.nan
    return indices, sigma


@njit(cache=True, error_model='numpy')
def _bouchy(vector, diff_vector, mean_rms, tmp):
    """
    bouchy_equation_line with a scalar mean_rms (tmp: work buffer of the
    same length as vector)
    """
    n = vector.shape[0]
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
def _bouchy_arr(vector, diff_vector, mean_rms, tmp):
    """
    bouchy_equation_line with a per-pixel mean_rms array (tmp: work buffer
    of the same length as vector)
    """
    n = vector.shape[0]
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
              passed_bounds, stats_updated):
    """
    The 'loop through all lines' block of lbl.science.general.compute_rv,
    with identical arithmetic. Arrays after min_line_width are updated in
    place.

    passed_bounds flags the lines whose meanxpix / meanblaze were set,
    stats_updated the lines whose rmsratio / npixline / chi2 were set.

    x_start_all / x_end_all are the floor()ed pixel positions of the line
    edges (from the wave -> pixel spline of the order), as int64.
    proj_models has shape (n_resproj, n_orders, n_pixels) (only read on the
    last iteration); proj / sproj have shape (n_resproj, n_lines).
    """
    npix = nwavegrid.shape[1]
    nproj = proj_models.shape[0]
    # work buffers (a line is at most npix pixels long): slices of these
    #   are used for the per-line vectors, instead of allocating them
    wbuf = np.empty(npix)
    tbuf = np.empty(npix)
    dbuf = np.empty(npix)
    diffbuf = np.empty(npix)
    sbuf = np.empty(npix)
    bbuf = np.empty(npix)
    rbuf = np.empty(npix)
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
        weight_mask = wbuf[:nseg]
        weight_mask[:] = 1.0
        if ww_ord[x_start] < wave_s:
            refdiff = ww_ord[x_start + 1] - wave_s
            wavediff = ww_ord[x_start + 1] - ww_ord[x_start]
            weight_mask[0] = 1 - refdiff / wavediff
        if ww_ord[x_end + 1] > wave_e:
            refdiff = wave_e - ww_ord[x_end]
            wavediff = ww_ord[x_end] - ww_ord[x_end - 1]
            weight_mask[nseg - 1] = 1 - (refdiff / wavediff)
        # mean xpix (bottleneck nansums) and mean blaze
        tmp = tbuf[:nseg]
        for i in range(nseg):
            tmp[i] = weight_mask[i] * (x_start + i)
        meanxpix[line_it] = _bn_nansum(tmp) / _bn_nansum(weight_mask)
        meanblaze[line_it] = blaze[order_num, (x_start + x_end) // 2]
        # segments
        sci_seg = sci_data[order_num, x_start:x_end + 1]
        model_seg = model[order_num, x_start:x_end + 1]
        d_seg = dbuf[:nseg]
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
        diff_seg = diffbuf[:nseg]
        for i in range(nseg):
            diff_seg[i] = (sci_seg[i] - model_seg[i]) * weight_mask[i]
        for i in range(nseg):
            tmp[i] = rms[order_num, x_start + i] * weight_mask[i]
        sum_rms = _np_sum(tmp)
        if not np.isfinite(sum_rms):
            continue
        mean_rms = sum_rms / sum_weight_mask
        # 1st derivative
        dv[line_it], sdv[line_it] = _bouchy(d_seg, diff_seg, mean_rms, tmp)
        if flag_last_iter:
            seg = sbuf[:nseg]
            # 0th derivative
            nfin_model = 0
            for i in range(nseg):
                if np.isfinite(model_seg[i]):
                    nfin_model += 1
            if nfin_model >= 2:
                # np.nanmean (model_seg has no NaN here)
                v1 = _np_sum(model_seg) / nseg
                for i in range(nseg):
                    seg[i] = model_seg[i] - v1
                d0v[line_it], sd0v[line_it] = _bouchy(seg, diff_seg,
                                                      mean_rms, tmp)
            # 2nd derivative
            for i in range(nseg):
                seg[i] = d2model[order_num, x_start + i] * weight_mask[i]
            d2v[line_it], sd2v[line_it] = _bouchy(seg, diff_seg, mean_rms,
                                                  tmp)
            # 3rd derivative
            for i in range(nseg):
                seg[i] = d3model[order_num, x_start + i] * weight_mask[i]
            d3v[line_it], sd3v[line_it] = _bouchy(seg, diff_seg, mean_rms,
                                                  tmp)
            # residual projection tables
            #   note: as in compute_rv, frac_diff_seg is diff_seg itself, so
            #   diff_seg is divided in place once per table
            bn_seg = bbuf[:nseg]
            frac_mean_rms = rbuf[:nseg]
            for ikey in range(nproj):
                for i in range(nseg):
                    seg[i] = (proj_models[ikey, order_num, x_start + i] *
                              weight_mask[i])
                    bn_seg[i] = (b_ratio[order_num, x_start + i] *
                                 norm[order_num, x_start + i])
                for i in range(nseg):
                    diff_seg[i] /= bn_seg[i]
                    frac_mean_rms[i] = mean_rms / bn_seg[i]
                pd_key, psd_key = _bouchy_arr(seg, diff_seg, frac_mean_rms,
                                              tmp)
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
            stats_updated[line_it] = True


# =============================================================================
# lowpassfilter kernel
# =============================================================================
@njit(cache=True, error_model='numpy')
def _select(a, n, k):
    """
    Quickselect: reorder a[:n] (no NaN) so that a[k] is the k-th smallest
    value, with a[:k] <= a[k] <= a[k + 1:n]. Returns a[k].
    """
    return _select_range(a, 0, n - 1, k)


@njit(cache=True, error_model='numpy')
def _select_range(a, lo, hi, k):
    """
    Quickselect within a[lo:hi + 1] (no NaN): reorder it so that a[k] is
    the value of rank k - lo in it, smaller ones before, larger ones after.
    Returns a[k].
    """
    while hi > lo:
        mid = (lo + hi) >> 1
        # median of three as pivot
        if a[mid] < a[lo]:
            a[mid], a[lo] = a[lo], a[mid]
        if a[hi] < a[lo]:
            a[hi], a[lo] = a[lo], a[hi]
        if a[hi] < a[mid]:
            a[hi], a[mid] = a[mid], a[hi]
        pivot = a[mid]
        i = lo
        j = hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            break
    return a[k]


@njit(cache=True, error_model='numpy')
def _bn_nanmedian(values, n_in):
    """bottleneck.nanmedian of values[:n_in] (float64); values is modified"""
    n = 0
    for i in range(n_in):
        if values[i] == values[i]:
            values[n] = values[i]
            n += 1
    if n == 0:
        return np.nan
    k = n >> 1
    med = _select(values, n, k)
    if n % 2 == 0:
        # largest value below the k-th: max of values[:k]
        amax = values[0]
        for i in range(1, k):
            if values[i] > amax:
                amax = values[i]
        return 0.5 * (med + amax)
    return med


@njit(cache=True, error_model='numpy')
def lowpass_windows(input_vect, width):
    """
    Window loop of lbl.core.math.lowpassfilter: for boxes of `width` pixels
    every width // 4 pixels (starting half a box before the vector), the
    mean pixel position (bottleneck nanmean) and the NaN-median of the values
    (bottleneck nanmedian), skipping boxes with < 3 pixels or < 3 finite
    values.

    :return: xmed, ymed (float64 arrays)
    """
    nvect = input_vect.shape[0]
    start = -width // 2
    stop = nvect + width // 2
    step = width // 4
    nmax = 0
    if stop > start:
        nmax = (stop - start + step - 1) // step
    xmed = np.empty(nmax)
    ymed = np.empty(nmax)
    buf = np.empty(max(width, 1))
    count = 0
    for iw in range(nmax):
        it = start + iw * step
        low_bound = it
        high_bound = it + width
        if low_bound < 0:
            low_bound = 0
        if high_bound > (nvect - 1):
            high_bound = nvect - 1
        npix = high_bound - low_bound
        if npix < 3:
            continue
        nfinite = 0
        for i in range(low_bound, high_bound):
            if np.isfinite(input_vect[i]):
                nfinite += 1
        if nfinite < 3:
            continue
        # bottleneck nanmean of the (integer) pixel positions
        asum = 0.0
        for i in range(low_bound, high_bound):
            asum += i
        xmed[count] = asum / npix
        # bottleneck nanmedian of the values
        for i in range(npix):
            buf[i] = input_vect[low_bound + i]
        ymed[count] = _bn_nanmedian(buf, npix)
        count += 1
    return xmed[:count], ymed[:count]


# =============================================================================
# FITPACK spline evaluation
# =============================================================================
@njit(cache=True, error_model='numpy')
def splev_group(t, cs, k, x, ext):
    """
    FITPACK splev (scipy's Fortran splev.f + fpbspl.f, der=0) of several
    splines that share the knots t and the degree k: cs[ispl] are their
    coefficients. The B-spline basis is computed once per point.

    The arithmetic is FITPACK's as compiled for scipy on arm64, where the
    compiler fuses h(i) + f * (t(li) - x) and sp + c(ll) * h(j) into fused
    multiply-adds: the values are identical to scipy's splev.

    :param t: knots (n)
    :param cs: coefficients (nspl, >= n - k - 1)
    :param k: spline degree
    :param x: points (m)
    :param ext: 0 extrapolate, 1 zero outside, 3 boundary value
                (2, raise, is not handled here)

    :return: values (nspl, m)
    """
    n = t.shape[0]
    nspl = cs.shape[0]
    m = x.shape[0]
    out = np.empty((nspl, m))
    tb = t[k]
    te = t[n - k - 1]
    lmin = k
    lmax = n - k - 2
    h = np.empty(k + 1)
    hh = np.empty(k + 1)
    # knot interval (0-based: t[ll] <= arg < t[ll + 1]), found by bisection
    #   for the first point, then moved step by step as FITPACK does
    ll = -1
    for i in range(m):
        arg = x[i]
        if arg != arg:
            for ispl in range(nspl):
                out[ispl, i] = np.nan
            continue
        if arg < tb or arg > te:
            if ext == 1:
                for ispl in range(nspl):
                    out[ispl, i] = 0.0
                continue
            elif ext == 3:
                if arg < tb:
                    arg = tb
                else:
                    arg = te
        if ll < 0:
            ll = np.searchsorted(t, arg, side='right') - 1
            if ll < lmin:
                ll = lmin
            if ll > lmax:
                ll = lmax
        while arg < t[ll] and ll > lmin:
            ll -= 1
        while arg >= t[ll + 1] and ll < lmax:
            ll += 1
        # fpbspl: the k + 1 non-zero B-splines at arg
        h[0] = 1.0
        for j in range(1, k + 1):
            for ii in range(j):
                hh[ii] = h[ii]
            h[0] = 0.0
            for ii in range(1, j + 1):
                li = ll + ii
                lj = li - j
                if t[li] == t[lj]:
                    h[ii] = 0.0
                    continue
                f = hh[ii - 1] / (t[li] - t[lj])
                h[ii - 1] = _fma(f, t[li] - arg, h[ii - 1])
                h[ii] = f * (arg - t[lj])
        # value of each spline
        for ispl in range(nspl):
            sp = 0.0
            for j in range(k + 1):
                sp = _fma(cs[ispl, ll - k + j], h[j], sp)
            out[ispl, i] = sp
    return out


# =============================================================================
# lbl_template kernels
# =============================================================================
@njit(cache=True, error_model='numpy')
def _np_quantile_select(buf, n, q):
    """
    numpy 'linear' quantile q (fraction) of buf[:n] (no NaN), found by
    selection instead of sorting; buf[:n] is reordered
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
    a = _select(buf, n, prev)
    if nxt == prev:
        b = a
    else:
        # the next value in order is the smallest of those after prev
        b = buf[prev + 1]
        for i in range(prev + 2, n):
            if buf[i] < b:
                b = buf[i]
    diff_b_a = b - a
    if gamma >= 0.5:
        return b - diff_b_a * (1 - gamma)
    return a + diff_b_a * gamma


@njit(cache=True, error_model='numpy')
def roll_ratio_sigma(flux, med, shifts, q_lo, q_hi):
    """
    For each shift: tmp = np.roll(flux, shift) / med and
    (np.nanpercentile(tmp, q_hi) - np.nanpercentile(tmp, q_lo)) / 2
    (q as fractions)
    """
    n = flux.shape[0]
    out = np.empty(shifts.shape[0])
    buf = np.empty(n)
    for s in range(shifts.shape[0]):
        shift = shifts[s]
        m = 0
        for i in range(n):
            j = (i - shift) % n
            if j < 0:
                j += n
            x = flux[j] / med[i]
            if x == x:
                buf[m] = x
                m += 1
        if m == 0:
            out[s] = np.nan
            continue
        low = _np_quantile_select(buf, m, q_lo)
        high = _np_quantile_select(buf, m, q_hi)
        out[s] = (high - low) / 2.0
    return out


@njit(cache=True, error_model='numpy')
def nanpercentile_rows(cube, qs):
    """
    np.nanpercentile(cube, qs * 100, axis=1) for a 2D float64 cube (qs as
    fractions): shape (len(qs), cube.shape[0])
    """
    nrow, ncol = cube.shape
    out = np.empty((qs.shape[0], nrow))
    buf = np.empty(ncol)
    for r in range(nrow):
        m = 0
        for c in range(ncol):
            x = cube[r, c]
            if x == x:
                buf[m] = x
                m += 1
        if m == 0:
            for iq in range(qs.shape[0]):
                out[iq, r] = np.nan
            continue
        srt = buf[:m]
        srt.sort()
        for iq in range(qs.shape[0]):
            out[iq, r] = _np_quantile_sorted(srt, m, qs[iq])
    return out
