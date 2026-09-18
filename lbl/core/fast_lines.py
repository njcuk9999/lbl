#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Numba kernel for the line-by-line loop of lbl.science.general.compute_rv
(same results as the original python loop, see lbl.core.npreplica for the
exact replicas of np.sum, bottleneck.nansum and bottleneck.nanstd).

Created on 2026-09-18
"""
import numpy as np
from numba import njit

from lbl.core import npreplica


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
    rms_value = 1 / np.sqrt(npreplica.np_sum(tmp))
    for i in range(n):
        tmp[i] = diff_vector[i] * vector[i]
    part1 = npreplica.np_sum(tmp)
    for i in range(n):
        tmp[i] = vector[i] * vector[i]
    value = part1 / npreplica.np_sum(tmp)
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
    rms_value = 1 / np.sqrt(npreplica.np_sum(tmp))
    for i in range(n):
        tmp[i] = diff_vector[i] * vector[i]
    part1 = npreplica.np_sum(tmp)
    for i in range(n):
        tmp[i] = vector[i] * vector[i]
    value = part1 / npreplica.np_sum(tmp)
    return value, rms_value


@njit(cache=True, error_model='numpy')
def line_loop(iteration, flag_last_iter, orders, wave_start, wave_end,
              x_start_all, x_end_all, mask_keep, nwavegrid, sci_data, rms,
              model, dmodel, d2model, d3model, blaze, b_ratio, norm,
              proj_models, min_line_width,
              dv, sdv, d0v, sd0v, d2v, sd2v, d3v, sd3v, frac_line_valid,
              meanxpix, meanblaze, rmsratio, npixline, chi2, proj, sproj,
              passed_bounds, stats_updated, nanstd_fma):
    """
    The 'loop through all lines' block of lbl.science.general.compute_rv,
    with identical arithmetic. Arrays after min_line_width are updated in
    place.

    passed_bounds flags the lines whose meanxpix / meanblaze were set,
    stats_updated the lines whose rmsratio / npixline / chi2 were set.
    nanstd_fma: npreplica.BN_NANSTD_FMA (how bottleneck.nanstd rounds).

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
        meanxpix[line_it] = npreplica.bn_nansum(tmp) / npreplica.bn_nansum(weight_mask)
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
        sum_weight_mask = npreplica.np_sum(weight_mask)
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
        sum_rms = npreplica.np_sum(tmp)
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
                v1 = npreplica.np_sum(model_seg) / nseg
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
            rmsratio[line_it] = (npreplica.bn_nanstd(diff_seg, nanstd_fma) /
                                 mean_rms)
            npixline[line_it] = nseg
            for i in range(nseg):
                x = diff_seg[i] / mean_rms
                tmp[i] = x * x
            chi2[line_it] = npreplica.bn_nansum(tmp)
            stats_updated[line_it] = True


