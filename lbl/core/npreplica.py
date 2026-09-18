#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Numba replicas of the numpy / bottleneck reductions used in the LBL hot
loops, written to give bit-identical results:

- np.sum of float64 is numpy's pairwise summation (blocks of up to 128
  values summed with 8 accumulators, longer arrays split in two halves)
- bottleneck.nansum is a plain sequential loop
- bottleneck.nanstd is a two-pass loop; depending on how bottleneck was
  compiled, "asum += ai * ai" may be a fused multiply-add (clang on arm64
  does this) or not: both variants are here and the self-test picks the one
  that matches the installed bottleneck
- np.nanpercentile (method 'linear') follows numpy's _quantile / _lerp
- bottleneck.nanmedian is the middle value (or the mean of the two middle
  values, as 0.5 * (a + b))

The fast (numba) code paths of LBL are only used when use_fast() is True:
FAST_KERNELS is on (parameter FAST_KERNELS, or environment variable
LBL_FAST_KERNELS=0 to switch it off) and self_test() found the replicas
identical to numpy / bottleneck on this machine. Otherwise the original
python code runs.

All kernels use error_model='numpy' so that a division by zero gives inf /
nan as numpy does, instead of raising.

Created on 2026-09-18
"""
import os
import warnings

import numpy as np
from numba import njit, types
from numba.extending import intrinsic
import llvmlite.ir as llvm_ir

# numpy's pairwise summation block size (numpy/_core/src/umath/loops_utils.h)
PW_BLOCKSIZE = 128
# switch for the fast code paths (see use_fast)
FAST_KERNELS = os.environ.get('LBL_FAST_KERNELS', '1') not in ['0', 'False',
                                                               'false']
# result of the self test (None: not run yet)
_SELF_TEST = None
# whether the installed bottleneck.nanstd uses a fused multiply-add
BN_NANSTD_FMA = True


# =============================================================================
# Exact replicas of numpy / bottleneck reductions
# =============================================================================
@intrinsic
def fma(typingctx, a, b, c):
    """a * b + c with a single rounding (llvm.fma.f64)"""
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
    multiple of 8) and summed as left + right. Written with an explicit
    stack (numba crashes when it loads a cached recursive function).
    """
    if n <= PW_BLOCKSIZE:
        return _block_sum(a, start, n)
    nstack = 128
    node_start = np.empty(nstack, dtype=np.int64)
    node_len = np.empty(nstack, dtype=np.int64)
    node_seen = np.zeros(nstack, dtype=np.bool_)
    values = np.empty(nstack)
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
            # both halves are summed: left then right on the value stack
            right = values[vtop - 1]
            left = values[vtop - 2]
            vtop -= 2
            values[vtop] = left + right
            vtop += 1
        else:
            n2 = nlen // 2
            n2 -= n2 % 8
            node_seen[top] = True
            top += 1
            node_start[top] = nstart + n2
            node_len[top] = nlen - n2
            node_seen[top] = False
            top += 1
            node_start[top] = nstart
            node_len[top] = n2
            node_seen[top] = False
            top += 1
    return values[0]


@njit(cache=True, error_model='numpy')
def np_sum(a):
    """np.sum of a 1D float64 array"""
    return 0.0 + _pairwise_sum(a, 0, a.shape[0])


@njit(cache=True, error_model='numpy')
def bn_nansum(a):
    """bottleneck.nansum of a 1D float64 array"""
    asum = 0.0
    for i in range(a.shape[0]):
        ai = a[i]
        if ai == ai:
            asum += ai
    return asum


@njit(cache=True, error_model='numpy')
def bn_nanstd(a, use_fma):
    """
    bottleneck.nanstd (ddof=0) of a 1D float64 array (use_fma: whether the
    installed bottleneck fuses asum += ai * ai, see BN_NANSTD_FMA)
    """
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
                if use_fma:
                    asum = fma(ai, ai, asum)
                else:
                    asum += ai * ai
        return np.sqrt(asum / count)
    return np.nan


@njit(cache=True, error_model='numpy')
def np_quantile_sorted(s, n, q):
    """
    numpy 'linear' quantile q (a fraction, as numpy computes it from a
    percentile: p / 100.) of the n first values of s, which must be sorted
    and free of NaN (numpy's _quantile / _get_indexes / _lerp)
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
def estimate_sigma(tmp, q_hi, q_lo, buf):
    """
    lbl.core.math.estimate_sigma (sigma=1): half the distance between the
    q_lo and q_hi np.nanpercentile values of tmp (NaNs ignored, NaN if no
    finite value). buf is a work array with len(buf) >= len(tmp).
    """
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
    srt = buf[:n]
    srt.sort()
    upper = np_quantile_sorted(srt, n, q_hi)
    lower = np_quantile_sorted(srt, n, q_lo)
    return (upper - lower) / 2.0


@njit(cache=True, error_model='numpy')
def select(a, n, k):
    """
    Quickselect: reorder a[:n] (no NaN) so that a[k] is the k-th smallest
    value, with a[:k] <= a[k] <= a[k + 1:n]. Returns a[k].
    """
    lo = 0
    hi = n - 1
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
def bn_nanmedian(values, n_in):
    """bottleneck.nanmedian of values[:n_in] (float64); values is modified"""
    n = 0
    for i in range(n_in):
        if values[i] == values[i]:
            values[n] = values[i]
            n += 1
    if n == 0:
        return np.nan
    k = n >> 1
    med = select(values, n, k)
    if n % 2 == 0:
        # largest value below the k-th: max of values[:k]
        amax = values[0]
        for i in range(1, k):
            if values[i] > amax:
                amax = values[i]
        return 0.5 * (med + amax)
    return med


# =============================================================================
# Self test and switch
# =============================================================================
def self_test(verbose: bool = False) -> bool:
    """
    Check that the replicas give the same bits as numpy / bottleneck on
    this machine (random arrays with NaNs, infs, ties), and find whether
    bottleneck.nanstd uses a fused multiply-add. Needs bottleneck (the LBL
    math functions use numpy's versions without it, which are not
    replicated here).

    :param verbose: bool, print the failures

    :return: bool, True if all replicas are identical
    """
    global BN_NANSTD_FMA
    try:
        import bottleneck as bn
    except ImportError:
        if verbose:
            print('npreplica: bottleneck not installed')
        return False
    from scipy.special import erf

    def same(x, y):
        return (x == y) or (np.isnan(x) and np.isnan(y))

    rng = np.random.default_rng(20260918)
    ok = True
    # percentiles of estimate_sigma, as lbl.core.math.estimate_sigma
    p1 = (1 - (1 - erf(1.0 / np.sqrt(2.0))) / 2) * 100
    q_hi = float(np.true_divide(p1, np.float64(100)))
    q_lo = float(np.true_divide(100 - p1, np.float64(100)))
    fma_votes = [0, 0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for trial in range(3000):
            n = int(rng.integers(1, 400)) if trial % 10 else \
                int(rng.integers(400, 3000))
            a = rng.normal(size=n) * 10 ** rng.uniform(-4, 4)
            if trial % 5 == 0:
                a = np.round(a)
            b = np.array(a)
            b[rng.uniform(size=n) < rng.uniform(0, 0.5)] = np.nan
            if trial % 17 == 0:
                b[rng.integers(n)] = np.inf
            # sums
            ok &= same(np_sum(a), np.sum(a))
            ok &= same(bn_nansum(b), bn.nansum(b))
            # nanstd, both variants
            ref = bn.nanstd(b)
            fma_votes[0] += same(bn_nanstd(b, False), ref)
            fma_votes[1] += same(bn_nanstd(b, True), ref)
            # median
            ok &= same(bn_nanmedian(np.array(b), n), bn.nanmedian(b))
            # estimate_sigma (two np.nanpercentile)
            if np.sum(np.isfinite(b)) == 0:
                ref = np.nan
            else:
                ref = (np.nanpercentile(b, p1) -
                       np.nanpercentile(b, 100 - p1)) / 2.0
            ok &= same(estimate_sigma(b, q_hi, q_lo, np.empty(n)), ref)
    if fma_votes[1] == 3000:
        BN_NANSTD_FMA = True
    elif fma_votes[0] == 3000:
        BN_NANSTD_FMA = False
    else:
        ok = False
    if verbose:
        print('npreplica self test: {0}, bottleneck nanstd fma: {1}'
              ''.format('OK' if ok else 'FAILED', BN_NANSTD_FMA))
    return bool(ok)


def set_fast_kernels(value: bool):
    """
    Switch the fast code paths on or off (parameter FAST_KERNELS)

    :param value: bool
    """
    global FAST_KERNELS
    FAST_KERNELS = bool(value)


def use_fast() -> bool:
    """
    Whether the fast (numba) code paths are used: FAST_KERNELS is on and
    the self test passed (run once, the first time this is called)
    """
    global _SELF_TEST
    if not FAST_KERNELS:
        return False
    if _SELF_TEST is None:
        _SELF_TEST = self_test()
        if not _SELF_TEST:
            warnings.warn('LBL: the numba replicas are not identical to '
                          'numpy/bottleneck here; using the original code')
    return _SELF_TEST
