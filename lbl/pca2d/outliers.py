"""Samples the model leaves unexplained, found and set to NaN.

What panel 5 of the report's sequence figure shows, the data less everything
the fit models (the star block, the observer block and the parity offset), is
noise where the model is right and something else where it is not: a cosmic
ray the MAD cut let through, a hot pixel, a sky residual. Against the noise
around it, measured where it is, a sample far outside that noise is not a
measurement of the star, and a corrected file is better without it.

    running_stats      the local median and robust sigma (1.4826 MAD) of each
                       row, in boxes as wide as the high pass (highpass.window)
    clip               NaN wherever a value is more than nsig of them away
    residual_outliers  the same on the fit's own residual, per exposure, as
                       the (even, odd) grid mask the correct stage applies

The corrected files take it through reconstruct's NaN mask, beside the samples
the fit gave no weight, when correct.nsig_cut is set in config.yaml; null
leaves every sample in.

A SAMPLE IS THE WRONG UNIT. One sample beyond 3 sigma is not an event: 0.27%
of pure noise is, and 1% of these residuals were, which is what asking for
correct.nsig_cut 3 threw away (2026-09-25). What is not noise is a run of
samples leaning the same way, a bump as wide as a line rather than a spike, so
the unit is the EXCURSION: every window from one sample up to
correct.excursion_elements resolution elements, each judged by the aggregate
significance of its sum,

    Z(w) = |sum of z over the window| / sqrt(V(w))

and flagged, with every sample in it, when Z is beyond correct.excursion_nsig.
Six sigma there is not six sigma on one sample: four samples at three sigma
each make six when the noise is independent, which is the point.

V(w) IS MEASURED, NEVER w. The grid oversamples the spectrograph on purpose
(0.5 km/s against a SPIRou pixel of 2.3 and a resolution element of 4.3), and
the Lanczos registration and the high pass correlate neighbours further, so
neighbouring samples are anything but independent and sqrt(w) would inflate
every excursion into a detection. The exact variance of a sum of w correlated
samples is

    V(w) = w + 2 * sum over k of (w - k) * rho_k

with rho_k the autocorrelation of the residual at lag k, measured on the run's
own cube by noise_correlation. On TOI-2120 (SPIRou, 0.5 km/s) rho_1 is about
0.9, so V(17) is near 90 rather than 17: a window of two resolution elements
holds about three independent measurements, not seventeen.

    window_variance     V(w) from the measured rho
    noise_correlation   rho_k of the residual, from the cube and the fit
    excursions          the mask of every window whose Z is beyond the threshold

ONE COLUMN, ACROSS THE EXPOSURES. The same z also says, column by column in
the OBSERVER's frame, how the exposures agree there. A column where too many
of them are flagged AND whose survivors still carry more variance than noise
is a column the model cannot describe in any exposure: a detector defect, a
telluric line the correction does not reach, a sky residual that stands still.
It goes for every exposure rather than for the few that happened to be caught.

    frac    the fraction of exposures flagged in that column, > correct.column_frac
    chi2    the reduced chi2 of the SURVIVORS, > correct.column_chi2

Both, not either. What each threshold means, measured rather than assumed
(scratchpad/null_chi2.py, 4096 columns of 321 rows of pure Gaussian noise, at
the 3 sigma per-sample clip): the survivors' reduced chi2 is 0.973, not 1,
because the clip takes the tails with it (expected_chi2 below is the closed
form). Its median over columns was 0.979 and its highest 1.271, so 1.5 is well
clear of noise. The clipped fraction reached 2.2% at most, so the 10% of
column_frac cannot be met by noise alone: it is what keeps a column with a
handful of genuinely bad exposures, whose survivors are clean, from being
thrown away whole.
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np

from .grids import pixel_shift

#: the light speed the resolution element is measured with, km/s
C_KMS = 299792.458


def expected_chi2(nsig):
    """E[z^2] of what survives a clip at `nsig`, for unit Gaussian noise.

    The number the column test's threshold is measured against: clipping
    removes the tails, so the survivors of a 3 sigma clip carry 0.973 and not
    1. A threshold below this would throw away pure noise.
    """
    a = float(nsig)
    phi = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    inside = math.erf(a / math.sqrt(2.0))
    return (inside - 2.0 * a * phi) / inside


def element_samples(resolution, dv, elements=1.0):
    """How many grid samples `elements` resolution elements cover.

    One element is c/R km/s wide, 4.28 for R = 70000, and the grid step is
    dv: at 0.5 km/s that is 8.6 samples, and two elements 17.
    """
    return max(1, int(round(float(elements) * (C_KMS / float(resolution))
                            / float(dv))))


def window_variance(width, rho):
    """The variance of the sum of `width` consecutive samples of unit variance
    whose autocorrelation is `rho` (rho[0] is lag 1).

        V(w) = w + 2 * sum over k < w of (w - k) * rho_k

    With rho all zero this is w, the independent case, and sqrt(V) is what an
    excursion's sum must be divided by to become a significance.
    """
    w = int(width)
    v = float(w)
    for k in range(1, min(w, len(rho) + 1)):
        v += 2.0 * (w - k) * float(rho[k - 1])
    return max(v, 1e-6)


def coherent_bias(mean, count, width, rho, min_rows=10):
    """(le Z du residu MOYEN sur les poses, colonne par colonne et sur `width`).

    Le troisieme axe, et le seul que ni la coupure ni l'exces de dispersion ne
    peuvent voir (2026-09-26). clip_weights compare chaque pose aux AUTRES
    POSES a la meme longueur d'onde, donc un residu present dans toutes les
    poses n'est aberrant pour aucune: sur la bande O2 a 1267 nm son |z| median
    reste a 0.67, comme sur une region temoin. La coupure locale en lambda ne
    le voit pas non plus tant qu'il reste sous son seuil dans chaque pose.

    Ce qui le revele est la MOYENNE sur les poses: l'etoile se deplace avec le
    BERV et s'y moyenne, ce qui est ancre dans le referentiel de l'observateur
    s'y ajoute. Avec N poses la moyenne gagne sqrt(N) = 17.8 contre le bruit,
    donc un biais de 0.2 sigma ressort a 3.5.

        Z(j) = |somme de la moyenne sur la fenetre| / sqrt(V(w) / N)

    avec V(w) la variance mesuree d'une somme de w echantillons correles
    (window_variance) et N le nombre de poses de la colonne. Chaque colonne de
    la fenetre porte le verdict de la fenetre, donc une region sort entiere.

    Sur TOI-2120: 21.9% du domaine au-dela de 3 sigma, 2.8% au-dela de 10, et
    les pires regions sont 1761, 1803, 1805 et 1764-1769 nm (methane et eau
    vers 1.76-1.81 um), larges de 35 a 51 echantillons, soit 4 a 6 elements de
    resolution: des raies entieres mal corrigees, pas des accidents.
    """
    mean = np.atleast_2d(np.asarray(mean, dtype=float))
    count = np.atleast_2d(np.asarray(count))
    rows, m = mean.shape
    w = max(1, int(width))
    ok = count >= int(min_rows)
    out = np.zeros((rows, m))
    variance = window_variance(w, rho)
    for p in range(rows):
        good = ok[p]
        if not good.any():
            continue
        filled = np.where(good, mean[p], 0.0)
        csum = np.concatenate([[0.0], np.cumsum(filled)])
        cnum = np.concatenate([[0], np.cumsum(good.astype(int))])
        took = cnum[w:] - cnum[:-w]
        total = csum[w:] - csum[:-w]
        rooms = np.where(good, count[p], 10 ** 9)
        least = np.minimum.reduce([rooms[i:m - w + 1 + i] for i in range(w)])
        sd = np.sqrt(variance / np.maximum(least, 1))
        here = np.where((took == w) & (sd > 0), np.abs(total) / sd, 0.0)
        for k in range(w):
            span = slice(k, k + m - w + 1)
            out[p, span] = np.maximum(out[p, span], here)
    return out


def excess_rms(chi2, count, width, length, min_rows=10):
    """(the windowed chi2 of each column, the significance of its excess over 1).

    A leftover anchored in the observer's frame need not have one sign, but it
    always has SCATTER the noise does not account for: chi2(j), the mean of
    z^2 over the exposures at that column, above 1. Averaged over a window of
    `width` columns, that average has an uncertainty of

        sqrt(2 / (N * width / length))

    with N the exposures the column has and `length` the correlation length of
    the residual in samples (1 + 2 sum rho), since a window of 17 samples on
    this grid holds about 4.7 independent columns and not 17. Every column of
    a window carries that window's verdict, so a region comes out whole.
    """
    chi2 = np.atleast_2d(np.asarray(chi2, dtype=float))
    count = np.atleast_2d(np.asarray(count))
    rows, m = chi2.shape
    w = max(1, int(width))
    ok = count >= int(min_rows)
    windowed = np.zeros((rows, m))
    sigma = np.zeros((rows, m))
    for p in range(rows):
        good = ok[p]
        if not good.any():
            continue
        filled = np.where(good, chi2[p], 0.0)
        csum = np.concatenate([[0.0], np.cumsum(filled)])
        cnum = np.concatenate([[0], np.cumsum(good.astype(int))])
        took = cnum[w:] - cnum[:-w]
        mean = np.divide(csum[w:] - csum[:-w], took,
                         out=np.zeros(m - w + 1), where=took > 0)
        rooms = np.where(good, count[p], 10 ** 9)
        least = np.minimum.reduce([rooms[i:m - w + 1 + i] for i in range(w)])
        independent = np.maximum(took / max(float(length), 1.0), 1.0)
        sd = np.sqrt(2.0 / np.maximum(least * independent, 1.0))
        for k in range(w):
            here = slice(k, k + m - w + 1)
            windowed[p, here] = np.maximum(windowed[p, here], mean)
            sigma[p, here] = np.maximum(sigma[p, here], sd)
    excess = np.divide(windowed - 1.0, sigma, out=np.zeros((rows, m)),
                       where=sigma > 0)
    return windowed, excess


def autocorrelation(z, lags, clip_at=5.0):
    """rho_1..rho_lags of rows of z, ignoring NaN and anything beyond clip_at.

    Measured on the residual itself, because what correlates neighbouring
    samples is the grid's oversampling, the Lanczos registration and the high
    pass together, and no formula for the three is worth trusting over the
    thing they did. The excursions themselves are left out (clip_at), so this
    is the correlation of the NOISE and not of what is being looked for.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    good = np.isfinite(z) & (np.abs(z) <= float(clip_at))
    x = np.where(good, z, 0.0)
    out = []
    var = float((x * x).sum())
    pairs = float(good.sum())
    if not var or not pairs:
        return [0.0] * int(lags)
    for k in range(1, int(lags) + 1):
        both = good[:, :-k] & good[:, k:]
        if not both.any():
            out.append(0.0)
            continue
        a, b = x[:, :-k][both], x[:, k:][both]
        # normalised by the variance of the samples that enter the product, so
        # rho_0 is 1 by construction whatever was thrown out
        norm = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        out.append(float((a * b).sum() / norm) if norm else 0.0)
    return out


def excursions(z, widths, rho, nsig):
    """The mask of every sample inside a window whose aggregate Z is beyond nsig.

    `z` is (rows, columns), `widths` the window lengths in samples, in
    ascending order; a window is judged only where all of its samples were
    measured. Every sample of a flagged window is flagged, which is why a wide
    excursion comes out as the band it is rather than as its centre.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    n, m = z.shape
    finite = np.isfinite(z)
    filled = np.where(finite, z, 0.0)
    zero = np.zeros((n, 1))
    csum = np.concatenate([zero, np.cumsum(filled, axis=1)], axis=1)
    cfin = np.concatenate([zero, np.cumsum(finite, axis=1)], axis=1)
    out = np.zeros((n, m), dtype=bool)
    for w in widths:
        w = int(w)
        if w < 1 or w > m:
            continue
        limit = float(nsig) * math.sqrt(window_variance(w, rho))
        total = csum[:, w:] - csum[:, :-w]
        count = cfin[:, w:] - cfin[:, :-w]
        hit = (count == w) & (np.abs(total) > limit)
        if not hit.any():
            continue
        # every sample covered by a flagged window: a start in [i - w + 1, i]
        starts = np.zeros((n, m), dtype=np.int32)
        starts[:, :m - w + 1] = hit
        run = np.cumsum(starts, axis=1)
        covered = run.copy()
        covered[:, w:] -= run[:, :-w]
        out |= covered > 0
    return out


def running_stats(values, window, step=None, min_valid=0.5, origin=0):
    """(local median, robust sigma) of `values` along its last axis.

    The median and 1.4826 times the median absolute deviation about it, in
    boxes of `window` samples centred every `step` (window // 4 by default) and
    interpolated linearly in between. NaN are ignored; a box with fewer than
    `min_valid` of its samples finite gives no estimate, and its neighbours'
    are carried across it. Robust on purpose: the statistic has to describe
    the noise around an outlier, not be pulled by it. `origin` is the index of
    the first sample in a longer array: the box centres sit at multiples of
    `step` of that array, so a slice of it gets the same boxes as the whole.
    """
    x = np.atleast_2d(np.asarray(values, dtype=float))
    n, m = x.shape
    window = int(window) | 1
    half = window // 2
    step = max(1, int(step) if step else window // 4)
    centres = np.unique(np.r_[0, np.arange((-int(origin)) % step, m, step), m - 1])
    med = np.full((n, m), np.nan)
    sig = np.full((n, m), np.nan)
    index = np.arange(m)
    pad = np.full(half, np.nan)
    for r in range(n):
        boxes = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([pad, x[r], pad]), window)[centres]
        count = np.isfinite(boxes).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mc = np.nanmedian(boxes, axis=1)
            sc = 1.4826 * np.nanmedian(np.abs(boxes - mc[:, None]), axis=1)
        ok = (count >= min_valid * window) & np.isfinite(sc) & (sc > 0)
        if ok.sum() < 2:
            continue
        med[r] = np.interp(index, centres[ok], mc[ok])
        sig[r] = np.interp(index, centres[ok], sc[ok])
    shape = np.shape(values)
    return med.reshape(shape), sig.reshape(shape)


def clip(values, window, nsig, **kwargs):
    """(values with NaN where they sit beyond nsig robust sigma of the local
    median, the mask of what was set to NaN)."""
    values = np.asarray(values, dtype=float)
    med, sig = running_stats(values, window, **kwargs)
    with np.errstate(invalid="ignore"):
        cut = np.abs(values - med) > float(nsig) * sig
    cut &= np.isfinite(values) & np.isfinite(sig)
    return np.where(cut, np.nan, values), cut


def _pieces(cube, fit, window):
    """What every pass over a cube's residual needs, read once."""
    from .twoframe import (cube_grid, load_cube, row_parity)
    m = np.asarray(cube_grid(cube)).size
    _, _, w1, meta = load_cube(cube, dtype=np.float32, columns=np.arange(0, 1))
    n = w1.shape[0]
    if len(fit["berv"]) != n:
        raise SystemExit("the fit has %d rows and the cube %d: it was not made on"
                         " this cube" % (len(fit["berv"]), n))
    delta = -pixel_shift(np.asarray(fit["berv"], dtype=float), float(fit["dv"]))
    reach = int(np.ceil(np.abs(delta).max())) + 2
    window = int(window) | 1
    return dict(n=n, m=m, meta=meta, parity=row_parity(meta, n) % 2,
                names=[os.path.basename(str(v)) for v in meta["filename"]],
                delta=delta, reach=reach, window=window,
                margin=reach + 16 + window)


def residual_z(cube, fit, window, block=40000, chunk=16, blocks=None,
               columns=None):
    """Yield the residual's z, chunk by chunk: (rows, c0, c1, inner, z).

    The residual is panel 5's: the cube less the star block carried into each
    row's frame, the star-frame means if the fit has them, the observer block
    and the parity offset, NaN where the fit gave no weight. It is divided by
    the local robust sigma around it (running_stats over `window` samples), so
    what comes out is comparable from one exposure and one wavelength to the
    next. The cube is read `block` columns at a time, each with a margin
    covering the largest shift, the Lanczos kernel and half the window, so
    every column comes out as from the whole cube; `block=None` reads it whole,
    and `blocks` stops after that many (for a measurement rather than a pass).
    `columns` restricts the pass to a (first, last) range of the grid, which is
    what a figure of one window needs.
    """
    from .twoframe import (LanczosShifter, carried_means, fit_means,
                           fit_templates, load_cube, star_model)
    part = _pieces(cube, fit, window)
    n, m = part["n"], part["m"]
    P, a = np.asarray(fit["P"], dtype=float), np.asarray(fit["a"], dtype=float)
    Q, b = np.asarray(fit["Q"], dtype=float), np.asarray(fit["b"], dtype=float)
    means, group = fit_means(fit, part["meta"], n, m)
    T, tgroup = fit_templates(fit, part["meta"], n, m)
    step = m if block is None else int(block)
    done = 0
    first, last = (0, m) if columns is None else (max(0, int(columns[0])),
                                                 min(m, int(columns[1])))
    for c0 in range(first, last, step):
        if blocks is not None and done >= int(blocks):
            return
        done += 1
        c1 = min(c0 + step, last)
        a0, b0 = max(0, c0 - part["margin"]), min(m, c1 + part["margin"])
        _, data, w, _ = load_cube(cube, dtype=np.float32,
                                  columns=np.arange(a0, b0))
        shifter = LanczosShifter(b0 - a0, a=8, max_shift=part["reach"])
        Tf = shifter.prepare(T[:, a0:b0]) if np.any(T[:, a0:b0]) else None
        inner = slice(c0 - a0, c1 - a0)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            rows = slice(start, stop)
            model = star_model(P[:, a0:b0], a[rows], shifter,
                               part["delta"][rows], stop - start, b0 - a0)
            if Tf is not None:
                model += carried_means(Tf, tgroup, shifter, part["delta"],
                                       start, stop)
            model += b[rows] @ Q[:, a0:b0]
            model += means[:, a0:b0][group[rows]]
            resid = np.where(w[rows] > 0, data[rows] - model, np.nan)
            med, sig = running_stats(resid, part["window"], origin=a0)
            with np.errstate(invalid="ignore"):
                z = (resid - med) / sig
            yield (start, stop), c0, c1, inner, z
        del data, w


def noise_correlation(cube, fit, window, lags, blocks=1, chunk=16, clip_at=5.0):
    """The residual's rho_1..rho_lags, measured on the run's own cube.

    A block of columns is enough: the whole point is the shape of the noise's
    correlation, which the grid step, the registration and the high pass set,
    not the wavelength. Averaged over the chunks of rows it sees.
    """
    weights, total = 0.0, np.zeros(int(lags))
    for (start, stop), _c0, _c1, inner, z in residual_z(
            cube, fit, window, chunk=chunk, blocks=blocks):
        rho = autocorrelation(z[:, inner], lags, clip_at=clip_at)
        count = float(np.isfinite(z[:, inner]).sum())
        total += count * np.asarray(rho)
        weights += count
    return list(total / weights) if weights else [0.0] * int(lags)


def residual_outliers(cube, fit, nsig, window, block=40000, chunk=16,
                      column_frac=None, column_chi2=None, column_min_rows=10,
                      excursion_nsig=None, excursion_samples=None, rho=None,
                      excess_nsig=None, excess_chi2=None, excess_samples=None,
                      excess_clip=10.0, bias_nsig=None, bias_samples=None,
                      bias_clip=5.0):
    """(file -> (2, grid) mask of what to NaN, a report).

    Row 0 of each mask is the even orders, row 1 the odd ones, as
    reconstruct.fit_weights_mask.

    What is flagged, per exposure:
      * with `excursion_nsig` and `excursion_samples`, every window from one
        sample to `excursion_samples` whose aggregate significance is beyond
        `excursion_nsig`, the whole window (excursions, window_variance). `rho`
        is the noise's autocorrelation; it is measured here when not given.
      * otherwise every sample beyond `nsig` robust sigmas on its own, which
        is what correct.nsig_cut asked for before the excursions existed.

    Then, with `column_frac` and `column_chi2`, every exposure also loses the
    columns the exposures disagree on as a body: more than `column_frac` of
    them flagged there and the survivors' reduced chi2 still above
    `column_chi2`, over at least `column_min_rows` exposures of that parity.
    """
    part = _pieces(cube, fit, window)
    n, m, names, parity = part["n"], part["m"], part["names"], part["parity"]
    out = {name: np.zeros((2, m), dtype=bool) for name in names}
    by_excursion = bool(excursion_nsig) and bool(excursion_samples)
    widths = (list(range(1, int(excursion_samples) + 1)) if by_excursion
              else [])
    # the residual's own correlation, which BOTH tests need: the excursions to
    # know what a window's sum is worth, the regions to know how many
    # independent columns a window holds. Measured with a correlation length of
    # 1 the regions came out 15% of the spectrum instead of 6% (2026-09-25).
    by_excess = bool(excess_nsig) and bool(excess_samples)
    by_bias = bool(bias_nsig) and bool(bias_samples)
    if (by_excursion or by_excess or by_bias) and rho is None:
        lags = max(len(widths) - 1, int(excess_samples or 0),
                   int(bias_samples or 0), 8)
        rho = noise_correlation(cube, fit, window, lags, blocks=1, chunk=chunk)
    rho = list(rho or [])
    # per order parity and per OBSERVER column, over every exposure: how many
    # were measured there, how many were flagged, and the z^2 of the rest
    wanted = bool(column_frac) and bool(column_chi2)
    # and, for the excess-RMS test, the chi2 of EVERY measured sample of the
    # column, not only of what survived the flagging: what is being looked for
    # is the scatter itself. Beyond excess_clip a sample is a cosmic ray rather
    # than a region, and the per-sample flagging is what deals with it.
    all_z2 = np.zeros((2, m))
    all_n = np.zeros((2, m), dtype=np.int32)
    # et, pour le biais coherent, la somme SIGNEE: ce qui est ancre dans le
    # referentiel de l'observateur s'y ajoute alors que l'etoile, qui bouge
    # avec le BERV, s'y moyenne
    all_z = np.zeros((2, m))
    bias_n = np.zeros((2, m), dtype=np.int32)
    seen = np.zeros((2, m), dtype=np.int32)
    taken = np.zeros((2, m), dtype=np.int32)
    kept = np.zeros((2, m), dtype=np.int32)
    sum_z2 = np.zeros((2, m), dtype=float)
    flagged = 0
    for (start, stop), c0, c1, inner, z in residual_z(
            cube, fit, window, block=block, chunk=chunk):
        measured = np.isfinite(z)
        if by_excursion:
            cut = excursions(z, widths, rho, excursion_nsig)
        elif nsig:
            cut = measured & (np.abs(z) > float(nsig))
        else:
            # the region test alone: nothing is flagged sample by sample, and
            # every measured sample counts towards its column's scatter
            cut = np.zeros(z.shape, dtype=bool)
        for i, r in enumerate(range(start, stop)):
            out[names[r]][int(parity[r])][c0:c1] = cut[i, inner]
        flagged += int(cut[:, inner].sum())
        if by_bias:
            steady = measured[:, inner] & (np.abs(z[:, inner]) <= float(bias_clip))
            signed = np.where(steady, z[:, inner], 0.0)
            rows_parity = parity[start:stop]
            for p in (0, 1):
                which = np.flatnonzero(rows_parity == p)
                if which.size:
                    all_z[p, c0:c1] += signed[which].sum(axis=0)
                    bias_n[p, c0:c1] += steady[which].sum(axis=0)
        if by_excess:
            inside = measured[:, inner] & (np.abs(z[:, inner]) <= float(excess_clip))
            square = np.where(inside, z[:, inner] ** 2, 0.0)
            rows_parity = parity[start:stop]
            for p in (0, 1):
                which = np.flatnonzero(rows_parity == p)
                if which.size:
                    all_z2[p, c0:c1] += square[which].sum(axis=0)
                    all_n[p, c0:c1] += inside[which].sum(axis=0)
        if wanted:
            rows_parity = parity[start:stop]
            for p in (0, 1):
                which = np.flatnonzero(rows_parity == p)
                if not which.size:
                    continue
                ok = measured[which][:, inner]
                gone = cut[which][:, inner]
                survivor = np.where(ok & ~gone, z[which][:, inner], np.nan)
                seen[p, c0:c1] += ok.sum(axis=0)
                taken[p, c0:c1] += gone.sum(axis=0)
                kept[p, c0:c1] += np.isfinite(survivor).sum(axis=0)
                sum_z2[p, c0:c1] += np.nansum(survivor ** 2, axis=0)

    frac = np.divide(taken, seen, out=np.zeros((2, m)), where=seen > 0)
    chi2 = np.divide(sum_z2, kept, out=np.zeros((2, m)), where=kept > 0)
    columns = np.zeros((2, m), dtype=bool)
    report = dict(columns=columns, frac=frac, chi2=chi2, seen=seen,
                  n_columns=0, added=0, thresholds=None, flagged=flagged,
                  rho=rho, widths=widths, excess=None, bias=None,
                  variance=[window_variance(w, rho) for w in widths])
    if by_bias:
        mean_z = np.divide(all_z, bias_n, out=np.zeros((2, m)), where=bias_n > 0)
        zbias = coherent_bias(mean_z, bias_n, int(bias_samples), rho,
                              min_rows=int(column_min_rows))
        regions = (zbias > float(bias_nsig)) & (bias_n >= int(column_min_rows))
        added = 0
        for name in names:
            added += int((regions & ~out[name]).sum())
            out[name] |= regions
        report["bias"] = dict(mean=mean_z, z=zbias, regions=regions,
                              n_columns=int(regions.sum()), added=added,
                              measured=int((bias_n >= int(column_min_rows)).sum()),
                              threshold=float(bias_nsig),
                              samples=int(bias_samples))
        columns = columns | regions
    if by_excess:
        # what stands still in the observer's frame: the columns whose scatter
        # over the exposures is more than the noise, over a window as wide as
        # the structures the river plots show
        length = 1.0 + 2.0 * sum(rho) if rho else 1.0
        column_chi2_all = np.divide(all_z2, all_n, out=np.zeros((2, m)),
                                    where=all_n > 0)
        windowed, excess = excess_rms(column_chi2_all, all_n,
                                      int(excess_samples), length,
                                      min_rows=int(column_min_rows))
        regions = (excess > float(excess_nsig)) & (all_n >= int(column_min_rows))
        if excess_chi2:
            regions &= windowed > float(excess_chi2)
        added = 0
        for name in names:
            added += int((regions & ~out[name]).sum())
            out[name] |= regions
        report["excess"] = dict(
            chi2=column_chi2_all, windowed=windowed, excess=excess,
            regions=regions, n_columns=int(regions.sum()), added=added,
            length=length, measured=int((all_n >= int(column_min_rows)).sum()),
            thresholds=(float(excess_nsig), float(excess_chi2 or 0)),
            samples=int(excess_samples))
        columns = columns | regions
    if wanted:
        columns = ((seen >= int(column_min_rows)) & (frac > float(column_frac))
                   & (chi2 > float(column_chi2)))
        added = 0
        for name in names:
            added += int((columns & ~out[name]).sum())
            out[name] |= columns
        report.update(columns=columns, n_columns=int(columns.sum()),
                      added=added,
                      thresholds=(float(column_frac), float(column_chi2)))
    return out, report
