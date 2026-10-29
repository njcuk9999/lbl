"""The telluric exponents (H2O and the dry absorbers) by the DERIVATIVE.

THE PRINCIPLE. The spectrum is divided by the modelled transmission and the
exponent kept is the one that makes the corrected spectrum the SMOOTHEST,
that is, the one that minimises the weighted dispersion of the
pixel-to-pixel gradient. A wrong exponent leaves at each telluric line a
residual shaped like a DERIVATIVE, which inflates that dispersion; the
stellar lines contribute the same amount to it whatever the exponent, so
the criterion depends only on the tellurics, even where they are a
minority. Unlike an extrapolation of the CCF amplitude of the telluric
lines against the exponent, it has no reason to run onto its bounds.

Two details of the formulation are essential; without them the fit ends on
its bounds for a large share of the exposures:

    the OPTICAL DEPTH, tau = -ln(T), carried on a UNIFORM log grid, then
    exponentiated: tau is multiplied by the exponent, which is linear,
    whereas interpolating already exponentiated transmissions onto the
    irregular grid of the spectrum mixes the two operations;

    the CONVOLUTION to the resolution of the instrument, AFTER the power and
    BEFORE the division. TAPAS is much finer than the spectrograph (a
    resolution element of 2.6 km/s at R = 115 000), and lines that are too
    narrow cannot cancel broadened lines.

WHAT IS FREE, for the optical:

    H2O             free PER FILE: the water column changes too fast for
                    a per-night model. And there is ALWAYS water: an
                    exponent on the lower bound is a fit failure, never a
                    dry night.
    the rest        as a single block, not per chemical species: in the
                    visible only O2 really matters among the dry species,
                    and they all follow the same airmass. This is also the
                    split that TAPAS gives in LBL (ABSO_WATER and
                    ABSO_OTHERS).
    the velocity    NOT free: the model is evaluated in the OBSERVER frame
                    (the ground), where the atmosphere is; a barycentric
                    wave solution is shifted back to it by the BERV before
                    the fit (tellu_clean.correct_tellu, BERV_TRANS).
"""
import numpy as np
from astropy import constants

C = constants.c.to("km/s").value
#: the FWHM of a Gaussian over its sigma, 2 sqrt(2 ln 2)
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
#: the 0-1 weights of the transmission (transmission_weights): a sigmoid of
#:    width transition_sigma centred on a transmission of depth_max, and
#:    zero where the transmission is under depth_saturated and on the wings
#:    of those lines, until it rises back above reject_saturated
WEIGHTS = dict(depth_max=0.6, depth_saturated=0.5, reject_saturated=0.8,
               transition_sigma=0.02)
#: the gradient pixels this many sigmas from its median lose half their
#:    weight, and all of it a few sigmas further (a sigmoid)
OUTLIER_SIGMA = 6.0
#: the step of the model grid, in km/s
STEP_KMS = 0.25

#: the water bands of the visible (380 to 700 nm), read on TAPAS: elsewhere
#: water does not exceed 1% depth, and a fit there is drowned by the stellar
#: lines. A reference for water_mask; the fit itself takes its windows from
#: TAPAS (telluric_windows)
WATER_BANDS_VISIBLE = ((586.0, 600.0), (644.0, 664.0), (686.0, 696.0))

#: THE DRY BANDS: ONLY WHERE IT CONSTRAINS SOMETHING.
#:
#: The dry component is not water and is not measured in the same places.
#: Read on TAPAS in 5 nm windows, the only windows of the visible (380 to
#: 700 nm) where it exceeds 5% depth:
#:
#:     575-585 nm   6%     marginal
#:     625-635 nm   70%    O2 gamma, and water is only 9% there
#:     685-690 nm   100%   B band, saturated, and water is only 2.6% there
#:     690-695 nm   100%   DISCARDED: water rises to 77% there, the two
#:                         species merge there and the fit can no longer
#:                         separate them
#:
#: Below 450 nm it drops to 0.2% maximum depth, that is, nothing.
#: Fitting it over the whole domain costs about fifty times more in
#: computation for information that only comes from the red, and adds the
#: noise of the blue half to a measurement that half does not constrain.
#: A reference, like WATER_BANDS_VISIBLE: the fit takes its windows from
#: TAPAS (telluric_windows)
DRY_BANDS_VISIBLE = ((625.0, 635.0), (685.0, 690.0))

#: THE FALLBACK MARK. The fallback value (0.35) falls exactly on a point of
#: the sweep, so an exposure that converges to 0.35 and an exposure that
#: failed would return the same number. This epsilon is added to the
#: fallback values only: three ten-millionths of the exponent, that is,
#: nothing for the physics, but `is_fallback` recognises them, and so does
#: a log.
FALLBACK_MARK = 1e-7


#: THE WINDOWS THE FIT SEES, read from TAPAS rather than hard-coded, so that
#: they follow any instrument's domain: the 5 nm windows where water
#: (exponent 1, at the resolution of the instrument) exceeds 15% depth, or
#: the dry component 25%. From 380 to 700 nm they give back the bands above
#: (585-600, 645-660, 625-635, 685-695 nm); further red they add 695-740
#: (water) and 755-770 nm (O2 A band). Over the whole spectrum, the minimum
#: of the objective stays in the same place, but its variation is diluted
#: by the stellar lines and the noise of the pixels without tellurics,
#: which do not depend on the exponent, and the sweep's safeguard sends
#: most exposures to the fallback.
WINDOW_NM = 5.0
WINDOW_WATER_DEPTH = 0.15
WINDOW_DRY_DEPTH = 0.25
#: the edge of the model grid, where the convolution makes up false lines
#: (45% depth at both ends): never a window
WINDOW_EDGE_NM = 2.0
#: below this, the windows say nothing and the whole spectrum is kept
WINDOW_MIN_PIXELS = 2000


def telluric_windows(model, wave):
    """The pixels of `wave` that fall in a window deep in water or in the
    dry component (WINDOW_*), read from the TAPAS model at exponent 1."""
    wave = np.asarray(wave, dtype=float)
    water = 1.0 - transmission(model, wave, 1.0, 0.0)
    dry = 1.0 - transmission(model, wave, 0.0, 1.0)
    low, high = np.nanmin(wave), np.nanmax(wave)
    index = np.floor((wave - low) / WINDOW_NM).astype(int)
    keep = np.zeros(wave.size, dtype=bool)
    for one in np.unique(index):
        inside = index == one
        start = low + one * WINDOW_NM
        if start < low + WINDOW_EDGE_NM or \
                start + WINDOW_NM > high - WINDOW_EDGE_NM:
            continue
        if np.nanmax(water[inside]) > WINDOW_WATER_DEPTH or \
                np.nanmax(dry[inside]) > WINDOW_DRY_DEPTH:
            keep[inside] = True
    return keep


def is_fallback(water, fallback=0.35):
    """Does this exponent come from the fallback rather than a measurement?"""
    return bool(abs(float(water) - (fallback + FALLBACK_MARK)) < FALLBACK_MARK / 10)


def band_mask(wave, bands):
    """The pixels that fall in one of the bands."""
    wave = np.asarray(wave, dtype=float)
    out = np.zeros(wave.shape, dtype=bool)
    for low, high in bands:
        out |= (wave > low) & (wave < high)
    return out


def water_mask(wave, bands=WATER_BANDS_VISIBLE):
    """The pixels of the water bands, where the H2O exponent is fitted."""
    wave = np.asarray(wave, dtype=float)
    out = np.zeros(wave.shape, dtype=bool)
    for lo, hi in bands:
        out |= (wave >= lo) & (wave <= hi)
    return out


def _spans_of(wave, gap_nm=1.0):
    """The intervals these wavelengths really cover.

    Returns a list of (start, end), skipping the gaps wider than `gap_nm`.
    This is what lets model_grid build the model only where it will be
    used: the three water bands of the visible only take up 40% of the
    interval they span.
    """
    wave = np.asarray(wave, dtype=float)
    wave = np.sort(wave[np.isfinite(wave)])
    if wave.size == 0:
        return []
    cuts = np.where(np.diff(wave) > gap_nm)[0]
    starts = np.concatenate([[0], cuts + 1])
    stops = np.concatenate([cuts, [wave.size - 1]])
    return [(float(wave[a]), float(wave[b])) for a, b in zip(starts, stops)]


def model_grid(spl_water, spl_dry, wave_min, wave_max, resolution,
               step_kms=STEP_KMS, bands=None):
    """The optical depths on a uniform log grid, and the kernel.

    tau rather than T, because tau is what gets multiplied by the exponent.

    `bands` restricts the grid to the useful intervals only. Without it the
    grid covers wave_min to wave_max CONTINUOUSLY, and the three water
    bands of the visible only take up 40% of that interval: 60% of the
    convolution, the most expensive step of the fit, would be spent where
    no pixel is fitted.

    Each piece is widened by the half-width of the kernel on either side,
    otherwise the convolution would be wrong on its own edges; the pieces
    that overlap after widening are merged. The result is numerically the
    same as the continuous grid inside the pieces, and correct at their
    edges by construction.
    """
    step = np.arctanh(step_kms / C)
    sig_pix = np.arctanh(C / resolution / FWHM_PER_SIGMA / C) / step
    half = int(5 * sig_pix)
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sig_pix) ** 2)
    kernel = kernel / kernel.sum()

    if bands:
        # the pieces, widened by the kernel and merged when they touch
        margin = (half + 1) * step
        # THE PIECES FALL ON THE CONTINUOUS GRID, not beside it: each start
        # is rounded down to the grid point below, so that the pieces sample
        # the same wavelengths as the continuous grid would. Otherwise the
        # two grids differ by a fraction of a step and np.interp samples the
        # model at shifted places (differences of 2e-3 in transmission).
        base = np.log(wave_min)
        spans = []
        for low, high in sorted(bands):
            lo = max(base, np.log(low) - margin)
            hi = min(np.log(wave_max), np.log(high) + margin)
            lo = base + np.floor((lo - base) / step) * step
            if hi <= lo:
                continue
            if spans and lo <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
            else:
                spans.append((lo, hi))
    else:
        spans = [(np.log(wave_min), np.log(wave_max))]
    if not spans:
        spans = [(np.log(wave_min), np.log(wave_max))]

    lnw = np.concatenate([np.arange(lo, hi, step) for lo, hi in spans])
    wave = np.exp(lnw)
    tau_water = -np.log(np.clip(spl_water(wave), 1e-6, 1.0))
    tau_dry = -np.log(np.clip(spl_dry(wave), 1e-6, 1.0))
    # the boundaries between pieces, to convolve each one on its own: a
    # straddling convolution would mix two non-adjacent regions
    edges = np.cumsum([0] + [len(np.arange(lo, hi, step)) for lo, hi in spans])
    return dict(wave=wave, tau_water=tau_water, tau_dry=tau_dry,
                kernel=kernel, pieces=edges)


def transmission(model, wave_ground, alpha_water, alpha_dry):
    """T on `wave_ground`: the power first, the convolution next."""
    raw = np.exp(-alpha_water * model["tau_water"]
                 - alpha_dry * model["tau_dry"])
    pieces = model.get("pieces")
    if pieces is None or len(pieces) <= 2:
        conv = np.convolve(raw, model["kernel"], mode="same")
    else:
        # EACH PIECE ON ITS OWN: a convolution straddling two pieces would
        # mix wavelengths that do not touch
        conv = np.empty_like(raw)
        for start, stop in zip(pieces[:-1], pieces[1:]):
            conv[start:stop] = np.convolve(raw[start:stop], model["kernel"],
                                           mode="same")
    return np.interp(wave_ground, model["wave"], conv)


def transmission_weights(trans, depth_max, depth_saturated, reject_saturated,
                         transition_sigma):
    """The 0-1 weights of the transmission: a sigmoid on the depth, and the
    saturated lines and their wings set to zero.

    THE RULE: sweeping in one direction, contamination is
    ENTERED at each saturated pixel and LEFT at the first pixel that rises
    back above `reject_saturated`; a pixel is contaminated if it is so in
    either of the two directions. In other words a pixel is contaminated
    when the last event met before it is a SATURATED pixel and not a return
    above the threshold, which a cumulative maximum of the event indices
    gives in one pass, without a Python loop over the pixels (this function
    is called hundreds of times per fit).
    """
    w_depth = 1.0 / (1.0 + np.exp(-(trans - depth_max) / transition_sigma))
    n = len(trans)
    if n == 0:
        return w_depth
    sat = trans < depth_saturated
    above = trans >= reject_saturated
    where = np.arange(n)
    contam = np.zeros(n, dtype=bool)
    for backwards in (False, True):
        this_sat = sat[::-1] if backwards else sat
        this_above = above[::-1] if backwards else above
        last_sat = np.maximum.accumulate(np.where(this_sat, where, -1))
        last_above = np.maximum.accumulate(np.where(this_above, where, -1))
        here = (last_sat >= 0) & (this_sat | (last_sat > last_above))
        contam |= here[::-1] if backwards else here
    return w_depth * (~contam)


def weighted_nanstd(data, weights):
    """The weighted standard deviation of `data` (with the unbiased
    weighted variance, for reliability weights); inf with fewer than 32
    usable points."""
    ok = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    d, w = data[ok], weights[ok]
    if d.size < 32:
        return np.inf
    mean = np.sum(w * d) / np.sum(w)
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    if denom <= 0:
        return np.inf
    return np.sqrt(np.sum(w * (d - mean) ** 2) / denom)


def gradient_noise(trans, noise=None):
    """The noise of np.gradient(sp / trans), pixel by pixel, from the noise
    of sp (None: the same everywhere). The noise follows the division: a
    pixel of sp with noise sigma becomes one of sigma / T in the corrected
    spectrum; and np.gradient takes (c[i+1] - c[i-1]) / 2 inside, c[1] - c[0]
    and c[-1] - c[-2] at the ends."""
    t = np.clip(np.asarray(trans, dtype=float), 1e-3, None)
    sigma = (np.ones_like(t) if noise is None
             else np.asarray(noise, dtype=float)) / t
    s2 = sigma ** 2
    var = np.empty_like(s2)
    if s2.size < 2:
        return np.sqrt(s2)
    var[1:-1] = (s2[2:] + s2[:-2]) / 4.0
    var[0] = s2[0] + s2[1]
    var[-1] = s2[-1] + s2[-2]
    return np.sqrt(var)


def objective(sp, trans, noise=None):
    """The weighted dispersion of the gradient of the corrected spectrum, IN
    UNITS OF ITS NOISE.

    Dividing by the transmission also divides the noise: a pixel of noise
    sigma becomes sigma / T, and without accounting for it every increase
    of the exponent inflates the noise of the deep lines. On a faint star
    this noise outweighs the telluric lines being removed: the cost rises
    monotonically with the exponent, with no minimum, and the water falls
    to the fallback. The gradient is therefore divided by its noise
    (gradient_noise), which follows the division by the transmission and,
    with `noise`, the photon noise of the exposure."""
    grad = np.gradient(sp / np.clip(trans, 1e-3, None))
    grad = grad / gradient_noise(trans, noise)
    w = transmission_weights(trans, **WEIGHTS)
    valid = w > 0.5
    if valid.sum() < 64:
        return np.inf
    p16, p84 = np.nanpercentile(grad[valid], [16, 84])
    sig = max((p84 - p16) / 2.0, 1e-12)
    dev = np.abs(grad - np.nanmedian(grad[valid])) / sig
    w = w / (1.0 + np.exp(np.clip(dev - OUTLIER_SIGMA, -500, 500)))
    return weighted_nanstd(grad, w)


#: the dry exponent free within this range of the airmass: the dry column
#:    follows it to a few percent (fit_exponents), and an unbounded fit can
#:    sink to zero on a blind pass, without a template
DRY_RANGE = (0.5, 1.5)


def fit_exponents(wave_ground, sp, spl_water, spl_dry, airmass,
                  resolution=115000.0, free_dry=True, fallback=0.35,
                  flat_span=1e-3, noise=None):
    """(H2O exponent, dry exponent) of THIS file, by the derivative.

    THE FIT IS DONE IN LOGARITHM, and that is what keeps it from reaching
    the bounds. There is ALWAYS water in the atmosphere (tens of thousands
    of infrared spectra say so), so a zero exponent is a fit failure and
    never a dry night. By fitting ln(alpha), alpha stays strictly positive
    whatever the simplex does, and a relative variation counts as much at
    the top as at the bottom of the range, which is the right metric for a
    column that varies by a factor of several. A fit in linear lets some
    exposures fall to zero or near it; in logarithm none does.

    `free_dry` is TRUE by default. The dry component does follow the
    airmass, but not with the coefficient 1 that TAPAS assumes: TAPAS is
    computed for a reference column, and the altitude of an observatory
    removes part of it (about 4% at 2400 m, a dry exponent of 0.96 times
    the airmass). Fixing it to the airmass would impose that systematic
    error on all the dry absorption; free, it is fitted with a scatter of a
    few percent around the airmass.

    `fallback` and `flat_span` carry the safeguard on the exposures without
    signal, described at the sweep below. It is HERE and not only in
    fit_campaign because LBL calls this function one exposure at a time and
    so has no campaign median at hand.
    """
    from scipy.optimize import minimize

    wave_ground = np.asarray(wave_ground, dtype=float)
    sp = np.asarray(sp, dtype=float)
    # the noise of each pixel of sp (`noise`, before the transmission is
    #   divided out); the same everywhere when not given
    noise = (np.ones_like(sp) if noise is None
             else np.asarray(noise, dtype=float))
    good = (np.isfinite(sp) & (sp > 0) & np.isfinite(wave_ground)
            & np.isfinite(noise) & (noise > 0))
    wave_ground, sp, noise = wave_ground[good], sp[good], noise[good]
    # THE PIECES COME FROM THE DATA, and not from a hard-coded list of
    # bands: the model is only built where there are pixels to explain. A
    # gap of more than 1 nm in the wavelengths separates two pieces; below
    # that it is only an order ending, and stitching them costs less than
    # separating them. Works for any instrument and any choice of bands.
    model = model_grid(spl_water, spl_dry, wave_ground.min() * 0.999,
                       wave_ground.max() * 1.001, resolution,
                       bands=_spans_of(wave_ground))
    # THE FIT ONLY SEES THE TELLURIC WINDOWS (telluric_windows):
    #   elsewhere the spectrum only brings stellar lines and noise, which
    #   do not depend on the exponent and drowned its variation under the
    #   safeguard of the sweep
    keep = telluric_windows(model, wave_ground)
    if keep.sum() >= WINDOW_MIN_PIXELS:
        wave_ground, sp, noise = wave_ground[keep], sp[keep], noise[keep]
        model = model_grid(spl_water, spl_dry, wave_ground.min() * 0.999,
                           wave_ground.max() * 1.001, resolution,
                           bands=_spans_of(wave_ground))

    low, high = DRY_RANGE[0] * float(airmass), DRY_RANGE[1] * float(airmass)

    def cost(p):
        water = float(np.exp(p[0]))
        dry = float(np.exp(p[1])) if free_dry else float(airmass)
        if not (1e-4 < water < 20) or not (low < dry < high):
            return np.inf
        return objective(sp, transmission(model, wave_ground, water, dry),
                         noise)

    # THE SWEEP FIRST, and it serves TWICE: it says whether the exposure has
    # enough to decide, and it gives the simplex a starting point on the
    # right side.
    #
    # An exposure without signal has no minimum at all: its cost RISES from
    # the smallest exponent on, so the simplex would push it to zero. Since
    # there is always water, this is a fit failure, and `fallback` is
    # returned rather than an invented value.
    #
    # THE SWEEP GOES DOWN TO 0.002, and this detail matters: an exposure
    # whose cost has a weak contrast still has a clear, INTERIOR minimum
    # (anywhere from 0.2 to 1.0), but a sweep that stopped at 0.02 can find
    # its lowest cost on its first point and take it for an edge, sending a
    # healthy exposure to the fallback.
    scan = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6,
                     1.0, 1.6, 2.5])
    # the sweep only moves the water; the dry component stays at its
    # starting value, which is enough to locate the dip in water. BOTH
    # parameters are always passed, as cost() reads p[1] when the dry
    # component is free.
    dry0 = np.log(max(float(airmass), 1e-3))
    costs = np.array([cost([np.log(a), dry0]) for a in scan])
    if not np.all(np.isfinite(costs)):
        return float(fallback) + FALLBACK_MARK, float(airmass)
    # WHAT "NOTHING TO GO ON" IS, the cost being in units of the noise: a
    #   minimum on an edge of the scan, or a curve flat to `flat_span`
    #   (0.1%), numerically. A faint star can have a real interior minimum
    #   with a span of only 0.5 to 2%, so the span alone cannot ask for
    #   more; a flat flux without tellurics, at the same noise, is no flat
    #   curve but a minimum at the scan's first point (spans of a few
    #   percent or more), which the edge test catches
    span = (costs.max() - costs.min()) / max(abs(costs.min()), 1e-30)
    jbest = int(np.argmin(costs))
    flat = span < flat_span
    on_edge = jbest in (0, len(scan) - 1)
    if flat or on_edge:
        # THE FALLBACK MUST BE RECOGNISABLE. `fallback` is 0.35, which is
        # ALSO a point of the sweep: an exposure that really converges to
        # 0.35 would return the same number as an exposure that failed, and
        # neither the log nor the header could tell them apart.
        #
        # So the value is shifted by an epsilon that is physically invisible
        # (0.35 against 0.3500001, i.e. three ten-millionths of the exponent)
        # but that makes the fallback recognisable on reading.
        return float(fallback) + FALLBACK_MARK, float(airmass)

    x0 = [np.log(scan[jbest]), np.log(max(float(airmass), 1e-3))]
    out = minimize(cost, x0=x0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-10, maxiter=300))
    water = float(np.exp(out.x[0]))
    dry = float(np.exp(out.x[1])) if free_dry else float(airmass)
    if not np.isfinite(water) or water < scan[0] or water > scan[-1]:
        return float(fallback) + FALLBACK_MARK, float(airmass)
    return water, dry


def fit_campaign(frames, spl_water, spl_dry, resolution=115000.0,
                 floor=0.05):
    """The H2O exponents of a campaign, with a safeguard on the SNR.

    `frames` is a list of (wave_ground, sp, airmass). Returns an array of
    exponents, one per exposure.

    WHY A SAFEGUARD. There is always water, so an exponent that collapses
    is a failure. The exposures that drift are those with a low SNR, and
    their cost curve is FLAT (about 1% variation over the sweep, where an
    exposure that works shows a clear minimum): the criterion simply has
    nothing to decide with, and forcing them would only invent a value.

    These exposures therefore get the campaign MEDIAN, which is the best
    estimate available for them. `floor` is the threshold below which the
    fit is considered not to have succeeded: 0.05, well below the exponents
    of the exposures that converge in the visible (5th percentile about
    0.15) and well above what the failures give (less than 0.01).
    """
    out = np.full(len(frames), np.nan)
    for i, (wave_ground, sp, airmass) in enumerate(frames):
        try:
            out[i] = fit_exponents(wave_ground, sp, spl_water, spl_dry,
                                   airmass, resolution)[0]
        except Exception:                                      # noqa: BLE001
            out[i] = np.nan
    usable = np.isfinite(out) & (out >= floor)
    if usable.any():
        out[~usable] = float(np.median(out[usable]))
    return out
