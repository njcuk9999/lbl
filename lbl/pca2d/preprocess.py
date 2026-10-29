"""BERV registration, high-pass filtering and per-pixel weight construction."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter

from .grids import doppler



def _fill_gaps(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate over non-finite samples; return (filled, good_mask).

    Filtering and splining both choke on NaNs. We fill them so the numerical
    operators stay well behaved, and carry the mask along so the affected
    pixels can be given zero weight at the end.
    """
    good = np.isfinite(y)
    if good.all():
        return y, good
    filled = np.array(y, dtype=np.float64, copy=True)
    if not good.any():
        filled[:] = 1.0
        return filled, good
    index = np.arange(y.size)
    filled[~good] = np.interp(index[~good], index[good], y[good])
    return filled, good


def highpass(y: np.ndarray, window: int, polyorder: int, mode: str):
    """Remove the slowly varying continuum / blaze.

    'divide'  : returns ln(y / savgol(y))       -- filter the flux itself
    'log_sub' : returns ln(y) - savgol(ln(y))   -- filter in the log domain

    Both return a quantity that sits at ~0 in the continuum and goes negative
    inside absorption lines. Non-positive flux is flagged rather than logged.
    """
    positive = np.isfinite(y) & (y > 0)
    filled, finite = _fill_gaps(np.where(positive, y, np.nan))
    good = positive & finite

    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, (filled.size - 1) | 1)

    if mode == "divide":
        low = savgol_filter(filled, window, polyorder)
        low = np.where(low > 0, low, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(filled / low)
    elif mode == "log_sub":
        log_flux = np.log(filled)
        out = log_flux - savgol_filter(log_flux, window, polyorder)
    else:
        raise ValueError("unknown highpass mode: %s" % mode)

    good &= np.isfinite(out)
    return np.where(good, out, 0.0), good


def lowpass(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """The smooth component alone (used for diagnostics / plots)."""
    filled, _ = _fill_gaps(y)
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, (filled.size - 1) | 1)
    return savgol_filter(filled, window, polyorder)


def erode_edges(good, pixels=1):
    """`good` without the last `pixels` valid samples on either side of every gap.

    The spline that carries a spectrum onto the grid is fed its gaps filled by
    a straight line, and a cubic spline's value between the last two valid
    pixels still bends toward that fill: about a quarter of the kink in the
    first interval, a fourteenth in the next. Beside a masked OH core that is
    sky-dominated flux pulled into what reads as a valid sample. Dropping the
    last valid pixel at every edge takes out the interval that carries most of
    it. The ends of the array are not gaps.
    """
    good = np.asarray(good, dtype=bool)
    out = good.copy()
    if pixels <= 0 or good.all() or not good.any():
        return out
    bad = ~good
    for k in range(1, int(pixels) + 1):
        out[..., :-k] &= ~bad[..., k:]      # a gap k samples to the right
        out[..., k:] &= ~bad[..., :-k]      # a gap k samples to the left
    return out


def drop_isolated(good, window=3):
    """Drop a valid sample that has a gap within `window` on BOTH sides.

    A sample can pass every quality cut and still be untrustworthy because of
    where it sits. Inside a masked region its high pass was built from a filter
    window mostly filled by interpolation, and a shift into another frame drew
    it from neighbours that are not there. On an image those samples are the
    speckles of colour inside a band of yellow, and they read as measurements.

    Acts along the last axis, so a single spectrum or a stack of them. The ends
    of the array are not gaps: a run that reaches the edge is bounded on one
    side only and survives.

        [1, nan, 5, nan, 2]            -> the 5 goes
        [1, nan, 3, 5, nan, 2]         -> the 3 and the 5 both go
        [1, nan, 5,5,5,5,3,5, nan, 2]  -> nothing goes, the run is long enough

    A run of valid samples bounded by gaps on both sides loses every sample
    that is within `window` of both ends, so runs up to 2*window - 1 long are
    emptied outright and a longer one keeps its middle.
    """
    good = np.asarray(good, dtype=bool)
    gap = ~good
    left = np.zeros_like(gap)
    right = np.zeros_like(gap)
    for k in range(1, int(window) + 1):
        left[..., k:] |= gap[..., :-k]
        right[..., :-k] |= gap[..., k:]
    return good & ~(left & right)


def register(wave_obs, values, good, berv, target_berv, grid,
             spline_order=3, mask_threshold=0.999):
    """Resample onto the common grid in the frame set by target_berv.

    The input wavelengths are in the observer frame: a stellar feature sits
    at lambda_obs = lambda_bary / D(BERV), D the relativistic Doppler factor
    sqrt((1 + v/c) / (1 - v/c)) (grids.doppler), i.e. a spectrum taken at high
    BERV has its stellar lines pushed blueward. Mapping each sample to
        lambda_bary = lambda_obs * D(BERV)
    therefore parks the star at rest. Registering to a non-zero target_berv
    simply divides D(target_berv) back out.

    BERV/dv is never an integer, so this is a genuine fractional-pixel
    resampling and is done with a cubic spline, not a pixel roll. The good-pixel
    mask is resampled linearly alongside; anything that picks up a contribution
    from a bad sample falls below mask_threshold and is flagged.
    """
    shift = float(doppler(berv) / doppler(target_berv))
    wave_shifted = wave_obs * shift
    values = np.asarray(values, dtype=np.float64)
    good = np.asarray(good)

    # THE SPLINE REQUIRES INCREASING ABSCISSAE, and a wavelength solution
    # is not always increasing: at the edge of an order, some exposures have
    # pixels whose wavelength GOES BACKWARD before moving forward again, and
    # scipy stops on "x must be increasing". Sorting them puts these pixels
    # back at the wavelength their solution gives them, which is the only
    # honest reading of what it says; order_quality has already removed them
    # from the mask, so they weigh on nothing, but the spline still needs an
    # ordered abscissa to exist. A sort has no effect when the ordering is
    # already increasing, which is the case everywhere else.
    if not np.all(np.diff(wave_shifted) > 0):
        order = np.argsort(wave_shifted, kind="stable")
        wave_shifted, values, good = (wave_shifted[order], values[order],
                                      good[order])
        # two pixels at the SAME wavelength remain forbidden: only the first
        # of each group is kept, the other one measures nothing new
        keep = np.ones(wave_shifted.size, dtype=bool)
        keep[1:] = np.diff(wave_shifted) > 0
        wave_shifted, values, good = (wave_shifted[keep], values[keep],
                                      good[keep])

    spline = InterpolatedUnivariateSpline(
        wave_shifted, values, k=spline_order, ext=1
    )
    out = spline(grid)

    mask_spline = InterpolatedUnivariateSpline(
        wave_shifted, np.asarray(good, dtype=np.float64), k=1, ext=1
    )
    out_good = mask_spline(grid) >= mask_threshold

    # anything outside the input range must not be trusted
    inside = (grid >= wave_shifted[0]) & (grid <= wave_shifted[-1])
    out_good &= inside
    out_good &= np.isfinite(out)

    return np.where(out_good, out, 0.0), out_good


def telluric_ramp(transmission, t_zero=0.5, t_one=1.0):
    """Linear weight ramp on telluric transmission.

    Full weight where the atmosphere was transparent, zero weight where the
    correction had to divide by less than t_zero of the light, and a straight
    line between. Deep telluric cores are corrected by dividing by a small
    number, which amplifies both the photon noise and any error in the
    transmission model, so the corrected flux there is not to be trusted even
    though it looks like a normal spectrum.

        w = clip((T - t_zero) / (t_one - t_zero), 0, 1)
    """
    transmission = np.asarray(transmission, dtype=np.float64)
    ramp = (transmission - float(t_zero)) / (float(t_one) - float(t_zero))
    ramp = np.clip(ramp, 0.0, 1.0)
    return np.where(np.isfinite(transmission), ramp, 0.0)


def photon_sigma(flux, s1d_weight, snr_band, snr_pixel_scale=1.0):
    """Per-pixel sigma of the log flux, from a photon-noise model.

    An s1d whose eflux column carries no usable error (all zeros) leaves the
    noise to be reconstructed. The number of detected photons in an s1d bin
    scales as flux * weight (the weight column is the accumulated blaze
    contribution), so
        sigma_ln = sigma_flux / flux ∝ 1 / sqrt(flux * weight).
    The proportionality constant is fixed per spectrum by forcing the median
    sigma_ln over the band to equal 1 / (snr_band * snr_pixel_scale), using the
    per-order extraction SNR from the header. This gets the pixel-to-pixel and
    the spectrum-to-spectrum weighting right; only the absolute chi2 scale is
    arbitrary, and PCA does not care about that.
    """
    counts = np.asarray(flux, dtype=np.float64) * np.asarray(s1d_weight, dtype=np.float64)
    counts = np.where(np.isfinite(counts) & (counts > 0), counts, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        shape = 1.0 / np.sqrt(counts)
    reference = np.nanmedian(shape)
    if not np.isfinite(reference) or reference <= 0:
        return np.full(counts.shape, np.nan)
    if not np.isfinite(snr_band) or snr_band <= 0:
        target = 1.0
    else:
        target = 1.0 / (snr_band * snr_pixel_scale)
    return shape * (target / reference)


def running_abs_diff(values, good, box):
    """Boxcar mean of |y[i] - y[i-1]|, over `box` pixels, ignoring gaps.

    A cumulative sum rather than a convolution, so the cost does not grow with
    the box: 540000 columns and a box of a few hundred is otherwise the slowest
    thing in the build.
    """
    y = np.where(good, values, np.nan)
    d = np.abs(np.diff(y))
    ok = np.isfinite(d)
    d = np.where(ok, d, 0.0)

    def boxsum(x):
        c = np.concatenate(([0.0], np.cumsum(x)))
        half = box // 2
        lo = np.clip(np.arange(x.size) - half, 0, x.size)
        hi = np.clip(np.arange(x.size) + half + 1, 0, x.size)
        return c[hi] - c[lo]

    total = boxsum(d)
    count = boxsum(ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0.25 * box, total / np.maximum(count, 1), np.nan)
    # d is defined between samples; put it back on the sample grid
    out = np.full(values.shape, np.nan)
    out[1:] = mean
    out[0] = mean[0] if mean.size else np.nan
    return out


def empirical_sigma(values, good, box=201):
    """Per-pixel noise estimated from the sample-to-sample scatter.

    For white noise of standard deviation s, the difference of two adjacent
    samples has standard deviation s*sqrt(2) and mean absolute value
    s*sqrt(2)*sqrt(2/pi) = 2s/sqrt(pi). So the boxcar mean of |dy| divided by
    2/sqrt(pi) = 1.1284 estimates s, and it does so WITHOUT trusting the
    photon-noise model.

    This exists because that model can be badly wrong in exactly the place it
    matters. Where the flux is tiny, at the edge of a domain, the formal sigma
    of ln f can be the smallest in the whole spectrum, because the photon
    error on a near-zero flux in the log is tiny, while the actual dispersion
    there is the largest. Those columns then receive many times more weight
    than they deserve, and a fit can put most of its star-block power into a
    few per cent of the columns, with components living on a handful of
    effective pixels.

    Measured on the high-passed values, where a real spectral line is resolved
    over many samples of the grid and contributes little to the
    sample-to-sample difference, while noise contributes all of it.
    """
    s = running_abs_diff(values, good, box) / (2.0 / np.sqrt(np.pi))
    return np.where(np.isfinite(s) & (s > 0), s, np.nan)


def highpass_against_template(values, good, template, window, polyorder,
                              already_log=False):
    """The high-pass taken on the RESIDUAL to the template, not on the
    spectrum.

    THE PROBLEM IT SOLVES. A Savitzky-Golay run over a whole spectrum is
    pulled down by every deep line: the smooth is not the continuum, it is
    the continuum minus a fraction of the line. As long as the line is the
    same from one exposure to the next, this bias is constant and the PCA
    does not see it. But a telluric that sweeps across the core of a
    stellar line with the BERV changes this bias FROM ONE EXPOSURE TO THE
    NEXT, and leaves in the star frame a residual that follows the BERV
    without being stellar.

    Over a BERV range of a few tens of km/s, a telluric next to a stellar
    line travels several line widths, and its depth at the line core can go
    from a few per cent to nearly saturated. The local smooth therefore
    changes intermittently, and not linearly in BERV, which makes it
    invisible to a simple correlation over the campaign.

    THE METHOD. The spectra being already in the star frame and on a common
    grid:

        residual  = ln(flux) - template    (the template is the same for
                                            all exposures, so it cannot
                                            bias one exposure more than
                                            another)
        residual -= savgol(residual)       (the high-pass works on a FLAT
                                            residual, with no deep line to
                                            pull the smooth)
        output    = residual + (template - savgol(template))

    The last term gives the spectrum back its high-passed stellar
    structure, taken once and for all on the template and therefore
    identical for all exposures. No line is masked, and nothing enters or
    leaves the mask depending on the BERV.

    `template` is in ln(flux), on the same grid as `values`.
    """
    values = np.asarray(values, dtype=float)
    template = np.asarray(template, dtype=float)
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, (values.size - 1) | 1)

    positive = np.asarray(good, dtype=bool) & np.isfinite(values)
    if already_log:
        # the per-order path already gives ln(flux): taking its logarithm
        # again would square it
        log_flux = np.where(positive, values, np.nan)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_flux = np.where(positive & (values > 0), np.log(
                np.where(positive & (values > 0), values, 1.0)), np.nan)
    usable = positive & np.isfinite(log_flux) & np.isfinite(template)
    if usable.sum() < window:
        return np.zeros_like(values), np.zeros(values.shape, dtype=bool)

    # THE TWO SAVGOLS DO NOT SEE THE SAME SUPPORT, and that is the whole
    # point.
    #
    # On an IDENTICAL support the formula would simplify, the savgol being
    # linear: (s - t) - sg(s - t) + t - sg(t) = s - sg(s), i.e. the
    # ordinary high-pass. The two then agree to rounding (2e-16).
    #
    # BUT THE SUPPORTS DIFFER. The exposure has masked pixels where a
    # saturated telluric eats the core of one of its lines; the median
    # template has none, since the telluric moves with the BERV and the
    # median erases it. The residual is therefore filtered on a support
    # with holes and the template on a full support, the two filters no
    # longer cancel, and the stellar line is returned from the FULL template
    # instead of being rebuilt from an exposure with holes. On a synthetic
    # line of 1.2 in ln with ten pixels masked at its core, the two formulas
    # differ by 0.2365, against 2e-16 without a mask.
    #
    # WHAT MATTERS IS THAT THE LOST PIXELS MOVE. A fixed mask would only
    # make a constant bias, which the PCA would absorb; it is the MOVEMENT
    # of the hole with the BERV that injects a signal. On a SYNTHETIC
    # spectrum kept identical, of which only the ten-pixel mask moves around
    # the core of a line of 1.2 in ln, one pixel of the neighbouring
    # continuum reads:
    #
    #     mask shift           classic      vs template
    #              -30 px       0.078476       0.078364
    #              -10 px       0.076784       0.077253
    #                0 px       0.071938       0.077854   <- the dip
    #              +10 px       0.076951       0.078194
    #              +30 px       0.077911       0.077810
    #
    #     dispersion            0.002265       0.000372   (6.1 times less)
    #
    # The ordinary high-pass breaks off when the hole falls on the core,
    # that is for a handful of BERVs only: the bias is intermittent, hence
    # invisible to a linear correlation with the BERV.
    residual, _ = _fill_gaps(np.where(usable, log_flux - template, np.nan))
    residual = residual - savgol_filter(residual, window, polyorder)

    # the template is filtered on ITS support, which is full
    filled_template, _ = _fill_gaps(np.where(np.isfinite(template), template,
                                             np.nan))
    star = filled_template - savgol_filter(filled_template, window, polyorder)

    out = residual + star
    usable &= np.isfinite(out)
    return np.where(usable, out, 0.0), usable


def median_template(rows, weights=None):
    """The template: the median of the already registered exposures, in
    ln(flux).

    Each order is normalised by its median before going in, so that a
    bright exposure does not weigh more than a faint one.
    """
    stack = np.asarray(rows, dtype=float)
    if stack.ndim != 2 or stack.shape[0] == 0:
        return None
    with np.errstate(invalid="ignore"):
        out = np.nanmedian(stack, axis=0)
    return out
