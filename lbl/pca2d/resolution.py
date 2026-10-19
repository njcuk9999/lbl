"""The star's spectra smoothed to the instrument's resolution, as LBL smooths its templates.

A star seen through a spectrograph carries nothing finer than one resolution
element, c/R: whatever a star-frame vector holds at higher frequency is noise
or the detector, and neither belongs to the star. LBL fits its velocities to a
Savitzky-Golay version of its template (calculate_savgol_template, on by
USE_SAVGOL_TEMPLATE): a cubic fitted under Gaussian weights whose FWHM is one
resolution element, over three of them. The fit's star-side vectors, the
per-parity star spectra of --mean star and the star components P, get exactly
that filter here. The observer components do not: they may carry pixel-level
detector structure, which is real at the pixel.
"""

import numpy as np

C_KMS = 299792.458


def fwhm_samples(resolution, dv, fraction=1.0):
    """`fraction` of a resolution element, c/R, in samples of dv km/s: LBL's
    rule, rounded and made odd, never under 3. One element is 9 samples for
    SPIRou (R = 70 000) and 7 for NIRPS (80 000) at 0.5 km/s.
    """
    n = int(np.round(float(fraction) * C_KMS / float(dv) / float(resolution)))
    n = max(n, 3)
    return n if n % 2 else n + 1


def noise_factor(fwhm, polyorder=3):
    """sum h^2 of the filter's taps: what smooth() does to the variance of white
    noise, away from any gap. A vector smoothed by it has its noise variance
    multiplied by this."""
    n = int(12 * fwhm) + 1
    impulse = np.full(n, 1e-9)
    impulse[n // 2] += 1.0
    taps = smooth(impulse, fwhm, polyorder) - smooth(np.full(n, 1e-9), fwhm, polyorder)
    return float(np.sum(taps ** 2))


def smooth(vec, fwhm, polyorder=3, deriv=0):
    """LBL's gaussian_weighted_savgol on one vector, value or derivative.

    `deriv=1` returns d(vec)/d(sample), which on a log-uniform grid is the
    velocity sensitivity: the local polynomial is fitted exactly as for the
    value, and its linear coefficient is kept instead of its constant one. That
    is LBL's flux_savgol_d1, and it is what a derivative of a spectrum must be
    here: np.gradient of noisy data is significantly worse.

    At every sample, a polynomial of `polyorder` fitted by weighted least
    squares over 3 x fwhm samples, the weights a Gaussian of that FWHM, and its
    value at the centre kept. Samples without support (zero or not finite: the
    vectors here are zero where the fit had no data) take no part in any fit
    and stay zero, so the support never grows into a gap. A sample whose window
    is more than half empty is dropped, which is LBL's rule.

    Written as moments rather than as LBL's loop over samples: the weighted
    normal equations of every window are correlations of the support and of the
    data with the kernel times powers of the offset, so the whole vector costs a
    handful of convolutions and one batch of small solves. Where nothing is
    missing that is the fixed Savitzky-Golay kernel; beside a gap it is LBL's
    refit with the missing samples' weights set to zero.
    """
    y = np.asarray(vec, dtype=float)
    ok = np.isfinite(y) & (y != 0.0)
    total = int(3 * fwhm)
    total += 1 - total % 2
    half = total // 2
    sigma = float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    k = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    u = k / half                      # offsets scaled to [-1, 1], for conditioning

    def corr(x, h):                   # sum_k h_k x_{i+k}
        return np.convolve(x, h[::-1], mode="same")

    m = ok.astype(float)
    my = np.where(ok, y, 0.0)
    n = polyorder + 1
    moment = [corr(m, g * u ** s) for s in range(2 * polyorder + 1)]
    count = corr(m, np.ones(total))
    live = ok & (count > 0.5 * total)
    idx = np.flatnonzero(live)
    out = np.zeros_like(y)
    if not idx.size:
        return out
    A = np.empty((idx.size, n, n))
    b = np.empty((idx.size, n, 1))
    for p in range(n):
        b[:, p, 0] = corr(my, g * u ** p)[idx]
        for q in range(n):
            A[:, p, q] = moment[p + q][idx]
    # u is k/half, so the polynomial's coefficient p is d^p/du^p / p!, and the
    # derivative per SAMPLE needs the chain rule back through that scaling
    solved = np.linalg.solve(A, b)[:, :, 0]
    if deriv >= solved.shape[1]:
        return out
    scale = 1.0
    for order in range(1, deriv + 1):
        scale *= order / float(half)
    out[idx] = solved[:, deriv] * scale
    return out


def smooth_rows(basis, fwhm, polyorder=3):
    """Each row smoothed, then made orthonormal again, as mstep leaves them."""
    out = np.array([smooth(row, fwhm, polyorder) for row in np.atleast_2d(basis)])
    for i in range(out.shape[0]):
        for j in range(i):
            out[i] -= np.dot(out[i], out[j]) * out[j]
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out
