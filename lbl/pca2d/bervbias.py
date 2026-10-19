"""The velocity bias that follows BERV, fitted by MCMC against V_tot.

A telluric line blended with a stellar line pulls the velocity measured on it
by an amount that depends on how far apart the two are. That separation is
the star's velocity in the telluric frame, the systemic velocity less the
BERV:

    V_tot = vrad / 1000 - BERV        (km/s; vrad from LBL in m/s)

and not the BERV alone, which is V_tot only for a star at rest (asked for on
2026-09-16: on SMETHELLS_20, at +2.6 km/s, it hardly matters; a star at
+-30 km/s may never bring its lines onto the tellurics at all). Superposed,
V_tot = 0, the blend is symmetric and nothing moves; far apart, nothing is
blended; in between, the pull is the derivative of a Gaussian:

    v(V_tot) = c + amp * V_tot * exp(-0.5 * (V_tot / sigma)**2)

odd in V_tot, zero at 0, extreme at V_tot = +-sigma, where it is worth

    peak = amp * sigma * exp(-1/2)

The width is fitted and quoted as the FWHM of that Gaussian, 2 sqrt(2 ln 2)
sigma, the number a blend of two lines has a known value for.

(written `amp*np.exp(-0.5*BERV/sigma)*BERV` when it was asked for, on
2026-09-16: the square is what makes the bias die away on both sides; taken
as written it grows without bound on one side). A straight line against
BERV, which the report drew before, has no such shape and no meaning here.
The functions below take V_tot, from `total_velocity`, wherever they say
`berv`: the shape is the same whatever the axis is called.

`amp` is signed, and the fit explores both signs: its prior is symmetric and
half the walkers start on each side. `amp` and `fwhm` are what the fit is
for; `c`, the velocities' arbitrary
zero, and `jitter`, the scatter LBL's error bars do not account for, are
fitted beside them and marginalised. Without the jitter the posterior would be
as narrow as LBL's error bars are optimistic: 13 m/s against 44 m/s of scatter
on SMETHELLS_20. The sampler is Goodman and Weare's affine-invariant stretch
move, emcee's algorithm, in a few lines of numpy: emcee is not in the
pipeline's environment, and four parameters do not need it.

    amp       uniform within +-1e4 (m/s)/(km/s), signed, since a bias pulls
              either way. A flat prior, asked for on 2026-09-16 in place of
              the modified Jeffreys 1/(|amp| + a0) it had until then: the
              Jeffreys one favours small amplitudes, and so the narrow widths
              whose bias would need a large one. On SMETHELLS_20 the delivered
              bias barely moves (-68.7 to -70.9 m/s, 7.0 to 7.2 sigma) and the
              corrected upper limit rises from 24.9 to 33.6 m/s (32.3 to 33.6
              over three seeds, 33.0 on a chain of 12 000 steps): what the
              data say, without the prior's pull toward zero
    fwhm      Gaussian, 5 +- 1.5 km/s, within 1 to 10 km/s (asked for on
              2026-09-17: the width of the Gaussian whose derivative the
              bias is, the blend of a stellar and a telluric line at a
              resolution of 70 000 to 80 000, is known to that precision).
              It replaced a log-uniform sigma over 1 to 60 km/s, which let a
              bias that is not there hide in the narrowest widths with a
              large amplitude, and let a slow drift of the velocities pass
              for a bias tens of km/s wide (GL725B: sigma 33 km/s)
    c         uniform, m/s: the DC level, the velocity far from the tellurics,
              which is not the median when the exposures sit more on one side
              of V_tot = 0 than the other
    slope     uniform within +-1e3 m/s per year: the DC level's drift over
              the campaign, c + slope * (t - t_mid), when the times are given
              (asked for on 2026-09-17). V_tot follows the season, and on a
              star whose V_tot spans a few km/s a drift of the velocities
              passes for a bias: GL725B corrected, Delta BIC +8.4 with the
              DC level held constant, -5.2 with it drifting
    jitter    log-uniform over 1e-3 to 1e4 m/s
"""

from __future__ import annotations

import numpy as np

#: the FWHM's prior: Gaussian, (mean, standard deviation), within bounds
FWHM_PRIOR = (5.0, 1.5)
FWHM_MIN, FWHM_MAX = 1.0, 10.0
#: sigma of a Gaussian per unit of its FWHM, 1 / (2 sqrt(2 ln 2))
SIGMA_PER_FWHM = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
JITTER_MIN, JITTER_MAX = 1e-3, 1e4
AMP_MAX = 1e4
#: bound on the DC level's drift, m/s per year
SLOPE_MAX = 1e3
NAMES = ("amp", "fwhm", "c", "jitter", "slope")
#: exposures with |V_tot| < FWHM_MAX a fit needs: the bias dies away beyond
#: that, and a star whose lines never come near the tellurics has no bias to
#: measure, only an amplitude the prior bounds
MIN_NEAR = 12
#: standard deviations from zero before a bias is called a bias. Below it the
#: posterior of the width is the prior's, and a peak and a width quoted from
#: it would describe the prior; an upper limit on the peak is what is said
DETECTED = 3.0
UNITS = ("(m/s)/(km/s)", "km/s", "m/s", "m/s")
C_KMS = 299792.458
#: |V_tot| below which the star's lines sit on the telluric lines, within
#: about one resolution element: the exposures the report lists by date
CLOSE_KMS = 4.0


def total_velocity(vrad, berv):
    """V_tot in km/s, the star's velocity in the telluric frame.

    vrad / 1000 - BERV, `vrad` in m/s (LBL's) and `berv` in km/s, composed
    relativistically: the wavelength ratios multiply, so the rapidities
    subtract. With +30 and -30 km/s that differs from the plain difference
    by 6e-7 km/s, far below anything the bias's width can see.
    """
    vrad = np.asarray(vrad, float) / 1000.0
    berv = np.asarray(berv, float)
    return C_KMS * np.tanh(np.arctanh(vrad / C_KMS) - np.arctanh(berv / C_KMS))


def sigma_of(fwhm):
    """The Gaussian's sigma for its FWHM."""
    return np.asarray(fwhm, float) * SIGMA_PER_FWHM


def shape(berv, amp, sigma):
    """The bias alone, without the offset: amp * B * exp(-B^2 / 2 sigma^2)."""
    berv = np.asarray(berv, float)
    return amp * berv * np.exp(-0.5 * (berv / sigma) ** 2)


def peak(amp, sigma):
    """The bias at BERV = sigma, its largest."""
    return amp * sigma * np.exp(-0.5)


def _columns(theta):
    """(amp, fwhm, c, ln jitter, slope) of rows that may lack the slope."""
    theta = np.atleast_2d(theta)
    slope = theta[:, 4] if theta.shape[1] > 4 else np.zeros(theta.shape[0])
    return theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], slope


def dc_level(c, slope, tau):
    """The velocity with no bias: c, drifting by `slope` m/s per year over
    `tau`, the years from the campaign's middle (None: no drift)."""
    if tau is None:
        return np.asarray(c, float)[..., None]
    return np.asarray(c, float)[..., None] \
        + np.asarray(slope, float)[..., None] * np.asarray(tau, float)


def log_likelihood(theta, berv, v, e, tau=None):
    """ln L of (amp, fwhm, c, ln jitter[, slope]), one row per walker, -inf
    outside the priors' bounds (the 2 pi left out). `tau`, years from the
    campaign's middle, lets the DC level drift; without it there is no
    slope."""
    amp, fwhm, c, ljit, slope = _columns(theta)
    inside = ((np.abs(amp) < AMP_MAX)
              & (fwhm > FWHM_MIN) & (fwhm < FWHM_MAX)
              & (np.abs(slope) < SLOPE_MAX)
              & (ljit > np.log(JITTER_MIN)) & (ljit < np.log(JITTER_MAX)))
    out = np.full(amp.size, -np.inf)
    if not inside.any():
        return out
    sigma = sigma_of(fwhm[inside])[:, None]
    model = dc_level(c[inside], slope[inside], tau) \
        + amp[inside][:, None] * berv[None, :] \
        * np.exp(-0.5 * (berv[None, :] / sigma) ** 2)
    var = e[None, :] ** 2 + np.exp(2 * ljit[inside])[:, None]
    out[inside] = -0.5 * np.sum((v[None, :] - model) ** 2 / var + np.log(var),
                                axis=1)
    return out


def log_prior(theta):
    """ln prior, up to a constant: Gaussian in the FWHM, flat in amp and c,
    log-uniform in the jitter (flat in its log, which is what is sampled)."""
    theta = np.atleast_2d(theta)
    mean, width = FWHM_PRIOR
    return -0.5 * ((theta[:, 1] - mean) / width) ** 2


def log_probability(theta, berv, v, e, tau=None):
    """ln posterior of (amp, fwhm, c, ln jitter[, slope]), one row per
    walker."""
    return log_likelihood(theta, berv, v, e, tau) + log_prior(theta)


#: Kass and Raftery (1995, JASA 90, 773) on a Bayes factor, read on the BIC's
#: scale, 2 ln B: below 2 not worth more than a mention, 2 to 6 positive, 6
#: to 10 strong, above 10 very strong. Negative: the data prefer no bias
BIC_WORDS = ((10.0, "very strong"), (6.0, "strong"), (2.0, "positive"),
             (0.0, "not worth a mention"))


def log_likelihood_null(v, e, tau=None):
    """ln L of (c, ln jitter[, slope]), no bias, one row per walker, as
    log_likelihood counts it (the 2 pi left out of both)."""
    def lnl(theta):
        theta = np.atleast_2d(theta)
        c, ljit = theta[:, 0], theta[:, 1]
        slope = theta[:, 2] if theta.shape[1] > 2 else np.zeros(c.size)
        var = e[None, :] ** 2 + np.exp(2 * ljit)[:, None]
        out = -0.5 * np.sum((v[None, :] - dc_level(c, slope, tau)) ** 2 / var
                            + np.log(var), axis=1)
        return np.where((ljit > np.log(JITTER_MIN))
                        & (ljit < np.log(JITTER_MAX))
                        & (np.abs(slope) < SLOPE_MAX), out, -np.inf)
    return lnl


def maximum(lnl, starts, bounds):
    """The highest ln L from several starting points, and where it is."""
    from scipy.optimize import minimize

    best = (-np.inf, None)
    for start in starts:
        start = np.clip(np.asarray(start, float), [b[0] + 1e-9 for b in bounds],
                        [b[1] - 1e-9 for b in bounds])
        found = minimize(lambda x: -float(lnl(x)[0]), start,
                         method="Nelder-Mead", bounds=bounds,
                         options={"xatol": 1e-6, "fatol": 1e-6,
                                  "maxiter": 4000, "maxfev": 8000})
        for x in (found.x, start):
            value = float(lnl(x)[0])
            if value > best[0]:
                best = (value, np.array(x, float))
    return best


def delta_bic(berv, v, e, chain_samples=None, chain_lp=None, tau=None):
    """(Delta BIC, ln L with the bias, ln L without), Delta BIC being
    BIC(no bias) - BIC(bias): positive when the data prefer the bias.

    BIC = k ln n - 2 ln L_max, with k = 4 (amp, fwhm, c, jitter) against 2
    (c, jitter), one more on each side with `tau`, the DC level's drift. The
    maxima are of the likelihood alone, the FWHM within its
    prior's bounds, and the FWHM is counted as free although its prior pins
    it: the conservative count. Each maximum is found from several starts:
    the best of the posterior's samples when there are some, and a grid of
    widths. The bias's width means nothing when its amplitude is zero, so the
    two models are nested only at a boundary the BIC does not know about
    (Davies' problem): the number is a guide on the Kass and Raftery scale,
    not a p-value.
    """
    berv, v, e = (np.asarray(x, float) for x in (berv, v, e))
    n = v.size
    excess = max(float(np.var(v - np.median(v)) - np.median(e) ** 2), 1.0)
    jit_bounds = (np.log(JITTER_MIN), np.log(JITTER_MAX))
    drift = tau is not None
    reach = float(np.max(np.abs(v - np.median(v)))) + 1.0
    c_bounds = (float(np.min(v)) - reach, float(np.max(v)) + reach)
    extra = [(-SLOPE_MAX, SLOPE_MAX)] if drift else []
    line = starting_line(v, e, tau)
    null, _ = maximum(log_likelihood_null(v, e, tau),
                      [[line[0], 0.5 * np.log(excess)] + line[1:],
                       [np.median(v), 0.5 * np.log(excess) + 1.0]
                       + [0.0] * drift,
                       [np.mean(v), 0.0] + line[1:]],
                      [c_bounds, jit_bounds] + extra)
    starts = []
    if chain_samples is not None and len(chain_samples):
        best = chain_samples[int(np.argmax(chain_lp))]
        starts.append(best)
    centre = starting_point(berv, v, e, tau)
    starts.append(centre)
    for fwhm in np.linspace(FWHM_MIN * 1.2, FWHM_MAX * 0.95, 6):
        starts.append([centre[0], fwhm] + list(centre[2:]))
    bias, _ = maximum(
        lambda theta: log_likelihood(theta, berv, v, e, tau), starts,
        [(-AMP_MAX, AMP_MAX), (FWHM_MIN, FWHM_MAX), c_bounds, jit_bounds]
        + extra)
    value = 2.0 * (bias - null) - (4 - 2) * np.log(n)
    return float(value), float(bias), float(null)


def bic_words(value, what="bias"):
    """What a Delta BIC says, on Kass and Raftery's scale; `what` is the
    model it is the evidence for, named when the data prefer none."""
    if not np.isfinite(value):
        return "n/a"
    if value < 0:
        return "no %s preferred" % what
    return next(word for floor, word in BIC_WORDS if value >= floor)


def stretch(log_prob, start, steps, rng, a=2.0):
    """Goodman and Weare's stretch move. Returns (chain, acceptance).

    `start` is (walkers, ndim) with an even number of walkers; `log_prob`
    takes a (n, ndim) array. The walkers are updated in two halves, each
    against the other, which is what keeps the move valid when it is done
    for many walkers at once.
    """
    pos = np.array(start, float)
    walkers, ndim = pos.shape
    lp = log_prob(pos)
    chain = np.empty((steps, walkers, ndim))
    accepted = 0
    halves = (np.arange(walkers // 2), np.arange(walkers // 2, walkers))
    for step in range(steps):
        for moving, fixed in (halves, halves[::-1]):
            z = ((a - 1.0) * rng.random(moving.size) + 1.0) ** 2 / a
            partners = pos[fixed[rng.integers(0, fixed.size, moving.size)]]
            proposal = partners + z[:, None] * (pos[moving] - partners)
            lp_new = log_prob(proposal)
            keep = np.log(rng.random(moving.size)) < \
                (ndim - 1) * np.log(z) + lp_new - lp[moving]
            pos[moving[keep]] = proposal[keep]
            lp[moving[keep]] = lp_new[keep]
            accepted += int(keep.sum())
        chain[step] = pos
    return chain, accepted / float(steps * walkers)


def starting_line(v, e, tau=None):
    """[c] or [c, slope]: the DC level, weighted, and its drift."""
    excess = max(float(np.var(v - np.median(v)) - np.median(e) ** 2), 1.0)
    w = 1.0 / (e ** 2 + excess)
    if tau is None:
        return [float(np.sum(w * v) / np.sum(w))]
    A = np.column_stack([np.ones_like(v), tau])
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], v * np.sqrt(w),
                               rcond=None)
    return [float(coef[0]),
            float(np.clip(coef[1], -0.5 * SLOPE_MAX, 0.5 * SLOPE_MAX))]


def starting_point(berv, v, e, tau=None):
    """The best (amp, fwhm, c, ln jitter[, slope]) on a grid of FWHM.

    For each width the model is linear in (c, amp[, slope]), so it is solved
    rather than searched; the jitter is the scatter the error bars leave
    over.
    """
    excess = max(float(np.var(v - np.median(v)) - np.median(e) ** 2), 1.0)
    w = 1.0 / (e ** 2 + excess)
    best = None
    for fwhm in np.linspace(FWHM_MIN * 1.05, FWHM_MAX * 0.98, 40):
        columns = [np.ones_like(berv), shape(berv, 1.0, sigma_of(fwhm))]
        if tau is not None:
            columns.append(tau)
        A = np.column_stack(columns)
        coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], v * np.sqrt(w),
                                   rcond=None)
        chi2 = float(np.sum(w * (v - A @ coef) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, fwhm, coef)
    _, fwhm, coef = best
    # a width whose bias no point reaches solves for any amplitude at all;
    # the walkers must start inside the prior's bounds
    amp = float(np.clip(coef[1], -0.01 * AMP_MAX, 0.01 * AMP_MAX))
    out = [amp, fwhm, float(coef[0]), 0.5 * np.log(excess)]
    if tau is not None:
        out.append(float(np.clip(coef[2], -0.5 * SLOPE_MAX, 0.5 * SLOPE_MAX)))
    return np.array(out)


def fit(berv, v, e, t=None, walkers=32, steps=2500, burn=1000, seed=0):
    """Posterior of the BERV bias of one velocity series.

    `t`, the times in days, lets the DC level drift linearly over the
    campaign (`slope`, m/s per year, about `t_mid`, the middle of the span);
    without it the level is a constant.

    Returns a dictionary: `samples` (n, 5) in (amp, fwhm, c, jitter, slope),
    the slope 0 without `t`,
    `acceptance`, and for each of amp, fwhm, c, jitter and peak its median
    and 16th and 84th percentiles, plus the amp-FWHM correlation of the
    posterior. None when there are too few points to fit four parameters,
    or fewer than MIN_NEAR within FWHM_MAX of zero, where the bias is.
    """
    berv, v, e = (np.asarray(x, float) for x in (berv, v, e))
    ok = np.isfinite(berv) & np.isfinite(v) & np.isfinite(e) & (e > 0)
    if t is not None:
        t = np.asarray(t, float)
        ok &= np.isfinite(t)
    if ok.sum() < 12 or np.ptp(berv[ok]) < FWHM_MIN:
        return None
    if np.sum(np.abs(berv[ok]) < FWHM_MAX) < MIN_NEAR:
        return None
    berv, v, e = berv[ok], v[ok], e[ok]
    tau, t_mid = None, np.nan
    if t is not None and np.ptp(t[ok]) > 0:
        t = t[ok]
        t_mid = 0.5 * float(t.min() + t.max())
        tau = (t - t_mid) / 365.25
    ndim = 4 if tau is None else 5
    rng = np.random.default_rng(seed)
    centre = starting_point(berv, v, e, tau)
    scale = np.array([max(abs(centre[0]) * 0.1, 1e-2), 0.3,
                      max(np.std(v) * 0.01, 1e-2), 0.1]
                     + ([max(abs(centre[-1]) * 0.1, 0.1)] if tau is not None
                        else []))
    start = centre + scale * rng.normal(size=(walkers, ndim))
    # half the walkers start on the other sign: a bias pulls either way, and
    # an ensemble that starts on one side only explores the other if it
    # happens to wander there. On SMETHELLS_20 the mirrored half rejoins the
    # delivered bias's negative mode within 500 steps, and the corrected
    # posterior keeps both signs, P(amp > 0) = 0.15 either way
    start[walkers // 2:, 0] *= -1
    start[:, 1] = np.clip(start[:, 1], FWHM_MIN + 1e-3, FWHM_MAX - 1e-3)
    chain, acceptance = stretch(
        lambda theta: log_probability(theta, berv, v, e, tau), start,
        steps, rng)
    flat = chain[burn:].reshape(-1, ndim)
    # the best of the posterior's samples starts the maximum the BIC needs
    thin = flat[::max(1, flat.shape[0] // 4000)]
    dbic, lnl_bias, lnl_null = delta_bic(
        berv, v, e, thin, log_probability(thin, berv, v, e, tau), tau)
    samples = np.column_stack([flat[:, 0], flat[:, 1], flat[:, 2],
                               np.exp(flat[:, 3]),
                               flat[:, 4] if ndim == 5
                               else np.zeros(flat.shape[0])])
    out = {"samples": samples, "acceptance": acceptance, "n": int(ok.sum()),
           "delta_bic": dbic, "lnl_bias": lnl_bias, "lnl_null": lnl_null,
           "t_mid": t_mid, "drift": tau is not None}
    for i, name in enumerate(NAMES):
        out[name] = np.percentile(samples[:, i], [50, 16, 84])
    peaks = peak(samples[:, 0], sigma_of(samples[:, 1]))
    out["peak"] = np.percentile(peaks, [50, 16, 84])
    out["peak_sigma"] = float(np.std(peaks))
    # how many standard deviations the peak is from no bias at all
    out["significance"] = (float(abs(np.median(peaks)) / np.std(peaks))
                           if np.std(peaks) > 0 else 0.0)
    out["amp_fwhm_r"] = float(np.corrcoef(samples[:, 0],
                                          samples[:, 1])[0, 1])
    if ndim == 5:
        out["amp_slope_r"] = float(np.corrcoef(samples[:, 0],
                                               samples[:, 4])[0, 1])
        out["fwhm_slope_r"] = float(np.corrcoef(samples[:, 1],
                                                samples[:, 4])[0, 1])
    else:
        out["amp_slope_r"] = out["fwhm_slope_r"] = np.nan
    out["p_positive"] = float(np.mean(samples[:, 0] > 0))
    # what can be said when nothing is seen: the bias is below this
    out["upper"] = float(np.percentile(np.abs(peaks), 95))
    out["detected"] = out["significance"] >= DETECTED
    return out


def envelope(result, grid, draws=400, seed=1):
    """(16th, 50th, 84th) percentiles of the bias curve over `grid`."""
    rng = np.random.default_rng(seed)
    samples = result["samples"]
    pick = samples[rng.integers(0, samples.shape[0], draws)]
    curves = pick[:, 0][:, None] * grid[None, :] \
        * np.exp(-0.5 * (grid[None, :] / sigma_of(pick[:, 1])[:, None]) ** 2)
    return np.percentile(curves, [16, 50, 84], axis=0)


def summary(result):
    """'peak -70.6 +9.6/-10.0 m/s, FWHM 5.2 km/s (7.1 sigma, \u0394BIC +40.2)',
    or the limit."""
    if result is None:
        return "not fitted"
    bic = ""
    if np.isfinite(result.get("delta_bic", np.nan)):
        bic = ", \u0394BIC %+.1f" % result["delta_bic"]
    if not result["detected"]:
        return ("none detected (%.1f sigma%s): |peak| < %.1f m/s at 95%%"
                % (result["significance"], bic, result["upper"]))
    p, lo, hi = result["peak"]
    return ("peak %.1f +%.1f/-%.1f m/s, FWHM %.1f km/s (%.1f sigma%s)"
            % (p, hi - p, p - lo, result["fwhm"][0], result["significance"],
               bic))
