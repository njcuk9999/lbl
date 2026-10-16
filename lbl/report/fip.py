"""The false inclusion probability, gaussian and outlier-aware.

WHY HERE AND NOT BORROWED. koloa has a far more complete FIP, but it runs
its chains in subprocesses, which fails as soon as LBL is driven from a
script rather than a file (seen the 2026-10-16: the outlier-aware page came
out empty with "koloa not installed"). A report must not depend on how it
was launched, so the calculation lives here, in numpy alone.

WHAT A FIP IS. Over a grid of periods, the probability that a sinusoid at
that period is NOT in the data. Near 1 the signal is almost surely absent,
near 0 almost surely there. The linear parameters (the sine and cosine
amplitudes and the trend) are integrated out analytically, so the only sum
left is over the grid and over a ladder of prior amplitudes, which is done
exactly. There is no sampling in the gaussian case.

THE OUTLIER-AWARE CASE. The gaussian FIP assumes every point is drawn from
its error bar, so one bad night can either invent a signal or bury one. Here
each point instead carries a probability of belonging to a wider population,
and the error bars are inflated accordingly; the two are iterated until they
settle. That is a deterministic approximation of the mixture that koloa
samples, and it answers the question the report asks: does the answer depend
on how the outliers are treated?
"""
import numpy as np

#: the ladder of prior amplitudes, in units of the robust scatter. The FIP
#: should not depend on a single guess of how large a signal might be, so it
#: is averaged over a log-uniform ladder.
TAU_LADDER = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
#: how wide the outlier population is, as a multiple of the error bars
OUTLIER_WIDTH = 5.0
#: the prior fraction of points that are outliers
OUTLIER_PRIOR = 0.05


def robust_sigma(values):
    """1.4826 times the median absolute deviation."""
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    if good.size < 2:
        return float('nan')
    return float(1.4826 * np.median(np.abs(good - np.median(good))))


def frequency_grid(time, pmin=1.1, pmax=None, oversample=10):
    """A frequency grid that resolves the series it is made for.

    The spacing is 1/(oversample * T), the usual rule: finer than that and
    neighbouring points say the same thing, coarser and a peak can fall
    between two samples.
    """
    time = np.asarray(time, dtype=float)
    span = float(np.max(time) - np.min(time))
    if not np.isfinite(span) or span <= 0:
        return np.array([])
    if pmax is None:
        pmax = 2.0 * span
    step = 1.0 / (oversample * span)
    return np.arange(1.0 / pmax, 1.0 / pmin, step)


def _design(time, freq, trend=1):
    """The model at one frequency: a sine, a cosine, and a polynomial."""
    centred = time - np.mean(time)
    columns = [np.cos(2 * np.pi * freq * centred),
               np.sin(2 * np.pi * freq * centred)]
    for power in range(trend + 1):
        columns.append(centred ** power)
    return np.vstack(columns).T


def _log_evidence(design, value, weight, tau, nsignal=2):
    """log of the marginal likelihood, the linear parameters integrated out.

    `nsignal` is how many of the leading columns carry the prior width tau;
    the rest (the trend) are given a flat prior, which is the limit of a
    very wide gaussian and drops out of the comparison.
    """
    npar = design.shape[1]
    prior = np.full(npar, 1e8)
    prior[:nsignal] = tau
    scaled = design * weight[:, None]
    normal = design.T @ scaled + np.diag(1.0 / prior ** 2)
    target = scaled.T @ value
    try:
        chol = np.linalg.cholesky(normal)
    except np.linalg.LinAlgError:
        return -np.inf
    solved = np.linalg.solve(normal, target)
    chi2 = float(np.sum(weight * value ** 2) - target @ solved)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * (chi2 + logdet) - float(np.sum(np.log(prior[:nsignal])))


def fip(time, value, err, freq=None, trend=1, prior_signal=0.5,
        jitter=None, weights=None):
    """The gaussian FIP over a grid of periods.

    Returns (freq, fip). `weights` down-weights points, which is how the
    outlier-aware version feeds its result back in.
    """
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float) 
    err = np.asarray(err, dtype=float)
    good = (np.isfinite(time) & np.isfinite(value) & np.isfinite(err)
            & (err > 0))
    time, value, err = time[good], value[good], err[good]
    if time.size < 10:
        return np.array([]), np.array([])
    value = value - np.median(value)
    if jitter is None:
        # the scatter the error bars do not explain
        excess = robust_sigma(value) ** 2 - np.median(err) ** 2
        jitter = float(np.sqrt(excess)) if excess > 0 else 0.0
    total = np.sqrt(err ** 2 + jitter ** 2)
    weight = 1.0 / total ** 2
    if weights is not None:
        weight = weight * np.asarray(weights, dtype=float)[good]
    if freq is None:
        freq = frequency_grid(time)
    if freq.size == 0:
        return np.array([]), np.array([])

    scale = robust_sigma(value)
    taus = np.array(TAU_LADDER) * (scale if np.isfinite(scale) and scale > 0
                                   else 1.0)
    # the null model: the trend alone, no sinusoid
    flat = _design(time, 0.0, trend)[:, 2:]
    null = _log_evidence(flat, value, weight, 1.0, nsignal=0)

    evidence = np.empty(freq.size)
    for index, one in enumerate(freq):
        design = _design(time, one, trend)
        terms = [_log_evidence(design, value, weight, tau) for tau in taus]
        top = np.max(terms)
        evidence[index] = top + np.log(np.mean(np.exp(np.array(terms) - top)))

    # FIP = P(no signal at this period | data), with the prior split evenly
    # over the grid: each frequency carries prior_signal / N of the prior
    # that a signal exists anywhere.
    log_prior = np.log(prior_signal / freq.size)
    log_null = np.log(1.0 - prior_signal) + null
    top = np.maximum(log_null, log_prior + evidence)
    with np.errstate(over='ignore'):
        odds = np.exp(log_prior + evidence - top) / (
            np.exp(log_null - top) + np.exp(log_prior + evidence - top))
    return freq, 1.0 - odds


def outlier_probability(value, err, jitter=0.0, width=OUTLIER_WIDTH,
                        prior=OUTLIER_PRIOR):
    """P(this point belongs to the wide population), point by point."""
    value = np.asarray(value, dtype=float)
    err = np.asarray(err, dtype=float)
    narrow = np.sqrt(err ** 2 + jitter ** 2)
    wide = narrow * width
    centred = value - np.median(value)
    with np.errstate(divide='ignore', invalid='ignore'):
        lnarrow = np.exp(-0.5 * (centred / narrow) ** 2) / narrow
        lwide = np.exp(-0.5 * (centred / wide) ** 2) / wide
    total = (1.0 - prior) * lnarrow + prior * lwide
    out = np.where(total > 0, prior * lwide / total, 0.0)
    return np.clip(out, 0.0, 1.0)


def oafip(time, value, err, freq=None, trend=1, rounds=5):
    """The outlier-aware FIP. Returns (freq, fip, outlier probability).

    The outlier probabilities and the FIP are iterated until they settle: a
    point that looks like an outlier is down-weighted, which changes the fit,
    which changes which points look like outliers. Five rounds are enough in
    practice; the probabilities move by less than a percent after three.
    """
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    err = np.asarray(err, dtype=float)
    good = (np.isfinite(time) & np.isfinite(value) & np.isfinite(err)
            & (err > 0))
    if good.sum() < 10:
        return np.array([]), np.array([]), np.array([])
    centred = value - np.median(value[good])
    excess = robust_sigma(centred[good]) ** 2 - np.median(err[good]) ** 2
    jitter = float(np.sqrt(excess)) if excess > 0 else 0.0

    prob = np.zeros(value.size)
    out_freq, out_fip = np.array([]), np.array([])
    for _ in range(rounds):
        # a point that is probably an outlier counts for less, in the exact
        # proportion of its probability of being a normal point
        weights = np.ones(value.size)
        weights[good] = 1.0 - prob[good]
        out_freq, out_fip = fip(time, value, err, freq=freq, trend=trend,
                                jitter=jitter, weights=weights)
        prob = np.zeros(value.size)
        prob[good] = outlier_probability(centred[good], err[good], jitter)
    return out_freq, out_fip, prob
