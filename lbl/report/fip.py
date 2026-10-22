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
and counts for less in proportion. That is a deterministic approximation of
the mixture that koloa samples, and it answers the question the report asks:
does the answer depend on how the outliers are treated?

SPEED. Every frequency of the grid (20 000 for six years of SPIRou) and every
prior width of the ladder is one small linear system. They are built and
solved together, a block of frequencies at a time, with batched Cholesky
factorisations: the same algebra as one at a time, in a few hundredths of a
second rather than seconds.
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


#: frequencies solved together: bounds the (block, points) arrays
BLOCK = 2048


def _evidence_grid(time, value, weight, freq, taus, trend=1):
    """log of the marginal likelihood at every frequency, averaged over the
    prior ladder `taus`: what _log_evidence gives one frequency and one tau
    at a time, for all of them together.

    The normal matrix of the model [cos, sin, trend] is assembled from
    weighted sums (one matrix product per block of frequencies), the priors
    added to its diagonal for each tau, and the whole stack factorised at
    once. A block whose stack is not positive definite falls back to the
    one-at-a-time path, which then answers -inf where it fails, as before.
    """
    centred = time - np.mean(time)
    poly = np.vstack([centred ** power for power in range(trend + 1)]).T
    n_poly = poly.shape[1]
    npar = 2 + n_poly
    wpoly = poly * weight[:, None]
    # the trend's own block, the same for every frequency
    pp = poly.T @ wpoly
    pv = wpoly.T @ value
    wv = weight * value
    chi0 = float(np.sum(wv * value))
    # the priors: tau on the sinusoid, flat (1e8) on the trend
    inv_prior = np.zeros((len(taus), npar))
    inv_prior[:, :2] = 1.0 / np.asarray(taus, dtype=float)[:, None] ** 2
    inv_prior[:, 2:] = 1.0 / 1e8 ** 2
    log_prior = 2.0 * np.log(np.asarray(taus, dtype=float))
    out = np.empty(freq.size)
    for start in range(0, freq.size, BLOCK):
        block = freq[start:start + BLOCK]
        phase = 2 * np.pi * block[:, None] * centred[None, :]
        cos, sin = np.cos(phase), np.sin(phase)
        k = block.size
        normal = np.empty((k, npar, npar))
        normal[:, 0, 0] = (cos * cos) @ weight
        normal[:, 1, 1] = (sin * sin) @ weight
        normal[:, 0, 1] = normal[:, 1, 0] = (cos * sin) @ weight
        normal[:, 0, 2:] = normal[:, 2:, 0] = cos @ wpoly
        normal[:, 1, 2:] = normal[:, 2:, 1] = sin @ wpoly
        normal[:, 2:, 2:] = pp
        target = np.empty((k, npar))
        target[:, 0] = cos @ wv
        target[:, 1] = sin @ wv
        target[:, 2:] = pv
        # one system per frequency and per tau
        stack = normal[:, None, :, :] + np.einsum(
            'tp,pq->tpq', inv_prior, np.eye(npar))[None, :, :, :]
        rhs = np.broadcast_to(target[:, None, :, None],
                              (k, len(taus), npar, 1))
        try:
            chol = np.linalg.cholesky(stack)
            solved = np.linalg.solve(stack, rhs)[..., 0]
        except np.linalg.LinAlgError:
            for j, one in enumerate(block):
                design = _design(time, one, trend)
                terms = [_log_evidence(design, value, weight, tau)
                         for tau in taus]
                top = np.max(terms)
                out[start + j] = top + np.log(np.mean(np.exp(
                    np.array(terms) - top)))
            continue
        chi2 = chi0 - np.einsum('fp,ftp->ft', target, solved)
        logdet = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)),
                              axis=-1)
        terms = -0.5 * (chi2 + logdet) - log_prior[None, :]
        top = np.max(terms, axis=1)
        out[start:start + k] = top + np.log(np.mean(np.exp(
            terms - top[:, None]), axis=1))
    return out


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

    evidence = _evidence_grid(time, value, weight, np.asarray(freq, float),
                              taus, trend)

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

    Each point counts for its probability of being a normal point. That
    probability is read off its distance to the median, against its error bar
    and the jitter, and NOT off the fit, so it is the same at every round:
    the second round is already the fixed point, and the FIP is computed once,
    with it. The five rounds this used to make gave the same values to the bit
    (checked on Welopoai and WASP-76, 2026-10-21) in five times the time.
    `rounds` is kept for the callers: 1 is the plain FIP with the
    probabilities beside it, any other value the outlier-aware one.
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
    prob[good] = outlier_probability(centred[good], err[good], jitter)
    # a point that is probably an outlier counts for less, in the exact
    # proportion of its probability of being a normal point
    weights = np.ones(value.size)
    if rounds > 1:
        weights[good] = 1.0 - prob[good]
    out_freq, out_fip = fip(time, value, err, freq=freq, trend=trend,
                            jitter=jitter, weights=weights)
    return out_freq, out_fip, prob
