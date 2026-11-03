"""The false inclusion probability, gaussian and outlier-aware, per frequency
and (alias_aware) over intervals, alias-aware.

WHERE IT COMES FROM. The false inclusion probability (FIP) of Hara et al.
(2022, A&A 663, A14, https://doi.org/10.1051/0004-6361/202140543), computed
here in numpy alone, with no sampling and no subprocess, so that the report
gives the same answer however LBL is launched (a script, a notebook, the
command line).

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
a mixture model in which each point's membership would be sampled with the
rest, and it answers the question the report asks: does the answer depend on
how the outliers are treated?

THE NOISE OF EACH MODEL, FROM ITS OWN RESIDUALS. The scatter the error bars
do not explain (the jitter) is estimated at every frequency from the
residuals of the best fit of a sinusoid there, and for the null model from
those of the trend alone; in the outlier-aware case each point's
probability of being an outlier is read off the same residuals. The
evidence of each model then carries its own noise normalisation. Taken from
the scatter of the whole series instead, the jitter holds the signal
itself: for a sinusoid of amplitude K it is about K, the gain of fitting
the sinusoid stays near 0.455 N whatever K is, and the FIP stops falling
with the amplitude (a 37 sigma planet can come out at 0.4). Fitting the
jitter of each model and comparing their maximised likelihoods is the
profile likelihood, whose ratio has, for many points, the distribution it
has with a known noise when there is no signal. On fewer points it is
optimistic, which the n / (n - p) correction of the scatter offsets
(_noise): on 1000 series of 60 epochs, half of noise alone, none of those
gets a FIP below 0.01, with or without a jitter to find, as before.

SPEED. Every frequency of the grid (20 000 for a six-year series) and every
prior width of the ladder is one small linear system. They are built and
solved together, a block of frequencies at a time, with batched Cholesky
factorisations: the same algebra as one at a time. On 200 epochs and 20 000
frequencies the whole grid takes about a second.
"""
import numpy as np

from lbl.report.stats import MAD_TO_SIGMA, robust_sigma

#: the ladder of prior amplitudes, in units of the robust scatter. The FIP
#: should not depend on a single guess of how large a signal might be, so it
#: is averaged over a log-uniform ladder.
TAU_LADDER = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
#: how wide the outlier population is, as a multiple of the error bars
OUTLIER_WIDTH = 5.0
#: the prior fraction of points that are outliers
OUTLIER_PRIOR = 0.05


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


def _noise(resid, err, outlier_aware, extra=None, fitted=0):
    """The noise model of fits, from their residuals (one fit per row of
    `resid`): (weights, log normalisation, outlier probabilities).

    The jitter of each fit is the robust scatter of its residuals less the
    median error bar, as the scatter the error bars do not explain. The
    scatter is that of the residuals of a fit of `fitted` parameters, which
    fall short of the noise by n / (n - fitted) in variance: corrected for
    it, a model with more parameters does not get a smaller jitter for
    having fitted the noise.

    Without outliers the weights are 1 / (err^2 + jitter^2) and the
    normalisation is the gaussian's, -1/2 sum log(err^2 + jitter^2). With
    them, each point's
    probability of being an outlier is read off its own residual
    (outlier_probability), the weights are (1 - P) / (err^2 + jitter^2), and
    the normalisation holds what the two populations add to that quadratic
    form with these memberships: the wide population's own term at the
    residual, the prior of each population and the entropy of the
    memberships, which together make the mixture's log likelihood at the
    fit (the memberships being its posterior there). Without them a
    frequency would gain by calling more points outliers. `extra`
    multiplies the weights, a down-weighting of the caller's.
    """
    resid = np.atleast_2d(resid)
    mad = np.median(np.abs(resid - np.median(resid, axis=1, keepdims=True)),
                    axis=1)
    n = resid.shape[1]
    dof = n / max(n - int(fitted), 1)
    jitter2 = np.clip(dof * (MAD_TO_SIGMA * mad) ** 2 - np.median(err) ** 2,
                      0.0, None)
    total = err[None, :] ** 2 + jitter2[:, None]
    if not outlier_aware:
        weight = 1.0 / total
        if extra is not None:
            weight = weight * extra[None, :]
        return weight, -0.5 * np.sum(np.log(total), axis=1), \
            np.zeros(resid.shape)
    wide = OUTLIER_WIDTH ** 2 * total
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        log_ratio = (0.5 * np.log(wide / total) - 0.5 * resid ** 2 / total
                     + 0.5 * resid ** 2 / wide)
        prob = 1.0 / (1.0 + (1.0 - OUTLIER_PRIOR) / OUTLIER_PRIOR
                      * np.exp(log_ratio))
    prob = np.clip(np.where(np.isfinite(prob), prob, 0.0), 0.0, 1.0)
    keep = 1.0 - prob
    weight = keep / total
    if extra is not None:
        weight = weight * extra[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -(np.where(prob > 0, prob * np.log(prob), 0.0)
                    + np.where(keep > 0, keep * np.log(keep), 0.0))
    norm = np.sum(-0.5 * keep * np.log(total)
                  + keep * np.log(1.0 - OUTLIER_PRIOR)
                  + prob * (np.log(OUTLIER_PRIOR) - 0.5 * np.log(wide)
                            - 0.5 * resid ** 2 / wide)
                  + entropy, axis=1)
    return weight, norm, prob


def _systems(cos, sin, poly, value, weight):
    """The normal matrices and right-hand sides of [cos, sin, trend], one per
    row of `weight` (and of cos and sin, None for the trend alone), with
    chi0 = sum weight value^2 of each."""
    n_poly = poly.shape[1]
    k = weight.shape[0]
    wv = weight * value[None, :]
    chi0 = np.sum(wv * value[None, :], axis=1)
    pp = np.einsum('kn,np,nq->kpq', weight, poly, poly)
    pv = wv @ poly
    if cos is None:
        return pp, pv, chi0
    npar = 2 + n_poly
    normal = np.empty((k, npar, npar))
    normal[:, 0, 0] = np.sum(cos * cos * weight, axis=1)
    normal[:, 1, 1] = np.sum(sin * sin * weight, axis=1)
    normal[:, 0, 1] = normal[:, 1, 0] = np.sum(cos * sin * weight, axis=1)
    normal[:, 0, 2:] = normal[:, 2:, 0] = (cos * weight) @ poly
    normal[:, 1, 2:] = normal[:, 2:, 1] = (sin * weight) @ poly
    normal[:, 2:, 2:] = pp
    target = np.empty((k, npar))
    target[:, 0] = np.sum(cos * wv, axis=1)
    target[:, 1] = np.sum(sin * wv, axis=1)
    target[:, 2:] = pv
    return normal, target, chi0


def _residuals(cos, sin, poly, value, err):
    """The residuals of the least-squares fit of [cos, sin, trend] (of the
    trend alone when cos is None) at the error bars, one row per frequency."""
    base = np.broadcast_to(1.0 / err ** 2, (1 if cos is None else cos.shape[0],
                                            err.size))
    normal, target, _ = _systems(cos, sin, poly, value, base)
    # the solve is regularised by a hair, for a frequency whose sine and
    #   cosine are degenerate on this sampling
    scale = np.einsum('kpp->kp', normal).max(axis=1)
    ridge = 1e-12 * scale[:, None, None] * np.eye(normal.shape[1])[None]
    beta = np.linalg.solve(normal + ridge, target[..., None])[..., 0]
    model = beta[:, -poly.shape[1]:] @ poly.T
    if cos is not None:
        model = model + beta[:, :1] * cos + beta[:, 1:2] * sin
    return value[None, :] - model


def _profiled(time, value, err, freq, taus, trend=1, outlier_aware=False,
              extra=None):
    """(log evidence at every frequency, log evidence of the null): each
    model with the noise of its own residuals (_noise), the evidence
    averaged over the prior ladder `taus` as in _evidence_grid."""
    centred = time - np.mean(time)
    poly = np.vstack([centred ** power for power in range(trend + 1)]).T
    n_poly = poly.shape[1]
    npar = 2 + n_poly
    # the null model: the trend alone, its noise from its own residuals
    weight, norm, _ = _noise(_residuals(None, None, poly, value, err), err,
                             outlier_aware, extra, n_poly)
    pp, pv, chi0 = _systems(None, None, poly, value, weight)
    pp = pp[0] + np.eye(n_poly) / 1e8 ** 2
    chi2 = chi0[0] - pv[0] @ np.linalg.solve(pp, pv[0])
    null = -0.5 * (chi2 + np.linalg.slogdet(pp)[1]) + norm[0]
    inv_prior = np.zeros((len(taus), npar))
    inv_prior[:, :2] = 1.0 / np.asarray(taus, dtype=float)[:, None] ** 2
    inv_prior[:, 2:] = 1.0 / 1e8 ** 2
    log_prior = 2.0 * np.log(np.asarray(taus, dtype=float))
    out = np.empty(freq.size)
    for start in range(0, freq.size, BLOCK):
        block = freq[start:start + BLOCK]
        phase = 2 * np.pi * block[:, None] * centred[None, :]
        cos, sin = np.cos(phase), np.sin(phase)
        weight, norm, _ = _noise(_residuals(cos, sin, poly, value, err), err,
                                 outlier_aware, extra, npar)
        normal, target, chi0 = _systems(cos, sin, poly, value, weight)
        stack = normal[:, None, :, :] + np.einsum(
            'tp,pq->tpq', inv_prior, np.eye(npar))[None, :, :, :]
        rhs = np.broadcast_to(target[:, None, :, None],
                              (block.size, len(taus), npar, 1))
        sign, logdet = np.linalg.slogdet(stack)
        solved = np.linalg.solve(stack, rhs)[..., 0]
        chi2 = chi0[:, None] - np.einsum('fp,ftp->ft', target, solved)
        terms = np.where(sign > 0, -0.5 * (chi2 + logdet), -np.inf) \
            - log_prior[None, :]
        top = np.max(terms, axis=1)
        out[start:start + block.size] = top + np.log(np.mean(np.exp(
            terms - top[:, None]), axis=1)) + norm
    return out, null


def memberships(time, value, err, frequency, trend=1):
    """Each point's probability of being an outlier against the best fit of
    a sinusoid at `frequency` (with the trend), read off its residual as
    the outlier-aware FIP reads it there."""
    centred = time - np.mean(time)
    poly = np.vstack([centred ** power for power in range(trend + 1)]).T
    phase = 2 * np.pi * float(frequency) * centred[None, :]
    resid = _residuals(np.cos(phase), np.sin(phase), poly, value, err)
    return _noise(resid, err, True, fitted=2 + poly.shape[1])[2][0]


def fip(time, value, err, freq=None, trend=1, prior_signal=0.5,
        jitter=None, weights=None, outlier_aware=False):
    """The gaussian FIP over a grid of periods.

    Returns (freq, fip). Without `jitter`, the noise of each model is
    estimated from its own residuals (_profiled); with `jitter` given, every
    point is taken at err^2 + jitter^2 for every model. `weights` multiplies
    the weights of the points. `outlier_aware` reads each point's
    probability of being an outlier off the residuals of each model too.
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
    extra = (np.asarray(weights, dtype=float)[good] if weights is not None
             else None)
    if freq is None:
        freq = frequency_grid(time)
    freq = np.asarray(freq, dtype=float)
    if freq.size == 0:
        return np.array([]), np.array([])

    scale = robust_sigma(value)
    taus = np.array(TAU_LADDER) * (scale if np.isfinite(scale) and scale > 0
                                   else 1.0)
    if jitter is None:
        evidence, null = _profiled(time, value, err, freq, taus, trend,
                                   outlier_aware, extra)
    else:
        weight = 1.0 / (err ** 2 + float(jitter) ** 2)
        if extra is not None:
            weight = weight * extra
        # the null model: the trend alone, no sinusoid
        flat = _design(time, 0.0, trend)[:, 2:]
        null = _log_evidence(flat, value, weight, 1.0, nsignal=0)
        evidence = _evidence_grid(time, value, weight, freq, taus, trend)

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
    # the ratio of the two densities in logs: far enough out, both densities
    #   underflow to zero, while the wide population stays the far likelier
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        log_ratio = (np.log(wide / narrow) - 0.5 * (centred / narrow) ** 2
                     + 0.5 * (centred / wide) ** 2)
        out = 1.0 / (1.0 + (1.0 - prior) / prior * np.exp(log_ratio))
    # no error bar, no estimate
    return np.clip(np.where(np.isfinite(out), out, 0.0), 0.0, 1.0)


def oafip(time, value, err, freq=None, trend=1, rounds=5):
    """The outlier-aware FIP. Returns (freq, fip, outlier probability).

    Each point counts for its probability of being a normal point, read at
    every frequency off its residual to the best fit there (and for the null
    model off its residual to the trend), against its error bar and the
    jitter of that fit (_noise). The probabilities returned are those at the
    frequency of the lowest FIP. `rounds`: 1 gives the gaussian FIP with
    those probabilities beside it, any other value the outlier-aware one.
    """
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    err = np.asarray(err, dtype=float)
    good = (np.isfinite(time) & np.isfinite(value) & np.isfinite(err)
            & (err > 0))
    if good.sum() < 10:
        return np.array([]), np.array([]), np.array([])
    out_freq, out_fip = fip(time, value, err, freq=freq, trend=trend,
                            outlier_aware=rounds > 1)
    prob = np.zeros(value.size)
    if out_freq.size:
        best = out_freq[int(np.argmin(out_fip))]
        centred = value[good] - np.median(value[good])
        prob[good] = memberships(time[good], centred, err[good], best, trend)
    return out_freq, out_fip, prob


#: the aliases a peak is checked against: the sidereal day and the year,
#: the two periods ground-based sampling imprints on every series
ALIAS_PERIODS = ((0.99727, '1 d'), (365.25, '1 yr'))


def alias_aware(time, value, err, freq=None, trend=1, prior_signal=0.5,
                outlier_aware=True):
    """The FIP of Hara et al. (2022, A&A 663, A14,
    https://doi.org/10.1051/0004-6361/202140543): over intervals one
    resolution element wide (1/T), the probability that NO signal has its
    frequency there, with every frequency of the grid competing for the
    same posterior. Returns dict(freq, fip, prob, best_period, best_fip).

    WHAT ALIAS-AWARE MEANS. fip() compares each frequency with the null
    model alone, so a peak and its alias are both "detected" when the data
    cannot tell them apart. Here the posterior of a single signal is spread
    over the whole grid and normalised once, so the two share it: neither
    gets below one half unless the data prefer it. The probability of an
    interval is the sum over the grid points inside it, a peak being spread
    over about `oversample` of them.

    OUTLIER-AWARE, by default: each point counts for its probability of
    being a normal one, at every frequency against the fit there (_noise,
    as oafip does); `prob` holds those of the best interval's frequency.
    Each model's jitter is that of its own residuals.
    """
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    err = np.asarray(err, dtype=float)
    good = (np.isfinite(time) & np.isfinite(value) & np.isfinite(err)
            & (err > 0))
    empty = dict(freq=np.array([]), fip=np.array([]), prob=np.array([]),
                 best_period=np.nan, best_fip=np.nan)
    if good.sum() < 10:
        return empty
    time, value, err = time[good], value[good], err[good]
    value = value - np.median(value)
    if freq is None:
        freq = frequency_grid(time)
    freq = np.asarray(freq, dtype=float)
    if freq.size == 0:
        return empty
    scale = robust_sigma(value)
    taus = np.array(TAU_LADDER) * (scale if np.isfinite(scale) and scale > 0
                                   else 1.0)
    evidence, null = _profiled(time, value, err, freq, taus, trend,
                               outlier_aware)
    # one posterior for the whole grid: the null, and a signal somewhere
    terms = np.log(prior_signal / freq.size) + evidence
    log_null = np.log(1.0 - prior_signal) + null
    top = max(log_null, float(np.nanmax(terms)))
    mass = np.exp(terms - top)
    total = np.exp(log_null - top) + np.sum(mass)
    # each interval, [f - 1/2T, f + 1/2T], summed through a cumulative sum
    half = 0.5 / float(np.ptp(time))
    cum = np.concatenate([[0.0], np.cumsum(mass)])
    low = np.searchsorted(freq, freq - half, side='left')
    high = np.searchsorted(freq, freq + half, side='right')
    out = np.clip(1.0 - (cum[high] - cum[low]) / total, 0.0, 1.0)
    best = int(np.argmin(out))
    prob = memberships(time, value, err, freq[best], trend)
    return dict(freq=freq, fip=out, prob=prob, best_period=1.0 / freq[best],
                best_fip=float(out[best]))


def alias_periods(period, low=None, high=None):
    """The aliases of a period with the day and the year: [(period, name)],
    those within [low, high] when given."""
    out = []
    for window, name in ALIAS_PERIODS:
        for sign in (1.0, -1.0):
            nu = abs(1.0 / period + sign / window)
            if nu <= 0:
                continue
            alias = 1.0 / nu
            if (low is None or alias >= low) and (high is None
                                                  or alias <= high):
                out.append((alias, name))
    return out
