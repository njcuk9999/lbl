"""The alias-aware FIP (lbl.report.fip.alias_aware): a peak and its alias
share one posterior, rather than being both "detected" as the per-frequency
FIP has them, and a few outliers do not move the answer."""
import numpy as np

from lbl.report import fip


def _series(seed=3, period=3.7, k=4.0, outliers=True):
    """A ground-based cadence (about every other night, 120-day seasons over
    three years), a sine of `k` m/s, 3 m/s errors, three bad nights."""
    rng = np.random.default_rng(seed)
    nights = [d for d in np.arange(0, 900)
              if (d % 365) < 120 and rng.random() < 0.5]
    time = np.sort(np.array(nights) + rng.uniform(-0.1, 0.1, len(nights)))
    value = k * np.sin(2 * np.pi * time / period) + rng.normal(0, 3, time.size)
    if outliers:
        value[[5, 40, 77]] += [60.0, -45.0, 80.0]
    return time, value, np.full(time.size, 3.0)


def test_the_planet_and_its_aliases_share_one_posterior():
    time, value, err = _series()
    out = fip.alias_aware(time, value, err)
    assert abs(out['best_period'] - 3.7) < 0.01
    # every interval's probability of a signal, and the null's, add to at
    #   most one: the posterior is shared, not handed out per frequency
    freq = out['freq']
    span = np.ptp(time)
    chosen = [np.argmin(out['fip'])]
    for alias, _name in fip.alias_periods(out['best_period'], 1.1, 2 * span):
        chosen.append(int(np.argmin(np.abs(freq - 1.0 / alias))))
    distinct = [chosen[0]] + [j for j in chosen[1:]
                              if abs(freq[j] - freq[chosen[0]]) > 1.0 / span]
    assert np.sum(1.0 - out['fip'][distinct]) <= 1.0 + 1e-9
    # the one-day alias is not a detection beside the planet
    one_day = [j for j in distinct[1:]
               if abs(1.0 / freq[j] - 1.0 / abs(1.0 / 3.7 - 1.0 / 0.99727))
               < 0.01]
    assert one_day and out['fip'][one_day[0]] > 0.5


def test_a_few_bad_nights_do_not_move_the_answer():
    clean = fip.alias_aware(*_series(outliers=False))
    dirty = fip.alias_aware(*_series(outliers=True))
    assert abs(dirty['best_period'] - clean['best_period']) < 0.01


def test_a_point_ever_further_out_stays_an_outlier():
    # both densities underflow near 193 error bars: the wide one stays the
    #   likelier, and the point must not come back with its full weight
    err = np.ones(31)
    for distance in (10.0, 50.0, 192.0, 193.0, 1000.0, 1e6):
        value = np.zeros(31)
        value[0] = distance
        assert fip.outlier_probability(value, err)[0] > 0.999
    # the same values as the densities themselves where they do not underflow
    value = np.linspace(-20.0, 20.0, 41)
    prior, width = fip.OUTLIER_PRIOR, fip.OUTLIER_WIDTH
    narrow = np.exp(-0.5 * value ** 2)
    wide = np.exp(-0.5 * (value / width) ** 2) / width
    expected = prior * wide / ((1.0 - prior) * narrow + prior * wide)
    assert np.allclose(fip.outlier_probability(value, np.ones(41)), expected,
                       rtol=1e-12, atol=0.0)


def _sine(n, k, seed, period=7.3, span=300.0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, span, n))
    value = k * np.sin(2 * np.pi * time / period + rng.uniform(0, 6.3)) \
        + rng.normal(0, 1, n)
    return time, value, np.ones(n)


def test_the_fip_keeps_falling_with_the_amplitude():
    # with the jitter taken from the scatter of the whole series, a sine of
    #   10 to 1000 error bars on 50 epochs stayed at a FIP of 0.1 to 0.6
    for k in (10.0, 100.0, 1000.0):
        out = fip.alias_aware(*_sine(50, k, seed=4), outlier_aware=False)
        assert abs(1.0 / out['best_period'] - 1.0 / 7.3) < 1.0 / 300.0
        assert out['best_fip'] < 1e-6, (k, out['best_fip'])
        out = fip.alias_aware(*_sine(50, k, seed=4))
        assert out['best_fip'] < 1e-6, (k, out['best_fip'])


def test_noise_alone_is_not_a_detection():
    rng = np.random.default_rng(7)
    for scatter in (1.0, 2.0):          # 2: a jitter the error bars miss
        lowest = 1.0
        for _ in range(40):
            time = np.sort(rng.uniform(0, 300.0, 60))
            value = rng.normal(0, scatter, time.size)
            for aware in (False, True):
                out = fip.alias_aware(time, value, np.ones(time.size),
                                      outlier_aware=aware)
                lowest = min(lowest, out['best_fip'])
        assert lowest > 0.05, (scatter, lowest)


def test_each_model_takes_the_jitter_of_its_own_residuals():
    """At the signal's frequency the residuals are the noise; under the null
    they hold the signal."""
    time, value, err = _sine(60, 20.0, seed=2)
    centred = time - np.mean(time)
    poly = np.vstack([np.ones_like(centred), centred]).T
    null = fip._residuals(None, None, poly, value, err)
    phase = 2 * np.pi * centred[None, :] / 7.3
    at_signal = fip._residuals(np.cos(phase), np.sin(phase), poly, value, err)
    w_null = fip._noise(null, err, False, fitted=2)[0]
    w_signal = fip._noise(at_signal, err, False, fitted=4)[0]
    assert np.median(w_signal) > 0.5          # jitter near zero
    assert np.median(w_null) < 0.01           # jitter of the signal, ~20


def test_a_given_jitter_is_the_fixed_noise_of_every_model():
    time, value, err = _sine(60, 3.0, seed=5)
    freq = fip.frequency_grid(time)
    fixed = fip.fip(time, value, err, freq=freq, jitter=0.0)[1]
    weight = 1.0 / err ** 2
    centred = value - np.median(value)
    scale = fip.robust_sigma(centred)
    taus = np.array(fip.TAU_LADDER) * scale
    null = fip._log_evidence(fip._design(time, 0.0)[:, 2:], centred, weight,
                             1.0, nsignal=0)
    evidence = fip._evidence_grid(time, centred, weight, freq, taus)
    log_prior = np.log(0.5 / freq.size)
    expected = 1.0 / (1.0 + np.exp(log_prior + evidence - np.log(0.5) - null))
    assert np.allclose(fixed, expected, rtol=1e-9, atol=1e-12)
