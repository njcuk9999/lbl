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
