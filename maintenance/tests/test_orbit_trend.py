"""The known orbits and an acceleration fitted together
(lbl.report.figures.fit_orbits_trend): K and the slope recovered, within
their errors, the orbit's shape held."""
import numpy as np

from lbl.report import figures


def test_k_and_the_acceleration_come_back():
    rng = np.random.default_rng(1214)
    time = 2455000.0 + np.sort(rng.uniform(0, 2000, 120))
    orbit = dict(name='b', period=1.58040433, k=14.0, e=0.0, omega=90.0,
                 t0=2455000.5)
    from lbl.science import planets as kepler
    slope = 2.0 / 365.25
    truth = 30.0 + slope * (time - time.mean()) + kepler.orbit_velocity(
        time, orbit)
    err = np.full(time.size, 2.0)
    vrad = truth + rng.normal(0, np.hypot(2.0, 1.5), time.size)
    fit = figures.fit_orbits_trend(time, vrad, err, [orbit])
    k, k_err = fit['coef'][2], fit['err'][2]
    a, a_err = fit['coef'][1] * 365.25, fit['err'][1] * 365.25
    assert abs(k - 14.0) < 4 * k_err
    assert abs(a - 2.0) < 4 * a_err
    # the jitter found is the one put in, roughly
    assert 0.5 < fit['jitter'] < 3.0
