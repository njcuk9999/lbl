"""lbl.report.kepfit: the joint fit of a star's transiting planets and
candidates finds the K it is given, with honest errors.

    PYTHONPATH=<this checkout> python -m pytest tests/test_kepfit.py
"""
import numpy as np

from lbl.report import kepfit
from lbl.science import planets

CANDIDATES = [dict(name='b', period=10.8772748, t0=2458626.968411),
              dict(name='c', period=83.9993991, t0=2458630.353329)]
KNOWN = [dict(name='x', period=300.0, t0=2460000.0, k=20.0, e=0.0,
              omega=90.0)]


def test_the_basis_is_zero_and_falling_at_the_transit():
    t0, period = CANDIDATES[0]['t0'], CANDIDATES[0]['period']
    values = kepfit.basis(np.array([t0 - 0.01, t0, t0 + 0.01]), CANDIDATES[0])
    assert abs(values[1]) < 1e-9 and values[0] > 0 > values[2]
    # a quarter of a period later, the star recedes least: -K
    assert abs(kepfit.basis(t0 + period / 4, CANDIDATES[0])[0] + 1) < 1e-9


def test_the_k_come_back_with_errors_that_match_their_scatter():
    rng = np.random.default_rng(3)
    found = []
    for _ in range(100):
        time = np.sort(rng.uniform(2460300, 2461100, 150))
        error = rng.uniform(2, 4, 150)
        vrad = (12 + 3.0 * kepfit.basis(time, CANDIDATES[0])
                + 5.0 * kepfit.basis(time, CANDIDATES[1])
                + planets.star_velocity(time, KNOWN)
                + rng.normal(0, 1, 150) * np.sqrt(error ** 2 + 1.5 ** 2))
        result = kepfit.fit(time, vrad, error, CANDIDATES, KNOWN)
        found.append([result['k'][0], result['k'][1], result['sk'][0]])
    found = np.array(found)
    assert abs(found[:, 0].mean() - 3.0) < 0.15
    assert abs(found[:, 1].mean() - 5.0) < 0.15
    # the error bar of K is its scatter, to 25%
    assert 0.75 < found[:, 0].std() / found[:, 2].mean() < 1.25


def test_the_model_leaves_out_the_candidate_asked():
    time = np.linspace(2460300, 2460400, 50)
    result = dict(offset=1.0, known=[], candidates=CANDIDATES,
                  k=np.array([2.0, 3.0]))
    full = kepfit.model(result, time)
    without_b = kepfit.model(result, time, skip=0)
    assert np.allclose(full - without_b,
                       2.0 * kepfit.basis(time, CANDIDATES[0]))


def test_the_reduced_chi2_says_how_far_the_error_bars_are():
    rng = np.random.default_rng(5)
    time = np.sort(rng.uniform(2460300, 2461100, 400))
    error = np.full(400, 2.0)
    signal = 4.0 * kepfit.basis(time, CANDIDATES[0])
    # error bars that hold: about 1 after the candidate, more before it
    honest = kepfit.fit(time, 7 + signal + rng.normal(0, 2.0, 400), error,
                        CANDIDATES[:1])
    assert honest['dof'] == 400 - 2
    assert abs(honest['chi2_after'] - 1) < 0.2 < honest['chi2_before'] - 1
    # a scatter twice the error bars: 4, and about 1 once the jitter is in
    noisy = kepfit.fit(time, 7 + signal + rng.normal(0, 4.0, 400), error,
                       CANDIDATES[:1])
    assert abs(noisy['chi2_after'] - 4) < 0.6
    assert abs(noisy['chi2_jitter'] - 1) < 0.1
