#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The known planets of a star fitted together (lbl.pca2d.keplerfit), as
circular orbits and as full Keplerians, the periods held, the jitter free;
and the report's figure of them (texreport.figure_phase: each planet with
the others taken out, its two orbits, the residuals under it) and its table
(texreport.star_numbers: K and Delta BIC of each planet from the joint
circular fit).

    PYTHONPATH=<this checkout> python -m pytest maintenance/tests/test_keplerfit.py

Created on 2026-11-12

@author: artigau
"""
import warnings

import numpy as np
import pytest

from lbl.pca2d import keplerfit as kf
from lbl.pca2d import texreport

#: planet b eccentric, planet c circular, around a campaign of 600 days
B = dict(period=5.3, k=8.0, ecosw=0.25, esinw=-0.15, lam=1.0)
C = dict(period=37.7, k=4.0, ecosw=0.0, esinw=0.0, lam=2.0)


def _series(seed=4, n=150, noise=1.5, planets=(B, C)):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(60000.0, 60600.0, n))
    t_ref = 0.5 * (t.min() + t.max())
    v = 30.0 + rng.normal(0.0, noise, n)
    for p in planets:
        v = v + kf.keplerian(t, p["period"], p["k"], p["ecosw"], p["esinw"],
                             p["lam"], t_ref)
    return t, v, np.full(n, noise)


def test_kepler_s_equation_and_the_circular_limit():
    mean = np.linspace(-10.0, 10.0, 501)
    for ecc in (0.0, 0.1, 0.5, 0.9):
        ecc_anom = kf.solve_kepler(mean, ecc)
        assert np.allclose(ecc_anom - ecc * np.sin(ecc_anom), mean,
                           atol=1e-10)
    t = np.linspace(0.0, 50.0, 200)
    assert np.allclose(kf.keplerian(t, 7.0, 3.0, 0.0, 0.0, 0.4, 10.0),
                       3.0 * np.cos(0.4 + 2 * np.pi * (t - 10.0) / 7.0))
    # a small eccentricity changes it smoothly, not by a jump
    near = kf.keplerian(t, 7.0, 3.0, 1e-6, 0.0, 0.4, 10.0)
    assert np.allclose(near, kf.keplerian(t, 7.0, 3.0, 0.0, 0.0, 0.4, 10.0),
                       atol=1e-4)


def test_both_planets_are_found_together():
    t, v, e = _series()
    fit = kf.fit_planets(t, v, e, [B["period"], C["period"]])
    circ, kep = fit["circular"], fit["keplerian"]
    b, c = kep.planets
    assert b["k"] == pytest.approx(B["k"], abs=3 * b["k_err"])
    assert c["k"] == pytest.approx(C["k"], abs=3 * c["k_err"])
    truth = np.hypot(B["ecosw"], B["esinw"])
    assert b["e"] == pytest.approx(truth, abs=3 * b["e_err"])
    assert b["omega"] == pytest.approx(
        np.degrees(np.arctan2(B["esinw"], B["ecosw"])), abs=3 * b["omega_err"])
    # both planets wanted, in either fit; b's eccentricity, not c's
    assert min(fit["planet_bic"]) > 10 and min(fit["kep_planet_bic"]) > 10
    assert fit["ecc_bic"][0] > 10 and fit["ecc_bic"][1] < 0
    # each fit against no planet, and the Keplerian above the circular by
    #   b's eccentricity
    assert fit["fit_bic"]["circular"] == pytest.approx(
        fit["null_bic"] - circ.bic)
    assert fit["fit_bic"]["keplerian"] > fit["fit_bic"]["circular"] > 100
    # one planet: without it, the fit is the one without planets
    one = kf.fit_planets(t, v, e, [B["period"]])
    assert one["planet_bic"] == pytest.approx([one["fit_bic"]["circular"]])
    assert one["kep_planet_bic"] == pytest.approx(
        [one["fit_bic"]["keplerian"]])
    # the Keplerian fit nests the circular one: never below it
    assert kep.lnl >= circ.lnl
    # what it leaves is the noise
    assert np.std(v - np.median(v) - kep.model(t)) == pytest.approx(1.5,
                                                                    rel=0.15)
    # each planet less the others: that planet alone
    assert np.allclose(kep.model(t) - kep.others(0, t), kep.planet(0, t))


def test_a_planet_fitted_alone_carries_the_other_as_noise():
    """Two planets at periods the campaign samples together: fitted alone,
    each takes the other as noise (its jitter, and K's error with it, grow
    by the other's amplitude); fitted together, each K is as precise as the
    noise allows"""
    rng = np.random.default_rng(11)
    # three-night runs every 20 days: the two periods alias on that window
    t = np.sort(np.concatenate([base + rng.uniform(0, 3, 4)
                                for base in np.arange(60000.0, 60400.0,
                                                      20.0)]))
    t_ref = 0.5 * (t.min() + t.max())
    v = (kf.keplerian(t, 3.0, 5.0, 0, 0, 0.3, t_ref)
         + kf.keplerian(t, 3.0 * 20 / 21, 5.0, 0, 0, 1.9, t_ref)
         + rng.normal(0.0, 0.5, t.size))
    e = np.full(t.size, 0.5)
    joint = kf.fit_planets(t, v, e, [3.0, 60.0 / 21], with_keplerian=False)
    alone = [kf.fit_planets(t, v, e, [p], with_keplerian=False)
             for p in (3.0, 60.0 / 21)]
    for i in range(2):
        k_joint = joint["circular"].planets[i]
        assert k_joint["k"] == pytest.approx(5.0, abs=4 * k_joint["k_err"])
        k_alone = alone[i]["circular"].planets[0]
        assert k_joint["k_err"] < k_alone["k_err"] / 3
        assert alone[i]["circular"].jitter > 2.0
    assert joint["circular"].jitter < 0.3


def test_a_circular_fit_alone_and_too_few_points():
    t, v, e = _series()
    fit = kf.fit_planets(t, v, e, [B["period"]], with_keplerian=False)
    assert fit["keplerian"] is None and fit["ecc_bic"] is None
    assert fit["circular"].planets[0]["k"] > 0
    assert kf.fit_planets(t[:9], v[:9], e[:9], [5.3]) is None
    assert kf.fit_planets(t, v, e, []) is None
    # the straight line: only when every period is under a third of it
    assert kf.drift_for(t, [5.3, 37.7]) and not kf.drift_for(t, [5.3, 300])


def _run(label, noise, seed):
    t, v, e = _series(seed=seed, noise=noise)
    return dict(t=t, v=v, e=e, label=label, star="TOI-TEST",
                longitude=-70.73)


PLANETS = [dict(name="TOI-TEST b", short="b", period=5.3, t0=2460010.3,
                k=7.5),
           dict(name="TOI-TEST c", short="c", period=37.7)]


def test_the_report_s_figure_and_table(tmp_path):
    before, after = _run("delivered", 3.0, 4), _run("corrected", 1.5, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for planets in (PLANETS, PLANETS[:1]):
            path = texreport.figure_phase(before, after, planets,
                                          str(tmp_path / "phase.pdf"))
            assert path and (tmp_path / "phase.pdf").stat().st_size > 0
    # the table's K and Delta BIC: those of the joint circular fit
    numbers = texreport.star_numbers(before, after, planets=[5.3, 37.7])
    joint = kf.fit_planets(after["t"], after["v"], after["e"], [5.3, 37.7],
                           with_keplerian=False)
    assert numbers["after"]["planets"][1][0] == pytest.approx(
        joint["circular"].planets[1]["k"])
    assert numbers["after"]["planet_bic"] == pytest.approx(
        joint["planet_bic"])


def test_the_two_mcmcs_find_the_orbits():
    t, v, e = _series()
    post = kf.posteriors(t, v, e, [B["period"], C["period"]], steps=600,
                         burn=300)
    circ, kep = post["circular"]["quantiles"], post["keplerian"]["quantiles"]
    # the circular MCMC has no eccentricity; the Keplerian one finds b's
    assert "e_0" not in circ and "e_0" in kep
    mid, low, high = kep["e_0"]
    truth = np.hypot(B["ecosw"], B["esinw"])
    assert low - 0.05 < truth < high + 0.05
    for i, planet in enumerate((B, C)):
        mid, low, high = kep["K_%d" % i]
        assert low - 0.5 < planet["k"] < high + 0.5
    assert 0.15 < post["keplerian"]["acceptance"] < 0.7
    assert post["circular"]["walkers"] >= kf.MIN_WALKERS


def test_the_report_s_mcmc_figure_and_table(tmp_path, monkeypatch):
    monkeypatch.setattr(kf, "MCMC_STEPS", 400)
    monkeypatch.setattr(kf, "MCMC_BURN", 200)
    monkeypatch.setattr(kf.posteriors, "__defaults__", (400, 200, 0))
    before, after = _run("delivered", 3.0, 4), _run("corrected", 1.5, 5)
    posts = texreport.planet_posteriors(before, after, PLANETS)
    assert set(posts) == {"delivered", "corrected"}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        path = texreport.figure_mcmc(posts, PLANETS, before, after,
                                     str(tmp_path / "mcmc.pdf"))
    assert path and (tmp_path / "mcmc.pdf").stat().st_size > 0
    table = texreport.mcmc_table(posts, PLANETS, before, after, "TOI-TEST")
    assert table.count("K (b, m/s)") == 1 and "e (c)" in table
    assert "0 (fixed)" in table and "--" not in table
    # no planet known: neither
    assert texreport.planet_posteriors(before, after, []) is None
    assert texreport.mcmc_table(None, PLANETS, before, after, "x") == ""
