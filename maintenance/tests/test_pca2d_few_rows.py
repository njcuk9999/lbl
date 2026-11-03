"""A cube with fewer rows than the star-frame median asks for still gets a star
spectrum, and says so (twoframe.star_frame_template).

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_pca2d_few_rows.py
"""
import numpy as np
import pytest

from lbl.pca2d import twoframe


def _rows(n, m=600, seed=4):
    rng = np.random.default_rng(seed)
    x = np.arange(m, dtype=float)
    star = sum(-0.2 * np.exp(-0.5 * ((x - c) / 3.0) ** 2)
               for c in rng.uniform(60, m - 60, 25))
    delta = rng.uniform(-6.0, 6.0, n)
    shifter = twoframe.LanczosShifter(m, a=8, max_shift=10)
    data = shifter.carry(shifter.prepare(star[None, :]), delta)[:, 0, :]
    data = data + rng.normal(0.0, 0.002, data.shape)
    w = np.full(data.shape, 1e4)
    w[:, :twoframe.EDGE] = 0.0
    w[:, -twoframe.EDGE:] = 0.0
    return data, w, shifter, delta, star


def _said(monkeypatch):
    said = []
    monkeypatch.setattr(twoframe, "log",
                        lambda msg, *a, **k: said.append(str(msg)))
    return said


@pytest.mark.parametrize("n", [12, 19])
def test_fewer_rows_than_min_spectra_still_give_the_star(monkeypatch, n):
    said = _said(monkeypatch)
    data, w, shifter, delta, star = _rows(n)
    template = twoframe.star_frame_template(data, w, shifter, delta)
    inner = slice(80, 520)
    assert np.count_nonzero(template[inner]) == template[inner].size
    assert np.corrcoef(template[inner], star[inner])[0, 1] > 0.99
    assert any("only %d weighted rows" % n in s for s in said)


def test_enough_rows_are_as_they_were(monkeypatch):
    said = _said(monkeypatch)
    data, w, shifter, delta, _ = _rows(30)
    # wider than the shifts: star columns 15 rows weight go, 25 rows stay
    w[:15, 260:340] = 0.0
    w[:5, 400:480] = 0.0
    template = twoframe.star_frame_template(data, w, shifter, delta)
    assert not np.any(template[290:310])
    assert np.all(template[430:450] != 0.0)
    assert not any("weighted rows" in s for s in said)


def test_two_rows_are_no_median_and_it_is_said(monkeypatch):
    said = _said(monkeypatch)
    data, w, shifter, delta, _ = _rows(2)
    template = twoframe.star_frame_template(data, w, shifter, delta)
    assert not np.any(template)
    assert any("NO STAR SPECTRUM" in s for s in said)


def test_rows_without_weight_do_not_count(monkeypatch):
    """A held-out half has no weight: the rest decide."""
    said = _said(monkeypatch)
    data, w, shifter, delta, _ = _rows(30)
    w[::2] = 0.0
    template = twoframe.star_frame_template(data, w, shifter, delta)
    assert np.count_nonzero(template[80:520]) == 440
    assert any("only 15 weighted rows" in s for s in said)
