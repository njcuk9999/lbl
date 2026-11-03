"""The running median of lbl.core.math.lowpassfilter: each box is placed at the
median position of its finite pixels, the last pixel of the vector is read, and
the numba kernel (lbl.core.fastmath.lowpass_windows) gives the values of the
plain loop."""
import numpy as np
import pytest

from lbl.core import fastmath
from lbl.core import math as mp


@pytest.fixture
def slow(monkeypatch):
    """lowpassfilter through its plain Python loop"""
    def run(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(mp, '_lowpass_fast_ok', lambda *a: False)
            return mp.lowpassfilter(*args, **kwargs)
    return run


def _ramp(npix=3000):
    return 0.01 * np.arange(npix, dtype=float)


@pytest.mark.parametrize('gap', [2, 5, 20, 200])
def test_a_ramp_with_a_gap_is_followed_exactly(gap, slow):
    ramp = _ramp()
    vect = ramp.copy()
    vect[1500:1500 + gap] = np.nan
    inner = slice(100, len(ramp) - 100)
    for func in (mp.lowpassfilter, slow):
        out = func(vect, 101)
        assert np.max(np.abs(out - ramp)[inner]) < 1e-10


def test_a_ramp_with_random_holes_is_followed_exactly():
    ramp = _ramp()
    vect = ramp.copy()
    vect[np.random.default_rng(4).random(len(vect)) < 0.3] = np.nan
    out = mp.lowpassfilter(vect, 101, k=3)
    assert np.max(np.abs(out - ramp)[100:-100]) < 1e-10


def test_the_last_pixel_is_read():
    """Only the last three pixels are finite: one box can use them"""
    vect = np.full(40, np.nan)
    vect[-3:] = [1.0, 2.0, 3.0]
    xmed, ymed = fastmath.lowpass_windows(vect, 8)
    assert len(xmed) > 0
    assert np.all(xmed == 38.0)
    assert np.all(ymed == 2.0)


def test_kernel_and_plain_loop_give_the_same_values(slow):
    rng = np.random.default_rng(7)
    for _ in range(20):
        npix = int(rng.integers(50, 2000))
        vect = rng.normal(1.0, 0.1, npix)
        vect[rng.random(npix) < rng.uniform(0, 0.6)] = np.nan
        if rng.random() < 0.5:
            start = int(rng.integers(0, npix))
            vect[start:start + int(rng.integers(1, 300))] = np.nan
        width = int(rng.integers(5, 300))
        fast = mp.lowpassfilter(vect, width)
        plain = slow(vect, width)
        assert np.array_equal(fast, plain, equal_nan=True)
