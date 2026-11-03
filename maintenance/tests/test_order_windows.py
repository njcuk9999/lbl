"""maintenance/windows/order_windows.py gives back what it proposed: a window
it moved off a stellar line to the middle of the line's order is still that
line's window on the next run.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_order_windows.py
"""
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "order_windows", os.path.join(HERE, "..", "windows", "order_windows.py"))
order_windows = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(order_windows)


def _spans():
    """Two orders, the sodium doublet on the red flank of the first (its
    blaze peaks at 586.6 nm)."""
    out = []
    for first, last, peak in ((583.0, 590.2, 586.6), (590.0, 597.2, 593.6)):
        wave = np.linspace(first, last, 4000)
        blaze = np.exp(-0.5 * ((wave - peak) / 2.2) ** 2)
        out.append((wave, blaze / blaze.max()))
    return out


def _tapas():
    wave = np.linspace(500.0, 700.0, 20001)
    return wave, np.ones(wave.size)


def test_a_moved_stellar_line_window_comes_back_the_same():
    spans, tapas = _spans(), _tapas()
    best, keep, moved = order_windows.propose(spans, tapas, 0.0, 1e4, 589.0,
                                              1.5)
    assert keep is not None and moved == 589.3
    proposed = best[1]
    assert abs(proposed - 586.6) < 0.2
    # the tool's own output, run again: the same line, the same window
    again, keep, moved = order_windows.propose(spans, tapas, 0.0, 1e4,
                                               proposed, 1.5)
    assert keep is not None and moved == 589.3
    assert again[1] == proposed and again[2] == 1.5


def test_a_telluric_window_of_another_order_is_not_taken_for_one():
    spans = _spans()
    assert order_windows.stellar_line(spans, 594.25) is None
    assert order_windows.stellar_line(spans, 589.6) == 589.3
    assert order_windows.stellar_line(spans, 586.5) == 589.3
