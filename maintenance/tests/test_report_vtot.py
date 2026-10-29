"""The report's velocity series (lbl.report.figures.page_sequence):
coloured by V_tot, the star's velocity in the telluric frame, when the
systemic velocity is known, by the BERV otherwise; V_tot composed
relativistically."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
from lbl.report import figures                           # noqa: E402


def _table():
    rjd = np.linspace(60000.0, 60300.0, 40)
    berv = 25.0 * np.sin(2 * np.pi * (rjd - rjd[0]) / 365.25)
    vrad = 4630.0 + np.random.default_rng(1).normal(0, 3, rjd.size)
    # as series.read_rdb gives it: a dict of columns
    return dict(rjd=rjd, vrad=vrad, svrad=np.full(rjd.size, 3.0), BERV=berv)


def test_vtot_is_the_systemic_velocity_less_the_berv():
    vrad = np.array([4630.0, 4640.0])
    vtot = figures.total_velocity(vrad, np.array([4.635, -20.0]), 4.63)
    # V_sys + (vrad - median) - BERV, to well under a m/s at these speeds
    assert abs(vtot[0] - (4.63 - 0.005 - 4.635)) < 1e-6
    assert abs(vtot[1] - (4.63 + 0.005 + 20.0)) < 1e-3


def test_coloured_by_vtot_only_with_a_systemic_velocity():
    table = _table()
    fig = figures.page_sequence(plt, table, 'star', 4.63)
    assert fig.coloured_by == 'V_tot'
    plt.close(fig)
    fig = figures.page_sequence(plt, table, 'star', None)
    assert fig.coloured_by == 'BERV'
    plt.close(fig)


def test_the_colours_change_hue_at_zero():
    cmap = figures.vtot_cmap()
    below, above = cmap(0.499), cmap(0.501)
    # dark on both sides, blue below and red above
    assert below[2] > below[0] and above[0] > above[2]
    assert max(below[:3]) < 0.5 and max(above[:3]) < 0.5


def test_the_telluric_exponents_against_the_airmass():
    rng = np.random.default_rng(3)
    airmass = rng.uniform(1.0, 1.9, 30)
    water = 0.4 * airmass * rng.uniform(0.6, 1.5, 30)
    water[:3] = figures.WATER_FALLBACK            # three at the fallback
    table = dict(rjd=np.arange(30.0), vrad=np.zeros(30), svrad=np.ones(30),
                 TLPEH2O=water, TLPEOTR=0.96 * airmass + 0.05, AIRMASS=airmass)
    fig = figures.page_telluric_airmass(plt, table, 'star')
    labels = [t.get_text() for ax in fig.axes
              for t in ax.get_legend().get_texts()]
    assert 'epochs (27)' in labels and 'fallback 0.35 (3)' in labels
    assert any(label.startswith('fitted: 0.960 x airmass') for label in labels)
    plt.close(fig)
    # no exponents (no telluric correction): no page
    assert figures.page_telluric_airmass(plt, dict(rjd=[1.0]), 'star') is None
