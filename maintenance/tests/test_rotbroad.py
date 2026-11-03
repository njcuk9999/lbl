"""ROTBROAD in lbl.science.general.spline_template: the template splines that
compute_rv reads are broadened by the rotation kernel of the given vsini, with
or without odd/even columns and Savitzky-Golay derivatives."""
import numpy as np
import pytest
from astropy.table import Table

from lbl.core import io
from lbl.core import math as mp
from lbl.science import general

C_MS = mp.speed_of_light_ms
#: width (sigma) of the template line, in m/s
SIGMA = 3000.0
#: template grid step, in m/s
STEP = 250.0


class Quiet:
    def __getattr__(self, name):
        def nothing(*args, **kwargs):
            return None
        return nothing


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(general, 'log', Quiet())
    monkeypatch.setattr(io, 'log', Quiet())


def _template(savgol):
    """One Gaussian line at 600 nm on a grid of constant velocity step; the
    Savitzky-Golay columns are its exact derivatives per pixel"""
    vel = np.arange(-300e3, 300e3, STEP)
    table = Table()
    table['wavelength'] = 600.0 * np.exp(vel / C_MS)
    gauss = 0.5 * np.exp(-0.5 * (vel / SIGMA) ** 2)
    flux = 1 - gauss
    table['flux'] = flux
    for name in ('flux_odd', 'flux_even'):
        table[name] = flux
    if savgol:
        # d^n/dv^n of -gauss, times STEP^n (per pixel of the grid)
        xx = vel / SIGMA
        derivs = [flux, gauss * xx / SIGMA,
                  -gauss * (xx ** 2 - 1) / SIGMA ** 2,
                  gauss * (xx ** 3 - 3 * xx) / SIGMA ** 3]
        for name in ('flux', 'flux_odd', 'flux_even'):
            for order, deriv in enumerate(derivs):
                key = '{0}_savgol_d{1}'.format(name, order)
                table[key] = deriv * STEP ** order
    return table


class Inst:
    def __init__(self, table, rotbroad, savgol):
        self.table = table
        self.params = dict(USE_SAVGOL_TEMPLATE=savgol, ROTBROAD=rotbroad,
                           COMPUTE_FLAG_SPURIOUS_SIG=10, RESPROJ_TABLES=None,
                           HP_WIDTH=500)

    def load_template(self, filename, get_hdr=False):
        return self.table


def _line_variance(spline):
    vel = np.arange(-60e3, 60e3, 50.0)
    depth = 1 - spline(600.0 * np.exp(vel / C_MS))
    return np.sum(vel ** 2 * depth) / np.sum(depth)


@pytest.mark.parametrize('savgol', [False, True])
@pytest.mark.parametrize('rotbroad', [(0.6, 10.0), [0.6, 10.0]])
def test_the_splines_compute_rv_reads_are_broadened(savgol, rotbroad):
    """Variance of the line: SIGMA**2 + var of the kernel, which is
    ((1 - eps) / 4 + 2 eps / 15) / (1 - eps / 3) vsini**2 (0.225 vsini**2
    for eps = 0.6)"""
    eps, vsini = rotbroad[0], rotbroad[1] * 1000.0
    expected = SIGMA ** 2 + ((1 - eps) / 4 + 2 * eps / 15) / (1 - eps / 3) \
        * vsini ** 2
    plain = general.spline_template(Inst(_template(savgol), None, savgol),
                                    'template', 0.0, '.')
    sps = general.spline_template(Inst(_template(savgol), rotbroad, savgol),
                                  'template', 0.0, '.')
    assert abs(_line_variance(plain['spline_odd']) / SIGMA ** 2 - 1) < 0.01
    for key in ('spline', 'spline_odd', 'spline_even'):
        assert abs(_line_variance(sps[key]) / expected - 1) < 0.02
    # the derivative is that of the broadened line
    vel = np.arange(-40e3, 40e3, 50.0)
    wave = 600.0 * np.exp(vel / C_MS)
    numeric = np.gradient(sps['spline_odd'](wave), vel)
    deriv = sps['dspline_odd'](wave)
    assert np.max(np.abs(deriv - numeric)) < 0.02 * np.max(np.abs(numeric))
