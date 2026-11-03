"""The mask: its lines from a template without the savgol derivatives."""
import numpy as np
from astropy.table import Table

from lbl.core import math as mp
from lbl.science import general

C_MS = mp.speed_of_light_ms


class _Inst:
    """What these functions read of an instrument: its parameters"""

    def __init__(self, **params):
        self.params = dict(USE_TQDM=False, PLOT=False,
                           OBJECT_COMPARISON='TEST')
        self.params.update(params)


def _lines(wave, centres, depth, sigma_kms):
    tau = np.zeros_like(wave)
    for centre in centres:
        tau += -np.log(1 - depth) * np.exp(
            -0.5 * (np.log(wave / centre) * C_MS / 1000 / sigma_kms) ** 2)
    return np.exp(-tau)


def test_lines_without_savgol_have_the_flux_and_a_weight_everywhere():
    # a template on a 0.5 km/s grid, with noise and points missing: the
    #   flux of a line is the template's (a mean of 5 points, not their
    #   sum), and every line found has a weight
    rng = np.random.default_rng(4)
    wave = general.get_magic_grid(600.0, 606.0, 500.0)
    centres = np.sort(rng.uniform(600.2, 605.8, 60))
    flux = _lines(wave, centres, 0.5, 3.0) * (1 + 0.002 * rng.normal(size=wave.size))
    flux[rng.choice(wave.size, 40, replace=False)] = np.nan
    table = Table(dict(wavelength=wave, flux=flux,
                       rms=np.full(wave.size, 0.002)))
    lines = general.find_mask_lines(_Inst(USE_SAVGOL_TEMPLATE=False), table)
    assert len(lines) > 100
    assert np.all(np.isfinite(lines['w_mask']))
    assert 0.9 < np.median(lines['value'][lines['w_mask'] < 0]) < 1.05
    # the minima: the flux at a line's centre, 1 - depth or deeper where
    #   two lines blend
    minima = lines['w_mask'] > 0
    assert np.all(lines['value'][minima] > 0.2)
    assert np.all(lines['value'][minima] < 1.05)
