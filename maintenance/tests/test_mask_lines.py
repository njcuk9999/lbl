"""The mask: its lines from a template without the savgol derivatives, the
systemic velocity of a mask against a model, and the names of the files
written."""
import os

import numpy as np
from astropy.table import Table

from lbl.core import io, math as mp, parameters
from lbl.instruments import default
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


def _model(velocity_kms, sigma_kms):
    """A model spectrum with lines at rest, and a mask of the same lines seen
    at a velocity (the template's frame)"""
    centres = np.sort(np.random.default_rng(6).uniform(602.0, 618.0, 30))
    wave = general.get_magic_grid(590.0, 630.0, 500.0)
    ratio = np.sqrt((1 + velocity_kms * 1000 / C_MS)
                    / (1 - velocity_kms * 1000 / C_MS))
    spectrum = _lines(wave, centres, 0.6, sigma_kms)
    table = Table(dict(ll_mask_s=centres * ratio, ll_mask_e=centres * ratio,
                       w_mask=np.ones(centres.size)))
    return table, wave, spectrum


def test_the_systemic_velocity_inside_and_on_the_edges_of_the_range():
    inst = _Inst()
    table, wave, spectrum = _model(37.0, 20.0)
    assert abs(general.mask_systemic_velocity(inst, table, wave, spectrum)
               - 37.0) < 0.5
    # a minimum on the first or the last point of the range: that edge, and
    #   no error
    for edge in (-200.0, 200.0):
        table, wave, spectrum = _model(edge, 3.0)
        assert general.mask_systemic_velocity(inst, table, wave,
                                              spectrum) == edge


def test_the_systemic_velocity_from_the_template_and_not_its_lines():
    """the template's flux against the model finds the star, where the
    lines' CCF is taken by a few spikes of weight (PROXIMA: three lines of a
    noisy end, near 1830 nm, with up to 28 000 times the median weight, and
    +157 km/s instead of -21): a template of 400 lines at -21.3 km/s, noise,
    a gap and a noisy end of huge spikes"""
    rng = np.random.default_rng(11)
    wave = general.get_magic_grid(1500.0, 1600.0, 500.0)
    centres = np.sort(rng.uniform(1502.0, 1598.0, 400))
    depth = rng.uniform(0.05, 0.5, 400)
    model = np.ones_like(wave)
    for centre, dep in zip(centres, depth):
        model *= _lines(wave, [centre], dep, 2.5)
    ratio = np.sqrt((1 - 21.3e3 / C_MS) / (1 + 21.3e3 / C_MS))
    flux = np.interp(wave, wave * ratio, model) * (
        1 + 0.003 * rng.normal(size=wave.size))
    flux[(wave > 1540.0) & (wave < 1545.0)] = np.nan
    end = wave > 1590.0
    flux[end] *= np.exp(rng.normal(size=end.sum()) * 0.8)
    template = Table(dict(wavelength=wave, flux=flux))
    velocity, dvs, ccf, signif = general.template_systemic_velocity(
        wave, flux, wave, model)
    assert abs(velocity - (-21.3)) < 0.3 and signif > 10
    assert np.isclose(ccf.max(), 1.0) and abs(dvs[np.argmax(ccf)]
                                                 + 21.3) < 0.6
    # through mask_systemic_velocity, with lines whose weights a spike took
    lines = Table(dict(ll_mask_s=[1595.0, 1500.5], ll_mask_e=[1595.0, 1500.5],
                       w_mask=[3e4, 1.0]))
    assert abs(general.mask_systemic_velocity(_Inst(), lines, wave, model,
                                              template) + 21.3) < 0.3


def test_the_mask_type_is_replaced_in_the_file_name_only(tmp_path):
    # a folder and an object whose names contain the mask type ('full')
    folder = tmp_path / 'fullrun' / 'masks'
    folder.mkdir(parents=True)
    mask_file = str(folder / 'LBL_Mask_fullmoon_full_default.fits')
    inst = default.Instrument('Default')
    inst.params = parameters.params.copy()
    inst.params.update(DATA_TYPE='SCIENCE', INSTRUMENT='Default',
                       DATA_SOURCE='None', OBJECT_SCIENCE='fullmoon',
                       OBJECT_COMPARISON='fullmoon')
    table = Table(dict(ll_mask_s=[600.0, 601.0], ll_mask_e=[600.0, 601.0],
                       w_mask=[1.0, -1.0]))
    inst.write_mask(mask_file, table, np.array([False, True]),
                    np.array([True, False]), 12.3, io.LBLHeader())
    written = sorted(os.listdir(folder))
    assert written == ['LBL_Mask_fullmoon_full_default.fits',
                       'LBL_Mask_fullmoon_neg_default.fits',
                       'LBL_Mask_fullmoon_pos_default.fits']


def test_only_the_photometric_bandpasses_count():
    """beyond H's red end (1809 nm) the template's lines are at another
    velocity, and deeper: the photometric bandpasses alone are used, and the
    star is found where its in-band lines are"""
    rng = np.random.default_rng(12)
    wave = general.get_magic_grid(1700.0, 1850.0, 500.0)
    centres = np.sort(rng.uniform(1702.0, 1848.0, 500))
    depth = np.where(centres > 1812.0, 0.6, 0.15)
    model = np.ones_like(wave)
    for centre, dep in zip(centres, depth):
        model *= _lines(wave, [centre], dep, 2.5)
    flux = np.empty_like(wave)
    for v, sel in ((-21.3, wave <= 1809.0), (150.0, wave > 1809.0)):
        ratio = np.sqrt((1 + v * 1e3 / C_MS) / (1 - v * 1e3 / C_MS))
        flux[sel] = np.interp(wave[sel], wave * ratio, model)
    flux *= 1 + 0.002 * rng.normal(size=wave.size)
    velocity, _, _, _ = general.template_systemic_velocity(wave, flux, wave,
                                                           model)
    assert abs(velocity + 21.3) < 0.3


def test_lines_of_excessive_weight_are_cut():
    """above 50 times the median weight for significant extrema (10 for the
    lines between maxima), a line is a spike and is cut; the strong real
    lines, up to about 30 times, stay"""
    weights = np.r_[np.ones(1000), 30.0, -40.0, 55.0, 3e4, -1200.0]
    table = Table(dict(ll_mask_s=np.arange(weights.size, dtype=float),
                       ll_mask_e=np.arange(weights.size, dtype=float),
                       w_mask=weights))
    kept, n_cut, limit = general.cut_excessive_weights(table, True)
    assert n_cut == 3 and np.isclose(limit, 50.0)
    assert set(np.abs(kept['w_mask'])) == {1.0, 30.0, 40.0}
    kept, n_cut, _ = general.cut_excessive_weights(table, False)
    assert n_cut == 5 and np.all(np.abs(kept['w_mask']) == 1.0)
