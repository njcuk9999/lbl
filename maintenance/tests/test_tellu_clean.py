"""The transmission correct_tellu divides out of a spectrum: the model its
exponents were fitted with, at the instrument's APPROX_RESOLUTION, defined
over the whole spectrum. A synthetic atmosphere, no TAPAS file."""
import numpy as np
from astropy import constants

from lbl.science import tellu_clean, tellu_derivative as td

C = constants.c.to("km/s").value
RESOLUTION = 140000.0


def _atmosphere(seed=2):
    """Water and dry transmissions, functions of the wavelength [nm]: narrow
    lines from 586.5 to 599.5 nm, as TAPAS gives them before the
    instrument's resolution"""
    rng = np.random.default_rng(seed)
    water_lines = np.sort(rng.uniform(586.5, 599.5, 60))
    water_depth = rng.uniform(0.05, 0.5, water_lines.size)
    dry_lines = np.sort(rng.uniform(586.5, 599.5, 15))
    dry_depth = rng.uniform(0.1, 0.5, dry_lines.size)

    def lines(centres, depths):
        def transmission(wave):
            wave = np.asarray(wave, dtype=float)
            tau = np.zeros_like(wave)
            for centre, depth in zip(centres, depths):
                width = centre * 1.5 / C
                tau += -np.log(1 - depth) * np.exp(
                    -0.5 * ((wave - centre) / width) ** 2)
            return np.exp(-tau)
        return transmission
    return lines(water_lines, water_depth), lines(dry_lines, dry_depth)


class _Inst:
    """What correct_tellu reads of an instrument: its parameters (those of the
    instrument classes for the kernel and the wavelength limits, which the
    division does not use)"""

    def __init__(self):
        self.params = dict(TELLUCLEAN_FORCE_AIRMASS=False,
                           APPROX_RESOLUTION=RESOLUTION,
                           TELLUCLEAN_TRANSMISSION_THRESHOLD=-1,
                           TELLUCLEAN_USE_TEMPLATE=False,
                           TELLUCLEAN_KERNEL_WID=1.4,
                           TELLUCLEAN_GAUSSIAN_SHAPE=2.2,
                           TELLUCLEAN_WAVE_LOWER=350,
                           TELLUCLEAN_WAVE_UPPER=750,
                           USE_SAVGOL_TEMPLATE=True, PLOT=False)


def _wave_map(start=570.0, stop=792.0, nord=10, step_kms=0.8):
    """Orders uniform in log wavelength, overlapping by 1 nm, from `start`
    past 750 nm (ESPRESSO goes to 790.8 nm)"""
    edges = np.linspace(start, stop, nord + 1)
    npix = int(np.log((edges[1] + 1) / edges[0]) / np.arctanh(step_kms / C)) + 1
    wave = np.zeros((nord, npix))
    for order in range(nord):
        wave[order] = edges[order] * np.exp(
            np.arange(npix) * np.arctanh(step_kms / C))
    return wave


def test_the_map_is_the_fit_model():
    # the transmission on any range is the fit's own model, at the same
    #   grid points: on the fit's pixels, the same to rounding
    spl_water, spl_dry = _atmosphere()
    wave = 586.0 * np.exp(np.arange(0, np.log(600 / 586.0), 0.8 / C))
    model = td.model_grid(spl_water, spl_dry, wave.min() * 0.999,
                          wave.max() * 1.001, RESOLUTION,
                          bands=td._spans_of(wave))
    fitted = td.transmission(model, wave, 0.7, 1.1)
    mapped = td.transmission_map(spl_water, spl_dry, wave, 0.7, 1.1,
                                 RESOLUTION)
    assert fitted.min() < 0.7
    assert np.allclose(mapped, fitted, rtol=0, atol=1e-8)
    # any shape, NaN where the wavelength is not finite
    grid = np.array([wave[:10], wave[10:20]])
    grid[1, 3] = np.nan
    out = td.transmission_map(spl_water, spl_dry, grid, 0.7, 1.1, RESOLUTION)
    assert out.shape == grid.shape
    assert np.isnan(out[1, 3]) and np.isfinite(np.delete(out.ravel(), 13)).all()


def test_the_division_is_by_the_fitted_model_and_reaches_past_750_nm():
    # a flat star seen through the atmosphere at the instrument's resolution,
    #   SNR 300: the transmission divided out is the fit's model at the
    #   exponents found, the true one to the error of those exponents, and
    #   it has a value at every pixel, also past TELLUCLEAN_WAVE_UPPER
    spl_water, spl_dry = _atmosphere()
    wave = _wave_map()
    truth = td.transmission_map(spl_water, spl_dry, wave, 0.6, 1.1,
                                RESOLUTION)
    flux = 9e4 * truth
    flux = flux + np.random.default_rng(3).normal(size=flux.shape) * np.sqrt(flux)
    e2ds = dict(flux=np.array(flux), wavelength=wave, AIRMASS=1.1,
                OBJECT='TEST', BERV=0.0, BERV_TRANS=0.0)
    out = tellu_clean.correct_tellu(_Inst(), '', e2ds, spl_dry, spl_water, '')
    water = out['pre_cleaned_exponent_water']
    dry = out['pre_cleaned_exponent_others']
    assert abs(water / 0.6 - 1) < 0.02 and abs(dry / 1.1 - 1) < 0.02
    abso = out['pre_cleaned_abso']
    model = td.transmission_map(spl_water, spl_dry, wave, water, dry,
                                RESOLUTION)
    usable = np.isfinite(abso)
    assert np.allclose(abso[usable], model[usable], rtol=0, atol=1e-12)
    lines = (truth < 0.9) & (truth > 0.5)
    assert lines.sum() > 200
    assert np.max(np.abs(truth[lines] / abso[lines] - 1)) < 0.01
    # past 750 nm: every pixel, no NaN
    red = wave > 750
    assert red.sum() > 10000
    assert np.all(np.isfinite(abso[red]))
    assert np.all(np.isfinite(out['pre_cleaned_flux'][red]))
