"""The telluric exponents by the derivative method: its cost is the
scatter of the corrected spectrum's gradient IN UNITS OF ITS NOISE, which
follows the division by the transmission. A synthetic atmosphere of known
exponents, no TAPAS file."""
import numpy as np
from astropy import constants

from lbl.science import tellu_derivative as td

C = constants.c.to("km/s").value


def _atmosphere(seed=2):
    """Water and dry transmissions, functions of the wavelength [nm]: narrow
    lines, as TAPAS gives them before the instrument's resolution"""
    rng = np.random.default_rng(seed)
    water_lines = np.sort(rng.uniform(586.5, 599.5, 60))
    water_depth = rng.uniform(0.2, 0.9, water_lines.size)
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


def _spectrum(spl_water, spl_dry, water, dry, counts, seed=5,
              resolution=115000.0):
    """A flat star seen through the atmosphere at these exponents, with the
    photon noise of `counts` per pixel; its noise, sqrt(F)"""
    wave = 586.0 * np.exp(np.arange(0, np.log(600 / 586.0), 0.8 / C))
    model = td.model_grid(spl_water, spl_dry, wave.min() * 0.999,
                          wave.max() * 1.001, resolution)
    flux = counts * td.transmission(model, wave, water, dry)
    rng = np.random.default_rng(seed)
    observed = flux + rng.normal(0.0, 1.0, flux.size) * np.sqrt(flux)
    return wave, observed, np.sqrt(np.clip(observed, 1, None))


def test_the_noise_follows_the_division():
    trans = np.array([1.0, 0.5, 0.25, 1.0])
    noise = np.array([2.0, 2.0, 2.0, 2.0])
    sigma = noise / trans
    expected = np.array([np.hypot(sigma[0], sigma[1]),
                         np.hypot(sigma[2], sigma[0]) / 2,
                         np.hypot(sigma[3], sigma[1]) / 2,
                         np.hypot(sigma[3], sigma[2])])
    assert np.allclose(td.gradient_noise(trans, noise), expected)


def test_a_faint_star_has_its_exponents():
    # a faint star keeps its exponents: the noise the division by the
    #   transmission amplifies, counted in the cost, does not hide its
    #   minimum or send it to the fallback
    spl_water, spl_dry = _atmosphere()
    for counts in (900.0, 40000.0):            # SNR 30 and 200 per pixel
        wave, sp, noise = _spectrum(spl_water, spl_dry, 0.5, 1.15, counts)
        water, dry = td.fit_exponents(wave, sp, spl_water, spl_dry, 1.2,
                                      noise=noise)
        assert not td.is_fallback(water), counts
        assert abs(water - 0.5) < 0.15, (counts, water)
        assert abs(dry - 1.15) < 0.2, (counts, dry)


def test_no_tellurics_gives_no_water_and_dry_stays_physical():
    # a spectrum without telluric lines: the fit finds next to no water, or
    #   falls back when its minimum is the scan's first point
    spl_water, spl_dry = _atmosphere()
    wave, sp, noise = _spectrum(spl_water, spl_dry, 0.0, 0.0, 900.0)
    water, _ = td.fit_exponents(wave, sp, spl_water, spl_dry, 1.2,
                                noise=noise)
    assert td.is_fallback(water) or water < 0.03
    # the dry exponent never leaves DRY_RANGE times the airmass
    wave, sp, noise = _spectrum(spl_water, spl_dry, 0.5, 1.15, 40000.0)
    _, dry = td.fit_exponents(wave, sp, spl_water, spl_dry, 1.2, noise=noise)
    assert td.DRY_RANGE[0] * 1.2 < dry < td.DRY_RANGE[1] * 1.2
