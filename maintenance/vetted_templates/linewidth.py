"""The line width of each candidate template, as the FWHM of its
autocorrelation in H band (1510-1750 nm): a rotating star's lines are
broader and its autocorrelation wider. Within one instrument, the stars
whose rotation periods say they are slow give the reference
(select_vetted_stars.py).

Reads the APERO templates copied into apero/ (named <INSTRUMENT>_<APERO
file name>) and, for the tags it prints, the previous selection.json;
writes linewidth.json (per template: fwhm_acf, the autocorrelation FWHM
[km/s], and dv, the sampling step [km/s]).

    python linewidth.py
"""
import glob
import json
import os
import numpy as np
from astropy import constants
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

C = constants.c.to("km/s").value
#: the FWHM of a Gaussian over its sigma, 2 sqrt(2 ln 2)
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def width(path):
    """(autocorrelation FWHM [km/s], sampling step [km/s]) of the template
    at `path`, in H band"""
    tab = fits.getdata(path, 1)
    wave = np.asarray(tab["wavelength"], float)
    flux = np.asarray(tab["flux"], float)
    nval = np.asarray(tab["n_valid"], float)
    # pixels that fewer than half the spectra cover are not trusted
    flux[nval < np.nanmax(nval) / 2] = np.nan
    keep = (wave > 1510) & (wave < 1750)
    wave, flux = wave[keep], flux[keep]
    # the s1d is on a uniform log grid: one pixel is a constant step in km/s
    dv = np.nanmedian(np.diff(np.log(wave))) * C
    good = np.isfinite(flux)
    # continuum: a running median over about 150 km/s
    box = int(150 / dv) | 1
    cont = median_filter(np.where(good, flux, np.nanmedian(flux)), box)
    # the lines alone, zero mean, the missing pixels at zero
    y = np.where(good, flux / cont - 1.0, 0.0)
    y[good] -= np.mean(y[good])
    # autocorrelation by FFT, zero-padded to twice the length so that it
    #   does not wrap around
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    lags = np.arange(n) * dv
    # a Gaussian fitted to the central +-20 km/s of the autocorrelation,
    #   mirrored to negative lags and normalised to its zero lag
    half = int(20 / dv)
    xs = np.concatenate([-lags[1:half][::-1], lags[:half]])
    ys = np.concatenate([acf[1:half][::-1], acf[:half]]) / acf[0]
    gauss = lambda x, a, s, b: a * np.exp(-0.5 * x ** 2 / s ** 2) + b
    (a, s, b), _ = curve_fit(gauss, xs, ys, p0=(1, 3, 0))
    return FWHM_PER_SIGMA * abs(s), dv


out = {}
for path in sorted(glob.glob("apero/*.fits")):
    name = os.path.basename(path)
    fw, dv = width(path)
    out[name] = dict(fwhm_acf=fw, dv=dv)
json.dump(out, open("linewidth.json", "w"), indent=1)
# the widths printed per instrument, narrowest first, with the period of
#   the stars the previous selection chose
sel = {("%s_%s" % (c["instrument"], c["file"])): c for c in json.load(open("selection.json"))["chosen"]}
for name, r in sorted(out.items(), key=lambda kv: (kv[0].split("_Template")[0], kv[1]["fwhm_acf"])):
    c = sel.get(name)
    tag = ("P=%.1f d" % c["prot"] if c and c["prot"] else "P=?") if c else "(passed over)"
    print("%-55s ACF FWHM %5.2f km/s  (step %.2f)  %s" % (name.replace("_sc1d_v_file", ""), r["fwhm_acf"], r["dv"], tag))
