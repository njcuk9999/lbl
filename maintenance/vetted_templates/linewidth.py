"""The line width of each candidate template, as the FWHM of its
autocorrelation in H band (1510-1750 nm): a rotating star's lines are
broader and its autocorrelation wider. Within one instrument, the stars
whose rotation periods say they are slow give the reference.
Usage: python linewidth.py  (reads apero/*.fits, writes linewidth.json)"""
import glob
import json
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

C = 299792.458


def width(path):
    tab = fits.getdata(path, 1)
    wave = np.asarray(tab["wavelength"], float)
    flux = np.asarray(tab["flux"], float)
    nval = np.asarray(tab["n_valid"], float)
    flux[nval < np.nanmax(nval) / 2] = np.nan
    keep = (wave > 1510) & (wave < 1750)
    wave, flux = wave[keep], flux[keep]
    dv = np.nanmedian(np.diff(np.log(wave))) * C
    good = np.isfinite(flux)
    # continuum: a running median over about 150 km/s
    box = int(150 / dv) | 1
    cont = median_filter(np.where(good, flux, np.nanmedian(flux)), box)
    y = np.where(good, flux / cont - 1.0, 0.0)
    y[good] -= np.mean(y[good])
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    lags = np.arange(n) * dv
    half = int(20 / dv)
    xs = np.concatenate([-lags[1:half][::-1], lags[:half]])
    ys = np.concatenate([acf[1:half][::-1], acf[:half]]) / acf[0]
    gauss = lambda x, a, s, b: a * np.exp(-0.5 * x ** 2 / s ** 2) + b
    (a, s, b), _ = curve_fit(gauss, xs, ys, p0=(1, 3, 0))
    return 2.3548 * abs(s), dv


out = {}
for path in sorted(glob.glob("apero/*.fits")):
    name = os.path.basename(path)
    fw, dv = width(path)
    out[name] = dict(fwhm_acf=fw, dv=dv)
json.dump(out, open("linewidth.json", "w"), indent=1)
sel = {("%s_%s" % (c["instrument"], c["file"])): c for c in json.load(open("selection.json"))["chosen"]}
for name, r in sorted(out.items(), key=lambda kv: (kv[0].split("_Template")[0], kv[1]["fwhm_acf"])):
    c = sel.get(name)
    tag = ("P=%.1f d" % c["prot"] if c and c["prot"] else "P=?") if c else "(passed over)"
    print("%-55s ACF FWHM %5.2f km/s  (step %.2f)  %s" % (name.replace("_sc1d_v_file", ""), r["fwhm_acf"], r["dv"], tag))
