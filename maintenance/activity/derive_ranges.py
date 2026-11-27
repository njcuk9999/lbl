"""
The windows and pseudo-continuum ranges of the activity lines no
high-resolution study defines: the lines of lbl/resources/activity_lines.yaml
with a `derived` entry, by the rule of the catalog's header and of
lbl/doc/ACTIVITY.md.

    derived: {window_like: <line>, ranges_like: <line>}

WINDOW (window_like): the half width in velocity of that line's window,
    about this line's centre (wave). Two windows of the catalog that
    overlap are cut at the midpoint of the two lines' centres.
RANGES (ranges_like): on each side, a range of the half width in velocity
    of that line's first range; its centre searched outward from the
    window's edge (a gap of one half width at least between them, so that
    the line's wings stay out) in steps of a quarter of that half width, up
    to SEARCH of the centre; kept: the one where PHOENIX models (TEFFS, log g 4.5,
    solar metallicity, at resolution R) are closest to their local upper
    envelope (the 90th percentile of the model in the range over its 99th
    percentile within ENVELOPE_HALF of the range's centre, the mean of the
    models), among the ranges that overlap no window of the catalog and
    where TAPAS (airmass 1) stays above MIN_TRANSMISSION over the range
    widened by +-BERV_SMEAR; the closest of equal ones. Beyond the TAPAS
    file (1850 nm), the tellurics are not checked.

Run by hand when the catalog changes, its output written into the catalog:

    python maintenance/activity/derive_ranges.py MODELS_DIR TAPAS_FILE

MODELS_DIR holds the PHOENIX files (lte03500-4.50-0.0...-HiRes*.fits[.gz])
and their wavelength file, in vacuum (as LBL uses them); TAPAS_FILE is
LBL's tapas_lbl.fits (WAVELENGTH, ABSO_OTHERS, ABSO_WATER, nm).
"""
import glob
import os
import sys

import numpy as np
import yaml
from astropy import constants
from astropy.io import fits
from astropy.table import Table

from lbl.science import activity_lines

C_KMS = constants.c.to('km/s').value
TEFFS = (3000, 3400, 3800)
R = 70000
SEARCH = 0.0025
BERV_SMEAR = 30.0
MIN_TRANSMISSION = 0.95
ENVELOPE_HALF = 2.5
#: samples per resolution element of the smoothed models
OVERSAMPLE = 6


def doppler(v):
    b = v / C_KMS
    return np.sqrt((1 + b) / (1 - b))


def velocity(ratio):
    return C_KMS * (ratio ** 2 - 1) / (ratio ** 2 + 1)


def half_velocity(lo, hi):
    """half the velocity width of [lo, hi]"""
    return velocity(np.sqrt(hi / lo))


def load_models(folder):
    models = []
    for teff in TEFFS:
        found = sorted(glob.glob(os.path.join(
            folder, 'lte%05d-4.50-0.0.PHOENIX*' % teff)))
        found = [f for f in found if '350-2550' in f] or found
        tag = '350-2550' in found[0]
        wfile = [w for w in sorted(glob.glob(os.path.join(
            folder, 'WAVE_PHOENIX*'))) if ('350-2550' in w) == tag][0]
        models.append((np.asarray(fits.getdata(wfile), float) / 10.0,
                       fits.getdata(found[0]).astype(float)))
    return models


def smooth(wave, flux, lo, hi):
    """the model between lo and hi [nm], at resolution R"""
    sel = (wave > lo) & (wave < hi)
    grid = np.exp(np.arange(np.log(lo), np.log(hi), 1 / (R * OVERSAMPLE)))
    fgrid = np.interp(grid, wave[sel], flux[sel])
    sigma = OVERSAMPLE / (2 * np.sqrt(2 * np.log(2)))
    x = np.arange(-int(5 * sigma), int(5 * sigma) + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return grid, np.convolve(fgrid, kernel / kernel.sum(), mode='same')


def main(models_dir, tapas_file, catalog=activity_lines.CATALOG):
    lines = yaml.safe_load(open(catalog))['lines']
    # the windows: published, or like another line's in velocity
    windows = {}
    for name, line in lines.items():
        spec = line.get('derived') or {}
        if 'window_like' in spec:
            lo, hi = lines[spec['window_like']]['window']
            v = half_velocity(lo, hi)
            windows[name] = [line['wave'] / doppler(v),
                             line['wave'] * doppler(v)]
        else:
            windows[name] = list(line['window'])
    # windows that overlap: cut at the midpoint of the two centres
    names = sorted(lines, key=lambda n: lines[n]['wave'])
    for a, b in zip(names[:-1], names[1:]):
        if windows[a][1] > windows[b][0]:
            mid = 0.5 * (lines[a]['wave'] + lines[b]['wave'])
            print('# %s and %s overlap: cut at %.4f' % (a, b, mid))
            windows[a][1], windows[b][0] = mid, mid
    models = load_models(models_dir)
    tap = Table.read(tapas_file)
    order = np.argsort(np.array(tap['WAVELENGTH'], float))
    tw = np.array(tap['WAVELENGTH'], float)[order]
    tt = (np.array(tap['ABSO_OTHERS'], float) *
          np.array(tap['ABSO_WATER'], float))[order]
    for name, line in lines.items():
        spec = line.get('derived') or {}
        lo, hi = windows[name]
        ranges, notes = line.get('continuum'), []
        if 'ranges_like' in spec:
            first = lines[spec['ranges_like']]['continuum'][0]
            v_pc = half_velocity(*first)
            half = line['wave'] * (doppler(v_pc) - 1)
            span = line['wave'] * SEARCH
            grids = [smooth(w, f, lo - span - 3 * ENVELOPE_HALF,
                            hi + span + 3 * ENVELOPE_HALF) for w, f in models]
            checked = tw[0] < lo - span and hi + span < tw[-1]
            ranges = []
            for side in (-1, 1):
                edge = lo if side < 0 else hi
                best = None
                for step in np.arange(2 * half, span, half / 4):
                    centre = edge + side * step
                    a, b = centre - half, centre + half
                    if any(a < w[1] and b > w[0] for w in windows.values()):
                        continue
                    if checked:
                        sel = ((tw > a / doppler(BERV_SMEAR)) &
                               (tw < b * doppler(BERV_SMEAR)))
                        if tt[sel].min() < MIN_TRANSMISSION:
                            continue
                    score = np.mean([
                        np.percentile(gf[(gw > a) & (gw < b)], 90) /
                        np.percentile(gf[np.abs(gw - centre) <
                                         ENVELOPE_HALF], 99)
                        for gw, gf in grids])
                    if best is None or score > best[0] + 1e-4:
                        best = (score, a, b)
                ranges.append([round(float(best[1]), 4),
                               round(float(best[2]), 4)])
                notes.append('%.4f' % best[0])
            notes.append('TAPAS checked' if checked else
                         'beyond TAPAS, not checked')
        if spec or windows[name] != list(line['window'] or []):
            print('  %s:' % name)
            print('    window: [%.4f, %.4f]' % (lo, hi))
            print('    continuum: %s' % ranges)
            if notes:
                print('    # envelope scores %s' % ', '.join(notes))


if __name__ == '__main__':
    main(*sys.argv[1:])
