"""
Where PCA2D's river-plot windows (output.windows of nominal.yaml) fall in
the orders of each instrument, and better ones: deep in the tellurics, in
the middle of an order's blaze.

For each instrument, the wavelength and the blaze of every order come from
a real file, read as PCA2D reads it (lbl.pca2d.readers); the tellurics from
LBL's TAPAS model (tapas_lbl.fits, water and the other molecules). A
window's THROUGHPUT is the lowest blaze inside it, relative to the peak of
the order that holds it best (the blaze smoothed, so that a line or a dead
column does not decide); its DEPTH is the share of its samples under 0.98
of transmission and its deepest transmission. Every window of the same width
within SEARCH nm of a configured one, inside the instrument's PCA2D domain,
is scored, and the best with a throughput of at least BLAZE_FLOOR is
proposed.

THE BLAZE, NOT THE MIDDLE OF THE ORDER (2026-10-25). The first placement
used the order's geometric centre, which is not where its blaze peaks: on
HARPS the sodium window, 588.25-589.75 nm, had a "centrality" of 0.62 and
sat on the red flank of order 56, its blaze falling from 0.75 to 0.45 of
the peak (586.6 nm). A river plot drawn there shows a noisier spectrum than
the method gives, which misleads on its quality. A window on a stellar line
that cannot reach BLAZE_FLOOR is now moved to the middle of its order's
blaze, and says so.

    python order_windows.py <models folder with tapas_lbl.fits> \
        INSTRUMENT:SOURCE:file [INSTRUMENT:SOURCE:file ...]
"""
import sys

import numpy as np
import yaml
from astropy.io import fits


#: how far from a configured window a better one is looked for [nm]
SEARCH = 15.0
#: the least throughput a proposed window has: its lowest blaze over the
#:    peak of its order
BLAZE_FLOOR = 0.8
#: the blaze is smoothed over this share of an order before its peak is read
SMOOTH = 0.05
#: a sample is in a telluric line under this transmission
ABSORBED = 0.98
#: the narrowest window proposed [nm]
MIN_WIDTH = 1.5
#: the windows on a STELLAR line, there to show that the correction leaves
#:    it alone (the sodium doublet, the magnesium triplet): kept on it, only
#:    moved this much [nm] to sit better in an order
KEEP = {589.3: 0.3, 517.5: 0.5}


def orders(instrument, source, path, models=None):
    """[(wave, blaze)] of every order, the blaze smoothed and divided by its
    peak, read as PCA2D reads the file (its blaze file looked for in calib/
    beside `models`, and in `models`)"""
    from scipy.ndimage import median_filter

    from lbl.pca2d import readers
    spectrum = readers.read(path, instrument, source, model_dir=models)
    if np.allclose(np.nanmax(np.atleast_2d(spectrum['blaze']), axis=1), 1.0) \
            and np.allclose(np.nanmin(np.atleast_2d(spectrum['blaze']),
                                      axis=1), 1.0):
        raise SystemExit('%s: no blaze found for %s (a flat 1), so no window'
                         ' can be placed by it' % (instrument, path))
    waves = np.atleast_2d(np.asarray(spectrum['wave'], dtype=float))
    blazes = np.atleast_2d(np.asarray(spectrum['blaze'], dtype=float))
    out = []
    for wave, blaze in zip(waves, blazes):
        good = np.isfinite(wave) & (wave > 0) & np.isfinite(blaze)
        if good.sum() <= 10:
            continue
        wave, blaze = wave[good], blaze[good]
        size = max(3, int(SMOOTH * wave.size) | 1)
        smooth = median_filter(blaze, size=size, mode='nearest')
        # the peak read away from the order's ends, where a blaze file can
        #   carry a spike (HARPS order 43: 1.0 on its first pixel, against
        #   0.92 in its middle)
        edge = max(1, int(SMOOTH * wave.size))
        peak = np.nanmax(smooth[edge:-edge])
        if not peak > 0:
            continue
        order = np.argsort(wave)
        out.append((wave[order], np.clip(smooth / peak, 0.0, 1.0)[order]))
    return out


def throughput(spans, lo, hi):
    """The lowest blaze in [lo, hi], over its order's peak, in the order
    that holds it best; -1 when no order holds it whole"""
    best = -1.0
    for wave, blaze in spans:
        if wave[0] <= lo and hi <= wave[-1]:
            inside = (wave >= lo) & (wave <= hi)
            if inside.any():
                best = max(best, float(np.min(blaze[inside])))
    return best


def blaze_middle(spans, lo, hi):
    """The wavelength where the blaze peaks, in the order that holds the
    middle of [lo, hi]; None when no order does"""
    mid = 0.5 * (lo + hi)
    held = [(abs(mid - 0.5 * (w[0] + w[-1])), w, b) for w, b in spans
            if w[0] <= mid <= w[-1]]
    if not held:
        return None
    _, wave, blaze = min(held, key=lambda one: one[0])
    edge = max(1, int(SMOOTH * wave.size))
    return float(wave[edge + int(np.argmax(blaze[edge:-edge]))])


def tellurics(tapas, lo, hi):
    """(share of samples absorbed, deepest transmission) in [lo, hi]"""
    wave, trans = tapas
    i0, i1 = np.searchsorted(wave, [lo, hi])
    part = trans[i0:i1]
    if part.size == 0:
        return 0.0, 1.0
    return float(np.mean(part < ABSORBED)), float(part.min())


def domain(instrument):
    """the instrument's PCA2D domain [nm] and its windows, from nominal.yaml"""
    import os
    import lbl.pca2d
    nominal = yaml.safe_load(open(os.path.join(os.path.dirname(
        lbl.pca2d.__file__), 'nominal.yaml')))
    family = {'NIRPS_HE': 'NIRPS', 'NIRPS_HA': 'NIRPS', 'HARPSN': 'HARPN'}.get(
        instrument, instrument)
    mine = (nominal.get('instruments') or {}).get(family) or {}
    general = nominal.get('general') or nominal
    dom = dict((general.get('domain') or {}), **(mine.get('domain') or {}))
    windows = ((mine.get('output') or {}).get('windows')
               or (general.get('output') or {}).get('windows') or [])
    return (float(dom.get('wave_min', 0)), float(dom.get('wave_max', 1e9)),
            [tuple(float(x) for x in str(w).split(':')) for w in windows])


def main(models, *specs):
    with fits.open(models.rstrip('/') + '/tapas_lbl.fits') as hdul:
        tab = hdul['ABSOSPEC'].data
        wave = np.asarray(tab['WAVELENGTH'], float)
        trans = np.asarray(tab['ABSO_OTHERS'], float) * np.asarray(
            tab['ABSO_WATER'], float)
    order = np.argsort(wave)
    tapas = (wave[order], trans[order])
    for spec in specs:
        # INSTRUMENT:SOURCE:file[:OTHER], OTHER an instrument whose windows
        #   to start from (HARPS-N has none of its own)
        parts = spec.split(':')
        instrument, source = parts[0], parts[1]
        seeds = None
        if len(parts) > 3 and parts[-1].isupper() and '/' not in parts[-1]:
            seeds = parts[-1]
            path = ':'.join(parts[2:-1])
        else:
            path = ':'.join(parts[2:])
        spans = orders(instrument, source, path, models)
        wmin, wmax, windows = domain(instrument)
        if seeds:
            windows = domain(seeds)[2]
        print('\n%s (%s): %d orders, %.1f-%.1f nm; PCA2D domain %.0f-%.0f nm'
              % (instrument, source, len(spans), min(w[0] for w, _ in spans),
                 max(w[-1] for w, _ in spans), wmin, wmax))
        for centre, width in windows:
            lo, hi = centre - width / 2, centre + width / 2
            cen = throughput(spans, lo, hi)
            share, deepest = tellurics(tapas, lo, hi)
            best = None
            # the same width, or narrower where the orders are short (the
            #   visible: 5 to 8 nm an order on HARPS), never under MIN_WIDTH
            widths = sorted({max(MIN_WIDTH, round(width * f * 4) / 4)
                             for f in (1.0, 0.75, 0.5)}, reverse=True)
            # by nearness: the windows were moved since KEEP was written
            #   (589.3 is now 589.0, 517.5 is 518.0)
            keep = next((reach for line, reach in KEEP.items()
                         if abs(line - centre) < 1.0), None)
            if keep is not None:
                widths = [width]
            reach = SEARCH if keep is None else keep
            for w in widths:
                for c in np.arange(centre - reach, centre + reach + 0.01,
                                   0.05 if keep is not None else 0.25):
                    a, b = c - w / 2, c + w / 2
                    if a < wmin or b > wmax:
                        continue
                    cc = throughput(spans, a, b)
                    if cc < BLAZE_FLOOR:
                        continue
                    sh, dp = tellurics(tapas, a, b)
                    if keep is not None:
                        # on the stellar line: the best centred, nearest
                        score = (round(cc, 3), -abs(c - centre))
                        if best is None or score > best[0]:
                            best = (score, c, w, cc, sh, dp)
                        continue
                    # the telluric content (a narrower window counts less),
                    #   then the throughput, then nearness
                    score = (round(sh * (1 - dp) * np.sqrt(w / width), 3), cc,
                             -abs(c - centre))
                    if best is None or score > best[0]:
                        best = (score, c, w, cc, sh, dp)
            # A STELLAR LINE THAT CANNOT SIT IN THE MIDDLE OF THE BLAZE: the
            #   window goes to the middle of its order's blaze instead
            if best is None and keep is not None:
                peak = blaze_middle(spans, lo, hi)
                if peak is not None:
                    a, b = peak - width / 2, peak + width / 2
                    sh, dp = tellurics(tapas, a, b)
                    best = ((0,), round(peak * 4) / 4, width,
                            throughput(spans, a, b), sh, dp)
                    print('  the line at %.1f nm sits on its blaze\'s flank'
                          ' (%.2f of the peak at worst): moved to the middle'
                          ' of the blaze' % (centre, cen))
            line = ('%s %7.2f:%-4g blaze %5.2f, %3.0f%% absorbed, deepest'
                    ' %.3f' % ('*' if keep is not None else ' ', centre, width, cen,
                               100 * share, deepest))
            if best is not None and (cen < BLAZE_FLOOR or best[1] != centre
                                     or best[2] != width):
                line += ('   ->  %7.2f:%g blaze %.2f, %3.0f%% absorbed,'
                         ' deepest %.3f' % (best[1], best[2], best[3],
                                            100 * best[4], best[5]))
            elif best is None:
                line += ('   ->  nothing above %.1f of the blaze within %g nm'
                         % (BLAZE_FLOOR, SEARCH))
            print(line)


if __name__ == '__main__':
    main(*sys.argv[1:])
