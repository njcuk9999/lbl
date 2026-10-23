"""
Where PCA2D's river-plot windows (output.windows of nominal.yaml) fall in
the orders of each instrument, and better ones: deep in the tellurics, as
close to the centre of an order as can be.

For each instrument, the wavelength of every order comes from LBL's own
class on a real file (get_wave_solution); the tellurics from LBL's TAPAS
model (tapas_lbl.fits, water and the other molecules). A window's
CENTRALITY is 1 at the centre of the order that holds it best and 0 at its
edge; its DEPTH is the share of its samples under 0.98 of transmission and
its deepest transmission. Every window of the same width within SEARCH nm
of a configured one, inside the instrument's PCA2D domain, is scored, and
the best with a centrality of at least CENTRAL is proposed.

    python order_windows.py <models folder with tapas_lbl.fits> \
        INSTRUMENT:SOURCE:file [INSTRUMENT:SOURCE:file ...]
"""
import sys

import numpy as np
import yaml
from astropy.io import fits

from lbl.core import base_classes
from lbl.instruments import select

#: how far from a configured window a better one is looked for [nm]
SEARCH = 15.0
#: the least centrality a proposed window has
CENTRAL = 0.6
#: a sample is in a telluric line under this transmission
ABSORBED = 0.98
#: the narrowest window proposed [nm]
MIN_WIDTH = 1.5
#: the windows on a STELLAR line, there to show that the correction leaves
#:    it alone (the sodium doublet, the magnesium triplet): kept on it, only
#:    moved this much [nm] to sit better in an order
KEEP = {589.3: 0.3, 517.5: 0.5}


def orders(instrument, source, path):
    """(low, high) of every order [nm], from LBL's wavelength solution"""
    args = base_classes.ParamDict()
    args.set('INSTRUMENT', instrument, source='windows')
    args.set('DATA_SOURCE', source, source='windows')
    args.set('DATA_DIR', '/tmp', source='windows')
    inst = select.load_instrument(args)
    wave = np.atleast_2d(np.asarray(inst.get_wave_solution(path),
                                    dtype=float))
    out = []
    for row in wave:
        good = np.isfinite(row) & (row > 0)
        if good.sum() > 10:
            out.append((float(row[good].min()), float(row[good].max())))
    return out


def centrality(spans, lo, hi):
    """1 at the centre of the order that holds [lo, hi] best, 0 at an edge;
    -1 when no order holds it whole"""
    best = -1.0
    for a, b in spans:
        if a <= lo and hi <= b:
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            # the window's far edge from the order's centre, in half-orders
            reach = max(abs(lo - mid), abs(hi - mid)) / half
            best = max(best, 1.0 - reach)
    return best


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
        spans = orders(instrument, source, path)
        wmin, wmax, windows = domain(instrument)
        if seeds:
            windows = domain(seeds)[2]
        print('\n%s (%s): %d orders, %.1f-%.1f nm; PCA2D domain %.0f-%.0f nm'
              % (instrument, source, len(spans), spans[0][0], spans[-1][1],
                 wmin, wmax))
        for centre, width in windows:
            lo, hi = centre - width / 2, centre + width / 2
            cen = centrality(spans, lo, hi)
            share, deepest = tellurics(tapas, lo, hi)
            best = None
            # the same width, or narrower where the orders are short (the
            #   visible: 5 to 8 nm an order on HARPS), never under MIN_WIDTH
            widths = sorted({max(MIN_WIDTH, round(width * f * 4) / 4)
                             for f in (1.0, 0.75, 0.5)}, reverse=True)
            keep = KEEP.get(round(centre, 1))
            if keep is not None:
                widths = [width]
            reach = SEARCH if keep is None else keep
            for w in widths:
                for c in np.arange(centre - reach, centre + reach + 0.01,
                                   0.05 if keep is not None else 0.25):
                    a, b = c - w / 2, c + w / 2
                    if a < wmin or b > wmax:
                        continue
                    cc = centrality(spans, a, b)
                    if cc < (CENTRAL if keep is None else -0.5):
                        continue
                    sh, dp = tellurics(tapas, a, b)
                    if keep is not None:
                        # on the stellar line: the best centred, nearest
                        score = (round(cc, 3), -abs(c - centre))
                        if best is None or score > best[0]:
                            best = (score, c, w, cc, sh, dp)
                        continue
                    # the telluric content (a narrower window counts less),
                    #   then the centrality, then nearness
                    score = (round(sh * (1 - dp) * np.sqrt(w / width), 3), cc,
                             -abs(c - centre))
                    if best is None or score > best[0]:
                        best = (score, c, w, cc, sh, dp)
            line = ('%s %7.1f:%-4g centrality %5.2f, %3.0f%% absorbed, deepest'
                    ' %.3f' % ('*' if keep is not None else ' ', centre, width, cen,
                               100 * share, deepest))
            if best is not None and (cen < CENTRAL or best[1] != centre
                                     or best[2] != width):
                line += ('   ->  %7.2f:%g centrality %.2f, %3.0f%% absorbed,'
                         ' deepest %.3f' % (best[1], best[2], best[3],
                                            100 * best[4], best[5]))
            elif best is None:
                line += '   ->  nothing central within %g nm' % SEARCH
            print(line)


if __name__ == '__main__':
    main(*sys.argv[1:])
