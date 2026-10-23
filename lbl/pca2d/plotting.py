"""Shared plotting conventions, so every figure says the same thing the same way.

One rule so far, and it earns its own module because getting it wrong is
invisible: **a missing sample is yellow.**

Matplotlib's default is to paint a NaN in the axes background colour, which on a
white figure is white and on a diverging colour map reads as "zero", the value
that means "continuum" in a high-passed spectrum. A gap and a flat continuum
then look identical, and a row of a cube that carries no data at all becomes a
pale stripe indistinguishable from a genuinely featureless spectrum. Yellow sits
outside every diverging map used here, so a gap can never be mistaken for a
measurement.
"""

from __future__ import annotations

import matplotlib
from .logger import log
import numpy as np

NAN_COLOUR = "#ffd400"      # banana



def _as_float(value):
    """A table value as a float: masked, empty or not a number is NaN
    (float() of a masked element warns, and the figure does not need it)"""
    if np.ma.is_masked(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def nan_cmap(name="RdBu_r", under=None, over=None):
    """A colour map that paints NaN yellow.

    Returns a copy, never the registered map: mutating a global colour map
    changes every later figure in the process, including ones drawn by code
    that never asked for this convention.
    """
    cmap = matplotlib.colormaps[name].copy()
    cmap.set_bad(NAN_COLOUR)
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)
    return cmap


#: how far a gap may be and still condemn a valid sample, in samples. The same
#: number the cube is built with; see config quality.isolated_window.
ISOLATED_WINDOW = 3


def live_mask(weights, window=ISOLATED_WINDOW):
    """Which samples of a shifted array are worth drawing.

    `weights > 0` is not enough. Carrying an array into another frame smears
    the mask by the shift, and what comes out has valid samples stranded inside
    regions that are otherwise gone; on screen they are speckles of colour in a
    band of yellow, and they read as measurements. This drops any sample with a
    gap within `window` on both sides, the same rule the cube is built with.
    """
    from .preprocess import drop_isolated

    return drop_isolated(np.asarray(weights) > 0, window)


def order_bounds(path):
    """(lo, hi) in nm for every order of a file's wavelength extension.

    One row per order, in the order the extension stores them, which is the
    same counting the cube uses: its builder assigns `parity = order % 2`, so
    the index returned here maps straight onto a cube row's parity. An order
    with no finite wavelength comes back as NaN rather than raising.
    """
    import numpy as np

    from .io import robust_open
    from .tfits import extensions_for

    try:
        with robust_open(path) as hdulist:
            # NOT a hardcoded extension name, for the reason given above
            _, e_wave, _, _ = extensions_for(hdulist)
            wave = np.asarray(hdulist[e_wave].data, dtype=float)
    except Exception:                                         # noqa: BLE001
        # UN SPECTRE DU VISIBLE N'A PAS D'EXTENSION D'APERO, et la table ne
        # connait que SPIRou et NIRPS: elle levait, l'appelant avalait
        # l'exception, et la parite n'etait JAMAIS choisie. Les deux parites
        # restaient alors empilees sur la meme page, ce qui met cote a cote
        # deux jeux d'ordres differents et fait pencher ce qui devrait etre
        # vertical. Vu le 2026-10-15 sur la fenetre 626-630 nm de HD 10700,
        # 174 lignes tracees au lieu de 87.
        #
        # On passe donc par le pont de LBL, qui sait lire n'importe quel
        # instrument, exactement comme le cube l'a lu.
        from astropy.io import fits

        from . import readers as _readers
        name = str(fits.getheader(path).get("INSTRUME", "")).strip().upper()
        # LA SOURCE SE MESURE, ELLE NE SE DEVINE PAS. Un instrument a
        # plusieurs classes de lecture (HARPS: ORIG, ESO, ESSP) et elles ne
        # lisent pas le meme fichier: l'une veut des polynomes d'en-tete que
        # ce fichier n'a pas. On essaie donc celles que LBL declare et on
        # garde la premiere qui rend une grille, ce qui est aussi la seule
        # facon de le savoir sans la configuration du run.
        wave = None
        for source in _readers.KNOWN.get(name, ()):
            try:
                wave = np.atleast_2d(np.asarray(
                    _readers.read(path, name, source)["wave"], dtype=float))
                break
            except Exception:                                 # noqa: BLE001
                continue
        if wave is None:
            raise
    wave = np.where(np.isfinite(wave) & (wave > 0), wave, np.nan)
    lo = np.full(wave.shape[0], np.nan)
    hi = np.full(wave.shape[0], np.nan)
    ok = np.isfinite(wave).any(axis=1)
    if ok.any():
        lo[ok] = np.nanmin(wave[ok], axis=1)
        hi[ok] = np.nanmax(wave[ok], axis=1)
    return lo, hi


def window_parity(wavelength_nm, path):
    """Which order parity owns a wavelength where two orders overlap.

    Consecutive echelle orders share their ends, so a window that falls in an
    overlap is measured twice: once near the middle of one order, where the
    blaze peaks and the extraction is at its best, and once at the very edge of
    its neighbour, where the throughput has fallen off and the same wavelengths
    are far noisier. Since the two parities are separate cube rows, both sets
    are drawn, and the figure then shows a measurement stacked above a poor
    copy of itself, which is worse than showing one.

    The order kept is the one whose middle the wavelength sits closest to,
    measured in units of that order's own half-width so that orders of
    different widths compare: 0 is dead centre, 1 is an edge.

    Returns (parity, offset, n_covering). `n_covering` is how many orders reach
    this wavelength at all, so 1 means there was nothing to choose and the
    caller can say so. (None, nan, 0) if the file cannot be read, in which case
    the caller should fall back to its coverage rule rather than fail: this
    sharpens a figure, it does not gate one.
    """
    try:
        lo, hi = order_bounds(path)
    except Exception:                                         # noqa: BLE001
        return None, float("nan"), 0
    return parity_from_bounds(wavelength_nm, lo, hi)


def parity_from_bounds(wavelength_nm, lo, hi):
    """The arithmetic of `window_parity`, with the file already read.

    Separated so it can be checked without a spectrum on disk: the order index
    of the winner is what decides the parity, and getting that off by one puts
    every overlapping window on the wrong side.
    """
    import numpy as np

    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    half = 0.5 * (hi - lo)
    with np.errstate(invalid="ignore"):
        offset = np.abs(wavelength_nm - 0.5 * (lo + hi)) / np.where(half > 0,
                                                                   half, np.nan)
        covers = np.isfinite(offset) & (offset <= 1.0)
    if not covers.any():
        # outside every order, or an unreadable solution: nothing to arbitrate
        return None, float("nan"), 0
    idx = np.where(covers)[0][np.argmin(offset[covers])]
    return int(idx % 2), float(offset[idx]), int(covers.sum())


def sample_source_file(cube_path, meta=None, source_dir=None):
    """One readable t.fits behind a cube, for anything that needs its geometry.

    The order boundaries are a property of the instrument, not of the night, so
    any exposure of the set answers the question and the first one that opens
    will do. Returns None if the config names no directory, the metadata holds
    no filenames, or none of them are on this machine, so that a figure drawn
    from a cube alone still draws.
    """
    import os

    directory = source_dir or source_directory(cube_path)
    if not directory or meta is None:
        return None
    try:
        names = [os.path.basename(str(v)) for v in meta["filename"]]
    except Exception:                                         # noqa: BLE001
        return None
    for name in names[:20]:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Correlations against ancillary quantities
# ---------------------------------------------------------------------------
# A component carries no label. That a1 varies on 83 days is a result; that it
# varies with the precipitable water column would say plainly that a "stellar"
# component is describing the atmosphere. This is the diagnostic that catches
# that, which is why the fit writes it itself rather than leaving it to a
# separate script somebody may forget to run.

#: cube metadata column -> label
ANCILLARY = [
    ("berv", "BERV"),
    ("airmass", "airmass"),
    ("snr_band", "band SNR"),
    ("seeing", "seeing"),
    ("humidity", "humidity"),
    ("exptime", "exp. time"),
    ("sun_elevation", "sun elev."),
    ("median_flux", "median flux"),
    ("mjdmid", "time"),
]
#: APERO's telluric-preclean water exponent, in the FluxA header, not the cube
WATER_KEY = "TLPEH2O"


def water_column(filenames, source_dir):
    """TLPEH2O per exposure, read from the t.fits themselves.

    NaN where the file is missing or the key absent, so a partial read shows as
    gaps in one row rather than silently dropping exposures and shifting every
    other correlation.
    """
    import os
    from astropy.io import fits

    out = np.full(len(filenames), np.nan)
    if not source_dir:
        return out
    for i, name in enumerate(filenames):
        path = os.path.join(source_dir, os.path.basename(str(name)))
        if not os.path.exists(path):
            continue
        try:
            with fits.open(path) as hdulist:
                for hdu in hdulist:
                    value = hdu.header.get(WATER_KEY)
                    if value is not None:
                        out[i] = float(value)
                        break
        except Exception:                                     # noqa: BLE001
            continue
    return out


def source_directory(cube_path):
    """Where the t.fits behind a cube live, per the config the build copied in.

    The cube stage drops the config it used into the cache directory, so the
    fit can recover the source folder without being told twice. Returns None if
    the file is absent, in which case the water row is simply omitted rather
    than the whole matrix failing.
    """
    import os

    path = os.path.join(cube_path, "cube_config.yaml")
    if not os.path.isdir(cube_path) or not os.path.exists(path):
        return None
    try:
        from .config import read_yaml, spectra_dir
        with open(path) as fh:
            cfg = read_yaml(fh) or {}
        # the object's own folder: input.directory is the root it hangs from,
        # and a lookup in the root found no file, so no water row, on every
        # run since the root and the object were split
        return spectra_dir(cfg) if (cfg.get("input") or {}).get("directory") else None
    except Exception:                                         # noqa: BLE001
        return None


def ancillary_table(meta, filenames, source_dir=None):
    """(labels, values) for every ancillary quantity that varies."""
    import os

    labels, values = [], []
    if meta is not None:
        by_name = {}
        for row in meta:
            by_name.setdefault(os.path.basename(str(row["filename"])), row)
        for key, label in ANCILLARY:
            if key not in getattr(meta, "colnames", []):
                continue
            col = np.array([_as_float(by_name[n][key]) if n in by_name
                            else np.nan for n in filenames])
            if np.isfinite(col).sum() > 10 and np.nanstd(col) > 0:
                labels.append(label)
                values.append(col)
    water = water_column(filenames, source_dir)
    if np.isfinite(water).sum() > 10:
        labels.append("water (%s)" % WATER_KEY)
        values.append(water)
    return labels, (np.array(values) if values else np.zeros((0, len(filenames))))


def rank_correlations(comps, values):
    """Spearman rho for every (component, ancillary) pair, NaN where undefined.

    Rank rather than Pearson: a coefficient has no reason to depend linearly on
    airmass, and one outlying night should not manufacture a correlation.
    """
    from scipy.stats import spearmanr

    rho = np.full((len(comps), values.shape[0]), np.nan)
    for i, (_, c) in enumerate(comps):
        for j in range(values.shape[0]):
            good = np.isfinite(c) & np.isfinite(values[j])
            # A constant on EITHER side has no rank correlation, and the test
            # is max minus min rather than a standard deviation: twelve copies
            # of the same airmass have a std of 2e-16, not 0, since their mean
            # is not exactly that value in binary. That let scipy be asked for
            # the correlation of a constant, which it answers with NaN and a
            # ConstantInputWarning per pair. NaN is the honest entry; this says
            # it without the noise.
            if (good.sum() > 10 and np.ptp(values[j][good]) > 0
                    and np.ptp(c[good]) > 0):
                rho[i, j] = spearmanr(c[good], values[j][good]).statistic
    return rho


def plot_correlations(rho, comps, labels, path, title="", flag=0.3):
    """The matrix, with the star and Earth blocks separated by a rule."""
    import matplotlib.pyplot as plt

    if rho.size == 0:
        log("  no ancillary quantity varies; skipping %s" % path)
        return
    plt.rcParams.update({"font.size": 9, "axes.grid": False})
    fig, ax = plt.subplots(figsize=(1.2 + 0.62 * len(labels), 1.4 + 0.34 * len(comps)))
    im = ax.imshow(rho, cmap=nan_cmap("RdBu_r"), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([n for n, _ in comps], fontsize=8)
    n_star = sum(1 for n, _ in comps if n.startswith("a"))
    if 0 < n_star < len(comps):
        ax.axhline(n_star - 0.5, color="k", lw=1.0)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if np.isfinite(rho[i, j]) and abs(rho[i, j]) >= flag:
                ax.text(j, i, "%.2f" % rho[i, j], ha="center", va="center",
                        fontsize=7, color="white" if abs(rho[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="Spearman rank correlation")
    ax.set_title(title or "coefficients against ancillary quantities", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log("  wrote %s" % path)


def write_correlation_table(rho, comps, labels, path):
    """The same numbers as plot_correlations, machine-readable."""
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["component", "quantity", "spearman_rho"])
        for i, (name, _) in enumerate(comps):
            for j, label in enumerate(labels):
                value = rho[i, j]
                writer.writerow([name, label,
                                 "" if not np.isfinite(value) else "%.4f" % value])
    log("  wrote %s" % path)


def report_correlations(rho, comps, labels, flag=0.7):
    """Print the pairs worth looking at, so a run says it without a PDF."""
    if rho.size == 0:
        return
    strong = [(abs(rho[i, j]), comps[i][0], labels[j], rho[i, j])
              for i in range(rho.shape[0]) for j in range(rho.shape[1])
              if np.isfinite(rho[i, j]) and abs(rho[i, j]) >= flag]
    if not strong:
        log("  no |rho| above %.2f: no component tracks an ancillary quantity"
              % flag)
        return
    log("  |rho| >= %.2f, i.e. a component that follows something known:" % flag)
    for _, name, label, value in sorted(strong, reverse=True):
        log("    %-4s %-22s %+.2f" % (name, label, value))


def rows_of(meta, fit):
    """Which cube rows the FIT used, in the fit's own order, or None if all.

    Shared by every figure that reads a cube and a fit together.

    A fit records the file and the order parity of every row it kept. Deriving
    them again from the cube instead, as this used to, ties a figure to whatever
    the row selection happened to be when the fit was made: the day the
    signal-to-noise cut became per object, every figure of every existing fit
    died on a shape mismatch, because the cube now yielded rows the fit had never
    seen. Aligning on what the fit says is immune to that.
    """
    import os

    names = [os.path.basename(str(v)) for v in meta["filename"]]
    parity = (np.asarray(meta["parity"], dtype=int)
              if "parity" in getattr(meta, "colnames", []) else np.zeros(len(names), int))
    where = {(n, int(p)): i for i, (n, p) in enumerate(zip(names, parity))}
    want = [(os.path.basename(str(n)), int(p))
            for n, p in zip(fit["filename"], fit["parity"])]
    if len(want) == len(names) and all(where.get(k) == i
                                       for i, k in enumerate(want)):
        return None                       # already row for row, the usual case
    missing = [k for k in want if k not in where]
    if missing:
        raise SystemExit("the fit has %d rows this cube does not hold, the first"
                         " being %s: this fit was made from another cube"
                         % (len(missing), missing[0]))
    return np.array([where[k] for k in want], dtype=int)



def raw_log_flux_window(cube_dir, source_dir, names, parities, grid, a0, b0):
    """ln(flux) before the high-pass, every cube row, on grid[a0:b0].

    From the cube's snippets when they are there. The cube build writes them
    while it has every spectrum open anyway, so a figure never reopens three
    hundred files to draw five of them. When a block has none yet, a cube
    built before snippets existed or a window added since, the spectra are
    read once, resampled onto the block alone rather than onto the whole grid,
    and the snippet is written for the next time.
    """
    import os

    import numpy as np

    from . import cache as _cache
    from . import tfits as sptf
    from .config import load_config, spectra_dir

    out = np.full((len(names), b0 - a0), np.nan, dtype=np.float32)
    stored = _cache.read_snippet(cube_dir, a0, b0)
    if stored is None:
        try:
            config = load_config(os.path.join(cube_dir, "cube_config.yaml"))
        except Exception as exc:                              # noqa: BLE001
            log("  no recipe in %s to re-read the spectra with (%s)" % (cube_dir, exc),
                "warn")
            return out
        directory = source_dir or spectra_dir(config)
        stored = {}
        for name in sorted({os.path.basename(str(v)) for v in names}):
            try:
                payload = sptf.read_tfits(os.path.join(directory, name))
            except Exception:                                 # noqa: BLE001
                continue
            stored[name] = sptf.raw_block(payload, grid, a0, b0, config)
        if stored:
            path = _cache.write_snippet(cube_dir, a0, b0, stored)
            log("  re-read %d spectra for columns %d-%d, kept in %s"
                % (len(stored), a0, b0, path))
    for i, name in enumerate(names):
        block = stored.get(os.path.basename(str(name)))
        if block is not None:
            # a snippet's two rows are the two ORDER parities, which is what the
            # detector has. A joint cube labels its rows 2 * object + parity, so
            # the label itself runs 0..2N-1 and indexed raw it walked off the end
            # of the block at the first row of the second object (2026-09-13).
            out[i] = block[int(parities[i]) % block.shape[0]]
    return out
