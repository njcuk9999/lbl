"""The Data Explorer tab: the rdb tables of DATA_DIR/lblrdb, any column
against any other, the false inclusion probability (FIP) of what is plotted,
and the template a table was measured against, with the TAPAS
transmission of each molecule under it.

What the page gets is already cut to what it shows: a table as its numeric
columns, the FIP as -log10 over a few thousand bins of period (the lowest
FIP of each bin, as koloa draws it), the template and TAPAS over the range
the plot shows, as the min and max of each bin of about one screen pixel
(every point once the range is narrow enough). The templates and the TAPAS
files are read once and kept.
"""
from __future__ import annotations

import glob
import os
import threading

import numpy as np

#: where LBL writes its tables, under DATA_DIR
RDB_FOLDER = "lblrdb"
#: the time of a table: LBL's own (BJD - 2400000), that of the FIP
TIME = "rjd"
#: the x of a table opened, the first one it has (the MJD asked for)
DEFAULT_X = ("MJDMID", "MJD-OBS", "MJDATE", "rjd")
DEFAULT_Y = ("vrad",)
DEFAULT_C = ("BERV",)
#: a column's error: the first of these it has (vrad svrad, fwhm sig_fwhm)
ERROR_PREFIXES = ("s", "sig_", "sig", "e_", "err_")
#: what names a row on the plot
LABELS = ("local_file_name", "FILENAME", "DATE-OBS")
#: the columns that are times: the FIP of a quantity is against TIME, and
#: is not asked of these
TIMES = ("rjd", "MJDMID", "MJD-OBS", "MJDATE", "BJD", "plot_date",
         "MJD", "JD")
#: the FIP over this many bins of log period (the lowest FIP of each)
FIP_BINS = 2000
#: the site of the per-molecule TAPAS file of an instrument (a file whose
#: name holds it, in models/tellurics)
SITES = {"SPIROU": "maunakea", "ESPADONS": "maunakea",
         "NIRPS_HE": "lasilla", "NIRPS_HA": "lasilla", "HARPS": "lasilla",
         "CARMENES": "calar", "ESPRESSO": "paranal", "HARPSN": "palma",
         "HARPS_N": "palma", "SOPHIE": "ohp", "EXPRES": "lowell",
         "MAROONX": "maunakea", "NEID": "kitt", "CPAPIR": "megantic"}
#: the molecules of a TAPAS file, the order they are drawn in
MOLECULES = ("H2O", "O2", "CH4", "CO2", "O3", "N2O", "NO2")

#: what a PCA2D run adds to the name of a star's folder (science/<star>_pca2d
#: or <star>_pca2d_<name>)
PCA2D_MARK = "_pca2d"

_LOCK = threading.Lock()
_NAMES = {}
_TEMPLATES = {}
_TAPAS = {}
_FIPS = {}
_SYSTEMIC = {}
_PLANETS = {}
#: two planets of a star whose periods agree this well are one, named by
#:    two catalogs
PLANET_SAME = 0.01
#: where a template's systemic velocity is found: the SYSTVELO [m/s] that
#:    LBL writes in the masks it makes from the template (its own
#:    measurement), else SIMBAD's
MASK_VELOCITY = "SYSTVELO"


# =============================================================================
# The tables
# =============================================================================
def names_of(path: str, data_dir: str = "") -> tuple:
    """(science, template) of a table, lbl_<science>_<template>.rdb: from
    its companion's header (LBL_OBJS, LBL_OBJC); else the name less the
    template the header names, the longest folder of DATA_DIR/science the
    name starts with, or the name less a template found on disk
    (template_of); else its first word."""
    from astropy.io import fits
    stem = os.path.splitext(os.path.basename(path))[0]
    rest = stem.split("_", 1)[1] if "_" in stem else stem
    companion = _companion(path)
    try:
        mtime = os.path.getmtime(companion)
    except OSError:
        mtime = None
    with _LOCK:
        kept = _NAMES.get(companion)
    if mtime is not None and kept and kept[0] == mtime:
        return kept[1]
    science = template = None
    if mtime is not None:
        try:
            header = fits.getheader(companion)
            science = header.get("LBL_OBJS")
            template = header.get("LBL_OBJC")
        except (OSError, ValueError):
            pass
    if not (science and rest.startswith(str(science) + "_")):
        science = None
        if template and rest.endswith("_" + str(template)):
            science = rest[:-len(str(template)) - 1]
        if not science:
            folders = glob.glob(os.path.join(data_dir or "", "science", "*"))
            known = sorted((os.path.basename(f) for f in folders),
                           key=len, reverse=True)
            science = next((name for name in known
                            if rest.startswith(name + "_")), None)
        if not science:
            found = template_of(path, data_dir)
            if found and rest.endswith("_" + found["object"]):
                science = rest[:-len(found["object"]) - 1]
        science = science or rest.split("_")[0]
        template = None
    science = str(science)
    template = str(template) if template else rest[len(science) + 1:]
    if mtime is not None:
        with _LOCK:
            _NAMES[companion] = (mtime, (science, template))
    return science, template


def object_flavour(science: str, template: str) -> tuple:
    """A table's star and its flavour: the star its science folder is of
    (GJ_1028_pca2d: GJ_1028), and what else its name says, the PCA2D
    variant and the template ('pca2d / GJ_1028_pca2d', 'VETTED_GL880')."""
    star = science.split(PCA2D_MARK)[0] or science
    variant = science[len(star):].lstrip("_")
    return star, ("%s / %s" % (variant, template) if variant else template)


def data_dir_of(path: str) -> str:
    """The data folder a table belongs to: DATA_DIR of DATA_DIR/lblrdb/x.rdb
    (its templates, masks and models are that folder's)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(path)))


def _homes(data_dir: str) -> list:
    """The data folders whose tables the explorer lists: DATA_DIR, and when
    it is one of a root's (ROOT/{INSTRUMENT}_{SOURCE}, lbl.gui.dispatch),
    the root's others and the root itself (the old layout), as (folder,
    its name in the root: '' for the root)."""
    from .dispatch import readers, split
    root, mine = split(data_dir)
    if not mine and not any(os.path.isdir(os.path.join(root, name))
                            for name in readers()):
        return [(os.path.abspath(data_dir or ""), "")]
    homes = [(os.path.join(root, name), name) for name in sorted(readers())
             if os.path.isdir(os.path.join(root, name, RDB_FOLDER))]
    if os.path.isdir(os.path.join(root, RDB_FOLDER)):
        homes.append((root, ""))
    # DATA_DIR's own first
    return sorted(homes, key=lambda h: h[1] != mine)


def rdb_files(data_dir: str) -> dict:
    """The rdb tables of DATA_DIR/lblrdb (and, in a root's DATA_DIR, those
    of the root's other ones and of the root itself: _homes), the newest
    first, each with its star and its flavour (object_flavour), its data
    folder's name in the root (home; own: DATA_DIR's). A table
    of every spectrum (lbl_X.rdb) carries that of its nights (lbl2_X.rdb,
    written in the same run) as 'nightly', which is not listed apart; an
    lbl2_ table without its lbl_ is."""
    from .dispatch import split
    homes = _homes(data_dir)
    mine = split(data_dir)[1]
    rows = []
    for home_dir, home in homes:
        found = {}
        for path in glob.glob(os.path.join(home_dir, RDB_FOLDER, "*.rdb")):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            name = os.path.basename(path)
            found[name] = dict(path=path, name=name, mtime=stat.st_mtime,
                               size=stat.st_size, nightly=None, home=home,
                               own=home == mine)
        for name, row in found.items():
            if name.startswith("lbl2_") and \
                    "lbl_" + name[len("lbl2_"):] in found:
                continue
            if name.startswith("lbl_"):
                twin = found.get("lbl2_" + name[len("lbl_"):])
                if twin is not None:
                    row["nightly"] = {key: twin[key]
                                      for key in ("path", "name", "mtime")}
            row["object"], row["flavour"] = object_flavour(
                *names_of(row["path"], home_dir))
            rows.append(row)
    # a run's two tables: as new as the newer of them; DATA_DIR's first
    rows.sort(key=lambda row: (not row["own"], -max(
        row["mtime"], (row["nightly"] or row)["mtime"])))
    return dict(folder=os.path.join(data_dir or "", RDB_FOLDER), files=rows,
                latest=rows[0]["path"] if rows else None)


def error_column(name: str, names) -> str | None:
    """The error column of a column (vrad: svrad, fwhm: sig_fwhm), None
    without one."""
    for prefix in ERROR_PREFIXES:
        if prefix + name in names:
            return prefix + name
    return None


def _listed(values) -> list:
    """An array as a list for JSON: a NaN or an inf as null (JSON has
    neither)."""
    values = np.asarray(values, dtype=float)
    return [v if np.isfinite(v) else None for v in values.tolist()]


def _first(options, names, fallback=None):
    for name in options:
        if name in names:
            return name
    return fallback


def read_table(path: str):
    from astropy.table import Table
    return Table.read(path, format="ascii.rdb")


def read_rdb(path: str, data_dir: str = "") -> dict:
    """A table as the page plots it: its numeric columns (null where not
    finite), each one's error column, a label per row, what it opens on,
    and the template it was measured against."""
    table = read_table(path)
    numeric = [name for name in table.colnames
               if table[name].dtype.kind in "fiu"]
    values = {name: _listed(table[name]) for name in numeric}
    errors = {name: error_column(name, numeric) for name in numeric}
    errors = {name: err for name, err in errors.items() if err}
    label = _first(LABELS, table.colnames)
    labels = ([str(v) for v in table[label]] if label
              else [str(i) for i in range(len(table))])
    x = _first(DEFAULT_X, numeric, numeric[0] if numeric else None)
    y = _first(DEFAULT_Y, numeric,
               numeric[1] if len(numeric) > 1 else x)
    berv = None
    if "BERV" in numeric:
        good = np.asarray(table["BERV"], dtype=float)
        good = good[np.isfinite(good)]
        if good.size:
            berv = [float(good.min()), float(good.max())]
    return dict(path=path, name=os.path.basename(path), rows=len(table),
                columns=numeric, values=values, errors=errors, labels=labels,
                time=TIME if TIME in numeric else None,
                times=[name for name in numeric if name in TIMES],
                x=x, y=y, c=_first(DEFAULT_C, numeric),
                berv=berv, template=template_of(path, data_dir))


# =============================================================================
# The FIP
# =============================================================================
def _robust_sigma(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def _binned(period, fip_values, edges):
    """-log10 of the lowest FIP in each bin of log period (koloa's
    _fip_curves): a peak is kept whatever the bins."""
    idx = np.clip(np.digitize(np.log10(period), edges) - 1, 0,
                  len(edges) - 2)
    best = np.ones(len(edges) - 1)
    np.minimum.at(best, idx, np.clip(fip_values, 1e-15, 1.0))
    return np.round(-np.log10(best), 4)


def planets_of(path: str) -> dict:
    """The published planets of a table's star, for its FIP: those of the
    catalogs LBL's PLANETS option reads (lbl.science.planets: a personal
    catalog of the data folder or of LBL first, then known_planets.yaml,
    from exoplanet.eu and the NASA Exoplanet Archive), the planets whose
    orbit is incomplete included (only their period is drawn), and those
    of ephemerides.yaml. The star is looked up by its folder's name less
    LBL's suffixes, the names of the companion's header, each also as
    SIMBAD writes it (GL699 as GJ 699). dict(star, planets: [dict(name,
    period, k, reference)], catalog)."""
    from astropy.io import fits
    from lbl.core.simbad import name_variants
    from lbl.science import planets as kepler
    from .dispatch import NAME_KEYS
    data_dir = data_dir_of(path)
    science, _ = names_of(path, data_dir)
    header = []
    try:
        head = fits.getheader(_companion(path))
        header = [str(head[key]).strip() for key in NAME_KEYS
                  if key in head and str(head[key]).strip()]
    except (OSError, ValueError):
        pass
    names = []
    for name in kepler.star_names(science, header):
        for one in [name] + name_variants(name):
            if one not in names:
                names.append(one)
    key = (tuple(names), data_dir)
    with _LOCK:
        if key in _PLANETS:
            return _PLANETS[key]
    out = dict(star=None, planets=[], catalog=None)
    found = kepler.find_star(names, data_dir)
    rows = []
    if found is not None:
        entry = found["entry"]
        rows += list(entry.get("planets") or [])
        rows += [one for one in entry.get("not_used") or []
                 if one.get("period")]
        out.update(star=found["star"],
                   catalog=os.path.basename(found["catalog"]))
    timed = kepler.ephemerides_for(names)
    if timed is not None:
        rows += timed["planets"]
        out["star"] = out["star"] or timed["star"]
        out["catalog"] = out["catalog"] or os.path.basename(timed["catalog"])
    seen = set()
    for row in rows:
        try:
            period = float(row.get("period"))
        except (TypeError, ValueError):
            continue
        name = str(row.get("name") or "?")
        if not np.isfinite(period) or period <= 0 \
                or kepler.normalise(name) in seen:
            continue
        # one planet named twice by two catalogs (eps Eridani b, eps Eri
        #   b): its period within PLANET_SAME of the other's
        if any(abs(period / one["period"] - 1) < PLANET_SAME
               for one in out["planets"]):
            continue
        seen.add(kepler.normalise(name))
        k = row.get("k")
        out["planets"].append(dict(
            name=name, period=period,
            k=float(k) if k is not None and np.isfinite(float(k)) else None,
            reference=row.get("reference")))
    out["planets"].sort(key=lambda one: one["period"])
    with _LOCK:
        _PLANETS[key] = out
    return out


def fip_of(path: str, column: str) -> dict:
    """The gaussian and the outlier-aware FIP of a column of a table,
    against its time (lbl.report.fip, as the report computes them), over
    periods from 1.1 days to twice the span, and each row's probability of
    being an outlier (at the period of the lowest outlier-aware FIP). A
    column without an error column is taken at its robust scatter (each
    model fits its own jitter on top anyway)."""
    from lbl.report import fip as lbl_fip
    stat = os.stat(path)
    key = (path, stat.st_mtime, column)
    with _LOCK:
        if key in _FIPS:
            return _FIPS[key]
    table = read_table(path)
    if TIME not in table.colnames or column not in table.colnames:
        return dict(column=column, error="no %s or %s" % (TIME, column))
    time = np.asarray(table[TIME], dtype=float)
    value = np.asarray(table[column], dtype=float)
    err_name = error_column(column, table.colnames)
    if err_name:
        err = np.asarray(table[err_name], dtype=float)
    else:
        err = np.full(value.size, _robust_sigma(value))
    out = dict(column=column, err=err_name)
    freq, gauss = lbl_fip.fip(time, value, err)
    if freq.size == 0:
        out.update(period=[], gauss=[], oa=[], prob=[],
                   error="fewer than 10 points with an error")
        return out
    _, aware, prob = lbl_fip.oafip(time, value, err, freq=freq)
    period = 1.0 / freq
    logp = np.log10(period)
    edges = np.linspace(logp.min(), logp.max() + 1e-9, FIP_BINS + 1)
    out.update(period=np.round(10 ** (0.5 * (edges[1:] + edges[:-1])),
                               6).tolist(),
               gauss=_binned(period, gauss, edges).tolist(),
               oa=_binned(period, aware, edges).tolist(),
               prob=_listed(prob),
               best=dict(gauss=dict(period=float(period[np.argmin(gauss)]),
                                    fip=float(np.min(gauss))),
                         oa=dict(period=float(period[np.argmin(aware)]),
                                 fip=float(np.min(aware)))),
               outliers=int(np.sum(np.asarray(prob) > 0.5)),
               points=int(np.sum(np.isfinite(time) & np.isfinite(value)
                                 & np.isfinite(err) & (err > 0))))
    with _LOCK:
        _FIPS[key] = out
    return out


# =============================================================================
# The template
# =============================================================================
def _companion(path: str) -> str:
    """The FITS LBL writes beside a table (lbl2_X.rdb: lbl_X.fits)."""
    folder, name = os.path.split(path)
    stem = os.path.splitext(name)[0]
    if stem.startswith("lbl2_"):
        stem = "lbl_" + stem[len("lbl2_"):]
    return os.path.join(folder, stem + ".fits")


def template_of(path: str, data_dir: str = "") -> dict | None:
    """The template a table was measured against: the object of its
    companion's header (LBL_OBJC), else of its name (lbl_<science>_
    <template>.rdb), in DATA_DIR/templates or, for a vetted one,
    DATA_DIR/models/template_*. dict(path, name, object, instrument) or
    None."""
    from astropy.io import fits
    data_dir = data_dir or os.path.dirname(os.path.dirname(path))
    folders = [os.path.join(data_dir, "templates")] + sorted(
        glob.glob(os.path.join(data_dir, "models", "template_*")))
    objc, instrument, mode = None, "", ""
    companion = _companion(path)
    if os.path.isfile(companion):
        try:
            header = fits.getheader(companion)
            objc = header.get("LBL_OBJC")
            instrument = str(header.get("LBLINSTR", "")).lower()
            mode = str(header.get("LBLIMODE", "")).lower()
        except (OSError, ValueError):
            pass
    if objc:
        names = [str(objc)]
    else:
        # lbl_<science>_<template>: the longest end of the name that is
        #   a template's object
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")[1:]
        names = ["_".join(parts[i:]) for i in range(1, len(parts))]
    suffixes = [s for s in ("%s_%s" % (instrument, mode), instrument) if s
                and s != "_"]
    for name in names:
        found = []
        for folder in folders:
            prefix = os.path.join(folder, "LBL_Template_%s_" % name)
            for candidate in glob.glob(glob.escape(prefix) + "*.fits"):
                rest = candidate[len(prefix):-len(".fits")]
                found.append((rest, candidate))
        if not found:
            continue
        # the instrument's own suffix first, else the shortest (GJ_1028:
        #   _nirps_he before _pca2d_nirps_he)
        found.sort(key=lambda item: (item[0] not in suffixes,
                                     len(item[0])))
        rest, candidate = found[0]
        return dict(path=candidate, name=os.path.basename(candidate),
                    object=name, instrument=instrument.upper() or None)
    return None


def _template_arrays(path: str):
    from astropy.io import fits
    mtime = os.path.getmtime(path)
    with _LOCK:
        kept = _TEMPLATES.get(path)
    if kept and kept[0] == mtime:
        return kept[1], kept[2]
    with fits.open(path) as hdul:
        hdu = hdul["TEMPLATE"] if "TEMPLATE" in hdul else hdul[1]
        wave = np.array(hdu.data["wavelength"], dtype=float)
        flux = np.array(hdu.data["flux"], dtype=float)
    order = np.argsort(wave)
    wave, flux = wave[order], flux[order]
    with _LOCK:
        _TEMPLATES[path] = (mtime, wave, flux)
    return wave, flux


def envelope(x, y, lo: float, hi: float, nbin: int, how: str = "minmax"):
    """The points of (x, y) within [lo, hi] (and one on each side), or, if
    there are more than 2 nbin of them, nbin bins of them: the min and the
    max of each (minmax: the line's envelope) or the min ('min': the
    depths of absorption lines). x is sorted."""
    i0 = max(int(np.searchsorted(x, lo, side="left")) - 1, 0)
    i1 = min(int(np.searchsorted(x, hi, side="right")) + 1, len(x))
    xs, ys = x[i0:i1], y[i0:i1]
    if xs.size <= 2 * nbin:
        return xs, ys
    starts = np.linspace(0, xs.size, nbin + 1).astype(int)[:-1]
    middle = 0.5 * (xs[starts] + xs[np.r_[starts[1:], xs.size] - 1])
    with np.errstate(invalid="ignore"):
        bottom = np.fmin.reduceat(ys, starts)
        if how == "min":
            return middle, bottom
        top = np.fmax.reduceat(ys, starts)
    return np.repeat(middle, 2), np.column_stack([bottom, top]).ravel()


# =============================================================================
# TAPAS
# =============================================================================
def tapas_file(data_dir: str, instrument: str = "") -> dict | None:
    """The TAPAS transmission under the template: a file of models/
    tellurics with a column per molecule (wavelength, H2O, O2, CH4...),
    the one of the instrument's site if there are several, else LBL's own
    tapas_lbl.fits (water, and the others together). dict(path, kind) or
    None."""
    from astropy.io import fits
    folder = os.path.join(data_dir or "", "models", "tellurics")
    molecules = []
    for path in sorted(glob.glob(os.path.join(folder, "*.fits"))):
        try:
            header = fits.getheader(path, 1)
        except (OSError, IndexError, KeyError, ValueError):
            continue
        columns = [str(header.get("TTYPE%d" % i, ""))
                   for i in range(1, int(header.get("TFIELDS", 0)) + 1)]
        if "wavelength" in columns and "H2O" in columns:
            molecules.append(path)
    if molecules:
        site = SITES.get(str(instrument).upper().replace("-", "_"), "")
        mine = [p for p in molecules
                if site and site in os.path.basename(p).lower()]
        return dict(path=(mine or molecules)[0], kind="molecules")
    lbl = os.path.join(folder, "tapas_lbl.fits")
    if os.path.isfile(lbl):
        return dict(path=lbl, kind="lbl")
    return None


#: the TAPAS samples a coarse bin holds (0.25 km/s each: 8 km/s)
TAPAS_COARSE = 32


def _transmission(values):
    """A TAPAS transmission within [0, 1] (1 where unknown)"""
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=1.0)
    return np.clip(values, 0.0, 1.0)


def _tapas_arrays(path: str, kind: str):
    """(wave, {molecule: transmission column}, coarse wave, {molecule:
    coarse transmission}): the file is mapped, not read, but for its
    coarse version (the min of each TAPAS_COARSE samples), made once for
    wide views."""
    from astropy.io import fits
    with _LOCK:
        kept = _TAPAS.get(path)
    if kept:
        return kept
    hdul = fits.open(path, memmap=True)
    data = hdul[1].data
    if kind == "molecules":
        wave = data["wavelength"]
        columns = {m: data[m] for m in MOLECULES if m in data.columns.names}
    else:
        wave = data["WAVELENGTH"]
        columns = {"H2O": data["ABSO_WATER"], "others": data["ABSO_OTHERS"]}
    starts = np.arange(0, len(wave), TAPAS_COARSE)
    ends = np.minimum(starts + TAPAS_COARSE, len(wave)) - 1
    coarse_wave = 0.5 * (np.asarray(wave[starts], dtype=float)
                         + np.asarray(wave[ends], dtype=float))
    coarse = {name: np.minimum.reduceat(_transmission(column), starts)
              for name, column in columns.items()}
    # the files are written sorted (LBL's, TAPAS's): only checked
    if not np.all(np.diff(coarse_wave) > 0):
        raise ValueError("%s is not sorted in wavelength" % path)
    kept = (wave, columns, coarse_wave, coarse, hdul)
    with _LOCK:
        _TAPAS[path] = kept
    return kept


def systemic(info: dict, data_dir: str) -> dict:
    """The systemic velocity of a template's star [km/s]: the SYSTVELO of a
    mask LBL made from it (DATA_DIR/masks/LBL_Mask_<object>_*, or beside a
    vetted template), else SIMBAD's (lbl.gui.simbad.ask, once per star).
    dict(value, source), value None when neither says."""
    key = (info["path"], data_dir)
    with _LOCK:
        if key in _SYSTEMIC:
            return dict(_SYSTEMIC[key])
    from astropy.io import fits
    obj = info.get("object") or ""
    found = dict(value=None, source=None)
    masks = sorted(glob.glob(os.path.join(data_dir, "masks",
                                          "LBL_Mask_%s_*.fits" % obj)))
    masks += sorted(glob.glob(os.path.join(os.path.dirname(info["path"]),
                                           "LBL_Mask_%s_*.fits" % obj)))
    for mask in masks:
        try:
            value = fits.getheader(mask).get(MASK_VELOCITY)
        except (OSError, ValueError):
            continue
        if value is not None and np.isfinite(float(value)):
            found = dict(value=round(float(value) / 1000.0, 4),
                         source=os.path.basename(mask))
            break
    if found["value"] is None and obj:
        from lbl.core.simbad import SUFFIXES
        from . import simbad
        star = SUFFIXES.sub("", obj)
        star = star[len("VETTED_"):] if star.upper().startswith("VETTED_") \
            else star
        answer = simbad.ask(star)
        if answer.get("rv") is not None:
            found = dict(value=round(float(answer["rv"]), 4),
                         source="SIMBAD (%s)" % answer.get("main", star))
    with _LOCK:
        _SYSTEMIC[key] = dict(found)
    return found


def doppler(velocity_kms: float) -> float:
    """The wavelength ratio of a velocity [km/s, positive receding]:
    relativistic, c from astropy."""
    from astropy import constants
    beta = float(velocity_kms) / constants.c.to("km/s").value
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))


def lines_of_interest() -> list:
    """The lines LBL measures the activity of (lbl/resources/
    activity_lines.yaml), in vacuum [nm]: label, species, wave, width (the
    window LBL measures in), kind."""
    from lbl.science.activity_lines import load_lines
    return [dict(label=line["LABEL"], species=line["SPECIES"],
                 wave=line["WAVE"], width=line["WIDTH"], kind=line["KIND"])
            for line in load_lines()]


def template_view(rdb: str, data_dir: str, lo=None, hi=None,
                  nbin: int = 1500, tapas: bool = True,
                  vsys=None) -> dict:
    """The template of a table over [lo, hi] nm (all of it when not given)
    and (tapas) the TAPAS transmission of each molecule over the same
    range, both cut to about nbin bins. `vsys` [km/s]: the template brought
    to its star's rest frame (its wavelengths divided by the Doppler factor
    of vsys), where the lines of a species are at their laboratory
    wavelengths; 'auto' for the systemic velocity found (systemic), None
    or '' for the template as it is. The TAPAS transmission stays in the
    telescope's frame."""
    info = template_of(rdb, data_dir)
    if info is None:
        return dict(template=None)
    wave, flux = _template_arrays(info["path"])
    found = systemic(info, data_dir)
    shift = None
    if vsys == "auto":
        shift = found["value"]
    elif vsys not in (None, ""):
        shift = float(vsys)
    if shift:
        wave = wave / doppler(shift)
    finite = np.isfinite(flux)
    full = [float(wave[finite].min()), float(wave[finite].max())]
    lo = full[0] if lo in (None, "") else float(lo)
    hi = full[1] if hi in (None, "") else float(hi)
    nbin = int(min(max(int(nbin), 100), 5000))
    x, y = envelope(wave, flux, lo, hi, nbin)
    out = dict(template=info, full=full, range=[lo, hi],
               x=np.round(x, 6).tolist(), y=_listed(np.round(y, 6)),
               vsys=dict(found, shift=shift))
    tapas = tapas_file(data_dir, info.get("instrument") or "") \
        if tapas else None
    if tapas is not None:
        twave, columns, cwave, coarse, _ = _tapas_arrays(tapas["path"],
                                                         tapas["kind"])
        i0 = max(int(np.searchsorted(twave, lo)) - 1, 0)
        i1 = min(int(np.searchsorted(twave, hi)) + 1, len(twave))
        use_coarse = i1 - i0 > 2 * nbin * TAPAS_COARSE
        swave = None if use_coarse else np.asarray(twave[i0:i1], dtype=float)
        molecules = {}
        for name, column in columns.items():
            if use_coarse:
                mx, my = envelope(cwave, coarse[name], lo, hi, nbin, "min")
            else:
                mx, my = envelope(swave, _transmission(column[i0:i1]), lo,
                                  hi, nbin, "min")
            # a molecule with nothing in the range: left out
            if my.size and np.nanmin(my) < 1 - 1e-4:
                molecules[name] = dict(x=np.round(mx, 6).tolist(),
                                       y=_listed(np.round(my, 6)))
        out["tapas"] = dict(file=os.path.basename(tapas["path"]),
                            kind=tapas["kind"], molecules=molecules)
    return out
