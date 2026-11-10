"""The spectra of a data root's Downloads folder, put where LBL reads them.

A data root holds the spectra as they were downloaded, and one LBL data
folder (DATA_DIR) per instrument and data source, named as LBL names them
(lbl.instruments.select.InstDict):

    ROOT/Downloads/                                 as downloaded
    ROOT/{INSTRUMENT}_{SOURCE}/science/{OBJECT}/    DATA_DIR is
                                                    ROOT/{INSTRUMENT}_{SOURCE}
    ROOT/{INSTRUMENT}/science/{OBJECT}/             an instrument LBL reads
                                                    from one source only

(NIRPS_HE_APERO, SPIROU_CADC, HARPS_ORIG, ESPRESSO, SOPHIE...). A wrap is
given one of these as its DATA_DIR, and LBL works in it as it always has.

What a file is comes from its headers alone:

- the LBL instrument and data source that read it: its INSTRUME, NIRPS's
  mode, and the layout of its extensions (targets.layout_source: APERO's
  t.fits, the CADC's, ESO's S2D...);
- whether LBL reads it: a spectrum (into science/), a calibration (a blaze,
  into calib/), or another product of its pipeline (a CCF, an S1D...), put
  aside in Downloads/unused/;
- a setting one template must not mix, which gets a folder of its own
  (<OBJECT>_EGGS): HARPS's EGGS, SOPHIE's HE, ESPRESSO's modes and
  binnings other than HR 1x1;
- the star: where it is in J2000, from the target's coordinates the headers
  carry (APERO's PP_RA, from Gaia; ESO's, the TNG's and the OHP's TARG
  ALPHA; the CFHT's OBJRA; EXPRES's RA), brought back to 2000 by the proper
  motion they give, else where the telescope pointed.

The files of one star go to one folder, whatever their observers called it
(Gl849, GL849, M112): two groups of files are one star when they carry the
same name, or when their places agree within their tolerance (TOLERANCE;
a place given without its proper motion may be that of the night, and is
given as much more as the star moves between 2000 and then). The two stars
of a binary are close and move together: their files are kept apart when
their names say two components of one system (GJ 15 A and GJ 15 B,
AlphaCenA and AlphaCenB), or when SIMBAD knows their names as two stars at
those places (BL Cet and UV Cet, 2 arcsec apart). Files whose place could
be either star's are put with the nearer, and said.

A star's folder is the one it already has in that DATA_DIR (a target folder
whose first spectrum is that star's); else the name it has in another
DATA_DIR of the root; else the name its headers give most often, APERO's
(DRSOBJN) first, then one SIMBAD knows at that place.

Nothing is overwritten: a file already in its place, byte for byte, is a
copy and is taken out of Downloads; one of that name that differs is left
there, and said. Every move is written in Downloads/.dispatched.jsonl (from,
to, when), so that it can be undone.

What a file needs before LBL can read it is done on the way: a gzipped
spectrum is gunzipped (LBL reads *.fits), the blaze SOPHIE's archive puts
in the spectrum's file is written apart as the calibration its header
names, a HARPS or HARPS-N e2ds whose night's blaze is missing gets one made
from it (fetch.spectra.empirical_blaze, ESO does not distribute them), and
a header astropy cannot write back is fixed (LBL rewrites the spectra when
it cleans their tellurics).
"""
from __future__ import annotations

import datetime
import filecmp
import gzip
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import targets

#: the folder of a data root the downloads are put in
DOWNLOADS = "Downloads"
#: under DOWNLOADS: the products LBL does not read, put aside
UNUSED = "unused"
#: under DOWNLOADS: every move, one JSON line each
RECORD = ".dispatched.jsonl"
#: the files read (the others are left where they are)
SPECTRA = (".fits", ".fits.gz", ".fit", ".hd5", ".h5")
#: INSTRUME (EXPRES: INSTRMNT) as the headers write it, and the family of
#:    LBL's instruments it is
FAMILIES = {"SPIROU": "SPIROU", "NIRPS": "NIRPS", "HARPS": "HARPS",
            "HARPN": "HARPSN", "HARPS-N": "HARPSN", "HARPSN": "HARPSN",
            "ESPRESSO": "ESPRESSO", "SOPHIE": "SOPHIE", "CORALIE": "CORALIE",
            "CARMENES": "CARMENES", "EXPRES": "EXPRES", "NEID": "NEID",
            "ESPADONS": "ESPADONS", "MAROONX": "MAROONX",
            "MAROON-X": "MAROONX"}
#: the target's coordinates the headers carry, most trusted first: who
#:    writes them, the keys of RA, Dec, their epoch and the proper motion,
#:    how RA is written (deg: degrees; packed: hhmmss.s as a number; text:
#:    'hh:mm:ss.s', '10h19m36.3', or degrees as a number), and the proper
#:    motion's unit [mas/yr]. RA's proper motion is mu_alpha cos(delta),
#:    as Gaia's (eps Eri: ESO's -0.974758 arcsec/yr, Gaia's -974.758)
CATALOGUES = (
    ("APERO", "PP_RA", "PP_DEC", "PP_EPOCH", "PP_PMRA", "PP_PMDE", "deg",
     1.0),
    ("ESO", "HIERARCH ESO TEL TARG ALPHA", "HIERARCH ESO TEL TARG DELTA",
     "HIERARCH ESO TEL TARG EPOCH", "HIERARCH ESO TEL TARG PMA",
     "HIERARCH ESO TEL TARG PMD", "packed", 1000.0),
    ("TNG", "HIERARCH TNG TEL TARG ALPHA", "HIERARCH TNG TEL TARG DELTA",
     "HIERARCH TNG TEL TARG EPOCH", "HIERARCH TNG TEL TARG PMA",
     "HIERARCH TNG TEL TARG PMD", "text", 1000.0),
    ("OHP", "HIERARCH OHP TARG ALPHA", "HIERARCH OHP TARG DELTA",
     "HIERARCH OHP TARG EPOCH", "HIERARCH OHP TARG PMA",
     "HIERARCH OHP TARG PMD", "packed", 1000.0),
    ("CFHT", "OBJRA", "OBJDEC", "OBJEQUIN", "OBJRAPM", "OBJDECPM", "text",
     1000.0),
    ("EXPRES", "RA", "DEC", "EPOCH", "PMRA", "PMDEC", "text", 1.0),
)
#: where the telescope pointed, when no target's coordinates are given
#:    (the TNG's RA is in hours and its DEC in no unit: its RA-DEG first)
POINTINGS = (("RA-DEG", "DEC-DEG"), ("RA", "DEC"))
#: how far [arcsec] a place may be from the star's: the target's
#:    coordinates (catalogues agree to an arcsec; observers round), a
#:    telescope's pointing (before the star is centred)
TOLERANCE = dict(catalogue=2.5, pointing=30.0)
#: files of the same name are one star if their places agree this well
#:    [arcsec] (a nickname of two programmes is two stars)
SAME_NAME = 300.0
#: names that say nothing of the star
GENERIC = {"", "OBJECT", "STAR", "TARGET", "UNKNOWN", "NONE", "TEST", "SKY",
           "NAME", "SCIENCE", "NULL"}
#: names of a body of the solar system: one name, a place each night
MOVING = {"SUN", "SOL", "SOLAR", "MOON", "LUNA"}
#: the headers that name the target, the pipeline's own first: APERO's
#:    DRSOBJN, the observer's OBJECT, the observatories' TARG NAME
NAME_KEYS = ("DRSOBJN", "OBJECT", "HIERARCH ESO OBS TARG NAME",
             "HIERARCH TNG OBS TARG NAME", "HIERARCH OHP TARG NAME",
             "OBJNAME")
#: the products of ESO's pipelines (PRO CATG) LBL reads, and those it reads
#:    as calibrations
ESO_SCIENCE = {"S2D_A"}
ESO_CALIB = {"BLAZE_A"}
#: the folders of a DATA_DIR whose files are the same whatever the
#:    instrument: one copy in the root, each DATA_DIR's a link to it. LBL's
#:    models are (PHOENIX spectra, TAPAS, RESPROJ tables, default masks and
#:    blazes): what is an instrument's own has its name in it
#:    (template_SPIROU_APERO, pca2d_NIRPS_HE_CADC, blaze_NIRPS_HE_ESO.fits)
SHARED = ("models",)
#: the SIMBAD names asked in one dispatch, at most (each unknown name takes
#:    a few seconds)
SIMBAD_MOST = 40
#: the headers read at once
WORKERS = 8
#: ESPRESSO's setting that needs no folder of its own (the most used; the
#:    archive search names its folders alike: fetch.spectra.ESPRESSO_MAIN)
ESPRESSO_MAIN = "HR11"


# -----------------------------------------------------------------------------
# The folders of a root
# -----------------------------------------------------------------------------
def _inst_dict() -> dict:
    from .lblinfo import _select
    return _select().InstDict


def folder_of(instrument: str, source) -> str:
    """The DATA_DIR of an instrument and a data source under a root:
    {INSTRUMENT}_{SOURCE}, or {INSTRUMENT} for an instrument LBL reads from
    one source only (ESPRESSO, SOPHIE, NEID)."""
    sources = _inst_dict().get(instrument) or {}
    if len(sources) <= 1:
        return instrument
    return "%s_%s" % (instrument, source)


def readers() -> dict:
    """Every DATA_DIR name of a root, and its (instrument, source)."""
    return {folder_of(inst, source): (inst, str(source))
            for inst, sources in _inst_dict().items() for source in sources}


def reader_of(data_dir: str):
    """(instrument, source) of a DATA_DIR named as a root's are, else None."""
    return readers().get(os.path.basename(os.path.normpath(data_dir or "")))


def split(data_dir: str):
    """A wrap's DATA_DIR as (root, folder): ROOT/NIRPS_HE_APERO as (ROOT,
    'NIRPS_HE_APERO'), any other folder as (it, '')."""
    path = os.path.normpath(os.path.abspath(os.path.expanduser(
        data_dir or "")))
    if reader_of(path):
        return os.path.dirname(path), os.path.basename(path)
    return path, ""


def downloads(root: str) -> dict:
    """The Downloads folder of a root, made if the root is there, and how
    many files wait in it (those put aside in UNUSED and hidden ones not
    counted)."""
    path = os.path.join(root, DOWNLOADS)
    if os.path.isdir(root) and not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
    files = spectra = unused = 0
    for top, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        aside = os.path.relpath(top, path).split(os.sep)[0] == UNUSED
        for name in names:
            if name.startswith("."):
                continue
            if aside:
                unused += 1
                continue
            files += 1
            spectra += name.lower().endswith(SPECTRA)
    return dict(path=path, exists=os.path.isdir(path), files=files,
                spectra=spectra, unused=unused)


def layout(root: str) -> dict:
    """What a root holds: its DATA_DIRs (each with its instrument, data
    source and number of targets), whether it is itself a data folder of
    the old kind (a science/ of its own), and its Downloads."""
    root = os.path.abspath(os.path.expanduser(root or ""))
    known = readers()
    folders = []
    if os.path.isdir(root):
        for entry in sorted(os.scandir(root), key=lambda e: e.name):
            if entry.name in known and entry.is_dir():
                inst, source = known[entry.name]
                folders.append(dict(name=entry.name, instrument=inst,
                                    source=source, path=entry.path,
                                    targets=len(targets.target_names(
                                        entry.path))))
    return dict(root=root, exists=os.path.isdir(root),
                legacy=os.path.isdir(os.path.join(root, "science")),
                legacy_targets=len(targets.target_names(root)),
                folders=folders, downloads=downloads(root))


def sharing(root: str, home: str) -> dict:
    """How each SHARED folder of a DATA_DIR of the root stands: 'linked'
    (a link to the root's), 'link' (none yet: a link will be made),
    'merge' (a folder of its own, its files to go to the root's), or
    'other' (a link elsewhere, left as it is)."""
    out = {}
    for name in SHARED:
        path = os.path.join(root, home, name)
        if os.path.islink(path):
            same = os.path.realpath(path) == os.path.realpath(
                os.path.join(root, name))
            out[name] = "linked" if same else "other"
        elif os.path.isdir(path):
            out[name] = "merge"
        else:
            out[name] = "link"
    return out


def share(root: str, home: str) -> dict:
    """Each SHARED folder of a DATA_DIR of the root made a link to the
    root's (ROOT/{INSTRUMENT}_{SOURCE}/models -> ../models): LBL's models
    are downloaded once for every instrument. A folder of the DATA_DIR's
    own is merged into the root's first: a file the root lacks is moved
    there, a copy of one it has is removed; one that differs stays where it
    is, and so does its folder, not linked. Returns {name: what was done:
    linked, merged, kept (with the files that differ), or why not}."""
    out = {}
    for name, state in sharing(root, home).items():
        mine = os.path.join(root, home, name)
        common = os.path.join(root, name)
        if state in ("linked", "other"):
            out[name] = state
            continue
        os.makedirs(common, exist_ok=True)
        differ = []
        if state == "merge":
            for here, _, files in os.walk(mine):
                for one in files:
                    src = os.path.join(here, one)
                    dest = os.path.join(common, os.path.relpath(src, mine))
                    if not os.path.exists(dest):
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        os.replace(src, dest)
                    elif _same(src, dest):
                        os.remove(src)
                    else:
                        differ.append(os.path.relpath(src, mine))
            if differ:
                out[name] = dict(kept=differ)
                continue
            for here, _, _ in sorted(os.walk(mine), key=lambda w: -len(w[0])):
                try:
                    os.rmdir(here)
                except OSError:
                    pass
            if os.path.exists(mine):
                out[name] = "not emptied"
                continue
        try:
            os.symlink(os.path.join(os.pardir, name), mine,
                       target_is_directory=True)
        except OSError as exc:
            os.makedirs(mine, exist_ok=True)
            out[name] = "no link (%s): a folder of its own" % exc
            continue
        out[name] = "merged" if state == "merge" else "linked"
    return out


# -----------------------------------------------------------------------------
# What one file is
# -----------------------------------------------------------------------------
def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _card(headers, key):
    for head in headers:
        if key in head:
            value = head[key]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _packed(value, hours: bool):
    """ESO's and the OHP's hhmmss.s (ddmmss.s) as a number, in degrees."""
    value = _number(value)
    if value is None:
        return None
    sign = -1.0 if value < 0 else 1.0
    value = abs(value)
    whole = int(value // 10000)
    minutes = int(value // 100) % 100
    seconds = value - 10000 * whole - 100 * minutes
    out = sign * (whole + minutes / 60.0 + seconds / 3600.0)
    return out * 15.0 if hours else out


def _text(value, hours: bool):
    """'hh:mm:ss.s', '10h19m36.281', '-04:38:26.0' in degrees; a number is
    taken as degrees."""
    if isinstance(value, (int, float)):
        return _number(value)
    text = str(value).strip()
    if not re.search(r"[:hdm ]", text.lower()):
        return _number(text)
    sign = -1.0 if text.startswith("-") else 1.0
    parts = [p for p in re.split(r"[:hdms\s]+", text.lstrip("+-").lower())
             if p]
    try:
        numbers = [float(p) for p in parts[:3]]
    except ValueError:
        return None
    numbers += [0.0] * (3 - len(numbers))
    out = sign * (numbers[0] + numbers[1] / 60.0 + numbers[2] / 3600.0)
    return out * 15.0 if hours else out


def _angle(value, how: str, hours: bool):
    if value is None:
        return None
    if how == "deg":
        return _number(value)
    if how == "packed":
        return _packed(value, hours)
    return _text(value, hours)


def _year(value):
    """An epoch as a year: a year, a JD or an MJD."""
    value = _number(value)
    if value is None:
        return None
    if value > 1.0e5:
        return 2000.0 + (value - 2451545.0) / 365.25
    if value > 3000.0:
        return 2000.0 + (value - 51544.5) / 365.25
    return value


def propagate(ra, dec, pmra, pmdec, years):
    """(ra, dec) [deg] moved by a proper motion [mas/yr, RA's with
    cos(dec)] over `years`."""
    cosd = max(math.cos(math.radians(dec)), 1e-6)
    return ((ra + pmra * years / 3.6e6 / cosd) % 360.0,
            max(-90.0, min(90.0, dec + pmdec * years / 3.6e6)))


def separation(a, b) -> float:
    """The angle between two (ra, dec) [deg], in arcsec."""
    ra1, dec1, ra2, dec2 = (math.radians(v) for v in (a[0], a[1], b[0],
                                                       b[1]))
    half = (math.sin((dec2 - dec1) / 2) ** 2 + math.cos(dec1) * math.cos(dec2)
            * math.sin((ra2 - ra1) / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(half)))) * 3600.0


def position(headers, year=None):
    """Where the target of a file is: dict(ra, dec [deg, J2000 when the
    proper motion is given, else as written], pm [mas/yr] or None, loose
    [yr]: how long the star may have moved away from that place (the place
    of a night written as if of 2000), how: 'catalogue' or 'pointing',
    via: who wrote it), or None."""
    for via, kra, kdec, kepoch, kpmra, kpmdec, how, unit in CATALOGUES:
        ra = _angle(_card(headers, kra), how, True)
        dec = _angle(_card(headers, kdec), how, False)
        epoch = _year(_card(headers, kepoch))
        if ra is None or dec is None or epoch is None or abs(dec) > 90:
            continue
        pmra, pmdec = _number(_card(headers, kpmra)), \
            _number(_card(headers, kpmdec))
        if pmra is not None and pmdec is not None and (pmra or pmdec):
            pm = (pmra * unit, pmdec * unit)
            ra, dec = propagate(ra, dec, pm[0], pm[1], 2000.0 - epoch)
            return dict(ra=ra, dec=dec, pm=pm, loose=0.0, how="catalogue",
                        via=via)
        loose = max(abs(epoch - 2000.0),
                    abs((year or 2000.0) - 2000.0))
        return dict(ra=ra % 360.0, dec=dec, pm=None, loose=loose,
                    how="catalogue", via=via)
    for kra, kdec in POINTINGS:
        ra = _angle(_card(headers, kra), "text", True)
        dec = _angle(_card(headers, kdec), "text", False)
        if ra is None or dec is None or abs(dec) > 90:
            continue
        return dict(ra=ra % 360.0, dec=dec, pm=None,
                    loose=abs((year or 2000.0) - 2000.0), how="pointing",
                    via=kra)
    return None


def norm(name) -> str:
    """A name as it is compared: capitals and digits, a planet's letter or
    number taken off (GJ 1214 b, TOI-1452.01), Gliese and Gl as GJ."""
    text = str(name or "").strip()
    text = re.sub(r"\.\d+$", "", text)
    text = re.sub(r"[\s_-]+[b-h]$", "", text)
    text = re.sub(r"[^A-Za-z0-9+]", "", text).upper()
    return re.sub(r"^(GLIESE|GL)(?=\d)", "GJ", text)


def component(name):
    """(system, letter) of a name that says one star of a multiple system
    (GJ 15 A, Gl15B, AlphaCenA: a capital after a digit, a small letter or a
    space), else None. APERO's names are capitals (ALPHACENB): their
    OBJECT is asked first."""
    text = re.sub(r"\.\d+$", "", str(name or "").strip())
    found = re.match(r"^(.*?[0-9a-z])[\s_-]*([A-D])$", text)
    if not found or not norm(found.group(1)):
        return None
    return norm(found.group(1)), found.group(2)


def clean(name) -> str:
    """A name as a folder of LBL's: letters, digits, + and -, the rest as
    single underscores (Eps_eri, KIC_12060710, G268-110)."""
    text = re.sub(r"\.\d+$", "", str(name or "").strip())
    return re.sub(r"[^A-Za-z0-9+-]+", "_", text).strip("_")


def _names(headers) -> list:
    out = []
    for key in NAME_KEYS:
        value = _card(headers, key)
        if value is None:
            continue
        value = str(value).strip()
        if norm(value) not in GENERIC and value not in out:
            out.append(value)
    return out


def _kind(family, source, name, headers, extnames, primary):
    """What LBL makes of a file: ('science' | 'calib' | 'unused', why)."""
    lower = name.lower()
    catg = str(_card(headers, "HIERARCH ESO PRO CATG")
               or _card(headers, "HIERARCH TNG PRO CATG") or "").strip()
    dpr = str(_card(headers, "HIERARCH ESO DPR CATG")
              or _card(headers, "HIERARCH TNG DPR CATG") or "").strip()
    if catg.upper() in ESO_CALIB or re.search(r"_(blaze|wave)_a\.fits",
                                              lower):
        return "calib", "a calibration"
    if dpr and dpr.upper() != "SCIENCE":
        return "unused", "a %s exposure" % dpr
    if source == "ESSP":
        return "science", ""
    if family in ("SPIROU", "NIRPS") and source in ("APERO", "CADC"):
        apero = str(_card(headers, "DPRTYPE") or "")
        if apero and not apero.upper().startswith(("OBJ", "POLAR")):
            return "unused", "a %s exposure" % apero
        if re.search(r"t\.fits(\.gz)?$", lower) or "tcorr" in lower:
            return "science", ""
        return "unused", "APERO's other products (LBL reads the t.fits)"
    if source == "ESO" or family == "ESPRESSO":
        if catg.upper() in ESO_SCIENCE or "HIERARCH LBL TELL FLOOR" in primary:
            return "science", ""
        return "unused", "%s (LBL reads the S2D_A)" % (catg or "a product")
    if family in ("HARPS", "HARPSN", "SOPHIE", "CORALIE"):
        if re.search(r"e2ds_a\.fits(\.gz)?$", lower):
            return "science", ""
        return "unused", "the DRS's other products (LBL reads the e2ds_A)"
    if family == "EXPRES":
        if "OPTIMAL" in extnames:
            return "science", ""
        return "unused", "no optimal extraction"
    if family == "CARMENES":
        if "nir" in lower and "vis" not in lower:
            return "unused", "CARMENES's NIR arm (LBL reads the VIS one)"
        if re.search(r"_b\.fits(\.gz)?$", lower):
            return "unused", "fibre B"
        return "science", ""
    if family == "ESPADONS":
        if re.search(r"i\.fits(\.gz)?$", lower):
            return "science", ""
        return "unused", "not an intensity spectrum (LBL reads the i.fits)"
    return "science", ""


def _setting(family, headers) -> str:
    """The setting of a file one template must not mix with the others, ''
    for the instrument's main one."""
    if family == "HARPS" and str(_card(headers, "HIERARCH ESO INS MODE")
                                 or "").strip().upper() == "EGGS":
        return "EGGS"
    if family == "SOPHIE" and str(_card(headers, "HIERARCH OHP INS FIBER")
                                  or "").strip().upper() == "HE":
        return "HE"
    if family == "ESPRESSO":
        mode = str(_card(headers, "HIERARCH ESO INS MODE") or "").strip()
        mode = mode.upper().replace("SINGLE", "").replace("MULTI", "")
        binning = "%s%s" % (_card(headers, "HIERARCH ESO DET BINX") or 1,
                            _card(headers, "HIERARCH ESO DET BINY") or 1)
        key = (mode or "HR") + binning
        return "" if key == ESPRESSO_MAIN else key
    return ""


def _blank(path: str) -> dict:
    stat = os.stat(path)
    return dict(path=path, name=os.path.basename(path), size=stat.st_size,
                instrument=None, source=None, family=None, kind=None,
                setting="", why="", names=[], component=None, place=None,
                year=None, blaze=None, split=False, extra=[])


def read(path: str) -> dict:
    """One file: its instrument and data source, what LBL makes of it
    (kind: science, calib, unused, or None and why), its setting, the names
    its headers give the target, the component of a binary they say, and
    the target's place (position)."""
    out = _blank(path)
    if path.lower().endswith((".hd5", ".h5")):
        return _maroonx(path, out)
    try:
        from astropy.io import fits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=False, lazy_load_hdus=False) as hdul:
                headers = [hdu.header for hdu in hdul]
                extnames = {str(hdu.name).upper() for hdu in hdul}
                primary = headers[0]
                source = targets.layout_source(targets._instrument(headers),
                                               hdul)
    except Exception as exc:                              # noqa: BLE001
        out["why"] = "not a FITS file astropy reads (%s)" % type(exc).__name__
        return out
    said = targets._instrument(headers)
    family = FAMILIES.get(re.sub(r"[^A-Z0-9-]", "", said))
    if family is None:
        out["why"] = ("no instrument in its header" if said == "?"
                      else "%s is not an instrument LBL reads" % said)
        return out
    out["family"] = family
    instrument = family
    if family == "NIRPS":
        mode = str(_card(headers, "DRSMODE")
                   or _card(headers, "HIERARCH ESO INS MODE") or "").upper()
        if mode not in ("HE", "HA"):
            out["why"] = "NIRPS, but neither HE nor HA (%s)" % (mode or "?")
            return out
        instrument = "NIRPS_" + mode
    sources = list((_inst_dict().get(instrument) or {}).keys())
    if family in ("HARPS", "HARPSN", "EXPRES", "NEID") and "FLUX" in extnames:
        source = "ESSP"
    elif family == "EXPRES":
        source = "ORIG"
    elif family == "NEID":
        # NEID's own L2 files: LBL reads NEID through the ESSP's only
        source = None
    elif family in ("SPIROU", "NIRPS", "HARPS", "HARPSN"):
        catg = str(_card(headers, "HIERARCH ESO PRO CATG") or "").upper()
        if source is None and catg in ESO_CALIB:
            source = "ESO"
        if source is None and re.search(r"_(blaze|wave)_a\.fits",
                                        out["name"].lower()):
            source = "ORIG"
    elif len(sources) == 1:
        source = sources[0]
    if source not in sources:
        out["why"] = ("%s: LBL reads it from %s, and its layout is none of"
                      " them" % (instrument, ", ".join(sources) or "nothing"))
        return out
    out.update(instrument=instrument, source=source)
    kind, why = _kind(family, source, out["name"], headers, extnames, primary)
    out.update(kind=kind, why=why)
    if kind == "calib":
        return out
    mjd = targets._first(headers, targets.MJD_KEYS)
    if mjd is not None and mjd > 2400000.5:
        mjd -= 2400000.5
    out["year"] = None if mjd is None else 2000.0 + (mjd - 51544.5) / 365.25
    out["setting"] = _setting(family, headers)
    out["names"] = _names(headers)
    if family == "EXPRES":
        # EXPRES names its stars by their HD number alone (10700)
        out["names"] = ["HD" + n if n.isdigit() else n for n in out["names"]]
    # the component a name says: the observer's spelling (APERO's DRSOBJN
    #   is in capitals) first
    for name in out["names"][1:] + out["names"][:1]:
        out["component"] = component(name)
        if out["component"]:
            break
    out["place"] = position(headers, out["year"])
    if kind == "science" and family in ("HARPS", "HARPSN", "SOPHIE",
                                        "CORALIE"):
        key = {"HARPS": "HIERARCH ESO DRS BLAZE FILE",
               "CORALIE": "HIERARCH ESO DRS BLAZE FILE",
               "HARPSN": "HIERARCH TNG DRS BLAZE FILE",
               "SOPHIE": "HIERARCH OHP DRS BLAZE FILE"}[family]
        blaze = _card(headers, key)
        out["blaze"] = os.path.basename(str(blaze).strip()) if blaze else None
        out["split"] = family == "SOPHIE" and "BLAZE_A" in extnames
    return out


def _maroonx(path: str, out: dict) -> dict:
    """A MAROON-X spectrum (HDF5): its arm from its name (_r_, _b_, or _x_
    for both, read by MAROONX_RED and MAROONX_BLUE alike), its target from
    the header the file stores."""
    lower = out["name"].lower()
    arms = [arm for arm, mark in (("RED", "_r_"), ("BLUE", "_b_"))
            if mark in lower or "_x_" in lower]
    if not arms:
        out["why"] = "MAROON-X? no arm (_r_, _b_, _x_) in its name"
        return out
    out.update(family="MAROONX", instrument="MAROONX", source=arms[0],
               extra=[("MAROONX", arm) for arm in arms[1:]])
    if "masterflat" in lower:
        out.update(kind="calib", why="a calibration")
        return out
    try:
        import pandas
        key = "header_red" if arms[0] == "RED" else "header_blue"
        header = pandas.read_hdf(path, key)
        header = dict(header.items()) if hasattr(header, "items") else {}
    except Exception as exc:                              # noqa: BLE001
        out.update(kind=None, why="its header could not be read (%s)"
                   % type(exc).__name__)
        return out
    out.update(kind="science", names=_names([header]))
    out["place"] = position([header])
    return out


# -----------------------------------------------------------------------------
# Which files are one star
# -----------------------------------------------------------------------------
def _sightings(records) -> list:
    """The records grouped by name and place (the files of a star of one
    night or programme, most often all of them): dict(names, norms,
    component, place, pm, loose, files)."""
    out = {}
    for index, rec in enumerate(records):
        place = rec.get("place")
        names = rec.get("names") or []
        key = (norm(names[0]) if names else "",
               None if place is None else (round(place["ra"], 4),
                                           round(place["dec"], 4),
                                           place["how"]),
               rec.get("component"))
        one = out.get(key)
        if one is None:
            one = out[key] = dict(names=Counter(), norms=set(),
                                  component=rec.get("component"),
                                  place=place, pm=None, loose=0.0, files=[])
        one["files"].append(index)
        for name in names:
            one["names"][name] += 1
            one["norms"].add(norm(name))
        if place is not None:
            one["loose"] = max(one["loose"], place["loose"])
            if place["pm"] is not None:
                one["pm"] = place["pm"]
    return list(out.values())


def _speed(*sightings) -> float:
    """The fastest proper motion of a star known among its sightings
    [mas/yr], 0 when none says."""
    speeds = [math.hypot(*s["pm"]) for s in sightings if s.get("pm")]
    return max(speeds) if speeds else 0.0


def tolerance(a, b, speed=None) -> float:
    """How far apart [arcsec] two sightings of one star may be: each its
    own (TOLERANCE), and for a place given without its proper motion, as
    far as the star moves between 2000 and the night."""
    speed = _speed(a, b) if speed is None else speed
    out = 0.0
    for one in (a, b):
        place = one["place"]
        out += TOLERANCE[place["how"]]
        if place["pm"] is None:
            out += speed * one["loose"] / 1000.0
    return out


def _two_components(a, b) -> bool:
    """Whether two sightings are two stars of one system by their names."""
    ca, cb = a.get("component"), b.get("component")
    return bool(ca and cb and ca[0] == cb[0] and ca[1] != cb[1])


def _same_name(a, b) -> bool:
    common = (a["norms"] & b["norms"]) - GENERIC
    if not common:
        return False
    if common & MOVING or a["place"] is None or b["place"] is None:
        return True
    return separation((a["place"]["ra"], a["place"]["dec"]),
                      (b["place"]["ra"], b["place"]["dec"])) <= max(
        SAME_NAME, tolerance(a, b))


def _link(a, b):
    """How close two sightings are as one star (0: by name; below 1: by
    place, a fraction of their tolerance), or None if they are not."""
    if _two_components(a, b):
        return None
    if _same_name(a, b):
        return 0.0
    if a["place"] is None or b["place"] is None:
        return None
    sep = separation((a["place"]["ra"], a["place"]["dec"]),
                     (b["place"]["ra"], b["place"]["dec"]))
    tol = tolerance(a, b)
    return sep / tol if sep <= tol else None


def stars(sightings, apart=frozenset()) -> list:
    """The sightings grouped into stars: the nearest pairs joined first,
    two groups joined only if every pair of their sightings may be one star
    (no two components, no two stars SIMBAD knows apart, two places of the
    targets' coordinates within their tolerance) so that a place far from
    both stars of a binary cannot join them. `apart`: pairs of sightings
    (frozenset of two indices) that are two stars."""
    count = len(sightings)
    links = {}
    for i in range(count):
        for j in range(i + 1, count):
            if frozenset((i, j)) in apart:
                continue
            how = _link(sightings[i], sightings[j])
            if how is not None:
                links[(i, j)] = how
    group = list(range(count))
    members = {i: [i] for i in range(count)}

    def allowed(first, second):
        for i in first:
            for j in second:
                a, b = sightings[i], sightings[j]
                if frozenset((i, j)) in apart or _two_components(a, b):
                    return False
                if (min(i, j), max(i, j)) in links:
                    continue
                # two places of the targets' coordinates that disagree
                if a["place"] and b["place"] \
                        and a["place"]["how"] == b["place"]["how"] \
                        == "catalogue":
                    return False
        return True

    for (i, j), _ in sorted(links.items(), key=lambda item: item[1]):
        gi, gj = group[i], group[j]
        if gi == gj or not allowed(members[gi], members[gj]):
            continue
        for k in members[gj]:
            group[k] = gi
        members[gi] += members.pop(gj)
    return [sorted(m) for m in members.values()]


def _simbad_names(sightings, groups, records, budget=SIMBAD_MOST):
    """The names SIMBAD is asked: those of the stars whose sightings carry
    more than one name (two names at one place may be two stars), and those
    of a star new to the root (a name SIMBAD knows there is a better
    folder than a programme's nickname), each spelling once; at most
    `budget`."""
    out = []
    for group in groups:
        norms = set()
        for i in group:
            norms |= sightings[i]["norms"]
        new = not any(records[k].get("folder") for i in group
                      for k in sightings[i]["files"])
        if len(norms - GENERIC) < 2 and not new:
            continue
        for i in group:
            for name, _ in sightings[i]["names"].most_common():
                if name not in out and norm(name) not in GENERIC:
                    out.append(name)
    return out[:budget], len(out) > budget


def _known(ident, sighting) -> bool:
    """Whether SIMBAD's star of a name is where a sighting is: a name that
    SIMBAD knows as a star elsewhere (a programme's nickname) says nothing
    of this one."""
    if not ident or ident.get("ra") is None or sighting["place"] is None:
        return False
    pm = None
    if ident.get("pmra") is not None and ident.get("pmdec") is not None:
        pm = (ident["pmra"], ident["pmdec"])
    simbad = dict(place=dict(ra=ident["ra"], dec=ident["dec"], pm=pm,
                             loose=0.0, how="catalogue"), pm=pm, loose=0.0)
    speed = max(_speed(sighting), math.hypot(*pm) if pm else 0.0)
    sep = separation((ident["ra"], ident["dec"]),
                     (sighting["place"]["ra"], sighting["place"]["dec"]))
    return sep <= tolerance(simbad, sighting, speed)


def apart_by_simbad(sightings, groups, idents) -> set:
    """Pairs of sightings of one group that SIMBAD knows as two stars: each
    has a name SIMBAD places there, and no such name is the other's star."""
    out = set()
    for group in groups:
        mains = {}
        for i in group:
            found = set()
            for name in sightings[i]["names"]:
                ident = idents.get(name)
                if _known(ident, sightings[i]):
                    found.add(norm(ident["main"]))
            mains[i] = found
        for a in group:
            for b in group:
                if a < b and mains[a] and mains[b] \
                        and not mains[a] & mains[b]:
                    out.add(frozenset((a, b)))
    return out


def star_folder(data_dir: str, ident: dict, setting: str = ""):
    """The target folder of a DATA_DIR that holds a star, as SIMBAD has it
    (ident: main, aliases, ra, dec in J2000, pmra, pmdec): the one whose
    first spectrum is the star's by its place or one of its names, in the
    same setting; None if there is none. The archive search fetches into
    it, rather than into a folder of the name it was asked."""
    pm = None
    if ident.get("pmra") is not None and ident.get("pmdec") is not None:
        pm = (float(ident["pmra"]), float(ident["pmdec"]))
    names = [n for n in [ident.get("main")] + list(ident.get("aliases") or [])
             if n]
    star = dict(names=Counter(names), norms={norm(n) for n in names},
                component=component(ident.get("name") or ""),
                place=dict(ra=float(ident["ra"]), dec=float(ident["dec"]),
                           pm=pm, loose=0.0, how="catalogue"),
                pm=pm, loose=0.0)
    best = None
    for name in targets.target_names(data_dir):
        paths = targets.files_of(data_dir, name)
        if not paths:
            continue
        try:
            rec = read(paths[0])
        except OSError:
            continue
        if rec.get("kind") != "science" or rec.get("setting") != setting:
            continue
        how = _link(star, _sightings([rec])[0])
        if how is not None and (best is None or how < best[0]):
            best = (how, name)
    return best[1] if best else None


# -----------------------------------------------------------------------------
# The plan: where each file goes
# -----------------------------------------------------------------------------
def _files(root: str) -> list:
    """The files of Downloads/ waiting (not hidden, not put aside)."""
    top = os.path.join(root, DOWNLOADS)
    out = []
    for here, dirs, names in os.walk(top):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                         and not (here == top and d == UNUSED))
        out += [os.path.join(here, n) for n in sorted(names)
                if not n.startswith(".")]
    return out


def _existing(root: str) -> list:
    """The target folders already in the root, each as a record of its
    first spectrum: in its DATA_DIRs (folder: (DATA_DIR name, folder)), and
    in a science/ of the root's own (DATA_DIR name '')."""
    from lbl.core.simbad import SUFFIXES
    out = []
    homes = [(name, os.path.join(root, name)) for name in readers()
             if os.path.isdir(os.path.join(root, name, "science"))]
    if os.path.isdir(os.path.join(root, "science")):
        homes.append(("", root))
    for home, data_dir in homes:
        for name in targets.target_names(data_dir):
            paths = targets.files_of(data_dir, name)
            if not paths:
                continue
            try:
                rec = read(paths[0])
            except OSError:
                continue
            if rec["place"] is None and not rec["names"]:
                continue
            base = SUFFIXES.sub("", name)
            if rec["setting"] and base.endswith("_" + rec["setting"]):
                base = base[:-len(rec["setting"]) - 1]
            rec.update(folder=dict(home=home, name=name, base=base,
                                   setting=rec["setting"],
                                   files=len(paths)),
                       names=rec["names"] + [base])
            out.append(rec)
    return out


def _folder_name(star, records, idents):
    """The name of a star's folders: its name in the root (its DATA_DIRs
    first, then a science/ of the root's own), else APERO's, else the name
    of its headers SIMBAD knows there, else the one they give most."""
    homes = [records[i]["folder"] for i in star if records[i].get("folder")]
    for new in (True, False):
        names = Counter()
        for one in homes:
            if bool(one["home"]) == new:
                names[one["base"]] += one["files"]
        if names:
            return names.most_common(1)[0][0]
    files = [records[i] for i in star if not records[i].get("folder")]
    for pick in ("apero", "known", "any"):
        names = Counter()
        for rec in files:
            if pick == "apero":
                chosen = [rec["drsobjn"]] if rec.get("drsobjn") else []
            elif pick == "known":
                chosen = [n for n in rec["names"]
                          if rec["place"] is not None
                          and _known(idents.get(n), dict(
                              place=rec["place"], pm=rec["place"]["pm"],
                              loose=rec["place"]["loose"]))]
            else:
                chosen = rec["names"]
            for name in chosen[:1]:
                names[clean(name)] += 1
        if names:
            best = max(names.values())
            return min((n for n, c in names.items() if c == best),
                       key=lambda n: (len(n), n))
    return "star"


def plan(root: str, idents=None, progress=None, ask_simbad=True) -> dict:
    """Where each file of a root's Downloads goes, and why: a plan to show
    before anything is moved (go). `idents`: SIMBAD's answers by name, kept
    between plans; `progress(done, total, step)`: told as it reads."""
    root = os.path.abspath(os.path.expanduser(root))
    idents = {} if idents is None else idents
    say = progress or (lambda *args: None)
    paths = _files(root)
    say(0, len(paths), "headers")
    records = [None] * len(paths)
    done = [0]
    lock = threading.Lock()

    def one(index):
        try:
            records[index] = read(paths[index])
        except OSError as exc:
            records[index] = dict(_blank(paths[index]), why=str(exc))
        with lock:
            done[0] += 1
            say(done[0], len(paths), "headers")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, range(len(paths))))
    for rec in records:
        drs = None
        if rec.get("names") and rec.get("family") in ("SPIROU", "NIRPS") \
                and rec.get("source") in ("APERO", "CADC"):
            drs = rec["names"][0]
        rec["drsobjn"] = drs
    say(len(paths), len(paths), "folders")
    existing = _existing(root)
    stars_of = [r for r in records if r.get("kind") == "science"] + existing
    sightings = _sightings(stars_of)
    groups = stars(sightings)
    warnings_ = []
    simbad = "not needed"
    names, cut = _simbad_names(sightings, groups, stars_of)
    asked = [n for n in names if n not in idents]
    if ask_simbad and asked:
        say(0, len(asked), "simbad")
        simbad = _ask_simbad(asked, idents, say)
        if cut:
            warnings_.append(dict(key="simbad_cut", args=[SIMBAD_MOST]))
    elif names:
        simbad = "asked" if ask_simbad else "not asked"
    if simbad == "unreachable":
        warnings_.append(dict(key="simbad_unreachable", args=[]))
    apart = apart_by_simbad(sightings, groups, idents)
    if apart:
        groups = stars(sightings, frozenset(apart))
    # each record its star (indices into stars_of)
    star_of = {}
    for number, group in enumerate(groups):
        for s in group:
            for i in sightings[s]["files"]:
                star_of[i] = number
    star_records = [[i for i, n in star_of.items() if n == number]
                    for number in range(len(groups))]
    warnings_ += _either(sightings, groups, apart, idents, stars_of)
    return _destinations(root, records, stars_of, star_records, idents,
                         warnings_, simbad)


def _either(sightings, groups, apart, idents, records) -> list:
    """The downloaded files whose place could be that of either star of a
    pair SIMBAD keeps apart, and whose names do not say which: put with
    the nearer, and said."""
    out = []
    home = {s: n for n, group in enumerate(groups) for s in group}
    for s, sighting in enumerate(sightings):
        if all(records[i].get("folder") for i in sighting["files"]) \
                or any(_known(idents.get(n), sighting)
                       for n in sighting["names"]):
            continue
        mine = home[s]
        for n, group in enumerate(groups):
            if n == mine or not any(frozenset((a, b)) in apart
                                    for a in groups[mine] for b in group):
                continue
            if any(_link(sighting, sightings[t]) is not None for t in group):
                out.append(dict(key="either_star", args=[
                    ", ".join(sighting["names"]) or "?",
                    _star_label(sightings, groups[mine]),
                    _star_label(sightings, group)]))
    return out


def _star_label(sightings, group) -> str:
    names = Counter()
    for s in group:
        names.update(sightings[s]["names"])
    return names.most_common(1)[0][0] if names else "?"


def _ask_simbad(names, idents, say) -> str:
    """SIMBAD's star of each name (fetch.stars.resolve, kept on disk), None
    for a name it does not know; 'unreachable' if it does not answer."""
    from .fetch.stars import resolve
    state = dict(failed=0, done=0)
    lock = threading.Lock()

    def one(name):
        with lock:
            if state["failed"]:
                return
        try:
            found = resolve(name)
        except ValueError:
            found = None
        except Exception:                                 # noqa: BLE001
            with lock:
                state["failed"] += 1
            return
        with lock:
            idents[name] = found
            state["done"] += 1
            say(state["done"], len(names), "simbad")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, names))
    return "unreachable" if state["failed"] else "asked"


def _destinations(root, records, stars_of, star_records, idents, warnings_,
                  simbad) -> dict:
    """Each file its place: science/<folder> of its DATA_DIR, calib/, or
    UNUSED; the folders shown star by star."""
    moves, left, unused = [], [], []
    rows = {}
    taken = {}
    for number, members in enumerate(star_records):
        base = _folder_name(members, stars_of, idents)
        homes = {}
        for i in members:
            folder = stars_of[i].get("folder")
            if folder and folder["home"]:
                homes.setdefault((folder["home"], folder["setting"]),
                                 []).append(folder)
        for i in members:
            rec = stars_of[i]
            if rec.get("folder"):
                continue
            for inst, source in [(rec["instrument"], rec["source"])] \
                    + list(rec.get("extra") or []):
                home = folder_of(inst, source)
                key = (home, rec["setting"])
                if key in homes:
                    name = max(homes[key], key=lambda f: f["files"])["name"]
                    new = False
                else:
                    name = base + ("_" + rec["setting"] if rec["setting"]
                                   else "")
                    name = _free(root, home, name, taken, number)
                    new = not os.path.isdir(os.path.join(root, home,
                                                         "science", name))
                taken.setdefault((home, name), number)
                row = rows.setdefault((home, name), dict(
                    star=number, home=home, folder=name, new=new,
                    instrument=inst, source=source, science=0,
                    names=Counter(), size=0))
                row["science"] += 1
                row["size"] += rec["size"]
                for one in rec["names"][:1]:
                    row["names"][one] += 1
                moves.append(dict(src=rec["path"], dest=os.path.join(
                    root, home, "science", name, _target_name(rec)),
                    kind="science", home=home, blaze=rec.get("blaze"),
                    split=rec.get("split"), family=rec.get("family"),
                    link=(inst, source) != (rec["instrument"],
                                            rec["source"])))
    for rec in records:
        if rec.get("kind") == "calib":
            for inst, source in [(rec["instrument"], rec["source"])] \
                    + list(rec.get("extra") or []):
                home = folder_of(inst, source)
                moves.append(dict(src=rec["path"], dest=os.path.join(
                    root, home, "calib", _target_name(rec)), kind="calib",
                    home=home, link=(inst, source) != (rec["instrument"],
                                                       rec["source"])))
        elif rec.get("kind") == "unused":
            unused.append(dict(path=rec["path"], name=rec["name"],
                               why=rec["why"], instrument=rec["instrument"]))
        elif rec.get("kind") is None:
            left.append(dict(path=rec["path"], name=rec["name"],
                             why=rec["why"]))
    calib = Counter(m["home"] for m in moves if m["kind"] == "calib")
    folders = sorted(rows.values(), key=lambda r: (r["star"], r["home"],
                                                   r["folder"]))
    for row in folders:
        row["names"] = [n for n, _ in row["names"].most_common()][:6]
    # the DATA_DIRs of the root, those this plan makes too: their models
    #   shared (SHARED)
    homes = sorted({m["home"] for m in moves} | {
        name for name in readers() if os.path.isdir(os.path.join(root,
                                                                 name))})
    shares = [dict(home=home, name=name, state=state) for home in homes
              for name, state in sharing(root, home).items()
              if state in ("link", "merge")]
    return dict(root=root, folders=folders, homes=homes, shares=shares,
                calibs=[dict(home=h, n=c) for h, c in sorted(calib.items())],
                unused=unused, left=left, warnings=warnings_, simbad=simbad,
                moves=moves, files=len(records),
                science=sum(r["science"] for r in folders),
                calib=sum(calib.values()))


def _target_name(rec) -> str:
    """A file's name in its folder: a gzipped spectrum without its .gz (it
    is gunzipped on the way)."""
    name = rec["name"]
    return name[:-3] if name.lower().endswith(".fits.gz") else name


def _free(root, home, name, taken, number) -> str:
    """A folder name of a DATA_DIR not already another star's (one of the
    root, or one this plan gives another star): name, name_2..."""
    out, count = name, 1
    while True:
        owner = taken.get((home, out))
        here = os.path.isdir(os.path.join(root, home, "science", out))
        if (owner is None and not here) or owner == number:
            return out
        count += 1
        out = "%s_%d" % (name, count)


# -----------------------------------------------------------------------------
# Moving the files
# -----------------------------------------------------------------------------
def _same(a: str, b: str) -> bool:
    try:
        return os.path.getsize(a) == os.path.getsize(b) and filecmp.cmp(
            a, b, shallow=False)
    except OSError:
        return False


def _record(root: str, src: str, dest: str, how: str) -> None:
    with open(os.path.join(root, DOWNLOADS, RECORD), "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps(dict(when=datetime.datetime.now().isoformat(
            timespec="seconds"), src=src, dest=dest, how=how)) + "\n")


def _write_fits(hdul, dest: str) -> None:
    part = "%s.%s.part" % (dest, uuid.uuid4().hex)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdul.writeto(part, output_verify="silentfix+ignore")
    os.replace(part, dest)


def _put(move: dict) -> str:
    """One file into its place (gunzipped, its blaze split out, its header
    fixed when it needs it); '' when done, else why not."""
    from astropy.io import fits
    src, dest = move["src"], move["dest"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if move.get("split"):
        with fits.open(src, memmap=False) as hdul:
            head = hdul[0].header.copy()
            blaze = move.get("blaze") or os.path.basename(dest).replace(
                "_e2ds_A", "_blaze_A")
            head["HIERARCH OHP DRS BLAZE FILE"] = blaze
            calib = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(dest))), "calib")
            os.makedirs(calib, exist_ok=True)
            if not os.path.exists(os.path.join(calib, blaze)):
                _write_fits(fits.HDUList([fits.PrimaryHDU(
                    hdul["BLAZE_A"].data)]), os.path.join(calib, blaze))
            _write_fits(fits.HDUList([fits.PrimaryHDU(hdul[0].data, head)]),
                        dest)
        return ""
    if src.lower().endswith(".gz"):
        part = "%s.%s.part" % (dest, uuid.uuid4().hex)
        with gzip.open(src, "rb") as raw, open(part, "wb") as handle:
            shutil.copyfileobj(raw, handle)
        os.replace(part, dest)
        return ""
    if move.get("family") in ("HARPS", "HARPSN") and move["kind"] == "science":
        # the DRS writes cards astropy cannot write back (ESO INS TEMP1
        #   RMS...): fixed here, or LBL stops on the first file it rewrites
        #   (its telluric cleaning)
        with fits.open(src, memmap=False) as hdul:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hdul.verify("exception")
            except Exception:                             # noqa: BLE001
                hdul.verify("silentfix+ignore")
                _write_fits(hdul, dest)
                return ""
    if move.get("link"):
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        return ""
    try:
        os.replace(src, dest)
    except OSError:
        shutil.move(src, dest)
    return ""


def _blazes(moves) -> list:
    """The blazes the e2ds just put in place name and their DATA_DIR does
    not have: (e2ds, blaze path), one e2ds per blaze."""
    out, seen = [], set()
    for one in moves:
        if one["kind"] != "science" or not one.get("blaze") \
                or one.get("split") or one.get("family") not in ("HARPS",
                                                                 "HARPSN"):
            continue
        calib = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            one["dest"]))), "calib")
        path = os.path.join(calib, one["blaze"])
        if path in seen or os.path.exists(path):
            continue
        seen.add(path)
        out.append((one["dest"], path))
    return out


def go(plan_: dict, progress=None) -> dict:
    """The moves of a plan: each file into its place, those of a DATA_DIR
    whose name is taken by a different file left in Downloads, the copies
    of a file already there taken out, the products LBL does not read put
    aside in Downloads/unused/; then the blazes the HARPS and HARPS-N
    spectra lack, the folders of Downloads left empty, and the models of
    each DATA_DIR made a link to the root's (share)."""
    from astropy.io import fits
    say = progress or (lambda *args: None)
    root = plan_["root"]
    top = os.path.join(root, DOWNLOADS)
    moves = plan_["moves"]
    out = dict(moved=0, copies=0, unused=0, made=[], kept=[], failed=[])
    # a file named twice (MAROON-X's _x_ in both arms) is linked first,
    #   then moved
    order = sorted(range(len(moves)), key=lambda k: not moves[k].get("link"))
    for count, k in enumerate(order):
        one = moves[k]
        say(count, len(moves), "moving")
        src, dest = one["src"], one["dest"]
        if not os.path.exists(src):
            out["failed"].append(dict(name=os.path.basename(src),
                                      why="no longer in Downloads"))
            continue
        if os.path.exists(dest):
            if src.lower().endswith(".gz") or one.get("split") \
                    or not _same(src, dest):
                out["kept"].append(dict(name=os.path.basename(src),
                                        dest=dest))
                continue
            if not one.get("link"):
                os.remove(src)
                _record(root, src, dest, "copy removed")
            out["copies"] += 1
            continue
        try:
            why = _put(one)
        except Exception as exc:                          # noqa: BLE001
            why = "%s: %s" % (type(exc).__name__, exc)
        if why:
            out["failed"].append(dict(name=os.path.basename(src), why=why))
            continue
        how = "linked" if one.get("link") else "moved"
        if os.path.exists(src) and not one.get("link"):
            # gunzipped, split or fixed: the file as downloaded put aside
            aside = os.path.join(top, UNUSED, "converted",
                                 os.path.relpath(src, top))
            os.makedirs(os.path.dirname(aside), exist_ok=True)
            os.replace(src, aside)
            how = "converted (as downloaded: %s)" % aside
        _record(root, src, dest, how)
        out["moved"] += 1
    for e2ds, blaze in _blazes(moves):
        try:
            from .fetch.spectra import empirical_blaze
            with fits.open(e2ds, memmap=False) as hdul:
                head = fits.Header()
                head["HIERARCH LBL BLAZE"] = (
                    "empirical", "envelope of " + os.path.basename(e2ds)[-40:])
                os.makedirs(os.path.dirname(blaze), exist_ok=True)
                _write_fits(fits.HDUList([fits.PrimaryHDU(
                    empirical_blaze(hdul[0].data), header=head)]), blaze)
            out["made"].append(os.path.basename(blaze))
        except Exception as exc:                          # noqa: BLE001
            out["failed"].append(dict(name=os.path.basename(blaze),
                                      why="%s: %s" % (type(exc).__name__,
                                                      exc)))
    for one in plan_["unused"]:
        src = one["path"]
        if not os.path.exists(src):
            continue
        aside = os.path.join(top, UNUSED, os.path.relpath(src, top))
        if os.path.exists(aside) and _same(src, aside):
            os.remove(src)
            _record(root, src, aside, "copy removed")
            out["copies"] += 1
            continue
        stem, count = aside, 1
        while os.path.exists(aside):
            count += 1
            aside = "%s_%d%s" % (os.path.splitext(stem)[0], count,
                                 os.path.splitext(stem)[1])
        os.makedirs(os.path.dirname(aside), exist_ok=True)
        os.replace(src, aside)
        _record(root, src, aside, "put aside: " + one["why"])
        out["unused"] += 1
    # the folders an archive was unpacked in, now empty
    for here, dirs, names in sorted(os.walk(top), key=lambda w: -len(w[0])):
        if here != top and not os.listdir(here) \
                and os.path.relpath(here, top).split(os.sep)[0] != UNUSED:
            try:
                os.rmdir(here)
            except OSError:
                pass
    # every DATA_DIR's models one folder of the root's
    out["shared"] = {}
    planned = {(one["home"], one["name"]) for one in plan_.get("shares")
               or []}
    for home in plan_.get("homes") or []:
        try:
            done = share(root, home)
        except OSError as exc:
            done = {name: str(exc) for name in SHARED}
        said = {k: v for k, v in done.items()
                if (home, k) in planned or v != "linked"}
        if said:
            out["shared"][home] = said
    say(len(moves), len(moves), "done")
    return out


# -----------------------------------------------------------------------------
# The window's dispatcher
# -----------------------------------------------------------------------------
class Dispatcher:
    """One plan or one dispatch at a time, followed by the page: plan(root)
    reads the headers in a thread, go(root) moves what the plan said."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = dict(state="idle", root="", done=0, total=0, step="")
        self._plan = None
        self.idents = {}

    def _say(self, done, total, step):
        with self.lock:
            self.state.update(done=done, total=total, step=step)

    def start_plan(self, root: str) -> dict:
        root = os.path.abspath(os.path.expanduser(root or ""))
        with self.lock:
            busy = self.state["state"] in ("planning", "moving")
            if not busy:
                self.state = dict(state="planning", root=root, done=0,
                                  total=0, step="headers",
                                  started=time.time())
                self._plan = None
        if busy:
            return self.status()
        threading.Thread(target=self._run_plan, args=(root,),
                         daemon=True).start()
        return self.status()

    def _run_plan(self, root):
        try:
            made = plan(root, self.idents, self._say)
        except Exception as exc:                          # noqa: BLE001
            with self.lock:
                self.state.update(state="error", error="%s: %s" % (
                    type(exc).__name__, exc))
            return
        with self.lock:
            self._plan = made
            self.state.update(state="planned", plan=_shown(made),
                              planned=time.time())

    def start_go(self, root: str) -> dict:
        root = os.path.abspath(os.path.expanduser(root or ""))
        with self.lock:
            made = self._plan
            ready = self.state["state"] == "planned" and made is not None \
                and made["root"] == root
            if ready:
                self.state.update(state="moving", done=0,
                                  total=len(made["moves"]), step="moving")
        if not ready:
            return dict(self.status(), error="no plan for %s" % root)
        threading.Thread(target=self._run_go, args=(made,),
                         daemon=True).start()
        return self.status()

    def _run_go(self, made):
        try:
            result = go(made, self._say)
        except Exception as exc:                          # noqa: BLE001
            with self.lock:
                self.state.update(state="error", error="%s: %s" % (
                    type(exc).__name__, exc))
            return
        with self.lock:
            self._plan = None
            self.state.update(state="done", result=result)

    def cancel(self) -> dict:
        """A plan dropped, or what a dispatch did put away."""
        with self.lock:
            if self.state["state"] in ("planned", "done", "error"):
                self._plan = None
                self.state = dict(state="idle", root=self.state["root"],
                                  done=0, total=0, step="")
        return self.status()

    def status(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state, default=str))


def _shown(made: dict) -> dict:
    """A plan as the page shows it: no list of moves, the files put aside
    and left counted by why."""
    def by_why(rows):
        counts = Counter(r["why"] for r in rows)
        return [dict(why=w, n=n, example=next(r["name"] for r in rows
                                              if r["why"] == w))
                for w, n in counts.most_common()]
    return dict(root=made["root"], folders=made["folders"],
                calibs=made["calibs"], shares=made["shares"],
                unused=by_why(made["unused"]),
                left=by_why(made["left"]), warnings=made["warnings"],
                simbad=made["simbad"], files=made["files"],
                science=made["science"], calib=made["calib"],
                n_unused=len(made["unused"]), n_left=len(made["left"]))
