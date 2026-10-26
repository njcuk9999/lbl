"""The targets of a data folder: DATA_DIR/science/<object>/, read from their
headers only, and kept so that a folder is read once.

What a row shows is what decides whether PCA2D has something to work with:
how many spectra, how bright (the SNR), and above all how far the BERV moves,
since the two frames separate only as far as it does. Every number is read
from the headers, never from the data, so that a folder of a thousand
spectra is listed in seconds; an index per data folder keeps each file's
numbers with its size and date, and a file is read again only if those
changed.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import threading

import numpy as np

#: where the indexes are kept, one per data folder
INDEX_HOME = os.path.expanduser("~/.lbl_pca2d_gui/scans")
#: bumped when the shape of an index changes
VERSION = 3
#: a folder made from another by LBL or PCA2D is not a target: the corrected
#: (<object>_pca2d, <object>_pca2d_<name>) and the cleaned (<object>_tc)
DERIVED = re.compile(r"(_pca2d(_.*)?|_tc(_.*)?)$")
#: the spectra of a folder, by extension (the instrument's own pattern, LBL's
#: INPUT_FILE, narrows it when the run is built)
PATTERNS = ("*.fits", "*.fits.gz", "*.hd5")
#: the keys the date is in, in order of preference: mid-exposure first
MJD_KEYS = ("MJDMID", "MJD-MID", "MJDATE", "MJD-OBS", "MJD_OBS",
            "HIERARCH OHP DRS BJD", "BJD", "MJD")
#: the BERV keys [km/s]
BERV_KEYS = ("BERV", "HIERARCH ESO QC BERV", "HIERARCH ESO DRS BERV",
             "HIERARCH OHP DRS BERV", "HIERARCH TNG QC BERV",
             "HIERARCH TNG DRS BERV")
#: the SNR keys for instruments that do not write APERO's EXTSNxxx
SNR_KEYS = ("SNR", "HIERARCH ESO QC ORDER60 SNR", "HIERARCH ESO DRS SPE EXT SN60",
            "HIERARCH TNG QC ORDER60 SNR", "HIERARCH OHP DRS SPE EXT SN36",
            "EEMSNR")
#: the effective temperature some pipelines write (APERO: OBJTEMP), which
#:    is what a wrap hands LBL as OBJECT_TEFF when the window has it
TEFF_KEYS = ("OBJTEMP", "PP_TEFF", "TEFF", "HIERARCH ESO OCS TARG TEFF")
#: the star's systemic velocity: APERO's PP_RV [m/s], when PP_RVS names its
#:    source, else ESO's target velocity [km/s]; 0 there means unknown.
#:    SPIRou's OBJRV is not read: it can be anything (280 km/s for
#:    KIC 12060710, whose lines LBL finds at 4 km/s)
VSYS_SOURCE_KEY, VSYS_APERO_KEY = "PP_RVS", "PP_RV"
VSYS_ESO_KEY = "HIERARCH ESO TEL TARG RADVEL"
#: the night a spectrum belongs to: MJD shifted so that a night does not
#: straddle midnight UTC anywhere it matters (Chile, Hawaii)
NIGHT_SHIFT = -0.3
#: the instruments whose LBL readers (DATA_SOURCE) differ by the LAYOUT of
#:    their files: APERO's own t.fits hold the flux in the primary HDU, with
#:    WAVEORDN in its header; the CADC's an empty primary and named
#:    extensions (FluxAB for SPIRou, FluxA for NIRPS); ESO's S2D a SCIDATA
#:    extension; the HARPS and HARPS-N DRS e2ds one image in the primary
LAYOUT_EXTENSIONS = {"FLUXAB", "FLUXA", "FLUXB"}


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def target_names(data_dir: str) -> list:
    """The target folders of DATA_DIR/science, sorted; a link to a folder
    elsewhere (the spectra kept on another disk) counts."""
    folder = os.path.join(data_dir or "", "science")
    if not os.path.isdir(folder):
        return []
    out = []
    for entry in sorted(os.scandir(folder), key=lambda e: e.name.lower()):
        if entry.name.startswith(".") or DERIVED.search(entry.name):
            continue
        if entry.is_dir():
            out.append(entry.name)
    return out


def files_of(data_dir: str, name: str) -> list:
    """The spectra of one target folder."""
    folder = os.path.join(data_dir, "science", name)
    out = []
    for pattern in PATTERNS:
        out += glob.glob(os.path.join(folder, pattern))
    return sorted(set(out))


def _instrument(headers) -> str:
    for head in headers:
        value = head.get("INSTRUME")
        if value:
            return str(value).strip().upper()
    return "?"


def layout_source(instrument: str, hdulist):
    """The DATA_SOURCE a file's layout says (LAYOUT_EXTENSIONS), or None
    where it does not tell."""
    family = re.sub(r"[^A-Z0-9]", "", str(instrument or "").upper()
                    .split("_")[0])
    names = {str(hdu.name).upper() for hdu in hdulist}
    primary = hdulist[0].header
    if family in ("SPIROU", "NIRPS"):
        if "SCIDATA" in names:
            return "ESO"
        if names & LAYOUT_EXTENSIONS:
            return "CADC"
        if primary.get("NAXIS", 0) and "WAVEORDN" in primary:
            return "APERO"
    elif family in ("HARPS", "HARPSN", "HARPN"):
        if "SCIDATA" in names:
            return "ESO"
        if len(hdulist) == 1 and primary.get("NAXIS", 0) == 2:
            return "ORIG"
    return None


def layout_of(data_dir: str, name: str):
    """(INSTRUME, DATA_SOURCE its layout says) of a target folder's first
    spectrum; (None, None) if it has none or it cannot be read."""
    from astropy.io import fits
    paths = files_of(data_dir, name)
    if not paths:
        return None, None
    try:
        with fits.open(paths[0], memmap=True) as hdulist:
            instrument = _instrument([hdu.header for hdu in hdulist])
            return instrument, layout_source(instrument, hdulist)
    except Exception:                                     # noqa: BLE001
        return None, None


def read_file(path: str) -> dict:
    """One spectrum's numbers, from its headers: instrument, object, SNR,
    BERV [km/s], MJD of mid-exposure, exposure time [s], and the data
    source its layout says. None where a header does not say."""
    from astropy.io import fits
    out = dict(instrument="?", object=None, snr=None, berv=None, mjd=None,
               exptime=None, teff=None, vsys=None, source=None)
    with fits.open(path, memmap=True) as hdulist:
        headers = [hdu.header for hdu in hdulist]
        out["source"] = layout_source(_instrument(headers), hdulist)
    out["instrument"] = _instrument(headers)
    if out["instrument"] == "ESPADONS":
        return _espadons(headers[0], out)
    # APERO writes EXTSNxxx and BJD on the science extension: its headers
    #   first, the primary's after
    ordered = sorted(headers, key=lambda h: not any(
        str(k).startswith("EXTSN") for k in h.keys()))
    snr = [_number(v) for h in ordered[:1] for k, v in h.items()
           if str(k).startswith("EXTSN")]
    snr = [v for v in snr if v is not None and v > 0]
    if snr:
        out["snr"] = round(float(np.median(snr)), 2)
    else:
        out["snr"] = _first(ordered, SNR_KEYS)
    out["berv"] = _first(ordered, BERV_KEYS)
    out["exptime"] = _first(ordered, ("EXPTIME", "EXPOSURE"))
    teff = _first(ordered, TEFF_KEYS)
    out["teff"] = teff if teff is not None and 2000 < teff < 50000 else None
    mjd = _first(ordered, MJD_KEYS)
    if mjd is not None and mjd > 2400000.5:
        mjd -= 2400000.5
    out["mjd"] = mjd
    out["vsys"] = _vsys(ordered)
    for key in ("DRSOBJN", "OBJECT", "OBJNAME"):
        value = next((h[key] for h in ordered if key in h), None)
        if value:
            out["object"] = str(value).strip()
            break
    return out


def _vsys(headers):
    """The star's systemic velocity [km/s] the headers give, or None."""
    for head in headers:
        source = str(head.get(VSYS_SOURCE_KEY, "") or "").strip()
        if VSYS_APERO_KEY in head and source and source.lower() != "none":
            value = _number(head[VSYS_APERO_KEY])
            if value is not None:
                return round(value / 1000.0, 4)
    value = _first(headers, (VSYS_ESO_KEY,))
    return value if value else None


def _first(headers, keys):
    for key in keys:
        for head in headers:
            if key in head:
                value = _number(head[key])
                if value is not None:
                    return value
    return None


def _espadons(header, out: dict) -> dict:
    """ESPaDOnS's i.fits carry no BERV: the barycentric correction is the
    one LBL's class computes (lbl.instruments.espadons), from the header's
    coordinates and the middle of the exposure; the SNR is Libre-ESpRIT's
    at 731 nm."""
    from lbl.instruments import espadons
    out["object"] = str(header.get("OBJECT", "")).strip() or None
    out["exptime"] = _number(header.get("EXPTIME"))
    try:
        mjd = espadons.mid_exposure(header)
        out["mjd"] = mjd
        berv, _ = espadons.barycentric(str(header["RA"]), str(header["DEC"]),
                                       round(mjd, 8))
        out["berv"] = round(berv, 4)
    except Exception:                                     # noqa: BLE001
        pass
    snr = espadons.order_snr(header)
    out["snr"] = None if not np.isfinite(snr) else float(snr)
    return out


def index_path(data_dir: str) -> str:
    digest = hashlib.sha1(os.path.abspath(data_dir).encode()).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(
        os.path.abspath(data_dir).rstrip("/")))
    return os.path.join(INDEX_HOME, "%s_%s.json" % (stem, digest))


def load_index(data_dir: str) -> dict:
    path = index_path(data_dir)
    try:
        with open(path) as handle:
            index = json.load(handle)
        if index.get("version") == VERSION:
            return index
    except (OSError, ValueError):
        pass
    return dict(version=VERSION, data_dir=os.path.abspath(data_dir),
                targets={})


def save_index(index: dict, data_dir: str) -> None:
    path = index_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(index, handle)
    os.replace(tmp, path)


class Scanner:
    """Reads the targets of a data folder in a thread, keeping what it has
    read in the index as it goes; what the window shows meanwhile is what is
    known so far."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.data_dir = None
        self.index = None
        self.done = 0
        self.total = 0
        self.current = ""
        self.stop = False

    def start(self, data_dir: str) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                if self.data_dir == os.path.abspath(data_dir):
                    return
                self.stop = True
        if self.thread is not None:
            self.thread.join(timeout=10)
        with self.lock:
            self.stop = False
            self.data_dir = os.path.abspath(data_dir)
            self.index = load_index(self.data_dir)
            self.done, self.total, self.current = 0, 0, ""
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def status(self) -> dict:
        with self.lock:
            return dict(running=bool(self.thread and self.thread.is_alive()),
                        done=self.done, total=self.total,
                        current=self.current, data_dir=self.data_dir)

    def _run(self) -> None:
        data_dir = self.data_dir
        names = target_names(data_dir)
        # ONE FILE PER FOLDER FIRST: every row gets its instrument, its
        #   object and its Teff in a second, then the rest of the files fill
        #   in its numbers
        files = {name: files_of(data_dir, name) for name in names}
        work = [(name, paths[0]) for name, paths in files.items() if paths]
        work += [(name, path) for name, paths in files.items()
                 for path in paths[1:]]
        with self.lock:
            self.total = len(work)
            # forget what is no longer there
            kept = {n: t for n, t in self.index["targets"].items()
                    if n in names}
            self.index["targets"] = kept
        changed = 0
        for name, path in work:
            if self.stop:
                break
            try:
                stat = os.stat(path)
            except OSError:
                continue
            key = os.path.basename(path)
            with self.lock:
                record = self.index["targets"].setdefault(name, {}).get(key)
                self.current = name
            if (record is None or record.get("size") != stat.st_size
                    or record.get("mtime") != int(stat.st_mtime)):
                try:
                    record = read_file(path)
                except Exception as exc:                  # noqa: BLE001
                    record = dict(unreadable=str(exc)[:120])
                record.update(size=stat.st_size, mtime=int(stat.st_mtime))
                with self.lock:
                    self.index["targets"][name][key] = record
                changed += 1
                if changed % 200 == 0:
                    with self.lock:
                        save_index(self.index, data_dir)
            with self.lock:
                self.done += 1
        with self.lock:
            # the files of a target that are gone
            for name in names:
                present = {os.path.basename(p) for p in files[name]}
                records = self.index["targets"].get(name, {})
                for key in list(records):
                    if key not in present:
                        del records[key]
            save_index(self.index, data_dir)
            self.current = ""

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.index or {}))


def summary(records: dict) -> dict:
    """One target's row: files, instrument, median SNR, median exposure
    time, the BERV span and the dates."""
    rows = [r for r in records.values() if "unreadable" not in r]
    def values(key):
        return np.array([r[key] for r in rows if r.get(key) is not None],
                        dtype=float)
    snr, berv, mjd, exp, teff = (values("snr"), values("berv"),
                                 values("mjd"), values("exptime"),
                                 values("teff"))
    instruments = sorted({r.get("instrument", "?") for r in rows})
    sources = sorted({r["source"] for r in rows if r.get("source")})
    objects = sorted({r.get("object") for r in rows if r.get("object")})
    nights = (len(np.unique(np.floor(mjd + NIGHT_SHIFT))) if mjd.size
              else 0)
    return dict(
        files=len(records), unreadable=len(records) - len(rows),
        instrument="/".join(instruments) if instruments else "?",
        source="/".join(sources) if sources else None,
        object=objects[0] if objects else None,
        snr=float(np.median(snr)) if snr.size else None,
        teff=float(np.median(teff)) if teff.size else None,
        exptime=float(np.median(exp)) if exp.size else None,
        berv_min=float(berv.min()) if berv.size else None,
        berv_max=float(berv.max()) if berv.size else None,
        berv_span=float(np.ptp(berv)) if berv.size else None,
        first=float(mjd.min()) if mjd.size else None,
        last=float(mjd.max()) if mjd.size else None,
        nights=nights)


def points(records: dict) -> dict:
    """One target's spectra for the plots: BERV, SNR, MJD and the systemic
    velocity [km/s, None where the headers do not say], aligned."""
    rows = [r for r in records.values()
            if "unreadable" not in r and r.get("berv") is not None]
    return dict(berv=[r["berv"] for r in rows],
                snr=[r.get("snr") for r in rows],
                mjd=[r.get("mjd") for r in rows],
                vsys=[r.get("vsys") for r in rows])


def incomplete(index: dict, data_dir: str, names=None) -> bool:
    """Whether some spectra of the folder are not in its index yet."""
    for name in names if names is not None else target_names(data_dir):
        records = (index.get("targets") or {}).get(name, {})
        if len(records) < len(files_of(data_dir, name)):
            return True
    return False
