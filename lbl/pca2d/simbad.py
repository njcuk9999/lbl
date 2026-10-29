"""A star as SIMBAD knows it, for the first section of a run's report.

    python -m lbl.pca2d.simbad NAME [NAME ...]

The folder a campaign lives in is a name somebody typed (a star's name,
<STAR>_<INSTRUMENT>), and the headers carry another (a TIC number, the
pipeline's spelling). SIMBAD's ident table takes any of them
(lbl.core.simbad.resolve, which also tries them as SIMBAD writes them:
TOI_1452 as TOI-1452), and its answer, SIMBAD's main identifier, is what the
tables are then asked about, all through SIMBAD's Python client (astroquery,
lbl.core.simbad):

    basic     position, spectral type, parallax, proper motion, systemic RV
    flux      B V G J H K
    ident     the identifiers worth quoting: TOI, TIC, Gaia DR3, 2MASS, GJ...
    mesFe_H   every Teff, log g and [Fe/H] SIMBAD lists, with its reference

Nothing here is needed to run anything. Each answer is kept under
~/.pca2d/simbad for a month, so that writing a report again does not depend
on the network, and a star SIMBAD cannot be asked about is a star the report
describes from its headers instead.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
import urllib.parse

import numpy as np

from lbl.core import base
from lbl.core import simbad as lbl_simbad

PAGE = "https://simbad.cds.unistra.fr/simbad/sim-id?Ident=%s"
CACHE = os.path.expanduser("~/.pca2d/simbad")
MAX_AGE = 30 * 86400
#: the identifiers quoted, in this order, the first of each kind only
KINDS = ("TOI-", "TIC ", "Gaia DR3 ", "2MASS J", "GJ ", "HD ", "HIP ", "LHS ",
         "Ross ", "G ")
#: folder suffixes that name an instrument, not a star: LBL's names for its
#: instruments, and their first part when a mode follows (<NAME>_<MODE>)
SUFFIXES = re.compile(r"_(%s)$" % "|".join(sorted(
    {re.escape(part) for name in base.INSTRUMENTS
     for part in (name, name.split("_")[0])}, key=len, reverse=True)), re.I)


def resolve(name):
    """SIMBAD's main identifier for `name`, or None."""
    found = lbl_simbad.resolve(name)
    return found["main"] if found else None


def tap(query):
    """The rows of an ADQL query, as dictionaries of strings ('' where
    SIMBAD has nothing, as its CSV answer had them)."""
    return [{key: "" if value is None else str(value)
             for key, value in row.items()}
            for row in lbl_simbad.tap(query)]


def _quoted(text):
    return "'%s'" % str(text).replace("'", "''")


def _float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def candidates(folder, header=None):
    """Every name a star goes by here, most specific first."""
    header = header or {}
    names = []
    gaia, release = header.get("GAIAID"), str(header.get("GAIADR") or "")
    if gaia and "DR" in release:
        names.append("%s %s" % (release.strip(), gaia))
    # APERO's own first: PP_OBJNS is the name its object database knows the
    # star by, which is very often SIMBAD's
    for key in ("PP_OBJNS", "PP_OBJN", "OBJECT", "OBJNAME",
                "ESO OBS TARG NAME", "DRSOBJN"):
        value = str(header.get(key) or "").strip()
        if value:
            names.append(value)
    names.append(SUFFIXES.sub("", str(folder)))
    names.append(str(folder))
    seen, out = set(), []
    for name in names:
        if name and name.upper() not in seen:
            seen.add(name.upper())
            out.append(name)
    return out


def _sexagesimal(value, hours):
    """HHMMSS.sss (ESO's packed form) or HH:MM:SS.s, in degrees."""
    text = str(value).strip()
    if not text:
        return None
    sign = -1.0 if text.startswith("-") else 1.0
    text = text.lstrip("+-")
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
    else:
        number = float(text)
        parts = [number // 10000, (number // 100) % 100, number % 100]
    while len(parts) < 3:
        parts.append(0.0)
    degrees = parts[0] + parts[1] / 60 + parts[2] / 3600
    return sign * degrees * (15.0 if hours else 1.0)


def header_position(header):
    """(ra, dec, epoch, whose) of the target, or None.

    APERO's PP_RA/PP_DEC when the file has them, a Gaia position at a stated
    epoch; otherwise the telescope's.

    Some observatories write the epoch of the target's coordinates (ESO
    TEL TARG EPOCH); others do not, and their OBJRA/OBJDEC are where the
    star was that night: a star of high proper motion (Barnard's star moves
    10 arcsec a year) can sit minutes of arc from its J2000 position. The
    epoch is then the night's.
    """
    header = header or {}
    # APERO's, when it is there: a Gaia position at a stated epoch (JD)
    ra, dec, jd = (_float(header.get(k)) for k in ("PP_RA", "PP_DEC",
                                                     "PP_EPOCH"))
    if ra is not None and dec is not None:
        return (ra, dec, 2000.0 + (jd - 2451545.0) / 365.25 if jd else 2000.0,
                "APERO's")
    for ra_key, dec_key, epoch_key in (
            ("ESO TEL TARG ALPHA", "ESO TEL TARG DELTA", "ESO TEL TARG EPOCH"),
            ("OBJRA", "OBJDEC", None)):
        if ra_key not in header or dec_key not in header:
            continue
        try:
            ra = _sexagesimal(header[ra_key], True)
            dec = _sexagesimal(header[dec_key], False)
        except (TypeError, ValueError):
            continue
        epoch = _float(header.get(epoch_key)) if epoch_key else None
        if epoch is None:
            mjd = _float(header.get("MJD-OBS") or header.get("MJDATE"))
            epoch = 2000.0 + (mjd - 51544.5) / 365.25 if mjd else 2000.0
        return ra, dec, epoch, "the telescope's"
    return None


def separation(ra1, dec1, ra2, dec2):
    """Angle between two positions, in arcseconds."""
    ra1, dec1, ra2, dec2 = np.radians([ra1, dec1, ra2, dec2])
    cosine = (np.sin(dec1) * np.sin(dec2)
              + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))) * 3600)


def fetch(main_id):
    """Everything the report quotes about one SIMBAD object."""
    where = "b.main_id = %s" % _quoted(main_id)
    basic = tap("SELECT b.oid, b.main_id, b.otype, b.ra, b.dec, b.sp_type,"
                " b.plx_value, b.plx_err, b.pmra, b.pmdec, b.rvz_radvel,"
                " b.rvz_err FROM basic AS b WHERE " + where)
    if not basic:
        return None
    row = basic[0]
    out = {"main_id": row["main_id"], "otype": row["otype"],
           "sp_type": row["sp_type"] or None}
    for key in ("ra", "dec", "plx_value", "plx_err", "pmra", "pmdec",
                "rvz_radvel", "rvz_err"):
        out[key] = _float(row.get(key))
    join = " JOIN basic AS b ON t.oidref = b.oid WHERE " + where
    out["mags"] = {r["filter"]: _float(r["flux"]) for r in
                   tap("SELECT t.filter, t.flux FROM flux AS t" + join)}
    idents = [r["id"] for r in tap("SELECT t.id FROM ident AS t" + join)]
    chosen = []
    for kind in KINDS:
        for ident in idents:
            if ident.startswith(kind) and ident != out["main_id"]:
                chosen.append(" ".join(ident.split()))
                break
    out["ids"] = chosen
    measures = []
    for r in tap("SELECT t.teff, t.log_g, t.fe_h, t.bibcode FROM mesFe_H AS t"
                 + join):
        teff = _float(r.get("teff"))
        if teff:
            measures.append({"teff": teff, "log_g": _float(r.get("log_g")),
                             "fe_h": _float(r.get("fe_h")),
                             "bibcode": r.get("bibcode") or ""})
    # the most recent first: a year is the first four characters of a bibcode
    measures.sort(key=lambda m: m["bibcode"][:4], reverse=True)
    out["teff"] = measures
    out["url"] = PAGE % urllib.parse.quote(out["main_id"])
    return out


def _cache_file(folder, cache):
    return os.path.join(cache, re.sub(r"[^A-Za-z0-9._+-]", "_", str(folder))
                        + ".json")


def offset(found, header):
    """How far SIMBAD's position, carried to the header's epoch, is from the
    header's own: (arcsec, epoch, whose position), or None."""
    position = header_position(header)
    if not position or found.get("ra") is None:
        return None
    ra, dec, epoch, whose = position
    years = epoch - 2000.0
    pmra, pmdec = found.get("pmra") or 0.0, found.get("pmdec") or 0.0
    there_dec = found["dec"] + pmdec * years / 3.6e6
    there_ra = found["ra"] + pmra * years / 3.6e6 / max(
        np.cos(np.radians(found["dec"])), 1e-6)
    return separation(there_ra, there_dec, ra, dec), round(epoch, 2), whose


def star(folder, header=None, cache=None, refresh=False):
    """What SIMBAD says of the star in `folder`, as a dictionary.

    Always a dictionary: `ok` says whether SIMBAD answered, `error` why not.
    `resolved_from` is the name that worked, `offset` how far SIMBAD's
    position is from the one in the header, in arcseconds, which is how a
    name that resolved to the wrong star would show. The offset is worked out
    on every call: it belongs to the header, not to SIMBAD's answer.
    """
    cache = cache or CACHE
    path = _cache_file(folder, cache)
    out = None
    if not refresh and os.path.exists(path) \
            and time.time() - os.path.getmtime(path) < MAX_AGE:
        with open(path, encoding="utf-8") as handle:
            kept = json.load(handle)
        if kept.get("ok"):
            out = kept
    if out is None:
        out = {"ok": False, "folder": str(folder), "error": None}
        try:
            for name in candidates(folder, header):
                main_id = resolve(name)
                if main_id:
                    found = fetch(main_id)
                    if found:
                        out.update(found, ok=True, resolved_from=name)
                        break
            else:
                out["error"] = ("none of %s is a name SIMBAD knows"
                                % ", ".join(candidates(folder, header)))
        except Exception as exc:                               # noqa: BLE001
            out["error"] = "%s: %s" % (type(exc).__name__, exc)
        if out["ok"]:
            out["retrieved"] = datetime.date.today().isoformat()
            os.makedirs(cache, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({k: v for k, v in out.items()
                           if not k.startswith("offset")}, handle, indent=1)
    if out["ok"]:
        measured = offset(out, header)
        for key in ("offset", "offset_epoch", "offset_from"):
            out.pop(key, None)
        if measured:
            out["offset"], out["offset_epoch"], out["offset_from"] = measured
    return out


def main(argv=None):
    import sys
    for folder in (argv if argv is not None else sys.argv[1:]):
        found = star(folder, refresh=True)
        print(json.dumps(found, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
