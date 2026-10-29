"""The known planets of a star, from the NASA Exoplanet Archive.

    python -m lbl.pca2d.archive TOI782 GL725B

One table, through the archive's TAP service: pscomppars, the confirmed
planets, one row each, with the archive's composite period, transit time and
RV semi-amplitude. A planet whose confirmation has been questioned in the
literature (pl_controv_flag = 1) is left out.

A star is found by the identifiers SIMBAD gives it (pca2d.simbad): its TIC
number above all, then its Gaia DR3, HD and HIP numbers and its names, since
some hosts carry no TIC in the archive. A host the archive names after its
TESS object of interest (TOI-1234) is also found by the folder's name, in
any spelling (TOI_1234_tc, TOI1234...).

Nothing here is needed to run anything. Each answer is kept under
~/.pca2d/exoplanets for a week, so that writing a report again does not
depend on the network, and a star the archive cannot be asked about has, as
far as the report can tell, no known planet.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

from lbl.core.simbad import toi_number

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CACHE = os.path.expanduser("~/.pca2d/exoplanets")
TIMEOUT = 30
MAX_AGE = 7 * 86400
#: the table asked, written in each kept answer: an answer kept without it,
#:  or from another table, is asked again
TABLE = "pscomppars"
#: two periods closer than this, relatively, are one planet
SAME_PERIOD = 1e-3


def _get(query):
    body = urllib.parse.urlencode({"query": query, "format": "csv"}).encode()
    request = urllib.request.Request(TAP, data=body,
                                     headers={"User-Agent": "LBL"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as handle:
        text = handle.read().decode("utf-8", "replace")
    if text.lstrip().startswith("<?xml"):
        found = re.search(r"<INFO[^>]*ERROR[^>]*>(.*?)</INFO>", text, re.S)
        raise RuntimeError((found.group(1) if found else text[:200]).strip())
    return list(csv.DictReader(io.StringIO(text)))


def _quoted(text):
    return "'%s'" % str(text).replace("'", "''")


def _float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def identifiers(facts, folder=None):
    """What to ask the archive about: {'tic', 'gaia', 'hd', 'hip', 'names'},
    from SIMBAD's facts and the folder's own name."""
    ids = list((facts or {}).get("ids") or [])
    names = [(facts or {}).get("main_id"), (facts or {}).get("resolved_from")]
    out = {"tic": None, "gaia": None, "hd": None, "hip": None}
    for ident in ids:
        if ident.startswith("TIC ") and out["tic"] is None:
            out["tic"] = ident[4:].strip()
        elif ident.startswith("Gaia DR3 ") and out["gaia"] is None:
            out["gaia"] = ident
        elif ident.startswith("HD ") and out["hd"] is None:
            out["hd"] = ident
        elif ident.startswith("HIP ") and out["hip"] is None:
            out["hip"] = ident
        names.append(ident)
    # a host named after its TOI, in any spelling, LBL's suffixes after it
    #   or not (TOI_1234_tc, TOI1234, TOI 1234, TOI-1234.01:
    #   lbl.core.simbad.toi_number), as the archive writes it: TOI-1234
    for text in [folder] + names:
        number = toi_number(text)
        if number:
            names.append("TOI-%s" % number)
            break
    # the archive writes GJ as Gl for some hosts, and names hosts as papers do
    spelled = set()
    for name in names:
        if not name:
            continue
        name = re.sub(r"^NAME ", "", str(name))
        spelled.add(name)
        if name.startswith("GJ "):
            spelled.add("Gl " + name[3:])
    out["names"] = sorted(spelled)
    return out


def confirmed(ids):
    """pscomppars rows of the star, as planet dictionaries, the controversial
    planets left out."""
    clauses = []
    if ids.get("tic"):
        clauses.append("tic_id = %s" % _quoted("TIC " + ids["tic"]))
    for key, column in (("gaia", "gaia_dr3_id"), ("hd", "hd_name"),
                        ("hip", "hip_name")):
        if ids.get(key):
            clauses.append("%s = %s" % (column, _quoted(ids[key])))
    if ids.get("names"):
        clauses.append("hostname in (%s)"
                       % ", ".join(_quoted(n) for n in ids["names"]))
    if not clauses:
        return []
    rows = _get("select pl_name, pl_letter, hostname, pl_orbper,"
                " pl_orbpererr1, pl_tranmid, pl_rvamp, pl_rvamperr1,"
                " pl_orbeccen from " + TABLE + " where pl_controv_flag = 0"
                " and (" + " or ".join(clauses) + ")")
    out = []
    for row in rows:
        period = _float(row.get("pl_orbper"))
        if not period:
            continue
        out.append({"name": row["pl_name"], "short": row.get("pl_letter")
                    or row["pl_name"], "period": period,
                    "period_err": _float(row.get("pl_orbpererr1")),
                    "t0": _float(row.get("pl_tranmid")),
                    "k": _float(row.get("pl_rvamp")),
                    "k_err": _float(row.get("pl_rvamperr1")),
                    "eccentricity": _float(row.get("pl_orbeccen")),
                    "source": "confirmed"})
    return sorted(out, key=lambda p: p["period"])


def _cache_file(ids, cache):
    key = ids.get("tic") or ids.get("gaia") or "+".join(ids.get("names") or [])
    key = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(key)) or "unknown"
    return os.path.join(cache, key + ".json")


def planets(facts, folder=None, cache=None, refresh=False):
    """The known planets of one star: {'ok', 'planets', 'error', ...}.

    `planets` is a list of dictionaries (name, short, period, period_err,
    t0 in BJD, k, k_err, eccentricity, source), shortest period first, all
    confirmed planets. `ok` is False when the archive could not be asked;
    the list is then what the cache last held, or empty.
    """
    ids = identifiers(facts, folder)
    cache = cache or CACHE
    os.makedirs(cache, exist_ok=True)
    path = _cache_file(ids, cache)
    kept = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            kept = json.load(handle)
        if kept.get("table") != TABLE:
            kept = None
        elif not refresh and time.time() - os.path.getmtime(path) < MAX_AGE:
            return kept
    try:
        found = confirmed(ids)
    except Exception as exc:                                  # noqa: BLE001
        if kept is not None:
            kept = dict(kept, stale=True, error=str(exc))
            return kept
        return {"ok": False, "planets": [], "error": str(exc), "ids": ids}
    out = {"ok": True, "planets": found, "error": None, "ids": ids,
           "table": TABLE, "retrieved": datetime.date.today().isoformat()}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    return out


def main(argv=None):
    from . import simbad

    for name in (argv if argv is not None else sys.argv[1:]):
        got = planets(simbad.star(name), folder=name)
        print("%s: %s" % (name, got.get("error") or "%d planet(s)"
                          % len(got["planets"])))
        for p in got["planets"]:
            print("  %-16s P = %.6f d  T0 %s  K %s  (%s)"
                  % (p["name"], p["period"], p["t0"], p["k"], p["source"]))


if __name__ == "__main__":
    main()
