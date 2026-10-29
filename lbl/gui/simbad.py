"""A target's effective temperature from SIMBAD, for the LBL tab: the
median of the Teff that SIMBAD's mesFe_H table holds for the star, those far
from its spectral type's left out, else its spectral type's (Pecaut &
Mamajek 2013). The answers are kept in DATA_DIR/gui/simbad_teff.json, one
question per target until asked again.

A target's name is its folder's, as APERO writes it ('-' as M, ' ' as _:
TOIM4508, GJ_1028); SIMBAD is asked that name, then the OBJECT of its
headers, then both as SIMBAD writes them (lbl.core.simbad.name_variants:
TOI_1452 as TOI-1452), until one is known. SIMBAD is asked through its
Python client (lbl.core.simbad), never through a web page.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

from lbl.core.simbad import SUFFIXES, name_variants, tap as _tap

#: where the answers are kept, in the data folder
CACHE = os.path.join("gui", "simbad_teff.json")
#: a measurement this far from its spectral type's Teff is left out (a
#:   catalogue's 4500 K for an M4.5V, 3000 K from its own paper)
SPT_TOLERANCE = 0.20
#: the questions asked at once
WORKERS = 4
#: a dwarf's Teff by spectral type: Pecaut & Mamajek (2013, ApJS 208, 9),
#:   Mamajek's table of 2022.04.16, as the archive search reads it
from .fetch.stars import SPT_TEFF, spectral_code  # noqa: E402,F401
from .fetch.stars import teff_from_spectral_type as teff_of_type  # noqa: E402

_lock = threading.Lock()


def candidates(name, header_object=None):
    """The names SIMBAD is asked, in order: the folder's, its headers'
    OBJECT, then both as SIMBAD writes them (APERO's M for '-' between
    characters and digits, _ for a space, Gliese as GJ)."""
    def spell(text):
        text = re.sub(r"(?<=[A-Za-z0-9])M(?=\d)", "-", text)
        text = re.sub(r"_+", " ", text).strip()
        return re.sub(r"^gliese\s*", "GJ ", text, flags=re.I)

    out = []
    for raw in (name, header_object):
        if not raw:
            continue
        raw = str(raw).strip()
        # then without LBL's suffixes, and as SIMBAD writes the catalogues
        #   (lbl.core.simbad.name_variants: TOI 1452 as TOI-1452)
        bare = spell(SUFFIXES.sub("", raw))
        for one in [raw, spell(raw)] + name_variants(bare):
            if one and one not in out:
                out.append(one)
    return out


def summarise(main, sptype, measured):
    """The answer for a star SIMBAD knows: its measurements' median, those
    more than SPT_TOLERANCE from its spectral type's Teff left out (all of
    them kept when that would leave none), else its spectral type's."""
    out = dict(state="done", main=main, sptype=sptype, teff=None, n=0,
               n_all=len(measured), source=None)
    typed = teff_of_type(sptype)
    kept = measured
    if typed and measured:
        near = [t for t in measured if abs(t - typed) <= SPT_TOLERANCE * typed]
        kept = near or measured
    if kept:
        out.update(teff=round(statistics.median(kept) / 10.0) * 10,
                   n=len(kept), source="mesFe_H")
    elif typed:
        out.update(teff=round(typed / 10.0) * 10, source="sptype")
    return out


def ask(name, header_object=None):
    """One target, as SIMBAD knows it: dict(state done/unknown/failed, main,
    sptype, teff, n (measurements kept), n_all, source mesFe_H/sptype,
    rv (its systemic velocity, km/s, None when SIMBAD has none), asked,
    error)."""
    names = candidates(name, header_object)
    try:
        for one in names:
            quoted = one.replace("'", "''")
            rows = _tap("SELECT b.main_id, b.sp_type, b.rvz_radvel, m.teff"
                        " FROM ident AS i JOIN basic AS b ON b.oid ="
                        " i.oidref LEFT JOIN mesFe_h AS m ON m.oidref = b.oid"
                        " WHERE i.id = '%s'" % quoted)
            if rows:
                measured = [float(r["teff"]) for r in rows if r["teff"]]
                out = summarise(str(rows[0]["main_id"]).strip(),
                                rows[0]["sp_type"], measured)
                out["main"] = re.sub(r"\s+", " ", out["main"])
                rv = rows[0].get("rvz_radvel")
                out["rv"] = float(rv) if rv not in (None, "") else None
                out["asked"] = one
                return out
    except Exception as exc:                              # noqa: BLE001
        return dict(state="failed", error="%s: %s" % (type(exc).__name__,
                                                      str(exc)[:160]))
    return dict(state="unknown", asked=names)


def _read(data_dir):
    path = os.path.join(data_dir, CACHE)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def teffs(data_dir, targets, refresh=False):
    """{name: answer} of the targets [dict(name, object)], from the folder's
    record when it has them (unless refresh), from SIMBAD otherwise; a
    failed question is not kept, so that it is asked again."""
    with _lock:
        known = _read(data_dir)
    # an answer kept without SIMBAD's systemic velocity (no rv) is asked
    #   again, once
    todo = [t for t in targets if refresh or t["name"] not in known
            or (known[t["name"]].get("state") == "done"
                and "rv" not in known[t["name"]])]
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            answers = list(pool.map(lambda t: ask(t["name"], t.get("object")),
                                    todo))
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with _lock:
            known = _read(data_dir)
            for target, answer in zip(todo, answers):
                answer["when"] = stamp
                if answer["state"] != "failed":
                    known[target["name"]] = answer
                else:
                    known.pop(target["name"], None)
            path = os.path.join(data_dir, CACHE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(known, handle, indent=1)
        failed = {t["name"]: a for t, a in zip(todo, answers)
                  if a["state"] == "failed"}
    else:
        failed = {}
    return {t["name"]: failed.get(t["name"]) or known.get(t["name"])
            for t in targets}
