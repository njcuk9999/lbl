"""The spectra of a star in the public archives, for LBL: searched and
fetched through lbl.gui.fetch, which knows where each instrument's
LBL-readable spectra are:

    SPIROU    the CADC, APERO's t.fits (LBL's SPIROU CADC)
    NIRPS     ESO, the S2D with ESO's telluric model (NIRPS_HE/HA ESO)
    HARPS     ESO, the e2ds of its DRS, the blaze of each night made from
              the spectra (ESO does not give it; HARPS ORIG)
    HARPSN    the TNG, espdr's S2D_A (HARPSN ESO)
    SOPHIE    the OHP, e2ds and their blaze (SOPHIE)
    ESPRESSO  ESO, the S2D_A (ESPRESSO)

A search asks every instrument at once (an archive that does not answer
holds the others up no more than its own time); what each one has is a
run of LBL, a group: its LBL instrument and source, its spectra. A fetch
writes a group's spectra into DATA_DIR/science/<folder>/ and its
calibrations into DATA_DIR/calib/, through each group's own getter, and
never moves or overwrites a file that is there. The folder then holds
ARCHIVE_FILE, what it came from, which the window reads to set the data
source and the file pattern of the wrap.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

#: the instruments asked, in the order the window shows them
INSTRUMENTS = ("SPIROU", "NIRPS", "HARPS", "HARPSN", "SOPHIE", "ESPRESSO")
#: each instrument's archive and the product LBL reads, said shortly
ARCHIVE_NAMES = dict(SPIROU="CADC · APERO t.fits", NIRPS="ESO · S2D + tellu",
                     HARPS="ESO · DRS e2ds",
                     HARPSN="TNG (IA2) · S2D_A", SOPHIE="OHP · e2ds",
                     ESPRESSO="ESO · S2D_A")
#: what the search keeps while it works (the headers of a star's HARPS-N
#:    exposures), in the fetch's cache (fetch.paths)
def _root():
    from .fetch.paths import cache
    return cache("archives")
#: what a fetched folder came from, beside its spectra (LBL reads *.fits)
ARCHIVE_FILE = ".lbl_archive.json"
#: the files fetched at once
WORKERS = 4
#: each instrument's observatory and its latitude [deg] (astropy's site
#:    registry, EarthLocation.of_site): a star that never rises
#:    MIN_ALTITUDE above its horizon is not asked for there
SITES = dict(SPIROU=("CFHT", 19.8267), NIRPS=("La Silla", -29.2567),
             HARPS=("La Silla", -29.2567), HARPSN=("the TNG", 28.7583),
             SOPHIE=("the OHP", 43.9308), ESPRESSO=("Paranal", -24.6274))
#: the least altitude [deg] a star reaches at its transit for an archive to
#:    be asked: 20 deg, an airmass of 2.9, wide on purpose (only what is out
#:    of reach is skipped)
MIN_ALTITUDE = 20.0
#: the requests asked at once in a search, and in one archive
MAX_REQUESTS, PER_ARCHIVE = 12, 2
#: the lines of the search's and fetch's log kept, and those the page gets
LOG_KEEP, LOG_SENT = 2000, 400
#: the window's archives, which the fetch's own lines go to while it works
_SINK = None
#: the fetch's levels as the window's log has them
_LEVELS = dict(value="info", warn="warn", error="error")


def _hear(message, level):
    """One line the fetch says (fetch.log.log), into the window's log too:
    it says what an archive is doing while it is slow."""
    if _SINK is not None and str(message).strip():
        _SINK.say("fetch: " + str(message).strip(),
                  _LEVELS.get(level, "info"))


def _hook_log():
    """The window listens to the fetch's lines, once."""
    try:
        from .fetch import log as fetch_log
    except Exception:                                     # noqa: BLE001
        return
    if _hear not in fetch_log.LISTENERS:
        fetch_log.LISTENERS.append(_hear)


def available():
    """None if the archive search can be used, else why not."""
    try:
        from .fetch import spectra                        # noqa: F401
    except Exception as exc:                              # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, exc)
    return None


def provenance(folder: str):
    """What a folder of spectra was fetched from (ARCHIVE_FILE), or None."""
    try:
        with open(os.path.join(folder, ARCHIVE_FILE),
                  encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def split_targets(text) -> list:
    """The stars of a search: names separated by commas, semicolons or new
    lines, each once, in their order."""
    out = []
    for name in re.split(r"[,;\n]+", str(text or "")):
        name = name.strip()
        if name and name not in out:
            out.append(name)
    return out


def out_of_reach(inst, dec):
    """Why a star of declination `dec` [deg] is not asked for in `inst`'s
    archive (it never rises MIN_ALTITUDE above its observatory's horizon),
    or None."""
    if inst not in SITES or dec is None:
        return None
    site, lat = SITES[inst]
    highest = 90.0 - abs(float(dec) - lat)
    if highest >= MIN_ALTITUDE:
        return None
    if highest < 0:
        return "Dec %+.1f deg: never rises above %s's horizon" % (float(dec),
                                                                  site)
    return ("Dec %+.1f deg: at most %.0f deg above %s's horizon (%.0f"
            " needed)" % (float(dec), highest, site, MIN_ALTITUDE))


def sexagesimal(ra, dec) -> str:
    """RA and Dec [deg] as 'hh:mm:ss.s +dd:mm:ss'."""
    ra = (float(ra) / 15.0) % 24.0
    h, m = int(ra), int(ra * 60) % 60
    sec = ra * 3600 - 3600 * h - 60 * m
    sign = "-" if dec < 0 else "+"
    d = abs(float(dec))
    dd, dm = int(d), int(d * 60) % 60
    dsec = d * 3600 - 3600 * dd - 60 * dm
    return "%02d:%02d:%04.1f %s%02d:%02d:%02.0f" % (h, m, sec, sign, dd, dm,
                                                   dsec)


def cell(target, inst) -> str:
    """The key of one star in one instrument, in a search's state."""
    return "%s|%s" % (target, inst)


def folder_for(data_dir: str, group: dict) -> str:
    """Where a group's spectra go: science/<object> if the folder is new or
    holds these spectra already (or came from the same LBL instrument and
    source); else science/<object>_<instrument>_<source>, so that one folder
    never mixes two readers."""
    base = group["obj"]
    names = {one["name"] for one in group["spectra"]}
    for name in (base, "%s_%s_%s" % (base, group["lbl"], group["source"])):
        path = os.path.join(data_dir, "science", name)
        if not os.path.isdir(path):
            return name
        mine = provenance(path)
        if mine and mine.get("lbl") == group["lbl"] \
                and mine.get("source") == group["source"]:
            return name
        here = {f for f in os.listdir(path) if f.endswith(".fits")}
        if not here or here <= names:
            return name
    return "%s_%s_%s" % (base, group["lbl"], group["source"])


class Archives:
    """One search and one fetch at a time, followed by the page."""

    def __init__(self):
        global _SINK
        self.lock = threading.Lock()
        self.search = dict(state="idle")
        self.fetch = dict(state="idle")
        self._groups = {}
        self._stop = False
        #: each search's number: a search stopped, or replaced by another,
        #:    has its late answers dropped
        self._search_id = 0
        self.log = []
        self.log_file = None
        _SINK = self

    def say(self, text: str, level: str = "info") -> None:
        """One line of the log the page shows under the search, kept in the
        window's logs as well (DATA_DIR/gui/logs/archives.log): the page
        keeps the last LOG_KEEP lines, the file every one."""
        now = datetime.datetime.now()
        line = dict(t=now.strftime("%H:%M:%S"), level=level, text=str(text))
        with self.lock:
            self.log.append(line)
            del self.log[:-LOG_KEEP]
            path = self.log_file
        if path:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write("%s | %-5s | %s\n" % (
                        now.strftime("%Y-%m-%d %H:%M:%S"), level, text))
            except OSError:
                pass

    def _log_into(self, data_dir: str) -> None:
        """The data folder whose logs the next lines go to."""
        from .history import GUI_LOGS
        with self.lock:
            self.log_file = (os.path.join(data_dir, GUI_LOGS, "archives.log")
                             if data_dir else None)

    # -- searching -----------------------------------------------------------
    def start_search(self, target: str, instruments, data_dir: str) -> dict:
        """Search the archives for one star or several (names separated by
        commas, semicolons or new lines): every star in every instrument
        at once (PER_ARCHIVE at a time in one archive)."""
        targets = split_targets(target)
        instruments = [i for i in (instruments or INSTRUMENTS)
                       if i in INSTRUMENTS]
        self._log_into(data_dir)
        with self.lock:
            if self.search.get("state") == "running":
                return self.status()
            self._search_id += 1
            number = self._search_id
            self.search = dict(
                state="running", target=", ".join(targets), targets=targets,
                main=None, teff=None, error=None,
                stars={t: dict(state="resolving") for t in targets},
                instruments={i: dict(state="searching") for i in instruments},
                cells={cell(t, i): dict(state="searching")
                       for t in targets for i in instruments},
                groups=[], started=time.time())
            self._groups = {}
        self.say("search: %s, in %s" % (", ".join(targets),
                                        ", ".join(instruments)))
        threading.Thread(target=self._search, args=(targets, instruments,
                                                    data_dir, number),
                         daemon=True).start()
        return self.status()

    def stop_search(self) -> dict:
        """Stop waiting for the archives: what has answered stays, what has
        not is marked stopped, and its late answer is dropped (a thread
        waiting on an archive cannot be killed, it is only no longer
        listened to)."""
        with self.lock:
            if self.search.get("state") != "running":
                return self.status()
            self._search_id += 1
            waiting = [key for key, about in self.search["cells"].items()
                       if about.get("state") == "searching"]
            for key in waiting:
                self.search["cells"][key] = dict(state="stopped")
            for about in self.search["stars"].values():
                if about.get("state") == "resolving":
                    about["state"] = "stopped"
            self._sum_instruments()
            self._sort_groups()
            self.search["state"] = "done"
        self.say("search stopped%s" % (
            ": no answer waited for from " + ", ".join(
                k.replace("|", " in ") for k in waiting) if waiting else ""),
            "warn")
        return self.status()

    def _current(self, number) -> bool:
        with self.lock:
            return number == self._search_id

    def _sort_groups(self):
        """The rows in the order of the stars asked, then of INSTRUMENTS
        (under the lock)."""
        order = {t: k for k, t in enumerate(self.search.get("targets") or [])}
        self.search["groups"].sort(key=lambda r: (
            order.get(r.get("target"), 0), INSTRUMENTS.index(r["inst"]),
            r["id"]))

    def _sum_instruments(self):
        """Each instrument's state over the stars (under the lock): still
        searching while one star is, else its spectra in all."""
        for inst in self.search["instruments"]:
            cells = [about for key, about in self.search["cells"].items()
                     if key.split("|", 1)[1] == inst]
            states = {c.get("state") for c in cells}
            for state in ("searching", "failed", "stopped"):
                if state in states:
                    break
            else:
                state = "skipped" if states == {"skipped"} else "done"
            self.search["instruments"][inst] = dict(
                state=state, n=sum(c.get("n", 0) for c in cells),
                error="; ".join(c["error"] for c in cells if c.get("error")))

    def _search(self, targets, instruments, data_dir, number):
        _hook_log()
        why = available()
        if why:
            with self.lock:
                self.search.update(state="failed", error=why)
            self.say("the archive search cannot be used: %s" % why, "error")
            return
        from .fetch import spectra
        ROOT = _root()
        os.makedirs(ROOT, exist_ok=True)
        gates = {inst: threading.Semaphore(PER_ARCHIVE) for inst in instruments}
        several = len(targets) > 1

        def star(target):
            """SIMBAD's name and Teff of a star, then its instruments"""
            try:
                ident = spectra.identify(target)
            except Exception as exc:                      # noqa: BLE001
                with self.lock:
                    self.search["stars"][target] = dict(
                        state="failed", error=str(exc)[:200])
                    for inst in instruments:
                        self.search["cells"][cell(target, inst)] = dict(
                            state="failed", error="SIMBAD: %s" % exc)
                    self._sum_instruments()
                self.say("SIMBAD does not know %s: %s" % (target, exc),
                         "error")
                return []
            teff = None
            try:
                from .fetch.stars import stellar_teff
                teff = stellar_teff(ident).get("teff")
            except Exception:                             # noqa: BLE001
                pass
            main = ident.get("main") or target
            ra, dec = ident.get("ra"), ident.get("dec")
            place = sexagesimal(ra, dec) if ra is not None and dec is not None \
                else None
            with self.lock:
                self.search["stars"][target] = dict(
                    state="done", main=main, teff=teff, ra=ra, dec=dec,
                    place=place)
                if not several:
                    self.search.update(main=main, teff=teff, place=place)
            self.say("SIMBAD: %s is %s%s%s" % (
                target, main, " (%s)" % place if place else "",
                ", Teff %.0f K" % teff if teff else ""))
            # an archive whose telescope never sees the star: not asked
            jobs = []
            for inst in instruments:
                why = out_of_reach(inst, dec)
                if why:
                    with self.lock:
                        self.search["cells"][cell(target, inst)] = dict(
                            state="skipped", error=why)
                        self._sum_instruments()
                    self.say("%s, %s: not asked, %s" % (
                        target, spectra.INSTRUMENTS[inst]["label"], why))
                else:
                    jobs.append((target, inst, ident, teff))
            return jobs

        def one(job):
            target, inst, ident, teff = job
            label = spectra.INSTRUMENTS[inst]["label"]
            said = "%s, %s" % (target, label) if several else label
            with gates[inst]:
                if not self._current(number):
                    return
                self.say("%s: asking %s" % (said, ARCHIVE_NAMES.get(inst,
                                                                    inst)))
                tick = time.time()
                try:
                    found = spectra.groups(target, inst, ROOT, ident)
                except Exception as exc:                  # noqa: BLE001
                    if not self._current(number):
                        return
                    with self.lock:
                        self.search["cells"][cell(target, inst)] = dict(
                            state="failed", error=str(exc)[:200])
                        self._sum_instruments()
                    self.say("%s: no answer after %.0f s: %s" % (
                        said, time.time() - tick, str(exc)[:200]), "error")
                    return
            if not self._current(number):
                self.say("%s: answered after the search was stopped, %.0f s;"
                         " not kept" % (said, time.time() - tick), "warn")
                return
            size = spectra.INSTRUMENTS[inst]["size"]
            rows = []
            for group in found:
                if not group["spectra"]:
                    continue
                key = "%s|%s:%s" % (target, inst, group["key"])
                with self.lock:
                    folder = folder_for(data_dir, group) if data_dir else \
                        group["obj"]
                    # two readers never share a folder: one already proposed
                    #   for another reader of this search gives this one its
                    #   reader's name
                    taken = {r["folder"] for r in self.search["groups"]}
                    if folder in taken:
                        folder = "%s_%s_%s" % (group["obj"], group["lbl"],
                                               group["source"])
                    here = os.path.join(data_dir or "", "science", folder)
                    have = sum(os.path.exists(os.path.join(here, x["name"]))
                               for x in group["spectra"])
                    row = dict(
                        id=key, target=target, teff=teff, inst=inst,
                        label=label, archive=ARCHIVE_NAMES.get(inst, inst),
                        lbl=group["lbl"], source=group["source"],
                        input=group["input"], params=group["params"],
                        n=len(group["spectra"]), have=have, folder=folder,
                        size_mb=round(size * (len(group["spectra"]) - have),
                                      1))
                    rows.append(row)
                    self.search["groups"].append(row)
                    self._groups[key] = group
            with self.lock:
                self.search["cells"][cell(target, inst)] = dict(
                    state="done", n=sum(r["n"] for r in rows))
                self._sum_instruments()
            if rows:
                for r in rows:
                    self.say("%s: %d spectra (%s %s), %d already here, %.0f MB"
                             " to fetch into science/%s, in %.0f s" % (
                                 said, r["n"], r["lbl"], r["source"],
                                 r["have"], r["size_mb"], r["folder"],
                                 time.time() - tick))
            else:
                self.say("%s: no public spectrum LBL reads, %.0f s" % (
                    said, time.time() - tick))

        # every star resolved at once, then every star in every archive at
        #   once, PER_ARCHIVE at a time in one archive
        with ThreadPoolExecutor(max_workers=min(8, len(targets)) or 1) as pool:
            jobs = [job for found in pool.map(star, targets) for job in found]
        if jobs:
            with ThreadPoolExecutor(max_workers=min(MAX_REQUESTS, len(jobs))
                                    ) as pool:
                list(pool.map(one, jobs))
        if not self._current(number):
            return
        with self.lock:
            self._sort_groups()
            self.search["state"] = "done"
            took = time.time() - self.search.get("started", time.time())
        self.say("search done in %.0f s" % took)

    # -- fetching ------------------------------------------------------------
    def start_fetch(self, data_dir: str, ids) -> dict:
        self._log_into(data_dir)
        with self.lock:
            if self.fetch.get("state") == "running":
                return self.status()
            rows = [r for r in self.search.get("groups", []) if r["id"] in ids]
            groups = [(r, self._groups[r["id"]]) for r in rows
                      if r["id"] in self._groups]
            if not groups:
                return dict(self.status(), error="nothing chosen")
            total = sum(r["n"] - r["have"] for r, _ in groups)
            # one bar per group: what it is, whose, where, how far
            bars = [dict(label=r["label"], target=r.get("target")
                         or self.search.get("main") or "",
                         folder=r["folder"], total=r["n"] - r["have"], done=0,
                         failed=0)
                    for r, _ in groups]
            self.fetch = dict(state="running", total=total, done=0,
                              failed=0, why="", current="", folders=[],
                              bars=bars)
            self._stop = False
            # each group's star and its Teff (a search of one star had them
            #   for the whole search)
            for r, _ in groups:
                r.setdefault("target", self.search.get("target"))
                if r.get("teff") is None:
                    r["teff"] = self.search.get("teff")
        threading.Thread(target=self._fetch, args=(data_dir, groups),
                         daemon=True).start()
        return self.status()

    def stop_fetch(self) -> None:
        self._stop = True
        self.say("fetch: stopping after the files under way", "warn")

    def _fetch(self, data_dir, groups):
        """The groups of different instruments at once (their archives and
        their calibrations are their own), those of one instrument one
        after the other (two stars of one night can share a calibration
        file), WORKERS files at a time in each."""
        calib = os.path.join(data_dir, "calib")
        os.makedirs(calib, exist_ok=True)
        by_inst = {}
        for index, (row, group) in enumerate(groups):
            by_inst.setdefault(row["inst"], []).append((index, row, group))

        def one_instrument(items):
            for index, row, group in items:
                if self._stop:
                    return
                self._fetch_group(data_dir, calib, index, row, group)

        with ThreadPoolExecutor(max_workers=len(by_inst)) as pool:
            list(pool.map(one_instrument, by_inst.values()))
        with self.lock:
            self.fetch["state"] = "stopped" if self._stop else "done"
            self.fetch["current"] = ""
            done, failed = self.fetch["done"], self.fetch["failed"]
        self.say("fetch %s: %d spectra, %d not fetched" % (
            "stopped" if self._stop else "done", done - failed, failed),
            "warn" if self._stop or failed else "info")

    def _fetch_group(self, data_dir, calib, index, row, group):
        """One group's spectra into its folder, and where they came from."""
        # asked again: a group fetched just before may have taken it
        with self.lock:
            mine = provenance(os.path.join(data_dir, "science", row["folder"]))
            if mine and (mine.get("lbl"), mine.get("source")) != (
                    row["lbl"], row["source"]):
                row["folder"] = "%s_%s_%s" % (group["obj"], row["lbl"],
                                              row["source"])
            science = os.path.join(data_dir, "science", row["folder"])
            os.makedirs(science, exist_ok=True)
        said = "%s, %s" % (row.get("target"), row["label"])
        todo = [one for one in group["spectra"]
                if not os.path.exists(os.path.join(science, one["name"]))]
        # started: the bar's time left, from how fast it has gone so far
        with self.lock:
            self.fetch["bars"][index].update(folder=row["folder"],
                                             total=len(todo),
                                             started=time.time())
            going = [b for b in self.fetch["bars"]
                     if b["done"] < b["total"]]
            self.fetch["current"] = ", ".join(
                "%s %s" % (b["target"], b["label"]) for b in going)
        self.say("fetch: %d %s spectra into science/%s (%d already there)"
                 % (len(todo), said, row["folder"],
                    len(group["spectra"]) - len(todo)))

        def get(one):
            if self._stop:
                return "stopped"
            try:
                why = group["get"](one, science, calib)
            except Exception as exc:                      # noqa: BLE001
                why = "%s: %s" % (type(exc).__name__, exc)
            with self.lock:
                bar = self.fetch["bars"][index]
                self.fetch["done"] += 1
                bar["done"] += 1
                if why:
                    self.fetch["failed"] += 1
                    bar["failed"] += 1
                    self.fetch["why"] = str(why)[:200]
                done, total = bar["done"], bar["total"]
            if why and why != "stopped":
                self.say("%s: %s not fetched: %s" % (
                    said, one["name"], str(why)[:200]), "warn")
            # one line every tenth of the way, and the last
            elif total and (done == total or done % max(1, total // 10) == 0):
                self.say("%s: %d of %d fetched" % (said, done, total))
            return why

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(get, todo))
        with self.lock:
            self.fetch["bars"][index]["ended"] = time.time()
        with open(os.path.join(science, ARCHIVE_FILE), "w",
                  encoding="utf-8") as handle:
            json.dump(dict(target=row.get("target"), inst=row["inst"],
                           lbl=row["lbl"], source=row["source"],
                           input=row["input"], params=row["params"],
                           archive=row["archive"], teff=row.get("teff"),
                           fetched=datetime.datetime.now().isoformat(
                               timespec="seconds")), handle, indent=1)
        with self.lock:
            self.fetch["folders"].append(row["folder"])

    def status(self) -> dict:
        with self.lock:
            # now: the server's clock, that the bars' times are read against
            return json.loads(json.dumps(dict(search=self.search,
                                              fetch=self.fetch,
                                              now=time.time(),
                                              log=self.log[-LOG_SENT:],
                                              unavailable=available())))
