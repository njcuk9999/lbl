"""Running wraps: one at a time, the others queued, each one's log kept and
handed to whoever watches it.

A run is an ordinary LBL wrap (wrap.py) started with the python of the
environment LBL is installed in, in its own process group so that Stop stops
everything it started (LBL and PCA2D start workers of their own). Its output
is cut into finished lines and redraws of a progress bar (cut_output), each
line given the level its colour says (LBL's and PCA2D's loggers colour info,
values, warnings and errors), and kept as a list of events that a page can
read from any point on: a page opened late, or reloaded, sees the whole run.

Every run leaves its folder, DATA_DIR/gui/runs/<id>/, with the wrap and
run.json (what ran, when, how it ended, which reports it wrote), which is
what the history reads back, and its full log, the colours stripped, in the
window's logs: DATA_DIR/gui/logs/<id>.log.
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import re
import signal
import subprocess
import threading
import time

#: a colour code, and the codes that say a level (both loggers use them)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEVELS = {"32": "info", "92": "info", "34": "value", "94": "value",
          "33": "warn", "93": "warn", "31": "error", "91": "error"}
#: what the status line follows: PCA2D's stages, LBL's recipes, and the
#: counters of the files they go through
STAGE = re.compile(r"\bstage (\w+)\s*$")
RECIPE = re.compile(r"^\s*LBL (Telluclean|Tellu ?Clean|Template|Mask|Compute"
                    r"|Compile?|Reset|Resmap|Pca2d|PCA2D)\s*$", re.I)
#: the object a recipe works on, the line after its banner's
OBJECT = re.compile(r'^\s*OBJECT_SCIENCE="([^"]*)"')
#: LBL's report, written at the end of an object's steps
REPORT = re.compile(r"^Report written: ")
#: the counters a step goes through, and how far it is: LBL's telluric
#:   cleaning and template ("for file 15 of 90"), compute ("Processing file
#:   1 / 90"), compile's progress bar ("| 63/63 ["), PCA2D's cube and
#:   correction ("reading spectra: 78/90 files", "correcting: 25/89 files")
COUNTERS = (re.compile(r"\bfile (\d+) of (\d+)\b"),
            re.compile(r"\bfile (\d+) / (\d+)\b"),
            re.compile(r"\|\s*(\d+)/(\d+) \["),
            re.compile(r"\b(\d+)/(\d+) files\b"))
#: how long a run is given to stop before it is killed [s]
GRACE = 10.0


def cut_output(buffer: str):
    """Cut what has been read from a run into finished lines, and the rest.

    tqdm draws a bar by rewriting one line: the bar, a carriage return, the
    bar again, and no newline until the loop is over. Returns (segments,
    rest): each segment is (text, transient), transient saying it ended in a
    carriage return, i.e. it is one draw of a bar that the next segment
    replaces; `rest` is what is not terminated yet, to be put in front of the
    next chunk (a pipe hands its output over wherever it happened to fill).
    (Ported from pca2d-preclean's window.)
    """
    segments = []
    start = i = 0
    while i < len(buffer):
        char = buffer[i]
        if char == "\n":
            segments.append((buffer[start:i], False))
            i = start = i + 1
        elif char == "\r":
            # a lone one ends a draw of a bar; "\r\n" is a line
            step = 2 if buffer[i + 1:i + 2] == "\n" else 1
            segments.append((buffer[start:i], step == 1))
            i = start = i + step
        else:
            i += 1
    return segments, buffer[start:]


def level_of(text: str) -> str:
    """The level a line's colour says (the last level code in it), or
    'plain'; LBL also writes it as |G| |I| |W| |E| after the time."""
    level = "plain"
    for code in ANSI.findall(text):
        for part in re.findall(r"[0-9]+", code):
            if part in LEVELS:
                level = LEVELS[part]
    if level == "plain":
        marker = re.search(r"\|([GIWE])\|", text)
        if marker:
            level = {"G": "info", "I": "value", "W": "warn",
                     "E": "error"}[marker.group(1)]
    return level


def strip(text: str) -> str:
    return ANSI.sub("", text)


def message_of(text: str) -> str:
    """A log line without its time and level marker."""
    plain = strip(text)
    return re.sub(r"^\S+ \S+\|[A-Z]\|\s?", "", plain)


class Job:
    """One run: its wrap, its process, its events."""

    def __init__(self, job_id: str, name: str, folder: str, wrap: str,
                 python: str, cwd: str, meta: dict):
        self.id = job_id
        self.name = name
        self.folder = folder
        self.wrap = wrap
        self.python = python
        self.cwd = cwd
        self.meta = meta
        self.state = "queued"
        self.created = time.time()
        self.started = None
        self.ended = None
        self.exit = None
        self.pid = None
        self.events = []          # dict(n, op, text, level), all of them
        self.status = ""
        self.stage = ""
        self.counter = ""
        self.cond = threading.Condition()
        self.transient = False
        self.stopping = False
        self.reports = []
        # the run's steps (wrapgen.steps), ticked off as the log names them
        self.steps = [dict(step, state="pending")
                      for step in (meta or {}).get("steps") or []]
        self.step_at = 0
        self.recipe = None

    # -- what a page sees ----------------------------------------------------
    def summary(self) -> dict:
        return dict(id=self.id, name=self.name, state=self.state,
                    pid=self.pid, python=self.python,
                    created=self.created, started=self.started,
                    ended=self.ended, exit=self.exit, stage=self.stage,
                    counter=self.counter, status=self.status,
                    wrap=self.wrap, folder=self.folder,
                    log=_log_of(self.folder),
                    events=len(self.events),
                    reports=self.reports, meta=self.meta,
                    steps=self.steps)

    def events_from(self, start: int, wait: float = 0.0):
        """The events from index `start` on, waiting up to `wait` seconds
        for new ones; (events, next index, finished)."""
        with self.cond:
            if wait and len(self.events) <= start \
                    and self.state in ("queued", "running"):
                self.cond.wait(timeout=wait)
            out = self.events[max(start, 0):]
            return (list(out), len(self.events),
                    self.state not in ("queued", "running"))

    # -- what the reader adds ------------------------------------------------
    def add(self, text: str, transient: bool, log=None) -> None:
        # the empty draw a bar starts with (a lone carriage return before
        #   its first frame): nothing to show, and its first frame is a line
        if transient and not strip(text).strip():
            return
        level = level_of(text)
        plain = strip(text)
        with self.cond:
            op = "replace" if self.transient and self.events else "add"
            self.events.append(dict(n=len(self.events), op=op,
                                    text=plain, level=level))
            # every event is kept, the page shows the whole log (a run of
            #   three targets: 15000 lines, 1.6 MB)
            self.transient = transient
            message = message_of(text)
            found = STAGE.search(message)
            if found:
                self.stage = "PCA2D " + found.group(1)
            if found:
                self._advance("pca2d_" + found.group(1))
            found = RECIPE.search(message)
            if found:
                self.stage = "LBL " + found.group(1).lower()
                self.recipe = _recipe_kind(found.group(1))
            found = OBJECT.search(message)
            if found and self.recipe:
                self._advance(self.recipe, found.group(1))
                self.recipe = None
            if REPORT.search(message.strip()):
                self._advance("report", done=True)
            found = next((m for m in (c.search(message) for c in COUNTERS)
                          if m), None)
            if found:
                self.counter = "%s / %s" % found.groups()
                done, total = (int(v) for v in found.groups())
                # how far the step going is, which its pill fills to
                for step in self.steps:
                    if step["state"] == "running" and total > 0:
                        step["progress"] = min(1.0, done / total)
            if message.strip() and not transient:
                self.status = message.strip()[:160]
            self.cond.notify_all()
        if log is not None and not transient:
            log.write(plain + "\n")

    def _advance(self, kind: str, obj=None, done: bool = False) -> None:
        """The log has reached a step: the first one of this kind (and
        object) from the current on; those before it are done."""
        for index in range(self.step_at, len(self.steps)):
            step = self.steps[index]
            if step["kind"] == kind and (obj is None or step["obj"] == obj):
                for before in self.steps[:index]:
                    if before["state"] in ("pending", "running"):
                        before["state"] = "done"
                    before.pop("progress", None)
                step["state"] = "done" if done else "running"
                step.pop("progress", None)
                # the event the log reached it at: a click on the step
                #   shows that line
                step.setdefault("at", len(self.events) - 1)
                self.step_at = index + 1 if done else index
                return

    def finish_steps(self) -> None:
        """At the end: every step done when the run succeeded; otherwise the
        one it was in failed (or was stopped), the others stay as they
        are."""
        for step in self.steps:
            if self.state == "done":
                step["state"] = "done"
            elif step["state"] == "running":
                step["state"] = "stopped" if self.state == "stopped" \
                    else "failed"

    def note(self, text: str, level: str = "info") -> None:
        """A line of the window's own (start, end, stop)."""
        stamp = datetime.datetime.now().strftime("%y%m%d %H:%M:%S.%f")[:-4]
        with self.cond:
            self.events.append(dict(n=len(self.events),
                                    op="add", text="%s | %s" % (stamp, text),
                                    level=level))
            self.transient = False
            self.cond.notify_all()

    def record(self) -> None:
        """run.json, what the history reads back."""
        out = self.summary()
        out.pop("events", None)
        with open(os.path.join(self.folder, "run.json"), "w") as handle:
            json.dump(out, handle, indent=1, default=str)


class Runner:
    """The queue: runs start in the order they were asked for, one at a
    time (a PCA2D fit can take most of a laptop's memory).

    A run does not depend on the window: its output goes to a file
    (run.raw) that the window follows, and its exit code to another (exit),
    written by the shell that starts it. Closing the window leaves a run
    going; opening it again on the same data folder finds it (adopt), and
    the runs still queued with it.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.order = []
        self.adopted = set()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, name: str, folder: str, wrap: str, python: str,
               cwd: str, meta: dict, job_id: str = None) -> Job:
        job_id = job_id or new_id(name)
        job = Job(job_id, name, folder, wrap, python, cwd, meta)
        job.note("queued: %s %s, in %s" % (python, wrap, cwd))
        job.record()
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.wake.set()
        return job

    def adopt(self, data_dir: str, runs_folder: str) -> None:
        """The runs of a data folder this window did not start: those still
        going (followed from their log) and those still queued (queued again,
        in their order); once per data folder."""
        data_dir = os.path.abspath(data_dir)
        with self.lock:
            if data_dir in self.adopted:
                return
            self.adopted.add(data_dir)
            known = set(self.jobs)
        found = []
        for path in glob.glob(os.path.join(data_dir, runs_folder, "*",
                                           "run.json")):
            try:
                with open(path) as handle:
                    record = json.load(handle)
            except (OSError, ValueError):
                continue
            if record.get("id") in known or record.get("state") not in (
                    "queued", "running"):
                continue
            found.append(record)
        found.sort(key=lambda r: r.get("created") or 0)
        for record in found:
            meta = record.get("meta") or {}
            if not meta.get("steps"):
                meta = dict(meta, steps=_steps_of(record.get("wrap")))
            job = Job(record["id"], record.get("name", ""), record["folder"],
                      record["wrap"], record.get("python", ""), data_dir,
                      meta)
            job.created = record.get("created") or time.time()
            job.started = record.get("started")
            job.pid = record.get("pid")
            if record["state"] == "running":
                if job.pid and _alive(job.pid):
                    job.state = "running"
                    job.note("found running (pid %s): following its log"
                             % job.pid, "warn")
                    threading.Thread(target=self._follow, args=(job, None),
                                     daemon=True).start()
                else:
                    # it ended while no window was there: its exit file says
                    #   how, its log says the rest
                    job.state = "running"
                    threading.Thread(target=self._follow, args=(job, None),
                                     daemon=True).start()
            else:
                job.note("queued in an earlier session", "warn")
            with self.lock:
                self.jobs[job.id] = job
                self.order.append(job.id)
        self.wake.set()

    def get(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id)

    def listing(self) -> list:
        with self.lock:
            return [self.jobs[j].summary() for j in self.order]

    def running(self):
        with self.lock:
            for job_id in self.order:
                if self.jobs[job_id].state == "running":
                    return self.jobs[job_id]
        return None

    def stop(self, job_id: str) -> bool:
        """Stop a run (its whole process group), or take a queued one off
        the queue."""
        job = self.get(job_id)
        if job is None:
            return False
        if job.state == "queued":
            job.state = "cancelled"
            job.ended = time.time()
            job.note("taken off the queue", "warn")
            job.record()
            return True
        if job.state != "running" or not job.pid:
            return False
        job.stopping = True
        job.note("stopping: SIGTERM to the run and everything it started",
                 "warn")
        try:
            os.killpg(job.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False

        def escalate():
            time.sleep(GRACE)
            if _alive(job.pid):
                job.note("still running after %.0f s: SIGKILL" % GRACE,
                         "warn")
                try:
                    os.killpg(job.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        threading.Thread(target=escalate, daemon=True).start()
        return True

    def _next(self):
        with self.lock:
            if any(self.jobs[j].state == "running" for j in self.order):
                return None
            for job_id in self.order:
                if self.jobs[job_id].state == "queued":
                    return self.jobs[job_id]
        return None

    def _loop(self) -> None:
        while True:
            self.wake.wait(timeout=1.0)
            self.wake.clear()
            job = self._next()
            if job is not None:
                self._run(job)
                self.wake.set()

    def _run(self, job: Job) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        job.state = "running"
        job.started = time.time()
        job.note("started: %s %s" % (job.python, os.path.basename(job.wrap)))
        raw = open(os.path.join(job.folder, "run.raw"), "ab")
        try:
            # the shell writes the exit code where a window that was closed
            #   meanwhile can still read it; the run is the session leader of
            #   its own group, so that Stop reaches what it starts
            proc = subprocess.Popen(
                ["/bin/sh", "-c", '"$0" -u "$1"; echo $? > "$2"',
                 job.python, job.wrap, os.path.join(job.folder, "exit")],
                cwd=job.cwd, env=env, stdout=raw, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            raw.close()
            job.note("could not start: %s" % exc, "error")
            job.state, job.ended, job.exit = "failed", time.time(), -1
            job.record()
            return
        raw.close()
        job.pid = proc.pid
        job.record()
        self._follow(job, proc)

    def _follow(self, job: Job, proc) -> None:
        """Follow a run's raw output until it has ended and all of it is
        read; then say how it ended."""
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        rest = ""
        raw_path = os.path.join(job.folder, "run.raw")
        exit_path = os.path.join(job.folder, "exit")
        while not os.path.exists(raw_path):
            time.sleep(0.2)
        from .history import run_log
        log_path = run_log(job.folder)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(raw_path, "rb") as raw, open(log_path, "w") as log:
            while True:
                chunk = raw.read(65536)
                if chunk:
                    segments, rest = cut_output(rest + decoder.decode(chunk))
                    for text, transient in segments:
                        job.add(text, transient, log)
                    log.flush()
                    continue
                ended = (proc.poll() is not None) if proc is not None \
                    else not _alive(job.pid)
                if ended:
                    # what was written between the last read and the end
                    chunk = raw.read()
                    if chunk:
                        segments, rest = cut_output(rest + decoder.decode(
                            chunk))
                        for text, transient in segments:
                            job.add(text, transient, log)
                    break
                time.sleep(0.2)
            if rest:
                job.add(rest, False, log)
        if proc is not None:
            proc.wait()
        code = None
        # a run stopped by a signal never reaches the line that writes the
        #   exit file: no need to wait for it
        for _ in range(1 if job.stopping else 10):
            try:
                with open(exit_path) as handle:
                    code = int(handle.read().strip())
                break
            except (OSError, ValueError):
                time.sleep(0.2)
        job.exit = code
        job.ended = time.time()
        job.reports = reports_since(job.cwd, job.started or job.created)
        took = duration(job.ended - (job.started or job.created))
        if job.stopping:
            job.state = "stopped"
            job.note("stopped after %s" % took, "warn")
        elif code == 0:
            job.state = "done"
            job.note("done in %s" % took)
        elif code is None:
            job.state = "failed"
            job.note("ended without an exit code (killed?) after %s" % took,
                     "error")
        else:
            job.state = "failed"
            job.note("ended with exit code %d after %s" % (code, took),
                     "error")
        job.finish_steps()
        job.record()
        self.wake.set()


def _log_of(folder: str) -> str:
    """Where a run's log is (history.run_log), without moving anything."""
    logs = os.path.join(os.path.dirname(os.path.dirname(folder)), "logs")
    return os.path.join(logs, os.path.basename(folder.rstrip(os.sep)) + ".log")


def _steps_of(wrap) -> list:
    """The steps of a run started before its steps were kept (run.json
    without them): planned again from its wrap, read and never run."""
    try:
        from . import wrapgen, wrapload
        rparams = wrapload.read(wrap)["rparams"]
        return wrapgen.steps(rparams)
    except Exception:                                     # noqa: BLE001
        return []


def _recipe_kind(banner: str) -> str:
    """A recipe banner's step kind: telluclean, template, mask, compute,
    compile (others as they are, lowercased)."""
    word = banner.lower().replace(" ", "")
    if word.startswith("compil"):
        return "compile"
    return word


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def new_id(name: str) -> str:
    """A run's id, and the name of its folder: when, and what."""
    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    return ("%s_%s" % (stamp, re.sub(r"[^A-Za-z0-9_+-]+", "_", name)))[:80]


def reports_since(data_dir: str, since: float) -> list:
    """The report PDFs a run wrote: DATA_DIR/reports and the PCA2D run
    folders, modified after it started."""
    out = []
    for pattern in ("reports/*.pdf", "pca2d/*/*/*.pdf", "pca2d/*/*/*/*.pdf"):
        for path in glob.glob(os.path.join(data_dir, pattern)):
            if "/superseded/" in path:
                continue
            try:
                if os.path.getmtime(path) >= since:
                    out.append(path)
            except OSError:
                pass
    return sorted(out)


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "%d h %02d min" % (hours, minutes)
    if minutes:
        return "%d min %02d s" % (minutes, secs)
    return "%d s" % secs
