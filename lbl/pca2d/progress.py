"""Telling the user what is happening, for the hour it takes.

A run reads three hundred spectra, builds a five gigabyte cube, iterates a
decomposition sixteen times and writes three hundred corrected files. Left to
itself it prints nothing for forty minutes, and there is no way to tell a
working run from a wedged one. Every loop that can take more than a few seconds
goes through here.

Two things, deliberately kept apart:

`bar` is a tqdm with `leave=False`, so a finished loop erases its own bar and
the scrollback holds the narration rather than a graveyard of completed
progress bars. It writes to stderr, and is drawn wherever a redrawn line means
something: a terminal, and the window that runs the pipeline. Into a log file
it would be a mile of half-drawn lines, so there the loop narrates itself
instead, one line every half minute saying how far it has got.

`log` is that narration: one line per thing that happened, timestamped
`YYMMDD HH:MM:SS.SS | message` and coloured by role. Green for progress, blue
for a number worth remembering, orange for something skipped, red for why the
run is stopping. `stage` bookends a phase and reports how long it took, because
"still going" and "this one is slow" look identical without it.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

from .logger import log

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:                                           # pragma: no cover
    _tqdm = None


# What the bars currently belong to, "sweep 3" say, prefixed to every bar's
# label so that a bar reading "coefficients" says which sweep it is part of.
_label = None


def set_label(label):
    """Name the unit of work every bar opened from now on belongs to.

    A setter and not a context manager on purpose: the loop that uses it has
    `break`s in it, and resetting once after the loop is simpler than
    re-indenting a hundred lines under a `with`.
    """
    global _label
    _label = label


def _labelled(desc):
    """`desc` with the current label in front of it, if there is one."""
    if _label and desc:
        return "%s: %s" % (_label, desc)
    return desc or _label or ""


#: how often a loop with no bar to draw says where it has got to
_SAY_EVERY = 30.0


def _drawn():
    """Whether a bar can be drawn where this run's output is going.

    A terminal, or the window that runs the pipeline: it sets PCA2D_COLOUR, it
    reads the run through a pipe, and it redraws a carriage-returned line in
    place (gui.cut_output), so a bar is a bar there too. A log file is neither,
    and a bar in one is a mile of half-drawn lines.
    """
    return sys.stderr.isatty() or os.environ.get("PCA2D_COLOUR") == "1"


def bar(iterable=None, total=None, desc="", unit="it", **kwargs):
    """A progress bar that leaves nothing behind when it finishes.

    Where no bar can be drawn, the loop narrates itself instead: one line every
    half minute saying how far it has got and how long is left. Without it, a
    run reading three hundred spectra off a shared disk says "cube footprint:
    2.86 GB" and then nothing at all for minutes, which is the one thing this
    module exists to prevent.

    Falls back to the plain iterable if tqdm is not installed, so nothing here
    is a hard dependency of the science.
    """
    desc = _labelled(desc)
    if not _drawn():
        return _Narrator(iterable, total, desc, unit)
    if _tqdm is None:                                         # pragma: no cover
        return iterable if iterable is not None else _Null()
    return _tqdm(iterable, total=total, desc=desc, unit=unit, leave=False,
                 file=sys.stderr, dynamic_ncols=True, **kwargs)


class _Narrator:
    """What a bar becomes when there is no screen to draw it on.

    Same shape as the bar it replaces (iterate it, or update and close it), and
    it says the same three things a bar says: how far, how fast, how much
    longer. Every `_SAY_EVERY` seconds and never faster, because the point is a
    log somebody reads, not a log somebody greps through.
    """

    def __init__(self, iterable=None, total=None, desc="", unit="it",
                 every=_SAY_EVERY):
        self.iterable = iterable
        self.unit = unit or "it"
        self.desc = desc or "working"
        self.every = every
        if total is None:
            total = len(iterable) if hasattr(iterable, "__len__") else None
        self.total = total
        self.done = 0
        self.started = self.last = time.time()

    def __iter__(self):
        for item in self.iterable if self.iterable is not None else ():
            yield item
            self.update()

    def update(self, n=1):
        self.done += n
        now = time.time()
        if now - self.last >= self.every:
            self.last = now
            self._say(now)

    def _say(self, now):
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        where = ("%d/%d" % (self.done, self.total) if self.total
                 else "%d" % self.done)
        left = ""
        if self.total and rate > 0 and self.done < self.total:
            left = ", about %s left" % human((self.total - self.done) / rate)
        log("%s: %s %s, %.1f %s/s%s"
            % (self.desc, where, self.unit + ("s" if self.done != 1 else ""),
               rate, self.unit, left), "value")

    def set_postfix_str(self, *a, **k):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Null:                                                  # pragma: no cover
    """What `bar` returns without tqdm, so callers need no special case."""

    def update(self, *a, **k):
        pass

    def close(self):
        pass

    def set_postfix_str(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@contextlib.contextmanager
def stage(title, level="info"):
    """Announce a phase, then say how long it took.

    The closing line is the useful one: it turns "the run has been going for
    forty minutes" into "the fit took seventeen of them", which is the
    difference between waiting and debugging.
    """
    log(title, level)
    started = time.time()
    try:
        yield
    finally:
        seconds = time.time() - started
        log("%s: %s" % (title, human(seconds)), "value")


def human(seconds):
    """A duration a person reads at a glance."""
    seconds = float(seconds)
    if seconds < 90:
        return "%.1f s" % seconds
    if seconds < 5400:
        return "%d min %02d s" % (int(seconds // 60), int(seconds % 60))
    return "%d h %02d min" % (int(seconds // 3600), int((seconds % 3600) // 60))


def size(nbytes):
    """Bytes, in the unit a person would have used."""
    nbytes = float(nbytes)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return "%.1f %s" % (nbytes, unit)
        nbytes /= 1024
    return "%.1f TB" % nbytes                                 # pragma: no cover
