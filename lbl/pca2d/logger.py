"""Timestamped, colour-coded console logging.

Convention (shared across projects):
    YYMMDD HH:MM:SS.SS | message

Colours encode the semantic role of the line:
    green  -> general info / progress narration
    blue   -> a reported numeric value (count, median, RMS, ...)
    orange -> something skipped, or a non-critical issue that does not stop the run
    red    -> explains why the run is stopping
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from datetime import datetime

_COLOURS = {
    "info": "\033[32m",    # green
    "value": "\033[34m",   # blue
    "warn": "\033[33m",    # orange/yellow
    "error": "\033[31m",   # red
}
_RESET = "\033[0m"

# The class the progress bars are built from, so that a message can ask them to
# step aside. None when tqdm is absent, and then there are no bars to collide.
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:                                           # pragma: no cover
    _tqdm = None

#: colour when writing to a terminal, or when asked for: the window that runs
#: the pipeline (pca2d.gui) reads these codes to colour its own log, and it is
#: a pipe, not a terminal
_USE_COLOUR = sys.stdout.isatty() or os.environ.get("PCA2D_COLOUR") == "1"


def set_colour(enabled: bool) -> None:
    """Force colour output on or off (auto-detected from the tty by default)."""
    global _USE_COLOUR
    _USE_COLOUR = enabled


# Below this many characters of message room, wrapping stops helping and starts
# turning one sentence into a column of single words.
_MIN_ROOM = 30


def _terminal_width():
    """The window's width in characters, or None when there is no window.

    None whenever stdout is not a terminal. A log file or a pipe has no width,
    and wrapping it would split across two lines the very message a grep is
    looking for; every background run in this package writes to a file.
    """
    if not sys.stdout.isatty():
        return None
    return shutil.get_terminal_size(fallback=(0, 0)).columns or None


def _wrap(message, room):
    """The message cut into lines of at most `room` characters.

    Never inside a word, and so never inside a path or a name like TOI-2120 or
    2-7v: a path that does not fit is left long rather than broken, because a
    broken path can no longer be copied back out of the terminal. Continuation
    lines keep the first line's indentation, so an indented block stays one.
    """
    lines = []
    for paragraph in str(message).split("\n"):
        indent = paragraph[: len(paragraph) - len(paragraph.lstrip())]
        wrapped = textwrap.wrap(paragraph.strip(), width=room,
                                initial_indent=indent, subsequent_indent=indent,
                                break_long_words=False, break_on_hyphens=False)
        lines.extend(wrapped or [indent])
    return lines


def stamp(now=None) -> str:
    """The convention's timestamp: `YYMMDD HH:MM:SS.SS`, to hundredths.

    Here rather than inside log() because the window that runs the pipeline
    (pca2d.gui) writes its own lines beside the run's, and the two must carry
    the same stamp rather than two spellings of the same idea.
    """
    now = now or datetime.now()
    return now.strftime("%y%m%d %H:%M:%S.") + "%02d" % (now.microsecond // 10000)


def log(message: str, level: str = "info") -> None:
    """Print a timestamped, colour-coded message, one stamp per printed line.

    level is one of 'info', 'value', 'warn', 'error'.

    In a terminal, a message longer than the window is wrapped to its width and
    EVERY line gets the timestamp, the same one, so the convention holds line by
    line (`YYMMDD HH:MM:SS.SS | message`) and a wrapped message still reads as
    one event rather than as a message followed by orphan text. Outside a
    terminal nothing is wrapped.

    Written through tqdm.write, not print. A bar lives on stderr and this on
    stdout, and in a terminal the two share one line: printed straight, a
    message landed at the end of whatever bar was being drawn, as in
    "...sweep/s, bases260910 10:44:44.68 | iter 0". tqdm.write clears every
    active bar, writes the message on its own line, and redraws the bars below
    it, so a message and a bar never share a line whichever of them came first.
    """
    prefix = "%s | " % stamp()
    width = _terminal_width()
    if width is not None and width - len(prefix) >= _MIN_ROOM:
        lines = _wrap(message, width - len(prefix))
    else:
        lines = str(message).split("\n")
    colour = _COLOURS.get(level, _COLOURS["info"]) if _USE_COLOUR else ""
    reset = _RESET if _USE_COLOUR else ""
    text = "\n".join("%s%s%s%s" % (colour, prefix, line, reset) for line in lines)
    if _tqdm is not None:
        _tqdm.write(text, file=sys.stdout)
    else:                                                     # pragma: no cover
        print(text)
    sys.stdout.flush()
