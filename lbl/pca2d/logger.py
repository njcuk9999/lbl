"""Console logging, through LBL's own logger.

pca2d runs inside LBL, so its messages read as LBL's: the same stamp, the
same level letter, the same colours, and in LBL's log file with the rest.
pca2d's four levels are LBL's four:

    info   -> G, green   general info / progress narration
    value  -> I, blue    a reported numeric value (count, median, RMS, ...)
    warn   -> W, yellow  something skipped, or a non-critical issue that does
                         not stop the run
    error  -> E, red     explains why the run is stopping

pypdf's own warnings ("Annotation sizes differ: 21 vs. 0", when the report's
pages are copied) are silenced here, since every module that reaches pypdf
imports this one first: they come with no stamp, and the report is right.
"""

from __future__ import annotations

import logging
import sys

# The class the progress bars are built from, so that a message can ask them to
# step aside. None when tqdm is absent, and then there are no bars to collide.
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:                                           # pragma: no cover
    _tqdm = None

#: pca2d's levels as LBL's logger names them
_LBL_LEVELS = {"info": "general", "value": "info", "warn": "warning",
               "error": "error"}

logging.getLogger("pypdf").setLevel(logging.ERROR)


def log(message: str, level: str = "info") -> None:
    """Print a message through LBL's logger, one stamp per line.

    level is one of 'info', 'value', 'warn', 'error'.

    Written with the progress bars stepped aside (tqdm.external_write_mode):
    a bar and LBL's logger share stderr, and in a terminal the two would
    otherwise share one line. The bars are cleared, the message written on
    its own lines, and the bars redrawn below it.
    """
    # imported here: lbl.core.io loads all of LBL's core
    from lbl.core import io
    write = getattr(io.log, _LBL_LEVELS.get(level, "general"))
    lines = str(message).split("\n")
    if _tqdm is not None:
        with _tqdm.external_write_mode(file=sys.stderr):
            for line in lines:
                write(line)
    else:                                                     # pragma: no cover
        for line in lines:
            write(line)
