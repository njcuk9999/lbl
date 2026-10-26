"""
The one way koloa talks to the console.

Every line reads 'YYMMDD HH:MM:SS.SS | message' and its colour says what it
is: green for what is happening, blue for a number being reported, orange
for something skipped or doubtful that does not stop the run, red for why
the run stops.

Created on 2026-09-27

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module log): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import os
import sys
import threading
from datetime import datetime
from typing import Optional


# =============================================================================
# Define variables
# =============================================================================
_LOG_COLORS = {
    'info': '\033[92m',        # green, what is happening
    'value': '\033[94m',       # blue, a number being reported
    'warn': '\033[38;5;208m',  # orange, skipped or doubtful, not fatal
    'error': '\033[91m',       # red, why the run stops
}


_LOG_RESET = '\033[0m'


#: set to False to silence koloa (the tests do)
VERBOSE = True


#: one line at a time: the archives of a star are asked side by side, and
#   two lines said at once must not run into each other (the GUI reads
#   them one by one)
_SAYING = threading.Lock()


# =============================================================================
# Define functions
# =============================================================================
def _say(line: str):
    """one whole line to the console, never cut by another thread's"""
    with _SAYING:
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


def _timestamp() -> str:
    """
    The current instant as 'YYMMDD HH:MM:SS.SS'

    :return: str, the timestamp
    """
    now = datetime.now()
    return now.strftime('%y%m%d %H:%M:%S') + f'.{now.microsecond // 10000:02d}'


def log(message: str, level: str = 'info'):
    """
    Print one status line, timestamped and colour-coded by its role

    :param message: str, what to say
    :param level: str, info, value, warn or error
    """
    if not VERBOSE and level not in ('warn', 'error'):
        return
    color = _LOG_COLORS.get(level, _LOG_COLORS['info'])
    # colours only when a human is reading
    if sys.stdout.isatty():
        _say(f'{color}{_timestamp()} | {message}{_LOG_RESET}')
    else:
        _say(f'{_timestamp()} | {message}')


def row(key: str, state: str, text: str = '', done: Optional[int] = None,
        total: Optional[int] = None):
    """
    Where an archive of a star being gathered is, for the table of
    koloa's GUI: one line, 'row: key | state | done | total | text', said
    only when the GUI runs koloa (KOLOA_PROGRESS=gui), which keeps it
    out of its log

    :param key: str, the archive (dace, carmenes, a source of
                koloa.published, tess, space)
    :param state: str, asking, ok (it has something), none, failed
    :param text: str, what it has, or why not
    :param done: int or None, the files read so far
    :param total: int or None, of how many
    """
    if os.environ.get('KOLOA_PROGRESS', '') != 'gui':
        return
    text = ' '.join(str(text).split())
    _say(f'{_timestamp()} | row: {key} | {state} | '
          f'{"" if done is None else int(done)} | '
          f'{"" if total is None else int(total)} | {text}')
