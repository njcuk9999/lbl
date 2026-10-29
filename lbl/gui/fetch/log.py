"""
How the archive search reports what it is doing.

Every line goes to the console as 'YYMMDD HH:MM:SS.SS | message', its
colour saying what it is: green for what is happening, blue for a number
being reported, orange for something skipped or doubtful that does not
stop the search, red for why it stops. The same message also goes to every
function of LISTENERS (the window's log of the search, lbl.gui.archives),
without the stamp or the colour.
"""
import sys
import threading
from datetime import datetime
from typing import Callable, List


# =============================================================================
# Define variables
# =============================================================================
_LOG_COLORS = {
    'info': '\033[92m',        # green, what is happening
    'value': '\033[94m',       # blue, a number being reported
    'warn': '\033[38;5;208m',  # orange, skipped or doubtful, not fatal
    'error': '\033[91m',       # red, why the search stops
}


_LOG_RESET = '\033[0m'


#: functions called with (message, level) for every line said
LISTENERS: List[Callable[[str, str], None]] = []


#: one line at a time: the archives of a star are asked side by side, and
#:    two lines said at once must not run into each other
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
    Print one status line, timestamped and colour-coded by its role, and
    hand it to the LISTENERS

    :param message: str, what to say
    :param level: str, info, value, warn or error
    """
    color = _LOG_COLORS.get(level, _LOG_COLORS['info'])
    # colours only when a human is reading
    if sys.stdout.isatty():
        _say(f'{color}{_timestamp()} | {message}{_LOG_RESET}')
    else:
        _say(f'{_timestamp()} | {message}')
    for listener in list(LISTENERS):
        try:
            listener(str(message), level)
        except Exception:  # a listener never stops the search
            pass
