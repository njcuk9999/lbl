"""
Where the archive search keeps what it fetched once (the TNG's pointings, a
star's exposures, SIMBAD's and Gaia's answers): ~/.cache/lbl/gui, or the
folder the environment variable LBL_GUI_CACHE names.

FROM KOLOA (github.com/eartigau/koloa, module paths), where it is
~/.cache/koloa or KOLOA_CACHE: LBL's window keeps a cache of its own.
"""
import os


def cache(*parts: str) -> str:
    """a path in the cache: cache('archive') is ~/.cache/lbl/gui/archive,
    or <LBL_GUI_CACHE>/archive"""
    base = os.environ.get('LBL_GUI_CACHE') or os.path.join(
        os.path.expanduser('~'), '.cache', 'lbl', 'gui')
    return os.path.join(os.path.abspath(os.path.expanduser(base)), *parts)
