"""
Where the archive search keeps what it has asked once (SIMBAD's and Gaia's
answers about a star, the TIC's Teff, the headers of a star's HARPS-N
exposures), and how it names itself to the archives.

The cache is ~/.cache/lbl/gui, or the folder the environment variable
LBL_GUI_CACHE names. Everything in it can be asked again: deleting it only
makes the next search slower.
"""
import os


#: how the archive search names itself to the archives (HTTP User-Agent)
USER_AGENT = 'LBL (github.com/njcuk9999/lbl)'


def cache(*parts: str) -> str:
    """a path in the cache: cache('tic') is ~/.cache/lbl/gui/tic, or
    <LBL_GUI_CACHE>/tic"""
    base = os.environ.get('LBL_GUI_CACHE') or os.path.join(
        os.path.expanduser('~'), '.cache', 'lbl', 'gui')
    return os.path.join(os.path.abspath(os.path.expanduser(base)), *parts)
