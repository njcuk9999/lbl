"""
LBL and PCA2D run without the window (lbl.gui): on a headless server, or from
a checkout where the window is not installed. Its own tests (test_gui_*.py)
are then not collected, and the few window checks in the other files skip
themselves (pytest.importorskip), so that the rest of the suite still runs.
"""
import importlib.util

try:
    _WINDOW = importlib.util.find_spec("lbl.gui") is not None
except ImportError:
    _WINDOW = False

#: pytest reads this: the window's own test files, left out without it
collect_ignore_glob = [] if _WINDOW else ["test_gui_*.py"]
