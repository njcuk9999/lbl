"""LBL's window: a browser page that builds, runs and follows LBL wraps with
the PCA2D precleaning, and finds the public spectra of a star.

    lbl_gui                    # opens the window in the browser

The window writes an ordinary LBL wrap (wrap.py, which runs by hand too) for
the targets and options picked, runs it in the environment LBL is installed
in, one run at a time, and shows its log as it goes; its history and what its
products weigh on disk are read back from the data folder.

KEPT APART FROM THE REST OF LBL: nothing of LBL imports lbl.gui, and lbl.gui
reads LBL only as a user would (its instruments' parameters, its wrap). Its
search of the public archives is lbl.gui.fetch, so that the window depends
on nothing outside LBL. Its page is static/; what makes the page's
references and background and the testers' guide is in maintenance/gui, for
the window's maintainers only.
"""
from lbl.core.base import __version__  # noqa: F401
