"""
The public spectra LBL reads, found and fetched from the archives: SPIRou's
t.fits at the CADC, NIRPS, HARPS and ESPRESSO at ESO, HARPS-N at the TNG,
SOPHIE at the OHP; each star resolved by SIMBAD (its coordinates, its motion
from Gaia, its Teff). What LBL's window (lbl.gui.archives) asks of it:
lblrun.groups, lblrun._ident, lblrun.INSTRUMENTS and the getters of each
instrument's files, and stars.stellar_teff.

FROM KOLOA (github.com/eartigau/koloa): the part of it this needs, module by
module, each definition kept as it was there (so that a fix there can be
carried here by comparing them), with two changes: koloa's own imports are
relative, and the team's shared repository of HARPS-N exposures
(koloa.shared) is left out, the TNG's archive being asked directly. LBL does
not depend on koloa.

What it keeps while it searches (the TNG's pointings, a star's exposures,
SIMBAD's answers) goes to ~/.cache/lbl/gui, or the folder LBL_GUI_CACHE
names (fetch.paths.cache).
"""
