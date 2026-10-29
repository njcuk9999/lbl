"""
The public spectra LBL reads, found and fetched from the archives, for the
window's archive search (lbl.gui.archives): SPIRou's t.fits at the CADC,
NIRPS, HARPS and ESPRESSO at ESO, HARPS-N at the TNG, SOPHIE at the OHP.

    stars          the star: SIMBAD's names, position and proper motion,
                   its effective temperature
    spectra        an instrument's runs of LBL for a star (groups) and how
                   each of their files is fetched
    observatories  how the archives are asked; SOPHIE's exposures
    tng            HARPS-N's exposures, from the headers of their CCF files
    log            the lines said while the archives are asked
    paths          the cache of what was asked once (~/.cache/lbl/gui, or
                   the folder LBL_GUI_CACHE names)

Every archive is asked with no account, for what is public; nothing here
needs anything outside LBL's own dependencies.
"""
