"""Two-frame weighted PCA precleaning of echelle spectra.

M components anchored in the stellar rest frame and N in the observer frame,
fitted together, so the fit decides which frame a spectral feature belongs to
rather than the analyst choosing a registration. Only the observer block is
divided out of the spectra, which then go to LBL.

    lbl_pca2d --object NAME      # cube, fit, figures, correct, lbl

    cli          the command line, the stages, the plan of a run
    config       the three-layer configuration and its defaults
    cube, build  the spectra onto one log-uniform grid, cached     (cube)
    twoframe     the M+N fit                                       (fit)
    figures      the report, one multipage PDF                     (figures)
    reconstruct  corrected spectra, exactly panel 3 of the report  (correct)
    lbl          LBL set up, and run, on both sets of spectra      (lbl)
    storage      products kept on an external disk, linked back
    cache        the rebuildable intermediates, and their snippets
    wizard       validate a configuration against the data

Input is extracted echelle spectra, order by order, read through LBL's own
instrument classes (readers). Documentation: lbl/doc/PCA2D.md
"""
__version__ = "0.3.0"
