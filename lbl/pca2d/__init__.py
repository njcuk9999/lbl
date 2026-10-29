"""Two-frame weighted PCA precleaning of echelle spectra (SPIRou, NIRPS).

M components anchored in the stellar rest frame and N in the observer frame,
fitted together, so the fit decides which frame a spectral feature belongs to
rather than the analyst choosing a registration. Only the observer block is
divided out of the spectra, which then go to LBL.

    pca2d-preclean --object TOI2120      # cube, fit, figures, correct, lbl

    cli          the command line, the stages, the plan of a run
    config       the three-layer configuration and its defaults
    cube, build  t.fits onto one log-uniform grid, cached         (cube)
    twoframe     the M+N fit                                       (fit)
    figures      the report, one multipage PDF                     (figures)
    reconstruct  corrected t.fits, exactly panel 3 of the report   (correct)
    lbl          LBL set up, and run, on both sets of spectra      (lbl)
    storage      products kept on an external disk, linked back
    cache        the rebuildable intermediates, and their snippets
    wizard       validate a configuration against the data

Input is APERO `t.fits`, order by order. Documentation:
https://eartigau.github.io/pca2d_preclean/
"""
__version__ = "0.3.0"
