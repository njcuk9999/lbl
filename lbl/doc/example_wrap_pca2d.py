#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example wrap file: LBL velocities with the PCA2D precleaning.

Copy it, change the five lines marked CHANGE, and run it:

    python example_wrap_pca2d.py

WHAT PCA2D DOES
---------------
Every exposure of a campaign holds two things that do not move together: the
star, whose lines shift with the barycentric velocity (BERV), and what sits in
the observer's frame, residual tellurics, airglow, detector features, which
does not. Over a campaign the same wavelength sample is seen at many BERVs.
PCA2D fits every exposure at once as

    ln(flux) - high pass =  star spectrum, in the star's frame (coefficient 1)
                          + n_star  components in the star's frame
                          + n_earth components in the observer's frame
                          + noise

and divides the observer-frame part out of the spectra. What is left is the
star, which LBL then measures like any other object. The fit decides which
frame a feature belongs to; nothing is masked by hand.

WHAT THIS FILE DOES, IN ORDER
-----------------------------
1. LBL measures the object as delivered, DATA_DIR/science/<object>
   (PCA2D compare=True), so that the two series can be compared.
2. PCA2D corrects it: DATA_DIR/science/<object>_pca2d, the same files under
   the same names, flux corrected. The fit, its figures and its own report
   go to DATA_DIR/pca2d/<object>/1-3/.
3. LBL measures the corrected object: template, mask, compute, compile,
   DATA_DIR/lblrdb/lbl_<object>_pca2d_<object>_pca2d.rdb.
4. The report, DATA_DIR/reports/<object>_pca2d_<object>_pca2d.pdf, is one
   PDF: pca2d's report (the star from SIMBAD, the delivered and corrected
   velocities side by side with what got better and what got worse, the
   river plots of the correction step by step, the components, their
   coefficients in time and their periodograms, the residuals), then LBL's
   own pages (the series, the false inclusion probability, the indicators).

With RUN_LBL_TELLUCLEAN on, which is the path for the optical spectrographs,
step 2 comes after LBL's telluric cleaning and corrects <object>_tc into
<object>_tc_pca2d.

AN EXAMPLE OF WHAT IT GIVES
---------------------------
TOI-2120, an M dwarf, 321 SPIRou exposures over a 37.9 km/s BERV span, deep
water bands across the stellar lines: the nightly velocity scatter goes from
47.98 to 14.20 m/s on the 80 nights both series share, a factor 3.4.

SIPS J1259-4336, NIRPS HE, 54 exposures, run through this very path on
2026-10-19: its corrected velocities equal those of the standalone pca2d
pipeline to 1.6e-6 m/s.

Not every star gains, and the fit can take part of a real signal with it.
50 m/s signals injected at 37.3 d into NIRPS spectra came out 12 to 14%
smaller after the correction, with a narrow BERV span (TOI-210, 7.6 km/s)
and a wide one (TOI-782, 58 km/s) alike; TOI-756 c (K = 273 m/s) kept 68% of
its amplitude. On a narrow span the components can follow the star's own
velocity night after night. The report's velocity pages are there to say
which case a star is: read them before trusting a gain.
https://eartigau.github.io/pca2d_preclean/ has the method and the results.

Created on 2026-10-19
"""
from lbl.recipes import lbl_wrap

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    rparams = dict()
    # -------------------------------------------------------------------------
    # The usual LBL settings: nothing here is specific to PCA2D
    # -------------------------------------------------------------------------
    rparams['INSTRUMENT'] = 'NIRPS_HE'                         # CHANGE
    rparams['DATA_SOURCE'] = 'CADC'                            # CHANGE
    rparams['DATA_DIR'] = '/path/to/lbl'                       # CHANGE
    rparams['DATA_TYPES'] = ['SCIENCE']
    # the folder under DATA_DIR/science; it is the object, whatever the
    #   headers call it
    rparams['OBJECT_SCIENCE'] = ['TOI756']                     # CHANGE
    # the object itself: each object its own template. PCA2D makes this the
    #   corrected object's template, built from the corrected spectra
    rparams['OBJECT_COMPARISON'] = ['TOI756']                  # CHANGE
    rparams['OBJECT_TEFF'] = [3300]
    # not for SPIRou or NIRPS from APERO, whose t.fits are already
    #   telluric-corrected; yes for the optical spectrographs
    rparams['RUN_LBL_TELLUCLEAN'] = False
    rparams['RUN_LBL_TEMPLATE'] = True
    rparams['RUN_LBL_MASK'] = True
    rparams['RUN_LBL_COMPUTE'] = True
    rparams['RUN_LBL_COMPILE'] = True
    rparams['SKIP_LBL_TEMPLATE'] = True
    rparams['SKIP_LBL_MASK'] = True
    rparams['SKIP_LBL_COMPUTE'] = True
    rparams['SKIP_LBL_COMPILE'] = True
    # the report: with PCA2D, pca2d's pages and LBL's in one PDF
    rparams['REPORT'] = True
    # -------------------------------------------------------------------------
    # PCA2D: the precleaning. Remove these lines and LBL is what it was.
    # -------------------------------------------------------------------------
    rparams['PCA2D'] = dict(
        # components fitted in the star's frame (stellar variability, a
        #   residual of the star's own shift); none of them is divided out
        n_star=1,
        # components fitted in the observer's frame, and divided out of the
        #   spectra: tellurics, airglow, detector. 3 for SPIRou and NIRPS,
        #   2 for the optical spectrographs (the nominal for HARPS)
        n_earth=3,
        # measure the delivered object too, first, for the comparison
        compare=True,
        # the figures of the correction (river plots, components...): they
        #   follow REPORT unless set here
        # figures=True,
        # --- what a row of the fit is. For NIRPS, by default, a night
        #     coadded, cut into bundles of at most 0.1 km/s of BERV (a visit
        #     of three or four exposures stays one row, a transit night is
        #     cut into several):
        # bundle_berv=0.05,                 # bundles of at most 50 m/s
        # bundle_minutes=20,                # or of at most 20 minutes
        # bundle_berv=None,                 # whole nights
        # nightly_stack=False,              # or every exposure on its own
        # --- anything else is a key of lbl/pca2d/nominal.yaml, which says
        #     what each one does and why it has the value it has. Examples:
        # keep the residual outliers rather than setting them to NaN
        # nsig_cut=0,
        # a sample is kept if most exposures weighted it, not all of them
        # correct=dict(mask_share=0.5),
        # more sweeps of the fit
        # **{'twoframe.iters': 24},
        # --- and to keep variants side by side, each its own LBL object
        # name='maskhalf',                  # -> <object>_pca2d_maskhalf
        # --- several stars of one instrument, one observer basis shared by
        #     all the objects of OBJECT_SCIENCE (they need shared nights)
        # joint=True,
        # --- redo the correction even if the same one is already on disk
        # overwrite=True,
    )
    # -------------------------------------------------------------------------
    # Run the wrapper code using the above settings
    # -------------------------------------------------------------------------
    lbl_wrap.main(rparams)

# =============================================================================
# End of code
# =============================================================================
