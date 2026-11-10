# Known limitations

State of branch `lbl-pca2d` at 0b9e2c9 (2026-11-03), after a full read of
the code. Everything found and fixed in that read is gone from this list;
what is here is known and left as it is. Numbers are measured, on the data
named.

## Left by decision

- **TOI 1452 with PCA2D:** the joint fit TOI 5579 + TOI 4860 + TOI 1452
  takes TOI 1452 from 8 to 48 m/s rms. The likely cause is the shared
  basis absorbing the mismatch between TOI 1452 and another star's
  template, but this is not tested.
- **O2 B band on HARPS and HARPS-N:** the telluric fit drops the last
  order, so the dry exponent comes from one 5 nm bin in the gamma band.

## Can change the velocities

**PCA2D on EXPRES** (HD 10700, 19 exposures; the one of 2023-12-28 is
-144 m/s off in every series, cause unknown):

| Series | rms (m/s) | median error (m/s) |
|---|---|---|
| delivered | 3.26 | 0.42 |
| LBL telluric cleaning | 3.02 | 0.38 |
| PCA2D, nominal | 3.01 | 0.56 |
| PCA2D, no residual clip | 3.15 | 0.41 |

- **The nominal residual clip is costly here.** `nsig_cut: 3.0` in
  `lbl/pca2d/nominal.yaml` blanks about 4% of the flux, including at
  490-560 nm where there are no tellurics. The errors grow by about 35%.
- **The O2 B band is over-corrected on delivered spectra.** Between deep
  line pairs the flux is multiplied by up to about 1.7 (1.67 and 1.76 in
  two measurements, near 689.02 nm). Seven
  components do not fix it; running PCA2D after LBL's telluric cleaning
  does.

**Other PCA2D**
- **`PCA2D_TO_OBSERVER` is False on ESPRESSO and NIRPS ESO.** Their grids
  are barycentric like EXPRES, which needed True; not tested, no data.

**Telluric cleaning**
- **One resolving power per class.** HARPS's line profile changed in 2015:
  the dry exponent over airmass goes from 0.914 to 0.831 on eps Eri, and
  from 0.903 to 0.817 on Gl 229.
- **Telluric lines sit to the blue of the TAPAS model** on three
  instruments: EXPRES -29 m/s, HARPS-N -22, HARPS -61 (medians).
  `TELLUCLEAN_DV0` is 0 everywhere and nothing fits it.
- **Water is biased +6.6% at SNR 5** (unbiased at SNR 10).

**Template and mask**
- **A star measured against its own template loses about 1/N of its
  velocities**, N its number of spectra: each spectrum is part of the
  template. A velocity added to the 40 NIRPS spectra of GJ 1028 came back
  0.978 times as large with BERVBIN, 0.983 with BSD; 1.000 with the 729
  of eps Eri. With few spectra and the star's own template, a planet's K is
  that much small.
- **The BSD template takes about twice as long as BERVBIN** (4.4 against
  2.1 min for 729 NIRPS spectra): the low passes of each spectrum, and the
  files read twice when the orders do not fit in `TEMPLATE_BSD_MEMORY`.
- **A BSD window much narrower than one resolution element shrinks the
  velocities** (0.84 at 0.4 resolution element on GJ 1028): see
  `TEMPLATE.md`. Nothing stops a user from setting it.
- **`MASK_EDGE_NOISE_FACTOR` defaults to 1.0**, while the derivative noise
  of a BERVBIN template is 1.80 times the propagated one. `MASK_EDGE_NSIG` = 3
  is therefore about 1.7 true sigma: on a synthetic sparse line list, 34% of
  the minima kept are real lines. (A BSD template gives the uncertainty
  of its derivative from the fit, `eflux_savgol_d1`, calibrated.)
- **`file_level` is one number per file** (BERVBIN). Where a file lacks a
  point, its bin moves to the other files' local level: a 1.0% step in the
  test.

**Compute**
- **Non-convergence is flagged only by `RESET_RV` and a large error:** a
  start 20 km/s off ends 30 km/s off.
- **The low-pass of the residual projections absorbs about 4%** of an even
  signal weaker than the noise.
- **`HP_WIDTH` has no effect in compute.**
- **Columns carried over from earlier files:** where a file does not reach
  a line, `MEANXPIX`, `MEANBLAZE`, `RMSRATIO`, `NPIXLINE` and `CHI2` hold
  the value from an earlier file.

**Instruments**
- **`get_berv` takes any `OBJECT_SCIENCE` containing `FP` for a
  Fabry-Perot:** an object named `HD_FP_5` silently gets a BERV of 0. This
  affects HARPS, HARPS-N, SOPHIE, CORALIE and MAROON-X.
- **MAROON-X's RJD is not barycentric.**
- **Copied limits:** `NIRPS_HE_ESO` has `NIRPS_HA_ESO`'s, and `Harps_ESSP`
  has HARPS-N's domain.
- **Different blaze normalisation:** `Harps_ORIG` uses degree 1, while
  `HarpsN_ORIG` uses 9.
- **EXPRES data:**
  - flux over variance is 0.83 to 0.84, not 1;
  - the Excalibur wavelengths leave the order median by more than 50 m/s
    over about the first 60 pixels of each calibrated range.

**Reports**
- **EXPRES telluric pages are empty.** `series.spectrum_path` finds
  spectra by an ISO timestamp, and EXPRES file names have none.
- **The telluric page may not divide by the blaze:** the Gl 229 curves
  slope across the window. Not verified.
- **Bad points still cost the outlier-aware FIP detections:** 8 bad points
  out of 100 bring detection down to 0.02.
- **`kepfit` jitter is biased low at small N.**

## Robustness

- **Unexpected NaN or crash on edge inputs:**
  - `odd_ratio_mean` returns NaN without a message when no line is near
    the median;
  - `lowpassfilter` raises `ZeroDivisionError` for a width of 3 or less;
  - `upper_envelope` is all NaN for a width below 3;
  - the PCA2D fit raises `ZeroDivisionError` when `min_pixel_snr` removes
    every sample.
- **A `MASK_FILE` without its type in the name** gets the pos, neg and
  full masks written to the same path. Found by reading, not run.
- **Wrap and parameter checks:**
  - `SKIP_LBL_RESMAP` or `RUN_LBL_NOISE` set in a wrap stops it;
  - `not_none` is not enforced;
  - the Generic class requires four `TELLUCLEAN_*` parameters that are no
    longer used.
- **Nights need the observatory's longitude:** without network access and
  without astropy's site registry in its cache, LBL stops. This is by
  design.
- **Downloads with a single source:**
  - The PHOENIX mirror (`STELLAR_MIRROR_URL`,
    github.com/eartigau/lbl-phoenix-mirror) holds only LBL's default grid:
    log g 4.5, Z 0, alpha 0, Teff 3000-6000 K. Another `OBJECT_LOGG` still
    needs the Goettingen site.
  - `MODEL_REPO_URL` (TAPAS, temperature gradients) is one server, a bare
    IP over plain HTTP, with no mirror. Its `tapas_lbl.fits` (181 MB, 104 MB
    with xz) is over GitHub's 100 MB per-file limit.

## Dead code, duplicates, stale text

- **Recipes that compute nothing:** `lbl_noise` and `lbl_resmap`.
- **Code and parameters never used:**
  - `cube.build_weights` (and with it `telluric.py` and several `weights`
    keys), `readers.parity_is_safe`, `fip.outlier_probability`,
    `general.bouchy_equation_line`, `general.get_scaling_ratio`;
  - in `compute_rv`: `model0`, `ratio_sci` and the commented-out science
    CCF;
  - in `lbl_template`: `frac_valid` and `flux_cube0`;
  - parameters `BERVBIN_*`, `MASK_SNR_MIN`, `TELLUCLEAN_KERNEL_WID`,
    `TELLUCLEAN_GAUSSIAN_SHAPE`, `TELLUCLEAN_WAVE_LOWER` and
    `TELLUCLEAN_WAVE_UPPER`.
- **Duplicates:**
  - two `frequency_grid` functions that differ at the upper end;
  - three copies of the robust scatter;
  - 193 of the 400 instrument method definitions are exact copies.
- **Stale comments and docstrings:**
  - "equation 11" (it is 10 in Artigau et al. 2022);
  - the docstrings of `bin_by_time`, `estimate_sigma`, `val_cheby` and
    `correct_rdb_drift`;
  - the description of `KW_RESET_RV` (10 iterations, against a limit
    of 15);
  - the `reconstruct.py` module docstring (T_p);
  - some comments in `nominal.yaml`;
  - `AddingAnInstrument.md` does not mention `science_flux_column`.
- **Small inconsistencies:**
  - `lbl_find` logs as `lbl_compute`;
  - the report path ignores `RDB_SUFFIX`;
  - with a different comparison star and no skip, the science template is
    built twice.
