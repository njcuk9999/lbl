# v0.68.001 (2026-09-23)

## Major changes

### [LBL] compile: residual projections

(e.g. DTEMP3000) per photometric band 
make_rdb_table gave one value per file for each residual projection, over
the whole wavelength domain. It now also gives one per photometric band, in
{KEY}_{band} and s{KEY}_{band} columns written next to {KEY} and s{KEY}
(e.g. DTEMP3000_j and sDTEMP3000_j).

The bands are the ones already defined in astro.bands, which the per-band
velocities use, so there is no second definition of a bandpass: a band is
used when the lines kept by the compilation hold at least
RESPROJ_BAND_MIN_LINES of them, and a file gets a value for that band when
at least RESPROJ_BAND_MIN_FRAC of those lines have one. The value is the
odd ratio mean of the lines of the band, as for the whole domain.

Two places in make_rdb_table: the bands and their columns, next to the cut
on the lines, and the measurement itself, next to the one over the whole
domain.

GL725B (SPIRou, 959 files, DTEMP3000): bands z, y, j, h, k, and

  column        median (K)   scatter   median error
  DTEMP3000         -7.41      1.79      0.42
  DTEMP3000_z       -3.79      2.46      1.46
  DTEMP3000_y       -3.80      2.38      1.37
  DTEMP3000_j       -5.60      2.19      1.15
  DTEMP3000_h       -6.66      2.15      0.60
  DTEMP3000_k      -14.05      2.37      1.16

Every column that existed before is unchanged, checked against the same
lblrv files compiled with this code without the patch.

### [LBL] astro: J band red edge 1349.441 nm (was 13494.41)

MKO/NSFCam.J from the SVO Filter Profile Service: 1148.178 to 1349.441 nm,
mean 1248.414 nm. With the typo the J band was never selected by
choose_bands, so the instruments that use astro.bands and cover the J band
(the NIRPS modes) now get their per-band J columns (vrad_j). SPIRou and
NIRPS-APERO use their own binned bands and do not change.


### [LBL] mask lines at the significant extrema of the template

lbl_mask put a line edge at every sign change of the derivative of the
template. Where the template is noisy the derivative crosses zero on noise
alone, so most edges were noise: on a template of 98 HARPS-N spectra of an F
star the minima of the mask had a median depth of 0.12% and the lines were
5.7 km/s wide, narrower than the lines of the star. The weight cut of
lbl_mask (10 x the median second derivative) then removed the strongest
lines, since that median was set by the noise extrema (Na D, Mg b and the
Ca I triplet were absent from the mask).

An extremum is now kept only where the derivative of the template is
significant on both sides of it. The savgol filter of lbl_template is
linear, so the uncertainty of its derivative follows from the coefficients
of the filter and the uncertainty of each point of the template
(sqrt(pi/2) rms / sqrt(nbin)). Walking along the spectrum, a maximum is
declared where z = dflux / sigma goes from above +MASK_EDGE_NSIG to below
-MASK_EDGE_NSIG, at the zero crossing next to the highest point of the
smoothed flux; minima the other way round. The weight cut is not applied to
these lines. No model and no line list are involved: it works on any
empirical template, optical or infrared.

The window of the filter comes from APPROX_RESOLUTION and the dispersion of
the template, as calculate_savgol_template computes it, and is checked
against the window measured on the template itself (which keeps both the
raw and the filtered flux), with a warning if they disagree.

The threshold adapts to the quality of the template: at 3 sigma, 59% of the
extrema are removed on the Kepler-21 template (98 HARPS-N spectra) and 7% on
the GL725B template (971 SPIRou spectra, SNR of about 1400 per point).

New parameters: MASK_SIGNIFICANT_EDGES (True), MASK_EDGE_NSIG (3.0),
MASK_EDGE_NOISE_FACTOR (1.0). mp.gaussian_weighted_savgol_coeffs gives the
coefficients of the filter (same filter as before).

Kepler-21 (HARPS-N, 98 files), RV scatter and median uncertainty per file:
  before                         6.22 m/s, 2.07 m/s
  significant edges              5.93 m/s, 1.90 m/s
  with COMPIL_MAX_PIXEL_WIDTH=500 (branch test-speed-260922-130215, needed
  in the optical, where 50 px is only 41 km/s and cuts the strongest lines)
  before                         5.86 m/s, 1.26 m/s
  significant edges              5.68 m/s, 1.27 m/s
GL725B (SPIRou, 971 files): unchanged (11.31 -> 11.38 m/s, 2.72 -> 2.73 m/s)


### [LBL] compile: COMPIL_MAX_PIXEL_WIDTH 500 px (was 50) for all the instruments

Lines wider than COMPIL_MAX_PIXEL_WIDTH are removed in lbl_compile. 50 px is
114 km/s with the SPIRou pixels (2.28 km/s) but only 41 km/s with the
HARPS-N pixels (0.818 km/s), so in the optical it removed the strongest
lines of the star (Mg b, Na D, Ca I, H alpha: 60 to 210 km/s from one local
maximum to the next), and it also removed the wider lines that the
significant extrema produce.

Kepler-21 (HARPS-N, 98 files), RV scatter and median uncertainty per file:
  significant edges, 50 px       5.93 m/s, 1.90 m/s
  significant edges, 500 px      5.68 m/s, 1.27 m/s
GL725B (SPIRou, 971 files): unchanged, 50 px is already 114 km/s there and
only 0.3% of the lines are wider
  current mask                   11.31 m/s, 2.72 m/s (both cuts)
  significant edges              11.38 -> 11.37 m/s, 2.73 m/s

### [LBL] 

Fixes for typo and warning 2 mentioned in #Issue71 (and found by Claude and possible a fix for Flavie's problem)