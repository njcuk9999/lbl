# The template: every pixel of every spectrum

LBL measures each spectrum against a template of the star, through the
template's first derivatives. Since 2026-11-09 the template is made without
resampling any spectrum: BSD, BERV Spectral Drizzle (`TEMPLATE_METHOD =
'BSD'`, the default; `'SCATTER'`, its name until 2026-11-10, is still
accepted, as are the `TEMPLATE_BSD_*` parameters): every
pixel of every spectrum keeps its own wavelength in the star's frame, and the
template and its derivatives are a local polynomial fitted through that cloud
of points. The method comes from
[templatepy](https://github.com/eartigau/templatepy). The former template, the
median of the spectra resampled into bins of BERV, is still there
(`TEMPLATE_METHOD = 'BERVBIN'`).

## The idea: the spectra oversample the star

Moved to the star's frame by its BERV (and the star's velocity, both
relativistic: `ln(wavelength) - atanh(v / c)`), each spectrum samples the
star at its own position within a pixel. Over a season the BERV spans tens of
km/s, tens of pixels, so the spectra together sample the star far more finely
than the detector: the 92 NIRPS spectra of Gl 317 give 96 points per km/s,
one spectrum gives one (a pixel is 1.00 km/s). The template is fitted through
these points; no spectrum is interpolated, so no interpolation error and no
correlated noise enter it.

## How it is made

Order by order (`lbl/science/template_scatter.py`):

1. Each spectrum's order is divided by the blaze and by its median; the s1d
   weights (the blaze, falling smoothly at the order edges and below
   `BLAZE_THRESHOLD`) say where the order is used.
2. At each point of the template's grid (the usual LBL grid, every 0.5 km/s
   for NIRPS, 1 km/s for SPIRou), a cubic is fitted by weighted least squares
   through the points within 1.5 FWHM (`lbl/core/gaussian_savgol.py`):
   - a Gaussian weight in wavelength, of FWHM `TEMPLATE_BSD_FWHM`
     resolution elements (`c / APPROX_RESOLUTION`);
   - each point weighted by `1 / sigma**2`, its noise being a low pass of its
     spectrum's residuals;
   - points further than `TEMPLATE_BSD_CLIP` times the RMS of the
     normalised residuals of their window are left out (a local scale: the
     cores of lines the cubic cannot follow are not clipped more than
     elsewhere);
   - points on both sides of the grid point, or no value: the polynomial is
     never extrapolated over the end of an order or a masked band;
   - nor drawn across a hole of the points: no value where the nearest point
     on either side is further than `TEMPLATE_BSD_MAX_GAP` window FWHM (a
     stretch every spectrum blanked, as PCA2D's corrected spectra have
     where the fit gave no weight), nor where less than
     `TEMPLATE_BSD_MIN_BALANCE` of the window's weight is on one side of
     the point (the edge of such a hole, where the cubic would be
     extrapolated from the other side). Without them, the cubic held by the
     few points beyond a hole swung far from the spectrum: on TOI-756's 167
     NIRPS spectra corrected by PCA2D, 222 template samples of 1180-1230 nm
     had an rms above 10% of the flux and 33 were 15% off the uncorrected
     template; with them, 0 and 0, for 2% of the samples.
3. A pixel of the detector whose residual, over the spectra, has a median
   beyond `TEMPLATE_BSD_OBS_CLIP` sigma is left out of all spectra: what
   stays on the detector while the star moves (telluric residuals, bad
   pixels).
4. Each spectrum is divided by a low pass (`HP_WIDTH`) of its ratio to the
   template: what the blaze division leaves.
5. Steps 2 to 4 are repeated (`TEMPLATE_BSD_NITER` passes).

The orders are then combined on the grid with the s1d weights, each brought
back to its level in the spectra so that they join without a step: all the
orders (`flux`), the odd ones and the even ones (`flux_odd`, `flux_even`).

Each fit is one pass over the points of its window: the normal equations of a
polynomial are moments of the weights, `sum(g w u**p)`, so these are summed,
with those of `g**2 w` for the covariance; the residuals come from the same
sums. No sort, no second pass.

With `SMALL_RV_SHIFT = False` (and no known orbits), the velocity of each
spectrum is measured first against the templates of the orders within 5 % of
`COMPIL_SLOPE_REF_WAVE`, pass after pass (the template of these orders made
again at each), until `TEMPLATE_RV_CONV_FRAC` (95 %) of the spectra changed
by less than `TEMPLATE_RV_CONV_NSIG` (2) times their uncertainty in the last
pass, `TEMPLATE_RV_MAX_PASSES` (10) at most; a warning says when they did
not. The passes start from the velocities of the object's velocity box
(below), so that a star that moves by km/s is not smeared in the first
template, whose velocities would then not converge. In each pass: a scan of
the shift, then Bouchy et al. (2001)
pixel by pixel on the template's Savitzky-Golay derivative. Each pixel gives
its own shift, `d_i = -(y_i - T_i) / T'_i`, of uncertainty `sigma_i / |T'_i|`
(their weighted mean is the projection on the derivative, Bouchy's optimal
weights), and the pixels are combined by LBL's mixture model
(`odd_ratio_mean`, a Gaussian and a flat floor of outliers; Artigau et al.
2022) rather than by the plain weighted mean. Bouchy's step is the first
order of the shift's expansion, right only for a small shift: it is
iterated, the template and its derivative taken again at each new shift,
until a step is 1 % of the spectrum's uncertainty or less (20 steps at
most; a spectrum still moving is named in a warning). From 3 km/s off,
with lines 4.5 km/s wide and no noise, one step leaves 1.4 km/s, two
169 m/s, three 0.3 m/s, five nothing; on the synthetic spectra of the
tests, from 3 km/s off with no scan, two steps (the method before
2026-11-24) left 74 and 112 m/s, the iteration less than 0.1 m/s. On synthetic spectra with 1 %
of bad pixels on the lines' flanks, after the scan, the velocities stay at
0.8 m/s rms (the photon noise's) where the least squares of before are
pulled to 22 m/s; without bad pixels the two agree (0.8 m/s).

With `PLANETS` and the star's orbits known, `SMALL_RV_SHIFT = True` makes
the template in one pass on the orbits; `False` starts from the orbits and
iterates on top of them, since the planets were in the spectra to start
with: `VSYS` is then the star's whole velocity, the orbits included (`VKEP`
their part), and `RVITER` in the header says the velocities were measured.
On GJ1 (NIRPS, 168 spectra, a K = 10 m/s signal injected, 2026-11-18), the
template's velocities give K = 10.9 +- 0.7 m/s free and 11.0 +- 0.7 m/s
starting from the injected orbit known (1.0 m/s of it added by the
iteration), and follow LBL's velocities of the same spectra to 3.9 m/s rms
(correlation 0.92).

Each spectrum's velocity, less their median, and its uncertainty (the
mixture model's) are written in the log once measured, one line per
spectrum (its number, file and MJD), and kept in the template's table of
spectra (`VSYS`, `VSYS_ERR`). The plot of the template's velocities shows
the iterations: the velocities at the start and after each pass, light to
dark, under those kept (against the spectrum's number and its MJD), and the
convergence, the share of the spectra that changed by less than
`TEMPLATE_RV_CONV_NSIG` sigma in each pass against `TEMPLATE_RV_CONV_FRAC`,
the rms change of each pass and the Bouchy steps each took (median and most
over the spectra).
PCA2D's star frame follows them (`registration.velocities`, by file name),
so that its star component is built where the star is and the PCA does not
take in part of a real signal; `lbl_wrap` makes the object's own template
this way before PCA2D, whatever template it is measured against
(`OBJECT_COMPARISON`), and makes it again when it was made with the velocity
taken as constant (`SMALLRV` in its header).

## The velocity box

Each spectrum's velocity is kept in a small box, one JSON file per object,
`DATA_DIR/velocities/<object>.json` (`lbl/science/velocity_box.py`), made
the first time a step needs a starting point and brought up to date as the
measurements improve. A spectrum's entry, under its file name: `v`, the
star's velocity in it, BERV taken out (relativistic composition: what the
template registers the spectrum at), `err`, `by` (what measured it), `mjd`,
`berv` and the date. An entry is replaced by a measurement of a higher rank
(`ccf` < `template` < `lbl`) or a newer one of the same rank, never by one
of a lower rank. The corrected spectra (`<object>_pca2d`) share their
object's box; an injection test (`<object>_inj`) has its own. The file is
written whole under a lock, so that two runs writing at once keep their
entries; `lbl_reset` deletes the boxes.

- **The first time** a spectrum is not in the box, its velocity is measured
  by a CCF against the PHOENIX model at the star's Teff (`OBJECT_TEFF`, the
  model the mask uses), as the mask's systemic velocity is
  (`general.sysvel_ccf`: ln flux on a grid of 0.5 km/s, high-passed, the
  photometric bands only, every shift at once by FFT): a peak below 5 sigma
  is not kept. The median of these velocities is the box's `systemic`, a
  clean systemic velocity of the star against the model. On TOI-904 (NIRPS,
  14 CADC spectra, 2026-11-24): 1 s a spectrum, systemic +2.848 km/s against
  LBL's median of +2.886 km/s for the same spectra, and the CCF velocities
  50 m/s rms from the template's.
- **The template** (`SMALL_RV_SHIFT = False`) starts from the box, less its
  systemic velocity (the template keeps its frame: `VSYS` stays relative),
  known orbits kept for the spectra the box has from a CCF only, and writes
  its velocities back (`by: template`).
- Without `OBJECT_TEFF` or a model, the passes start from the BERV alone (or
  the orbits), as before, and a warning says so.

## Explicit derivatives

With `u = (x - x0) / r` (`x` is ln(wavelength), `r` the radius of the
window), the cubic at `x0` is `c0 + c1 u + c2 u**2 + c3 u**3`, so

    T = c0,   T' = c1 / r,   T'' = 2 c2 / r**2,   T''' = 6 c3 / r**3

These are the derivatives compute uses (columns `flux_savgol_d0` to `d3`,
per step of the grid as before). Their uncertainties come from the fit (the
sandwich covariance of the weighted least squares, scaled by the mean square
of the normalised residuals): on synthetic spectra the scatter of the
template and of its derivative about the noise-free template is 0.99 times
these uncertainties.

## Parameters

| parameter | default | what |
|---|---|---|
| `TEMPLATE_METHOD` | `'BSD'` | `'BSD'` (or its old name `'SCATTER'`) or `'BERVBIN'` (the FP and other calibrations always use `'BERVBIN'`) |
| `TEMPLATE_BSD_FWHM` | 1.0 | FWHM of the window, in resolution elements |
| `TEMPLATE_BSD_NITER` | 4 | passes per order |
| `TEMPLATE_BSD_CLIP` | 5.0 | clip of the local outliers, in units of the RMS of their window |
| `TEMPLATE_BSD_OBS_CLIP` | 2.0 | clip of the detector pixels, in sigma of their median residual |
| `TEMPLATE_BSD_MAX_GAP` | 0.5 | no value where the nearest point on either side is further than this many window FWHM (0: no limit) |
| `TEMPLATE_BSD_MIN_BALANCE` | 0.1 | no value where less than this share of the window's weight is on one side of the point (0: no limit) |
| `TEMPLATE_BSD_MEMORY` | 2.0 | memory for the orders read at once [GB]; more orders take more reads of the files |

## What the template file holds

The same columns as before, so that the mask and compute read it unchanged,
and one more:

| column | BSD |
|---|---|
| `flux`, `flux_odd`, `flux_even` | the fitted value (the same as `*_savgol_d0`) |
| `eflux`, `eflux_odd`, `eflux_even` | its uncertainty, from the fit |
| `rms`, `rms_odd`, `rms_even` | `eflux * sqrt(nbin) / sqrt(pi / 2)`, so that the mask, which reads the BERVBIN `rms` as the spread of `nbin` medians, gets `eflux` back |
| `flux_savgol_d0` to `d3` (and odd, even) | the value and derivatives, per step of the grid |
| `eflux_savgol_d1` | the uncertainty of the first derivative: the mask's significant edges (`MASK_SIGNIFICANT_EDGES`) use it instead of propagating a grid savgol window |

The header keyword `LBLTMPLT` is `LBL_BSD` (`LBL_SCATTER` for a template made
before the rename). The method is part of the
template's provenance token: a template made by the other method is made
again.

## The width of the window: not too narrow

Each spectrum is part of the template it is measured against. A narrow window
copies each spectrum's own noise into the template, sharply, at that
spectrum's own pixels; when compute moves the template, that copy moves away
from the noise it came from, which pulls the velocity of the spectrum towards
the template's: all velocities shrink, more so as the SNR falls and with fewer
spectra. A velocity of 15 m/s added to the 40 NIRPS spectra of GJ 1028 (their
wave solutions times the relativistic Doppler ratio), through the whole of
LBL, came back:

| template | fraction given back |
|---|---|
| BSD, 0.4 resolution element (templatepy's 3 steps of the grid) | 0.838 |
| BSD, 1 resolution element (default) | 0.983 |
| BERVBIN | 0.978 |

With the template held fixed (made from the spectra without the added
velocity), compute gives back 1.000 with either template: the pull is the
template's. With 729 bright spectra (eps Eri) all three give back 1.000. The
last 2 % with 40 spectra is about `1/N`: any template made of the spectra it
measures has it (see `KnownLimitations.md`).

## How it compares with the BERV bins

| | BERVBIN | BSD |
|---|---|---|
| GJ 1028 (NIRPS, 40 spectra): rms | 5.76 m/s | 5.73 m/s |
| eps Eri (NIRPS, 729 spectra): rms within the nights | 1.470 m/s | 1.472 m/s |
| a star moving by +-1.5 km/s (synthetic, `SMALL_RV_SHIFT = False`): velocities found to | 18 m/s | 3 m/s |
| the template of 729 NIRPS spectra | 2.1 min | 4.4 min |

The velocities are the same where the BERV bins worked; the uncertainties,
the derivatives and the stars that move are better.

## Code and tests

- `lbl/core/gaussian_savgol.py`: the local polynomial through scattered
  points (`scatter_savgol`), numba.
- `lbl/science/template_scatter.py`: reading the orders, the passes, the
  velocities, the combination of the orders.
- `lbl/recipes/lbl_template.py`: the choice of the method; the BERV bins are
  in `_bervbin_template`.
- Tests: `maintenance/tests/test_gaussian_savgol.py` (derivatives,
  uncertainties, outliers) and `maintenance/tests/test_template_recipe.py`
  (the template within its uncertainty, the velocities of a moving star, no
  pull of the velocities with the default window).
