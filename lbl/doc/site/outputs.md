# What LBL writes

## The folders of a DATA_DIR

```
ROOT/Downloads/                 as downloaded
ROOT/models/                    PHOENIX, TAPAS, vetted templates: shared
ROOT/<INSTRUMENT>_<SOURCE>/     a DATA_DIR
    science/<obj>/              the spectra (yours)
    science/<obj>_tc/           telluric-cleaned (optical)
    science/<obj>_pca2d/        corrected by PCA2D
    science/<obj>_residuals/    PCA2D's residuals (with save_residuals)
    templates/                  LBL_Template_<obj>_<instrument>.fits
    masks/                      the lines
    lblrv/<obj>_<template>/     the velocity of every line, one file per spectrum
    lblrdb/                     the time series (.rdb)
    lblreftable/                the reference table of the lines
    pca2d/<obj>/<run>/          PCA2D's fit, figures and report
    pca2d/cache/                PCA2D's cubes, made again when needed
    reports/                    the reports (PDF)
    gui/runs/<run>/             what the window ran: wrap.py, run.raw, run.json
    calib/                      the blazes
    models -> ../models
```

Only `science/<obj>/` is irreplaceable: everything else is made again by
running. Beside each product, a small `.inputs` file says what it was made
from, and a product is made again when its inputs change (spectra added, a
new correction). Corrected spectra to give a colleague: `science/<obj>_pca2d/`.

## The tables

`lblrdb/` holds plain text tables, readable in any language (tab
separated, a header line, a line of dashes):

- `lbl_<object>_<template>.rdb`: one row per spectrum;
- `lbl2_<object>_<template>.rdb`: the spectra of each night combined into
  one row.

Their main columns:

| column | what |
|---|---|
| `rjd` | the time: BJD - 2 400 000 |
| `vrad`, `svrad` | the velocity and its uncertainty, m/s |
| `d2v`, `sd2v`, `d3v`, `sd3v` | the projections on the template's second and third derivatives (line shape) |
| `dW`, `sdW` | the change of the lines' width |
| `DTEMP<T>`, `sDTEMP<T>` | the change of effective temperature (DTEMP3500 when asked; one per band, `_j`, `_h`...) |
| `vrad_<nm>`, `svrad_<nm>` | the velocity of each wavelength band |
| `vrad_achromatic`, `vrad_chromatic_slope`, `CRX` | the velocity at a reference wavelength, its slope with wavelength |
| `dEW_<species>_<um>`, `sdEW_...` | the change of equivalent width of an activity line (H alpha, He, Na...) |
| header keys | the spectra's own (BERV, airmass, SNR...), copied |

The [Data Explorer](tab_explore.md) plots any of them against any other.

## Reports

With `REPORT` (on by default), each run writes a PDF in `reports/`: a
summary, a page per step, the velocities and their periodogram with the
false inclusion probability ([Hara et al.
2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...663A..14H/abstract)),
the semi-amplitudes of known planets at their ephemerides and, with PCA2D,
its own pages first (the river plots, the components, the velocities before
and after).

The known planets (the NASA Exoplanet Archive's confirmed planets, and any
period the configuration gives) are fitted together, so that none is read
through another's signal, and twice, whatever their number: as circular
orbits and as full Keplerians (K, e, omega and the phase free, a Gaussian
of width 0.3 on e cos omega and e sin omega keeping a few outlying points
from being fitted by an orbit of e near 1), the periods
held at the published ones, each fit with an offset, a straight line in
time when every period is under a third of the campaign, and a free jitter
(`lbl/pca2d/keplerfit.py`). The figure of the folded velocities shows each
planet with the others taken out, its two orbits with each fit's K and e,
and under it, touching it, the residuals to the whole Keplerian fit. A
table gives each fit's Delta BIC (against no planet, the Keplerian against
the circular), each planet's in each fit and each planet's eccentricity;
the summary table gives each planet's K and Delta BIC from the joint
circular fit. Each of the two fits is then sampled by an MCMC of
its own (from its maximum likelihood, priors uniform in K, the phase, e and
omega): a figure of the posteriors of K, e and omega, a corner of each
fit (every pair of parameters, the covariances, delivered and corrected
overlaid) and a table of every parameter (median, 16th and 84th
percentiles) close the star's pages.
