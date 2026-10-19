# PCA2D: two-frame precleaning inside LBL

`lbl.pca2d` fits every exposure of an object as a star spectrum in the
star's rest frame, plus a few components anchored in the star frame and a
few anchored in the observer frame (residual tellurics, airglow, detector
features). The observer-frame part is divided out of the spectra, which
LBL then measures like any other object. Method and results:
<https://eartigau.github.io/pca2d_preclean/>.

## Turning it on

One key in the wrap's run parameters:

```python
from lbl.recipes import lbl_wrap

rparams = dict()
rparams['INSTRUMENT'] = 'NIRPS_HE'
rparams['DATA_SOURCE'] = 'CADC'
rparams['DATA_DIR'] = '/data/nirps/lbl'
rparams['DATA_TYPES'] = ['SCIENCE']
rparams['OBJECT_SCIENCE'] = ['TOI756']
rparams['OBJECT_COMPARISON'] = ['TOI756']
rparams['OBJECT_TEFF'] = [3300]
rparams['RUN_LBL_TEMPLATE'] = True
rparams['RUN_LBL_MASK'] = True
rparams['RUN_LBL_COMPUTE'] = True
rparams['RUN_LBL_COMPILE'] = True
# the precleaning, with the nominal settings
rparams['PCA2D'] = dict(n_star=1, n_earth=3)

lbl_wrap.main(rparams)
```

Without `PCA2D`, nothing of `lbl.pca2d` is imported and LBL runs exactly as
before.

## What it does to the tree

| path | what |
|---|---|
| `science/<object>/` | read, never written |
| `science/<object>_pca2d/` | the corrected spectra, same names, same format |
| `pca2d/<object>/request.yaml` | the configuration that ran |
| `pca2d/<object>/<M>-<N>/` | the fit, the figures, the report (PDF) |
| `pca2d/cache/` | the registered cubes, rebuildable |

Every step after it (template, mask, compute, compile, report) runs on
`<object>_pca2d`, and the comparison follows when it was the object itself.
A different `OBJECT_COMPARISON` is used as it is, uncorrected.

With `RUN_LBL_TELLUCLEAN` the precleaning runs after the telluric cleaning,
on `<object>_tc`, and writes `<object>_tc_pca2d`: this is the path for the
optical spectrographs.

## The dict

| key | default | |
|---|---|---|
| `n_star` | 1 | star-frame components fitted |
| `n_earth` | 3 | observer-frame components fitted and divided out |
| `figures` | True | the run's figures and its report |
| `overwrite` | False | redo even when the same correction is on disk |
| `name` | None | `<object>_pca2d_<name>`, to keep variants side by side |
| `joint` | False | every science object against one observer basis |

Anything else is a key of the pca2d configuration,
`lbl/pca2d/nominal.yaml`, which holds the nominal settings and their
reasons: as `'section.key'`, as a dict per section, or by a short name.

```python
rparams['PCA2D'] = dict(n_star=1, n_earth=3,
                        nsig_cut=0,                        # short name
                        correct=dict(mask_share=0.5),      # a section
                        **{'twoframe.iters': 24})          # a dotted key
```

The short names: `velocity_term`, `iters`, `shrink`, `nsig_cut`,
`bias_nsig`, `column_frac`, `column_chi2`, `weight`, `mask`, `mask_share`,
`width_kms` (or `high_pass`), `dv`, `nightly_stack`, `min_rjd`, `max_rjd`,
`drs_modes`, `min_pixel_snr`, `windows`, `cache_directory`. An unknown key
stops the run before anything is read, with the list of what is accepted.

For NIRPS, the DRS mode follows the LBL instrument (`NIRPS_HE` keeps the HE
files). The folder is the object: the OBJECT keyword of the files is not
checked against its name.

## Running it again

A correction already on disk, from the same settings and the same input
files, is kept and not made again. A different one moves the previous
`<object>_pca2d` and every LBL product made from it (lblrv, lblrdb,
templates, masks, plots) to `pca2d/<object>/superseded/<date>/`, so that
`SKIP_LBL_*` never measures a correction against a template made from
another one. Nothing is deleted; the folder and its size are printed.

## The pca2d window

`pca2d-gui`, from pca2d-preclean, is a front end to this: it writes an
ordinary wrap file with a `PCA2D` dict (`DATA_DIR/run_<object>_pca2d.py`)
and runs it with this LBL. The file can be edited and run again by hand.
