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
before. This is all it takes: no other package, no window. LBL from its
`lbl-pca2d` branch has everything.

```
git clone -b lbl-pca2d https://github.com/njcuk9999/lbl.git
pip install -e ./lbl
```

## What it does to the tree

A commented example, with what the precleaning does and what it gave on two
stars: [example_wrap_pca2d.py](example_wrap_pca2d.py).

In order: LBL measures the delivered object (`compare=True`), PCA2D corrects
it, LBL measures the corrected object, and the report is one PDF, pca2d's
report followed by LBL's.

| path | what |
|---|---|
| `science/<object>/` | read, never written |
| `science/<object>_pca2d/` | the corrected spectra, same names, same format |
| `pca2d/<object>/request.yaml` | the configuration that ran |
| `pca2d/<object>/<M>-<N>/` | the fit, the figures, the report (PDF) |
| `pca2d/cache/` | the registered cubes, rebuildable |
| `reports/<object>_pca2d_<object>_pca2d.pdf` | THE report: pca2d's, then LBL's |

The report, when `REPORT` is on: the star as SIMBAD describes it, the
delivered and corrected velocities on the exposures they share with what got
better and what got worse, the river plots of the correction step by step in
each window, the components and their share of the variance, their
coefficients in time, against BERV, airmass and the rest, and their
periodograms, the weight spectrum, every setting that ran; then LBL's pages
for the corrected object.

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
| `compare` | True | LBL measures the delivered object too, first, for the report |
| `figures` | as `REPORT` | the river plots and the other figures of the correction |
| `overwrite` | False | redo even when the same correction is on disk (as `SKIP_LBL_PCA2D=False`, below) |
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
`width_kms` (or `high_pass`), `dv`, `nightly_stack`, `bundle_minutes`,
`bundle_berv`, `bundle_max`, `min_rjd`, `max_rjd`, `drs_modes`,
`min_pixel_snr`, `windows`, `cache_directory`. An unknown key stops the run
before anything is read, with the list of what is accepted.

## What a row of the fit is

Each row of the fit is one exposure, a whole night, or a bundle of a night:

| setting | rows |
|---|---|
| `nightly_stack=False` | every exposure on its own: the most information, the most memory |
| `nightly_stack=True` (nominal for NIRPS) | every night coadded: NIRPS's three or four exposures of a visit become one row |
| `bundle_berv=0.1` (the default) | with stacking, each night cut into bundles of at most 0.1 km/s of BERV |
| `bundle_minutes=20` | or of at most 20 minutes |
| `bundle_max=6` | or of at most six exposures |

The cube is in the observer's frame, so a coadd smears the star by the BERV
range of what it holds, and a long sequence is not one sky either: over a
five-hour WASP-76 transit night the BERV moves by 600 m/s and the airmass, so
the water column, and the airglow change. The bundles keep that apart and
still coadd what does not change. With the default 100 m/s, an ordinary NIRPS
visit stays one row: over twenty targets, the median BERV range of a night is
4 to 35 m/s. The limits can be combined, and a bundle closes as soon as one of
them is reached; `bundle_berv=None` gives whole nights again. They only cut
stacked nights: the stacking itself is `nightly_stack`'s. Every exposure is
then corrected with its own amplitudes, solved against the fitted basis, and
matched to the row it went into by the list the cube keeps of them
(`members.csv` in the cube). The report shows the rows: every night, its
exposures by BERV and the bundles it was cut into, before the river plots.

In the river plots, the thick green line is where a feature fixed in the
observer's frame falls in each row: the observer's residuals run parallel to
it, and the star's lines stand vertical.

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

`SKIP_LBL_PCA2D`, beside LBL's other `SKIP_LBL_*` in the wrap, overrides
that choice:

| `SKIP_LBL_PCA2D` | the correction on disk |
|---|---|
| `None` (or not given) | kept if the settings, the input files and `nominal.yaml` are those it was made with, made again otherwise |
| `True` | kept even if they changed, which a warning says |
| `False` | made again in every case, as `overwrite=True` |

A correction with no `done.json` beside it was interrupted and is made again
in every case. `SKIP_LBL_PCA2D=True` with `overwrite=True` stops the run:
they ask opposite things.

## Speed

Two stages work on several exposures at once, with results identical to the
bit to one at a time:

| key | default | |
|---|---|---|
| `input.workers` | 4 | the cube: exposures read, resampled and high-passed in processes, coadded here in the files' order |
| `correct.workers` | 4 | the correction: exposures refitted and written in threads |

`workers=1` in the PCA2D dict's `input` or `correct` section gives one at a
time. Memory grows with them: on a 17 GB machine, run one wrap at a time. A
script read from standard input (`python - <<EOF`) builds its cube one
exposure at a time, since spawned processes cannot re-read it.

## The command line

`lbl_pca2d` is pca2d's own command line (`lbl.pca2d.cli`), for research runs
kept side by side under their own names, variants and joint fits, outside an
LBL tree: `lbl_pca2d --object TOI2120 --n-star 1 --n-earth 3`. It reads
`config.yaml` in the working directory, in the format of
`lbl/pca2d/nominal.yaml`, and that nominal file itself when there is none.

## The pca2d window

`pca2d-gui`, from pca2d-preclean (branch `lbl-wrapper`), is only a front end
to this: it writes an ordinary wrap file with a `PCA2D` dict
(`DATA_DIR/run_<object>_pca2d.py`) and runs it with this LBL. The file can be
edited and run again by hand. None of the computation is in it.
