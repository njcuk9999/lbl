# PCA2D: two-frame precleaning inside LBL

`lbl.pca2d` fits every exposure of an object as a star spectrum in the
star's rest frame, plus a few components anchored in the star frame and a
few anchored in the observer frame (residual tellurics, airglow, detector
features). The observer-frame part is divided out of the spectra, which
LBL then measures like any other object.

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
rparams['PCA2D'] = dict(n_star=1, n_earth=7)

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
| `science/<object>_residuals/` | with `save_residuals`: the fit's residual of each exposure, same names, same format, the flux replaced by `ln f - savgol(ln f) - model` (below) |
| `pca2d/<object>/request.yaml` | the configuration that ran |
| `pca2d/<object>/s<M>-o<N>/` | the fit, the figures, the report (PDF): M = n_star, N = n_earth |
| `pca2d/cache/` | the registered cubes, rebuildable |
| `pca2d/components/` | the components of every fit, observer and star (below) |
| `models/pca2d_<INSTRUMENT>_<DATA_SOURCE>/` | the observer bases one keeps (below): no cleanup touches `models/` |
| `reports/<object>_pca2d_<object>_pca2d.pdf` | THE report: pca2d's, then LBL's |

The report, when `REPORT` is on: the star as SIMBAD describes it, the
delivered and corrected velocities on the exposures they share with what got
better and what got worse, the river plots of the correction step by step in
each window, the components and their share of the variance, their
coefficients in time, against BERV, airmass and the rest, and their
periodograms, the weight spectrum, every setting that ran; then LBL's pages
for the corrected object.

The report is typeset with pdflatex, from any TeX distribution (TeX Live,
MiKTeX, MacTeX). LBL looks for it on the PATH, then, outside Windows, on the
PATH of your login shell (so a window started from a desktop launcher finds
the same one as a terminal); `LBL_PDFLATEX` set to the full path of pdflatex
overrides both. Without it, the report is the figures bound into one PDF.

Every step after it (template, mask, compute, compile, report) runs on
`<object>_pca2d`, and the comparison follows when it was the object itself.
A different `OBJECT_COMPARISON` is used as it is, uncorrected.

With `RUN_LBL_TELLUCLEAN` the precleaning runs after the telluric cleaning,
on `<object>_tc`, and writes `<object>_tc_pca2d`: this is the path for the
optical spectrographs.

## The dict

| key | default | |
|---|---|---|
| `n_star` | 1 | the star's spectra, the template included: 1 is the template alone, not adjusted in depth; 2 adds one star-frame component with a free amplitude (never divided out); 0 is refused |
| `n_earth` | the instrument's | observer-frame components fitted and divided out: without it, the class's `PCA2D_N_EARTH` (7 for NIRPS, 2 for HARPS); with `input_name`, all of the basis' |
| `compare` | True | LBL measures the delivered object too, first, for the report |
| `figures` | as `REPORT` | the river plots and the other figures of the correction |
| `overwrite` | False | redo even when the same correction is on disk (as `SKIP_LBL_PCA2D=False`, below) |
| `save_residuals` | False | also write the fit's residual of every exposure, star and observer components removed, as a log, into `science/<object>_residuals/` (below) |
| `name` | None | `<object>_pca2d_<name>`, to keep variants side by side |
| `joint` | False | every science object against one observer basis |
| `output_name` | the object(s) | the name this fit's components are saved under (below) |
| `input_name` | None | the output name of another fit, whose observer components are taken frozen (below) |

Anything else is a key of the pca2d configuration,
`lbl/pca2d/nominal.yaml`, which holds the nominal settings and their
reasons: as `'section.key'`, as a dict per section, or by a short name.
What depends on the instrument (how its files are read, its domain, its
resolving power, its number of observer components, the modes kept, the
windows of the figures) is not in that file: it comes from the instrument's
LBL class, its `PCA2D_*` parameters (`lbl/core/parameters.py` for the
defaults, `lbl/instruments/*.py` for each instrument), and the photometric
bands are LBL's (`lbl/core/astro.py`, u to K). Adding an instrument to
PCA2D is giving its class these parameters.

```python
rparams['PCA2D'] = dict(n_star=1, n_earth=7,
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

## The residuals (`save_residuals`)

With `save_residuals=True`, the correct stage also writes, for every
exposure the fit has a row of, its residual: panel 5 of the sequence
figure, the high-passed log flux less the whole model,

    r = ln f - savgol(ln f) - [S(T + P a) + mu + Q b]

the star block carried into the exposure's frame with its star-frame means,
the observer block and its per-parity mean. Zero where the model explains
the data, -0.01 where the flux is 1 % below it, NaN where the fit gave the
sample no weight. It is computed on the fit's grid from the cube, with the
fit's own amplitudes, then put back on each order's own pixels (linearly:
a spline through noise would invent structure).

Each file is a copy of its source, same name, same format (a `t.fits` for
APERO), whose flux is replaced by `r`; the other extensions (wavelengths,
blaze...) are as they were, and `PCA2RES = T` in the primary header says
what the flux is. They go in `science/<object>_residuals/`
(`<object>_residuals_<name>/` for a named correction, `name`), which the
window lists among the rebuildable products, not as a target. A fit whose
rows are nights or bundles (`nightly_stack`, `bundle_minutes`) has no
residual of one exposure: none is written, the log says so. Building them
needs a temporary array of the cube's size, beside them, removed at the
end.

They are not part of what makes a correction (as the figures): asked of a
correction already on disk without them, the correction is made again to
write them (unless `SKIP_LBL_PCA2D=True`, which keeps it as it is). A new
correction moves the previous residuals aside with the corrected spectra
(`superseded/`).

## What a row of the fit is

Each row of the fit is one exposure, a whole night, or a bundle of a night:

| setting | rows |
|---|---|
| `nightly_stack=False` | every exposure on its own: the most information, the most memory |
| `nightly_stack=True` | every night coadded: the few exposures of a visit become one row |
| `bundle_berv=0.1` (the default) | with stacking, each night cut into bundles of at most 0.1 km/s of BERV |
| `bundle_minutes=20` | or of at most 20 minutes |
| `bundle_max=6` | or of at most six exposures |

The default (`nightly_stack`) is the instrument's, its `PCA2D_NIGHTLY_STACK`.
A night runs from local noon to the next local noon at the observatory, in
local mean solar time from the longitude of the site the instrument's LBL
class names (`EARTH_LOCATION`): the nights of LBL's own compile, so a night on
Mauna Kea is not cut at noon UT. The time of an exposure is read in its file
name when it is there (APERO's and ESO's names of exposures carry it), else
in its headers: DATE-OBS of the flux's extension or of the primary header
(where ESO's S2D has it), else MJD-OBS. The date in the name of an ESO
archive product (`ADP.<date>`, and the window's archive search names
ESO's NIRPS spectra after it) is when the archive processed it, and is
never taken. The cube records that longitude, and its
cache key holds it, so a cube coadded at another boundary is built again.
The cube is in the observer's frame, so a coadd smears the star by the BERV
range of what it holds, and a long sequence is not one sky either: over a
transit night of several hours the BERV moves by hundreds of m/s and the
airmass, so the water column, and the airglow change. The bundles keep that
apart and still coadd what does not change. With the default 100 m/s, an
ordinary visit of a few exposures (a few tens of m/s of BERV) stays one row.
The limits can be combined, and a bundle closes as soon as one of
them is reached; `bundle_berv=None` gives whole nights again. They only cut
stacked nights: the stacking itself is `nightly_stack`'s. Every exposure is
then corrected with its own amplitudes, solved against the fitted basis, and
matched to the row it went into by the list the cube keeps of them
(`members.csv` in the cube). The report shows the rows: every night, its
exposures by BERV and the bundles it was cut into, before the river plots.

In the river plots, the thick green line is where a feature fixed in the
observer's frame falls in each row: the observer's residuals run parallel to
it, and the star's lines stand vertical. The rows are ordered by the
observer's BERV, the one the line follows, so the line ALWAYS runs
monotonically down the page, from the upper left, whatever frame the
spectra are delivered in: on a grid already barycentric (ESO's S2D,
NIRPS's S2D_BLAZE_TELL_CORR_A), the BERV the fit applies is zero for every
row, and the observer's comes from LBL's profile (`get_berv_transmission`).
A line that zigzags means the rows are not in that order. Yellow is NaN
(no data), said by a little yellow square above the colour bar on every
page.

The DRS modes kept are the instrument's (`PCA2D_DRS_MODES`: an LBL
instrument named after a mode keeps that mode's files). HARPS gives two
groups, its two fibres, which have their own resolution and velocity zero
point: HAM (`HARPS`, `HARPS+NIRPS`) and EGGS (`EGGS`, `EGGS+NIRPS`). A run
keeps the group most of its files are in, read in their headers before
the cube is named, and says the counts (`instrument modes: 56 EGGS+NIRPS,
1 HARPS+NIRPS; kept EGGS and EGGS+NIRPS`); the others are rejected.
`drs_modes` in the PCA2D dict (a plain list) overrides it. The folder is
the object: the OBJECT keyword of the files is not checked against its
name.

The run stops before the fit when the exposures that pass the quality cuts
make no more rows (exposures, or nights when stacked) than the fit has
components (`n_earth` plus the star's free ones): each row would be fitted
exactly and the corrections would be noise. The error gives how many
passed and the commonest reasons the others were rejected; the warning
that counts the rejected exposures gives them too. In a joint fit, each
object's own cube needs one row: its rows join the others'.

The correct stage stops too, before writing anything, when the common mask
(the samples at least `mask_share` of the exposures weighted, all of them
by default) keeps under 5% of the samples any exposure weighted: the
corrected files would be all but NaN, and LBL's template and mask made
from them fail. It happens when a few exposures weight almost nothing (a
pixel SNR under `min_pixel_snr` nearly everywhere, as in faint HARPS EGGS
campaigns); the error names the three that weight least. In a joint fit it
is a warning for that object and the run goes on.

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
| `None` (or not given) | kept if the settings, the input files and the nominal settings (the general block of `nominal.yaml` and the instrument's own `PCA2D_*` settings, read without comments, windows or figures) are those it was made with, made again otherwise |
| `True` | kept even if they changed, which a warning says |
| `False` | made again in every case, as `overwrite=True` |

A correction with no `done.json` beside it was interrupted and is made again
in every case. `SKIP_LBL_PCA2D=True` with `overwrite=True` stops the run:
they ask opposite things.

## Components: saved, and read back

Every fit writes its components to `DATA_DIR/pca2d/components/`, named as
LBL names its templates, the instrument last, under the fit's `output_name`
(the object, or the joint fit's objects, when it has none):

| file | what |
|---|---|
| `PCA2D_Observer_<output_name>_<instrument>.fits` | the observer basis: grid, components, mask, their power, the training stars, every row's amplitudes |
| `PCA2D_Star_<object>_<output_name>_<instrument>.fits` | each star of the fit: its spectrum per order parity, its star-frame components and their amplitudes |

Another star takes those observer components frozen with `input_name`: the
output name (this instrument's file), the file name, or a path. Only its own
star block and the amplitudes are fitted; `n_earth` is all of them unless
given. A basis made from spectra of another DATA_SOURCE (another pipeline)
stops the run.

A basis that is to stay is KEPT in
`DATA_DIR/models/pca2d_<INSTRUMENT>_<DATA_SOURCE>/` (`models/pca2d_SPIROU_CADC`),
beside LBL's other models, which nothing deletes (`lbl_reset` empties
`templates/`, a cleanup of the PCA2D products empties `pca2d/`):
`lbl.pca2d.basis.keep(name, DATA_DIR, instrument)` moves it there (the window
has a *keep* button beside the basis, and in its Runs tab a *Freeze basis*
box that copies a run's basis there under a name of your own), or put the
file there yourself:
a basis in any other subfolder of `models/` is found as well. `input_name`
looks in `models/` first (the run's own `pca2d_` folder before the
others), then in `pca2d/components/`; a fit whose `output_name` is that of
a kept basis updates it where it is kept. Without `n_earth`, a frozen basis
is divided out whole. The
stars' files stay in `pca2d/components/`.

```python
# the basis: a joint fit of stars whose BERV sweeps, under its own name
rparams['PCA2D'] = dict(n_star=1, n_earth=7, joint=True, output_name='2026a')
# PCA2D_Observer_2026a_nirps_he.fits, read by a star that cannot separate
#   the two frames itself (a narrow BERV range)
rparams['PCA2D'] = dict(n_star=1, input_name='2026a', name='basis')
```

The cube must be on the basis' grid (same instrument and domain settings),
or the fit stops; rows more than 60 days outside the basis' dates are
counted in a warning. The basis enters the correction's stamp by its
content (`BASISID`), so another basis makes the correction again.

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
LBL tree: `lbl_pca2d --object STAR --instrument NIRPS_HE --data-source CADC
--n-star 1 --n-earth 3`. It reads `config.yaml` in the working directory, in
the format of `lbl/pca2d/nominal.yaml`, and that nominal file itself when
there is none; the instrument and its data source are LBL's names, as in a
wrap (or `lbl.instrument` and `lbl.data_source` in the configuration).

## The LBL window

`lbl_gui` (`lbl.gui`) is only a front end to this: it
writes an ordinary wrap file, with a `PCA2D` dict when the precleaning is on
(`DATA_DIR/gui/runs/<run>/wrap.py`), and runs it with this LBL. The file can
be edited and run again by hand. None of the computation is in it.
