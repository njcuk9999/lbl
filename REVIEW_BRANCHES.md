# Reviewing lbl-pca2d: one branch per topic

`lbl-pca2d` is 111 commits ahead of `developer` (184 files, +131,782 and
-807 lines, 73k of them two YAML planet catalogues), written between
2026-09-18 and 2026-10-26. That is too much to review as one diff, and
many commits mix several topics. The plan: split it into `developer-{topic}`
branches, each one a pull request into `developer` that can be read,
tested and accepted (or refused) on its own.

This file is the map of that work. It belongs to no topic branch, and goes
away once every topic is merged or dropped.

## How each branch is built

1. **From today's `developer`**, not from the commit `lbl-pca2d` started
   from (`dbcd813`). `developer` has moved on by 10 commits since, and
   every branch must be a clean diff against what is there now.
2. **In a git worktree** of its own, so that the clone being worked in
   stays on `lbl-pca2d`.
3. **Without tracking `developer`.** Git makes a branch created from
   `origin/developer` track it, and a plain `git push` would then go to
   `developer`. Right after creating it: `git branch --unset-upstream`.
4. **Commits that hold one topic are cherry-picked** with `-x`, which
   writes the original hash into the message. Commits that mix topics are
   split by file and by hunk, and the new commit message names the
   original.
5. **Conflicts with `developer` keep `developer`'s change.** Where
   `developer` changed code that the topic rewrites, its change is carried
   into the rewrite as a separate commit that says so, so that it can be
   checked on its own.
6. **Check that nothing was lost or added:** `git diff <the topic's last
   commit on lbl-pca2d> <branch>` must show exactly `developer`'s own
   changes since `dbcd813`, apart from the pieces carried into a rewrite
   (step 5), which the pull request lists.
7. **Tests:** the topic's tests from `tests/`, and, for code that claims
   to change no result, the same data run through `developer` and through
   the branch, with every product compared bit for bit.
8. **Push, then a pull request** into `developer` that says what changes,
   why, what changes for every user (defaults, outputs, files moved), how
   it was verified, and in which order to read the files.

## The final check

Once every topic branch is merged (first in a throwaway branch),
`git diff lbl-pca2d <that merge>` must be empty, this file aside: then
every line of `lbl-pca2d` went through exactly one review.

## Tools to read a branch against `developer`

- **The pull request on GitHub**, "Files changed" tab: split view (gear
  menu), `developer` on the left, the branch on the right; a "Viewed"
  checkbox per file keeps track of what has been read, and comes back
  unticked if the file changes; comments sit on the line.
- **VS Code with GitLens**, with the LBL clone open in the window:
  `Cmd+Shift+P`, "GitLens: Compare References...", `origin/developer`
  then the branch; the "Search & Compare" view lists every file, a click
  opens the side-by-side diff. The "GitHub Pull Requests" extension shows
  the pull requests themselves in VS Code.
- **The command line:** `git difftool -d origin/developer developer-speed`
  (FileMerge, `opendiff`, comes with macOS).

The left/right view is for the ~35 existing LBL files that change. The new
code (`lbl/pca2d`, `lbl/gui`, `lbl/report`) has nothing on the left, and is
better read as files, in the order its pull request gives.

## Already on `developer`

Two changes of `lbl-pca2d` reached `developer` another way and get no
branch; what differs between the two versions is looked at when
`lbl-pca2d` is brought up to date with `developer`.

| `lbl-pca2d` | What | On `developer` as |
|---|---|---|
| `aa278c7` | mask lines at the template's significant extrema (`MASK_SIGNIFICANT_EDGES`) | merge `1644971`, then `dca3023` |
| `95bf4f4` | NIRPS_HE/CADC loaded the NIRPS_HA class | `d6a5454` |

## The topics

In an order that respects what each needs.

| Branch | What | Commits on `lbl-pca2d` | Needs |
|---|---|---|---|
| `developer-speed` | Numba kernels (`lbl/core/fastmath.py`), faster FITS I/O, template splines, medians; options `COMPUTE_NPROC` (exact parallel compute) and `NOISE_SAMPLING_NSTEP` (default 4, the original sampling). Claims identical results. | `0e5f9f7` to `59344b8` (18) | |
| `developer-console` | Readable console: one block per file, the last iteration only (`logger.py`) | `eeed2f4` | |
| `developer-compile-width` | `COMPIL_MAX_PIXEL_WIDTH` from 50 to 500 px, every instrument | `6015c37` | |
| `developer-compute-fixes` | Rough CCF: zeros and gaps in the model threw the velocity 300 km/s off (GJ 1214); the model velocity reused only with `SKIP_DONE`. Template checks: an order off the template's grid gives no verdict, under 50% valid is a warning naming the orders. | `4028367`, `9653c37`, `b26888d` | |
| `developer-telluclean` | TAPAS evaluated in the observer's frame; the water exponent from the derivative, fitted on the telluric windows, with a fallback that says so (`tellu_derivative.py`); the two passes fixed (the second divides by the template, which is made again after it) | `27d5e99`, `18928e4`, `74dfea7`, `8ce0754`, `1297521`, `02d9538`, `4f99dfc`, `0cdd107`, `96adeb4`, `6ec9bf4`, `852572d`, `601f778`, and the template step of `4075d47` | |
| `developer-espadons` | ESPaDOnS, the `i.fits` the CADC distributes; its zero point from the tellurics | `3dd154c`, `e8aa5c7` (its report figure goes with the report) | |
| `developer-models` | `models/` in subfolders (`phoenix`, `tellurics`, `resproj`, `masks`), an old `models/` sorted at the next run (`io.model_path`, `io.sort_models`) | the `models/` part of `909da88` | |
| `developer-activity` | Differential equivalent widths of activity lines, measured in compute, written in the rdb | `187c34b`, `e3c7b87`, `f2cbb88`, the activity part of `e63d305` | |
| `developer-planets` | `PLANETS`: the template corrected for the BERV and the known Keplerians; the catalogues and the script that builds them | `4b3a809`, the catalogue part of `85a1740` | |
| `developer-lbl-report` | `lbl/report`: one PDF per run; the velocity series coloured by BERV; periodogram with an alias-aware, outlier-aware FIP; joint fit of transiting planets and TESS candidates; known orbits fitted with an acceleration | `50e0748`, `06c4ae7`, `a9133b8`, `c764f50`, `b21a5c5`, `f090d22`, `24e2262`, `d9afe09`, `05007c7`, `38ae5db`, `573c2cd`, `48ed942`, `a5a36ed`, `3106b34`, the report parts of `85a1740`, `e63d305`, `e8aa5c7` | planets, activity |
| `developer-vetted` | `VETTED_<star>` comparison templates for NIRPS and SPIRou, kept in `models/template_<INSTRUMENT>_<SOURCE>`; how they were chosen (`lbl/doc/vetted_templates`); the science template's path in compute | `b0c06f6`, `f8a047f`, the vetted part of `909da88` | models |
| `developer-pca2d` | `lbl.pca2d`, the `lbl_pca2d` recipe and command, its place in `lbl_wrap`, `nominal.yaml`, `lbl/doc/PCA2D.md`; bases saved, read back and kept; another star's template (`template_import.py`); figure windows in the middle of the blaze (`lbl/doc/windows`) | `265e3c2` to `d6d096d`, the PCA2D commits (44), and the PCA2D parts of `4b3a809`, `909da88` | planets, report, telluclean, vetted, models |
| `developer-gui` | `lbl/gui` and the `lbl_gui` command; the archive search taken from koloa (`lbl/gui/fetch`); the guide's generator | `19b2c3c`, `a48ba25`, `930fef0`, `c95886f`, the window's part of `909da88` | pca2d, vetted, models |

`developer-pca2d` (+29k lines) can be stacked as three branches: the
algorithm (cube, fit, correction, recipe, `lbl_wrap`), then its report and
figures, then the saved bases and another star's template.

Two older branches cover part of the same ground and are compared with
these rather than ignored: `developer_speed1` to `developer_speed8` (an
earlier, step by step version of the speed work) and `developer-report`
(an earlier `lbl_report`, 2026-09-25), whose name is why ours is
`developer-lbl-report`.

## To decide during the review

**What changes for every user of LBL**, not only for PCA2D:

- the compile width, 50 to 500 px;
- the telluric cleaning, all of it;
- the rough CCF's gap filter (`ROUGH_CCF_GAP_STEPS`, default 20);
- the console's output;
- `models/` sorted into subfolders in every `DATA_DIR` at the next run
  (an LBL without this change then downloads TAPAS again);
- a new dependency, `pypdf`.

**What goes into the package:**

- 73k lines of planet catalogues (`known_planets.yaml`,
  `candidates.yaml`);
- `personal_planet_catalog.yaml`: is it meant to be public?
- the window's background images and `references.json`;
- the code taken from koloa (`lbl/gui/fetch`): attribution and licence.

## Where each branch stands

### `developer-speed`

Built 2026-10-26 on `developer` at `826f39d`. The 18 commits cherry-picked
with `-x`; two conflicts:

- `lbl/science/general.py`: `developer`'s fix of
  [issue 71](https://github.com/njcuk9999/lbl/issues/71) in `compute_rv`.
  With more than one RESPROJ table, the residual was divided in place
  (`frac_diff_seg = diff_seg`, then `/=`), so every table after the first,
  and `RMSRATIO` and `CHI2`, were computed on an already divided residual;
  `developer` now divides a copy. The speed work replaces that per-line
  loop by a numba kernel, `fastmath.line_loop`, written to reproduce the
  old arithmetic, bug included. The fix is carried into the kernel in a
  commit of its own (a buffer for the divided copy).
- `lbl/recipes/lbl_compute.py`: the imports only (`developer`'s
  `traceback` beside the speed work's).

Check of step 6: the branch against `59344b8` (the last speed commit)
differs by `developer`'s changes exactly, but for the per-line loop above.

`lbl-pca2d` has the same kernel, so it needs the same fix when it is
brought up to date with `developer`.

Verification, 2026-10-26: the 39 SPIRou spectra of KIC 7847566 (CADC),
with two RESPROJ tables (DTEMP3000, DTEMP3500), run through template,
mask, compute and compile by `developer` and by the branch, each in a
`DATA_DIR` of its own:

- template, the three masks, the reference table, the 38 lblrv files and
  the rdb tables (198 columns): bit for bit the same, with
  `COMPUTE_NPROC` 1 and 2. What differs is the paths, the dates of the
  run, and the two new parameters in the parameter table;
- the whole run: 135 s with `developer`, 33 s with the branch (30 s with
  `COMPUTE_NPROC=2`);
- the same run without the issue 71 commit differs where that bug acts,
  and nowhere else: the second table (DTEMP3500 and its band columns),
  `RMSRATIO`, `CHI2` and `CHI2_VALID_CDF`. The check would have caught it.

The scripts (`run_one.py`, `compare_runs.py`, `compare_tables.py`) are
with the runs in `~/ClaudeScratch/lbl-review/261026_developer-speed`,
which expires 14 days after its last use.

Pushed as `developer-speed`; no pull request yet.
