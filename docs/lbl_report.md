# `lbl_report`: the detailed report of an LBL run

A recipe that runs at the very end of an LBL reduction and writes one PDF
describing what is in the data of one object, plus a tar archive holding every
figure as its own file. It reads the products of a run that already happened;
it never recomputes velocities (with one small exception, below) and never
writes into the run's own directories.

Branch: `developer-report` on `github.com/njcuk9999/lbl`, on top of
`developer`. Everything below is in that branch:
`lbl/science/report.py` (~3500 lines, all of the report),
`lbl/recipes/lbl_report.py` (the recipe), plus the `REPORT_*` parameters in
`lbl/core/parameters.py` and three lines in `lbl/recipes/lbl_wrap.py`.

## What it is for, and what it is not

It is a starting point for an analysis: periodograms, false inclusion
probabilities, known planets, drifts, river plots and per-line diagnostics,
all computed without a human in the loop. The first page of every report says
so in a red box, and asks that the LBL paper (Artigau et al. 2022), the DTemp
paper (Artigau et al. 2024) and, for APERO spectra, the APERO paper (Cook et
al. 2022) be cited if anything from it is published.

It is not vetted science. Nothing in it is inspected, clipped or argued with.

## Running it

Through the wrapper, where it is on by default for `SCIENCE` data:

```python
rparams['RUN_LBL_REPORT'] = True      # wrap_default.txt / lbl_wrap.py
```

On its own, on a run that is already done:

```python
import sys
sys.argv = ['lbl_report']             # the recipe parses argv
from lbl.recipes import lbl_report
lbl_report.main(instrument='SPIROU', data_source='CADC', data_type='SCIENCE',
                data_dir='/path/to/lbl',
                object_science='TOI2120_PCA2D_0-7',
                object_comparison='TOI2120_PCA2D_0-7')
```

`config_file=` works as in any other recipe. The arguments the recipe accepts
are in `ARGS_REPORT` (`lbl/recipes/lbl_report.py`).

Cost: a few minutes. On 316 SPIRou spectra it takes about 3 to 4 minutes, most
of it reading science files for the river plots and lblrv tables for the
per-line uncertainties.

## What it reads

Everything through the accessors of the instrument class, never by opening
files by hand:

| what | from | if missing |
| --- | --- | --- |
| velocities and indicators | `lblrdb/lbl_<obj>_<tpl>.rdb` | the report stops: this one is required |
| per-line velocities, BERV | `lblrv/<obj>_<tpl>/*.fits` | the per-line section and the BERV figures are skipped |
| the spectra | `science/<obj>/` | the river plots and the line plot say so and are skipped |
| template, mask | `templates/`, `masks/` | the template panel and the line plot are skipped |
| TAPAS transmission | `models/tapas_lbl.fits` via `tellu_clean.get_tapas_lbl` | that panel says "TAPAS not available" |

A file that is not on disk any more (a broken symlink, an archive that moved)
is skipped with a warning and counted; it does not stop the report. The one
thing that is recomputed is the velocity of a single spectrum, the one at the
median signal to noise, so that the debug plot of the lines can be drawn with
the same mask, reference table, template splines and blaze as the run.

## What it writes

Under `<DATA_DIR>/lblreport/<OBJECT_SCIENCE>_<OBJECT_COMPARISON>/`:

```
lbl_report_<obj>_<tpl>.pdf        the report
lbl_report_<obj>_<tpl>.tex        its source, kept
lbl_figures_<obj>_<tpl>.tar.gz    every figure, one PDF each, for a paper
figures/                          the same figures, loose
```

and, one level up, in `<DATA_DIR>/lblreport/`: `exoplanet_eu_catalog.csv` and
`cache/` (see *Network*). The `figures/` directory is emptied at the start of
each run, so it always holds that report and nothing else.

Without a LaTeX installation the run stops after the `.tex` and the figures,
and says so. `pdflatex` needs `geometry, graphicx, longtable, booktabs,
pdflscape, hyperref, float, xcolor`.

## What is in the PDF

Ten sections, in this order:

1. **What was processed** — the LBL parameters of the run (`OBJECT_SCIENCE`,
   `OBJECT_COMPARISON`, `INSTRUMENT`, `DATA_SOURCE`, `DATA_TYPE`), what SIMBAD
   knows about the star, the number of spectra and nights, the first and last
   observation in rjd and in calendar dates, the template, the mask and its
   systemic velocity, the rdb file and the LBL version.
2. **Known and suspected planets** — matched on the position of the star from
   SIMBAD (30 arcsec) and on its names, with the periods, the masses, the
   amplitude K of the NASA archive and links to the discovery and parameter
   papers on ADS.
3. **The velocities folded on the planets** — the velocities with their long
   term drift removed, folded on the period of each planet. The periods are
   held at the catalogue value and one sinusoid per planet is fitted at once
   (weighted linear least squares), so the fold of a planet has the sinusoids
   of the other planets taken out of its points. Phase 0 is the transit, the
   conjunction or the periastron when the catalogue has one. The fitted K, its
   error and the catalogue K are tabulated.
4. **Radial velocity and the other indicators** — every column of the rdb file
   that has an error column gets a time series (rjd at the bottom, calendar
   dates on top), a straight-line drift with its 1 sigma envelope, a
   Lomb-Scargle periodogram and a FIP periodogram. One landscape table holds
   the statistics of all of them. Velocities of a slice of the domain
   (`vrad_1673nm`) or of half a detector (`vrad_h_0-2044`) stay in the table
   and get no figure of their own.
5. **Radial velocity against the BERV** — the correlation, the slope with its
   1 sigma envelope; a slope here means something fixed in the frame of the
   observatory.
6. **Sampling** — the spectral window and the periodogram of the BERV taken at
   the times of the observations.
7. **River plots** — three wavelengths per photometric band (a quarter, a half
   and three quarters of the way through the part of the band the instrument
   covers), each 1500 km/s wide, in the rest frame of the star, sorted in BERV
   so what belongs to the Earth walks across the plot. Three panels: the
   spectra, the same with the median spectrum and each row's own median taken
   out, and the median spectrum in wavelength.
8. **What each line is worth** — the median uncertainty of every line of the
   mask over up to 200 lblrv files, on a page of its own against the template
   and against the TAPAS transmission (water in blue, the other molecules in
   red), then the velocity precision of each sub-domain in bins 5 percent wide
   in wavelength, the lines of a bin combined as photon noise.
9. **The lines of one spectrum** — the debug plot of `lbl_compute`
   (`PLOT_COMPUTE_LINES`) for the spectrum at the median signal to noise, one
   order per photometric band, over the 20 lines closest to the middle of that
   order, each line in its own colour with its edges dashed. The lower panel is
   `spectrum / template - 1`.
10. **How the numbers were obtained**, then a numbered reference section.

Every periodogram carries dashed markers at 1 day, a year, half a year and a
third of a year, and the periods of the known planets.

## Parameters

All in `lbl/core/parameters.py`, all overridable on the call or in the config:

| parameter | default | what it does |
| --- | --- | --- |
| `REPORT_SUBDIR` | `lblreport` | where the reports go under `DATA_DIR` |
| `REPORT_PERIOD_MIN` | 1.0 | shortest period of the periodograms [days] |
| `REPORT_PERIOD_MAX` | None | longest; None means the time span of the run |
| `REPORT_FIP_PRIOR` | 0.5 | prior probability of a signal, for the FIP |
| `REPORT_MAX_RIVER_FILES` | 300 | spectra read for the river plots, spread evenly in time |
| `REPORT_RIVER_WIDTH` | 1500.0 | width of a river plot [km/s] |
| `REPORT_MAX_LINE_FILES` | 200 | lblrv files read for the per-line uncertainties |
| `REPORT_EXOPLANET_EU` | True | look for the planets of the star |
| `REPORT_EXOPLANET_EU_URL` | `https://exoplanet.eu/catalog/csv/` | the catalogue |
| `REPORT_CACHE_DAYS` | 30.0 | how long a catalogue answer is kept [days] |

## Network, and the cache

Three services are used, all of them optional: SIMBAD (TAP, for the position,
the spectral type and every identifier of the star), the catalogue of
exoplanet.eu (the whole CSV, a few megabytes), and the NASA exoplanet archive
(TAP, for K and the references, and as a stand-in for exoplanet.eu when that
one does not answer).

Answers are kept in `<DATA_DIR>/lblreport/cache/`, one file per query named by
an md5 of it, and reused while they are younger than `REPORT_CACHE_DAYS`. The
exoplanet.eu CSV is kept in `<DATA_DIR>/lblreport/` and downloaded again once
it is older than that; if the download fails and an old copy is there, the old
copy is read with a warning. A second report in the same data directory
therefore touches the network only for queries it has not made before, and a
run with no network at all still produces everything but the star's identity
and its planets.

## Things worth knowing before extending it

- **Object names built by a pipeline.** SIMBAD is asked for the object name,
  then for that name with its last underscore-separated piece dropped, again
  and again (`TOI2120_PCA2D_0-7` → `TOI2120_PCA2D` → `TOI2120`). Nothing about
  any particular pipeline is hard-coded, and a leftover shorter than three
  characters is not tried.
- **BERV units.** `inst.get_berv()` returns m/s; the header key `KW_BERV` is in
  km/s. The report keeps everything in m/s internally. For the ESO modes
  (HARPS, HARPS-N, ESPRESSO) `get_berv` is zero because the wave solution is
  already barycentric: the rest frame of the river plots uses LBL's BERV
  (`rdata['berv_lbl']`) and the BERV figures use the header one
  (`rdata['berv']`). Mixing them shifts the river plots by 15 km/s and is the
  first thing to check if lines drift.
- **Doppler shifts are relativistic**, here as everywhere in LBL:
  `ratio_to_velocity` and `velocity_to_wavelength` in `report.py`.
- **Photometric bands** come from `astro.bands`. A mistyped edge (the J band of
  the versions of `astro.py` before the fix: `maximum=13494.41`) is put back
  where the band's mean wavelength says it is, by `band_edges`, so a report run
  against an older LBL still draws J in the right place.
- **TAPAS** in LBL carries two components only, `ABSO_WATER` and `ABSO_OTHERS`,
  from 355 to 1850 nm, so the K band has no telluric curve and the figure says
  so. `telluric_absorption` colours any `ABSO_<species>` column it finds, so a
  richer file would be drawn species by species with no code change.
- **Dense curves are thinned before plotting** (`bin_for_display`), the
  telluric one by the lowest point of each bin so a narrow line still shows,
  and scatter plots of tens of thousands of points are rasterised. Keeping the
  PDF at a few megabytes is deliberate.
- The figures of a previous report are deleted at the start of a run. If you
  add a figure, add it to the `figures` list as well or it will not travel in
  the tar archive.

## Code map (`lbl/science/report.py`)

- reading the run: `get_report_data`, `find_indicators`, `time_series`,
  `line_uncertainties`, `river_plot_data`, `river_plots`, `median_snr_file`
- statistics: `remove_drift`, `basic_stats`, `lomb_scargle`,
  `fip_periodogram`, `window_function`, `correlation`, `fit_sinusoids`,
  `sinusoid_amplitude`, `phase_bins`, `precision_bins`
- catalogues: `simbad_candidates`, `simbad_target`, `simbad_identifiers`,
  `exoplanet_eu_planets`, `nasa_archive_planets`, `planets_from_nasa`,
  `same_planet`, `name_variants`, `cache_read`, `cache_write`, `cache_key`
- figures: `plot_indicator`, `plot_phase_folds`, `plot_rv_vs_berv`,
  `plot_window`, `plot_river`, `plot_line_precision`, `plot_precision_bins`,
  `line_edge_plot`, `shade_bands`, `date_axis`, `save_figure`
- LaTeX: `latex_header`, `disclaimer`, `latex_table`, `latex_figure`,
  `cite`, `bibliography`, `compile_pdf`, `make_tarball`, `make_report`

`make_report` is the one entry point: it builds the body section by section
and is where a new section is added.

## Dependencies

Nothing beyond what LBL already requires (numpy, scipy, astropy, matplotlib
and the standard library); `pyproject.toml` says so next to the dependency
list. The only external requirement is a LaTeX installation for the PDF.
