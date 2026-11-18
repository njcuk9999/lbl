# Running and following

## Run

- **The name** of the run: its folder, `DATA_DIR/gui/runs/<name>/`.
- **The wrap** that will run, built from the other tabs; edit it here if
  you want (it then stops following the tabs until a setting changes, which
  rebuilds it). **Save** writes it in the data folder as `wrap_<name>.py`;
  **Load** reads a wrap of yours back into the tabs.
- **Run** puts it in the queue: one run at a time, in the environment LBL is
  installed in. A wrap not edited by hand is built again from the settings
  as Run is pressed (and Save), so that what runs is what the tabs say,
  even when the text shown dates from before a setting changed. A wrap
  edited by hand that no longer says what the settings say (another
  `FAKE_KEPLERIAN`, `OBJECT_SCIENCE`...) is not run before asking, with
  the parameters that differ: yes runs it as it is, and the steps planned
  are those of that text; no writes it again from the settings, to look at
  before pressing Run again.
- **The steps**: one pill per step (LBL on the delivered spectra, PCA2D's
  cube, fit, figures and correction, LBL on the corrected spectra, the
  report), each filled as its counter goes. Every step that runs has its
  pill: a template or a mask whose step is unticked in the LBL settings but
  that is not on disk is made all the same before the velocities: its pill
  is dashed, its tip says why; a step the log names that the plan did not
  have gets a dashed pill where it comes. A step done says how long it took
  (`template: 1m32s`). After the last, how long is left and when the run
  should end: the step going from its counter when it has one, each step
  to come from how long the same kind took already in this run (the
  corrected spectra's LBL steps like the delivered ones', a second PCA2D
  pass like the first), else from the median of the earlier runs of the
  data folder (their `run.json` keeps when each step began and ended); a
  step with neither is said, not guessed.
- **The plots**, between the steps and the log, as LBL draws them, without
  stopping the run: the template once made, the velocities it measured on
  its spectra, with their uncertainties, against the spectrum's number and
  its MJD (an injection test's signal drawn over them), and the spectra
  registered at them, coloured by epoch (`SMALL_RV_SHIFT` unticked;
  each spectrum's velocity is written in the log too), then the other
  steps' (the mask's CCF, compute's CCF and lines, PCA2D's fit: the variance
  share of each component, its eigenvalue, and the rank correlation of each
  component's amplitude with the observing conditions, one plot per pass;
  compile's lines: each line's velocity and error bar against its
  wavelength, for the epoch of median `svrad` and over all epochs, colour
  and opacity by error bar (the smallest darkest and opaque), y range the
  3rd to 97th percentiles of the lines; compile's bands, the telluric
  correction); a plot drawn
  for every spectrum is shown for the first spectrum of each object only.
  One fixed frame: the newest as it comes, or the one picked with the
  arrows or the list, which stays until **follow the newest** is ticked
  again. They are PDFs in `gui/runs/<name>/plots/` (`index.json` lists
  them): the window gives the run that folder (`LBL_LIVE_PLOTS`,
  `lbl/science/liveplot.py`), and LBL saves its plots there instead of
  opening windows that wait to be closed.
- **The log**: everything the run prints, timestamped, coloured by level;
  progress bars redraw in place. The arrows go to its top (it stops
  following) and its bottom (it follows again). **Stop** ends the run
  cleanly.

The log is kept in `gui/runs/<name>/run.raw`, with the wrap as it ran and
`run.json` (when, how it ended, the reports it wrote).

## Runs

One row per run made from the window, read back from the disk
(`gui/runs/*/run.json`, so a run made before the window was restarted is
there too): its name, its state (done, failed, stopped, cancelled), its start
and duration, and links to its reports (LBL's, PCA2D's, its components). A
failed run keeps its log: open it to see why.

The runs of every instrument folder of the data root are listed, newest
first, not only those of the folder chosen: a run of `NIRPS_HE_CADC` stays
in the list when the window moves to `HARPS_ORIG`. When they come from more
than one folder, a **folder** column names each run's, the current folder's
in bold; LBL's reports (`reports/*.pdf`) below are listed the same way.
Freezing a run's basis acts on the run's own folder.

## Freezing a run's PCA2D basis

A run that fitted PCA2D left its observer basis (what the corrections leave
in our frame) in `pca2d/components/`, named after its targets; the next fit
of the same targets writes over it. **Freeze basis**, in the run's row,
keeps it for good under a name of yours:

1. Tick **Freeze basis** and type a name (`nirps_2026a`): it is checked as
   it is typed.
2. **Freeze** copies the basis into
   `models/pca2d_<INSTRUMENT>_<SOURCE>/PCA2D_Observer_<name>_<instrument>.fits`
   (the root's `models/`, shared by its instrument folders), its name
   (`BASISNAM`) the one typed, its identity (`BASISID`) unchanged, the run
   it came from written in it (`FROMRUN`).
3. From then on, the PCA2D settings of that instrument and data source
   offer it among the observer bases: a star that cannot separate the two
   frames itself (a narrow BERV range) uses it frozen (`input_name`). No
   cleanup removes it.

A name is refused, with why, when it is:

- no file name: letters, digits, `+`, `-` and `_` only, a letter or a digit
  first and last, 64 characters at most;
- the name of a basis already there, kept or in `pca2d/components/`, of any
  instrument, in any case;
- the name of a target of the folder: its next fit would write a basis of
  that name, and a fit updates a kept basis of its name.

A basis already frozen (the same `BASISID`, under any name) is not frozen
twice: the row says the name it has. The row also says when there is
nothing to freeze: the run read a frozen basis, a later fit of the same name
wrote over its basis, the basis is no longer on disk, or the run is not
over.

## The window's help

{{help run_name | Run name}}

{{help wrap | The wrap}}

{{help load | Load a wrap}}

{{help steps | The steps}}

{{help run_plots | The plots of the run}}
