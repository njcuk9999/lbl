# Running and following

## Run

- **The name** of the run: its folder, `DATA_DIR/gui/runs/<name>/`.
- **The wrap** that will run, built from the other tabs; edit it here if
  you want (it then stops following the tabs until rebuilt). **Save** writes
  it in the data folder as `wrap_<name>.py`; **Load** reads a wrap of yours
  back into the tabs.
- **Run** puts it in the queue: one run at a time, in the environment LBL is
  installed in.
- **The steps**: one pill per step (LBL on the delivered spectra, PCA2D's
  cube, fit, figures and correction, LBL on the corrected spectra, the
  report), each filled as its counter goes. Every step that runs has its
  pill: a template or a mask whose step is unticked in the LBL settings but
  that is not on disk is made all the same before the velocities, and its
  pill (dashed) says "missing"; a step the log names that the plan did not
  have gets a dashed pill where it comes.
- **The log**: everything the run prints, timestamped, coloured by level;
  progress bars redraw in place. The arrows go to its top (it stops
  following) and its bottom (it follows again). **Stop** ends the run
  cleanly.

The log is kept in `gui/runs/<name>/run.raw`, with the wrap as it ran and
`run.json` (when, how it ended, the reports it wrote).

## Runs

One row per run of this window: its name, its state (done, failed, stopped,
cancelled), its start and duration, and links to its reports (LBL's, PCA2D's,
its components). A failed run keeps its log: open it to see why.

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
