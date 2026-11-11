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
  report), each filled as its counter goes.
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

## The window's help

{{help run_name | Run name}}

{{help wrap | The wrap}}

{{help load | Load a wrap}}

{{help steps | The steps}}
