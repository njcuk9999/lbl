# When something looks wrong

## The window and the data

| what you see | what to check |
|---|---|
| the window does not open | another program holds port 8765: the window takes the next free one and prints its address; or start it with `--port` |
| files in `Downloads/` are not counted | they cannot be dispatched: the tree lists them under `Downloads/` with why (not a spectrum, an empty download, a file astropy cannot read) |
| the first plan of a big folder is slow | each header is read once, the first time (on an external disk, the disk is slow to answer); the next plans read only new files |
| a star is split in two folders | its files carry no place, or places far apart: rename one folder like the other, or move its files into it |
| two stars of a binary in one folder | their names do not say the component and SIMBAD could not be asked: give their files names with A and B, plan again |
| "SIMBAD did not answer" | the stars are told apart by their places and names only; plan again later |
| "APERO's database could not be fetched" | the new stars are named as SIMBAD names them; plan again later, or choose another naming |
| a `models/` is not a link | that disk or system makes no links: it stays the instrument folder's own (see [Your data](data_root.md#the-shared-models-and-the-systems-that-refuse-links)) |
| a target name does not resolve | SIMBAD is asked variants of it (`HD10700_EXPRES` gives HD 10700); else type the Teff by hand |

## The run

| what you see | what to check |
|---|---|
| the log is silent for long | a long step without a progress bar, or memory: look at the warnings atop the log and at the free memory |
| "two instruments ticked" | keep one: one instrument per run |
| a template is missing | LBL makes it, and says so, before measuring against it |
| the report is unchanged after adding spectra | a product is made again when its inputs change: run the target again |
| velocities follow the BERV | tellurics: run PCA2D; look at the BERV coverage and the SNR against v_tot |
| a planet's amplitude shrinks after PCA2D | the BERV span is too small: a frozen basis, an injection test |
| poor velocities with the star's own template | few spectra or little BERV coverage: another star's template, or a vetted one |

## Starting again

- The window's state: delete `~/.lbl_pca2d_gui.json`.
- A product: the [Cleanup](tab_cleanup.md) tab deletes what a run makes
  again; untick the matching "keep" step (LBL settings) to make it again
  without deleting it.
