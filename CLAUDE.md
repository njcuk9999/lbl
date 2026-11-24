# Working in this repository

## LBL is the arithmetic, nothing else

This branch holds the measurement, the PCA2D precleaning and the reports,
all run by wraps (`lbl.recipes.lbl_wrap`; `lbl/doc/example_wrap_pca2d.py`) and by the
commands `lbl_pca2d`, `lbl_setup`, `lbl_reset`. It holds no window, no
search or fetch of public archives and no dispatch of downloaded files:
those belong to a separate package, which imports LBL and which LBL never
imports. Do not add such code here, nor options, parameters or tests that
exist only for it.

## The documentation is always kept up to date

LBL's documents are the Markdown files of `lbl/doc/` (`PCA2D.md`,
`TEMPLATE.md`, `KnownLimitations.md`, `AddingAnInstrument.md`) and the
README. **Every change of what LBL does updates them in the same commit**:
a new or renamed option, file, folder, step or recipe; a default changed; a
behaviour changed (what a run writes, what a reset deletes...); a new
instrument or data source. Do not consider a change done while the
documentation still describes the old behaviour.

Pages are in English, terse: every fact and number kept, no filler; no em
dashes.

## Tests

```
python -m pytest maintenance/tests
```
