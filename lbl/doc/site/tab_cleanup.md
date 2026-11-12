# Cleanup, Documentation, References

## Cleanup

What takes the disk in the root and its instrument folders, by kind: the
caches the next run remakes, the products a run makes again (cleaned and
corrected spectra, per-spectrum velocities, templates), the results, the
downloads waiting. Tick, then delete.

- Only what can be made again is offered for deletion; your spectra and
  your results are counted, never offered.
- Nothing is deleted while a run works in the root (a run started from a
  terminal too); where the system cannot tell, nothing is deleted.
- Corrected spectra are someone's results once shared: tar them first.

The kinds it weighs:

{{cleanup}}

### Full reset of LBL

Under the list, one row per instrument folder with what LBL's own reset
(`lbl_reset`, `RUN_LBL_RESET` in a wrap) would delete there, by part, its
size and its number of files, and a **Reset** button: after a confirmation
that lists every part, the window runs `lbl_reset` on that folder, in a
process of its own, and says how many files went. A folder whose instrument
and source its name does not tell (a lone DATA_DIR is reset with the
instrument and source of the Settings tab) has no button.

{{help reset_lbl | Full reset of LBL}}

## Documentation

These pages: see [About this documentation](about_docs.md).

## References

The method's paper ([Artigau et al.
2022](https://ui.adsabs.harvard.edu/abs/2022AJ....164...84A/abstract)), the
papers that measured their velocities with LBL (each with the sentence
that says so), and the credits of the window's background.
