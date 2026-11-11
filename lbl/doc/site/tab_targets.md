# Targets

The Targets tab says which spectra a run measures. Under the tabs, the
**instrument folder** (`DATA_DIR`) every other tab works in: choosing one
sets the instrument and the data source of the wrap. Then **My targets**
(the folder's stars) or the [Archive search](archives.md).

## The table

One row per folder of `science/`: the instrument, the number of spectra and
of nights, the median SNR, the exposure time, the BERV range and its span,
the first and last dates, the Teff. Their headers are read once and kept;
**Refresh** reads the folder again (new targets, new spectra) and keeps what
was read.

- **Tick** the targets of a run. One instrument per run; several targets of
  one instrument make one **joint** PCA2D fit.
- Click a column to sort; the boxes above the table hide instruments.
- The tiles above say what is ticked: targets, spectra, dates, nights, SNR,
  BERV coverage.

## Four plots to judge a data set

- **BERV coverage**: the spectra per 3 km/s of BERV. PCA2D separates the
  star from our frame only as far as the BERV moves.
- **SNR against BERV**: one dot per spectrum.
- **SNR against v_tot**: where the star's lines fall against the tellurics;
  shaded where they fall on them.
- **When**: one tick per spectrum. The RJD bounds keep a time range for
  PCA2D (the plots grey out what they leave out).

## The window's help

{{help vtot | v_tot}}

{{help bounds | RJD bounds}}
