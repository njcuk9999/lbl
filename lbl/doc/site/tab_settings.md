# LBL and PCA2D settings

What the run does. Every field writes a parameter of the wrap; the Run tab
shows the wrap as it stands, and it can be edited there.

## LBL settings

- **Instrument, data source, Python**: set by the instrument folder; the
  Python is that of the environment LBL is installed in.
- **Per target**: the template each target is measured against (its own,
  another star's, or one of LBL's vetted templates) and its temperature
  (read from the headers, else asked of SIMBAD).
- **The steps**: which of LBL's recipes run (telluric cleaning, template,
  mask, velocities, tables), what is kept from a previous run, the report,
  the known planets.

{{help instrument | Instrument}}

{{help source | Data source}}

{{help python | Python}}

{{help comparison | Template (OBJECT_COMPARISON)}}

{{help vetted | Vetted templates}}

{{help teff | Teff}}

{{help resproj | DTEMP3500}}

## LBL's steps

{{help sw_RUN_LBL_TELLUCLEAN | Telluric cleaning (RUN_LBL_TELLUCLEAN)}}

{{help sw_RUN_LBL_TEMPLATE | Template (RUN_LBL_TEMPLATE)}}

{{help sw_RUN_LBL_MASK | Mask (RUN_LBL_MASK)}}

{{help sw_RUN_LBL_COMPUTE | Velocities (RUN_LBL_COMPUTE)}}

{{help sw_RUN_LBL_COMPILE | Tables (RUN_LBL_COMPILE)}}

{{help sw_SKIP_LBL_TEMPLATE | Keep the template (SKIP_LBL_TEMPLATE)}}

{{help sw_SKIP_LBL_MASK | Keep the mask (SKIP_LBL_MASK)}}

{{help sw_SKIP_LBL_COMPUTE | Keep the velocities (SKIP_LBL_COMPUTE)}}

{{help sw_SKIP_LBL_COMPILE | Keep the tables (SKIP_LBL_COMPILE)}}

{{help overwrite_all | Make everything again}}

{{help sw_REPORT | Report (REPORT)}}

{{help sw_PLANETS | Known planets (PLANETS)}}

{{help sw_SMALL_RV_SHIFT | Small velocity changes (SMALL_RV_SHIFT)}}

## Injection test

Whether PCA2D removes a signal: Keplerian signals added to the BERV of
every target's spectra, one row per signal (one to start with, **add a
signal** for more). The wrap gets them as `FAKE_KEPLERIAN`, one list per
key, one entry per signal:

```
rparams['FAKE_KEPLERIAN'] = dict(k=[5.0, 3.0], period=[12.3, 40.0],
                                 phase=[0.0, 0.25], e=[0.0, 0.2],
                                 omega=[90.0, 30.0])
```

`k` in m/s and `period` in days must be given; `phase` (0), `e` (0) and
`omega` (90 deg) have defaults; `t0` (the conjunction) or `tp` (the
periastron), in BJD, can replace the phase in a wrap written by hand. The
velocity of each spectrum is the sum of the signals at its BJD (the time
LBL's tables give, `rjd` + 2 400 000), composed relativistically with its
BERV in every instrument class's `get_berv` (`lbl/science/injection.py`),
which LBL's recipes and PCA2D all read: a larger BERV is a larger measured
velocity. The spectra are not changed: the telluric lines stay where they
are.

Each target is measured as `<target>_inj`: `science/<target>_inj/` holds a
link to each of its spectra (a hard link, else a copy, on a disk without
links) and the signals, `fake_keplerian.yaml`; what is derived from it
(`<target>_inj_tc`, `<target>_inj_pca2d`) takes them from there. The
target's own template, velocities and PCA2D correction are not touched; its
template becomes `<target>_inj`'s when it was its own (another star's, or a
vetted one, stays). Run again with other signals, what LBL made from
`<target>_inj` is moved to `pca2d/<target>_inj/superseded/<date>/`, and
PCA2D makes its correction again (the signals are in its request). In the
[Data Explorer](tab_explore.md), the tables are the target's, flavours
`inj / ...` and `inj_pca2d / ...`. The [report](outputs.md#reports) says
what was recovered.

{{help injection | Injection test (FAKE_KEPLERIAN)}}

## PCA2D settings

On or off, what to do with a correction already on disk, and the options of
the fit and the correction. Every option, its default and what it does: the
[PCA2D options](pca2d_options.md); the method: [PCA2D](../PCA2D.md).

{{help pca2d_on | PCA2D on or off}}

{{help skip_pca2d | A correction already on disk (SKIP_LBL_PCA2D)}}
