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

## PCA2D settings

On or off, what to do with a correction already on disk, and the options of
the fit and the correction. Every option, its default and what it does: the
[PCA2D options](pca2d_options.md); the method: [PCA2D](../PCA2D.md).

{{help pca2d_on | PCA2D on or off}}

{{help skip_pca2d | A correction already on disk (SKIP_LBL_PCA2D)}}
