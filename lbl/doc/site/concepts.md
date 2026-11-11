# Key ideas

## Line by line

A velocity of a few m/s moves a line by a tiny fraction of a pixel (1 m/s
is 5.3e-6 nm at 1600 nm). LBL measures that shift for **each line** of the
star separately, as the projection of the difference between a spectrum and
the template on the template's derivative, and then combines the thousands
of line velocities with an outlier-resistant mean: a line hit by a
telluric, a cosmic ray or a bad pixel is down-weighted instead of pulling
the whole velocity.

## The template and the lines

The **template** is the star's spectrum at high signal-to-noise, made from
its own spectra (or another star's, or one of LBL's vetted templates). LBL
builds it from every pixel of every spectrum shifted into the star's frame,
without resampling (see [The template](../TEMPLATE.md)).

The **mask** is the list of lines measured on that template: a line runs
from one maximum of the template to the next, and only maxima and minima
that are significant (a derivative above 3 sigma on both sides) are kept.

## Two frames

Each spectrum carries two kinds of features. The star's lines are fixed in
the **star's frame**; the Earth's atmosphere (telluric absorption, sky
emission) and the detector are fixed in **our frame**. Between the two,
the barycentric velocity (**BERV**) changes through the year by up to
about 30 km/s, depending on the star's ecliptic latitude.

The pipelines correct the tellurics; what their corrections leave is still
fixed in our frame, and enters the velocities. **PCA2D** fits all the
spectra of a star at once, the star in its frame and a few components in
ours, and divides the latter out; LBL then measures the corrected spectra.
It needs the BERV to move (the two frames separate only as far as it
moves), or a basis learned on other stars. See [PCA2D](../PCA2D.md).

## Words used in these pages

| word | meaning |
|---|---|
| data root | the folder you give the window: its `Downloads/`, one data folder per instrument and source, the shared `models/` |
| DATA_DIR | LBL's data folder: one instrument and one source (`NIRPS_HE_APERO/`), with `science/<star>/` and everything LBL makes |
| target, object | a star, one folder of `science/` (`OBJECT_SCIENCE` in a wrap) |
| wrap | the Python script that runs LBL: its parameters (`rparams`) and `lbl_wrap.main(rparams)` |
| template | the star's high-SNR spectrum the exposures are measured against (`OBJECT_COMPARISON`) |
| vetted template | one of LBL's templates of slowly rotating stars, by temperature, for faint data or few epochs |
| mask | the lines measured, made from the template |
| BERV | the barycentric velocity of the observer towards the star |
| v_tot | where the star's lines fall against the tellurics: its systemic velocity less the BERV |
| rdb table | LBL's velocities, one row per spectrum (`lbl_`) or per night (`lbl2_`), plain text |
| RJD | BJD - 2 400 000, the time of LBL's tables |
| DTEMP | an exposure's change of effective temperature, from the lines' response to temperature |
