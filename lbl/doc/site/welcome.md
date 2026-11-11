# Welcome

LBL measures precise radial velocities **line by line**: each spectral line
of the star gives its own velocity against a template, and the lines are
combined with an outlier-resistant mean ([Artigau et al.
2022](https://ui.adsabs.harvard.edu/abs/2022AJ....164...84A/abstract)).
On the `lbl-pca2d` branch, LBL can first clean the spectra with **PCA2D**,
which fits what the telluric and sky corrections leave behind (fixed in our
frame) apart from the star (fixed in its own frame) and divides it out.

`lbl_gui` is a window, in your browser, that does everything LBL does from
one place: it sorts your downloads, finds public spectra, writes the wrap,
runs it, follows it, and shows the velocities. Every run it makes is an
ordinary LBL wrap you can also run by hand.

{{version}}

## Quick start

1. Install LBL from its `lbl-pca2d` branch and start the window: see
   [Installing and starting](install.md).
2. In the **Folders** tab, give the data root (a folder of your own, for
   instance `~/lbl_data`).
3. Put the spectra you download in `~/lbl_data/Downloads/`, as they come,
   or fetch public ones with the **Archive search** of the Targets tab.
4. **Dispatch**: the window reads their headers, shows a plan, then moves
   each spectrum into `ROOT/<INSTRUMENT>_<SOURCE>/science/<star>/`. See
   [Your data](data_root.md) and [Naming the stars](naming.md).
5. **Targets**: choose the instrument folder, tick a target.
6. **LBL settings** and **PCA2D settings**: the defaults are a good start.
7. **Run**: read the wrap, press Run, follow the steps and the log.
8. **Runs** opens the reports; the **Data Explorer** sums up each star
   (its parameters, its published planets) and plots any column of the
   velocity tables against any other.

## The tabs, left to right

| tab | what it is for | page |
|---|---|---|
| Folders | the data root, its Downloads, their dispatch, the folder tree | [Your data](data_root.md) |
| Targets | the instrument folder, its targets, four plots to judge them, the archive search | [Targets](tab_targets.md) |
| LBL settings | instrument, source, Python, each target's template and Teff, LBL's steps | [Settings](tab_settings.md) |
| PCA2D settings | the precleaning: on or off, its model, its bases, its correction | [Settings](tab_settings.md) |
| Run | the wrap, the queue, the steps and the log of the run going | [Running](tab_run.md) |
| Runs | every run of this window, its log and its reports | [Running](tab_run.md) |
| Data Explorer | the star and its published planets, the velocity tables, column against column, their periodograms, the template | [Data Explorer](tab_explore.md) |
| Cleanup | what takes the disk, and what can go | [Cleanup](tab_cleanup.md) |
| Documentation | these pages | [About](about_docs.md) |
| References | the method's paper and the papers that used it | [Cleanup](tab_cleanup.md) |

Every **(i)** of the window opens its help text; each of them is also in
these pages, under the tab it belongs to (search for its name above).
