# Installing and starting

## Installing LBL

PCA2D and the window are on the `lbl-pca2d` branch of LBL.

1. Get the code:

   ```
   git clone https://github.com/njcuk9999/lbl.git
   cd lbl
   git checkout lbl-pca2d
   ```

2. A Python 3.12 environment (conda here, or venv): LBL asks for
   `>=3.12,<3.13`.

   ```
   conda create --name lbl-env python=3.12
   conda activate lbl-env
   ```

3. Install LBL in place, with its pinned dependencies (astropy, numpy,
   numba, scipy...):

   ```
   pip install -U -e .
   ```

4. Check:

   ```
   lbl_gui --help
   ```

`-e` installs in place: `git pull` updates LBL; run `pip install -U -e .`
again when its dependencies change. The commands installed are `lbl_gui`
(this window), `lbl_pca2d`, `lbl_setup` and `lbl_reset`. Full
instructions, and the branch for the Digital Research Alliance clusters, on
[lbl.exoplanets.ca](https://lbl.exoplanets.ca).

## Starting the window

```
lbl_gui --data-dir ~/lbl_data
```

The window is a page served by your own computer, on
`http://127.0.0.1:8765` (or the next free port), opened in your browser;
nothing leaves the computer but the requests to the archives, SIMBAD and
LBL's model servers.

| option | what it does |
|---|---|
| `--data-dir` | the data root to open (else the last one, or the one typed in the Folders tab) |
| `--port` | the port to serve on (8765 by default; the next free one when it is taken) |
| `--no-browser` | do not open the page (open the address it prints yourself) |

What you set is kept between sessions, in `~/.lbl_pca2d_gui.json`: the
root, the tab, the targets ticked, the settings. Delete that file to start
from the defaults.

## What else LBL uses

- **Internet**, the first time: SIMBAD (each star's temperature and
  systemic velocity), the PHOENIX model spectra and the TAPAS transmission
  (downloaded once into the root's `models/`; a mirror when the main site is
  down), and APERO's database of star names (a few MB, kept in
  `~/.cache/lbl/gui/`; see [Naming the stars](naming.md)).
- **pdflatex** (a TeX distribution), optional: it typesets the PCA2D report;
  without it, the report is a PDF of its figures.
- **Disk**: the spectra, the caches of PCA2D, the corrected copies; the
  [Cleanup](tab_cleanup.md) tab shows what can go.
- **Memory**: a few GB for a PCA2D cube of a few hundred nights.

## Without the window

Everything the window runs is an LBL wrap, kept with the run in
`DATA_DIR/gui/runs/<name>/wrap.py`: run it again by hand with the Python of
the environment (`python wrap.py`), on a cluster, or edit it. A wrap of your
own can be read back into the window (Run tab, Load a wrap).
