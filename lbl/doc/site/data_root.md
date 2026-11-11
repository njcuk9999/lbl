# Your data: the root and its folders

The window works on a **data root**, a folder of yours. It holds what you
download, one LBL data folder (`DATA_DIR`) per instrument and data source,
and the models every instrument shares:

```
~/lbl_data/                          the root (--data-dir, or the Folders tab)
    Downloads/                       what you download, as it comes
        unused/                      what LBL does not read, put aside
    NIRPS_HE_APERO/                  a DATA_DIR: NIRPS in HE mode, from APERO
        science/GL699/               one star, one folder: its spectra
        calib/                       the blazes
        models -> ../models          a link to the root's models
        templates/ masks/ lblrv/ lblrdb/ pca2d/ reports/ gui/
    HARPS_ORIG/
    SPIROU_CADC/
    models/                          PHOENIX, TAPAS, vetted templates, kept bases
```

A `DATA_DIR` is named after the LBL instrument and data source that read
its spectra, or the instrument alone when LBL reads it from one source
only. It is an ordinary LBL data folder: a wrap runs there as it always
has, the window or not.

{{folders}}

## Dispatch

Put the files in `Downloads/` as you download them: any names, any
instruments, in folders or not, gzipped or not. The badge of the Folders
tab counts those that can be dispatched. **Dispatch** reads their headers
(never their data), shows a **plan**, and moves nothing before you press
**Move the files**:

- **The reader**: the instrument (`INSTRUME`; NIRPS's HE or HA mode) and the
  data source, from the layout of the file (APERO's `t.fits`, ESO's
  `S2D_A`, the CADC's, the HARPS DRS's `e2ds_A`...).
- **What it is**: a spectrum LBL reads (into `science/<star>/`), a
  calibration (a blaze, into `calib/`, under the name the spectra's headers
  give it), or another product of its pipeline (a CCF, an S1D...), put
  aside in `Downloads/unused/`.
- **A setting** a template must not mix gets a folder of its own:
  HARPS's EGGS, SOPHIE's HE, ESPRESSO's settings other than HR 1x1
  (`<star>_EGGS`).
- **The star**: the files of one star go to one folder whatever their
  observers called it (HR1084, Eps_eri and K3_01 are all eps Eri), matched
  by where the star is in J2000, from the target's coordinates in the
  headers brought back to 2000 by their proper motion. The two stars of a
  binary stay apart: by their names (GJ 15 A and GJ 15 B), else by SIMBAD,
  which knows BL Cet and UV Cet as two stars 2 arcsec apart.

On the way: gzipped spectra are gunzipped, SOPHIE's blaze is split out of
its spectra, a header astropy cannot write back is fixed, and the HARPS and
HARPS-N spectra whose blaze ESO does not give get one made from them (one a
night; a long series takes a few minutes, said as it goes). A blaze made so
carries `HIERARCH LBL BLAZE = 'empirical'` and the instrument of its
spectrum (`INSTRUME`): put back in `Downloads/`, it goes to the `calib/` of
its instrument again. One made before the instrument was written in it
(until 2026-11-11) is placed by its name, as the DRS names its files
(`HARPS.<date>_blaze_A.fits`, `HARPN.<date>_blaze_A.fits`).

Nothing is overwritten: a copy of a file already in its place is taken out
of `Downloads/`; a different file of the same name is left there, and said.
Every move is written in `Downloads/.dispatched.jsonl` (from, to, when), so
it can be undone.

A star already in the root keeps its folder. A new one is named as the
**Naming of new stars** menu says: see [Naming the stars](naming.md).

## What stays in Downloads

What cannot be dispatched is not counted as waiting: the tree lists it under
`Downloads/`, with why (not a spectrum, an empty file from a download that
did not finish, a FITS file astropy cannot read...). Remove it, or download
it again.

## Files from elsewhere

**Add files and dispatch** opens the system's own file picker: the files
picked are copied (or moved, if ticked) into a folder of `Downloads/` of
their own, then planned as Dispatch does.

## The tree

Below, the root as a tree, open: `Downloads/` and what waits there, each
instrument folder with its targets and their number of spectra, its
calibrations and its models. **open** makes an instrument folder the one
every other tab works in. The pencil beside a target renames it (see
[Naming the stars](naming.md#renaming-by-hand)).

## Large roots and external disks

Each file's header is read once: what was read is kept in
`Downloads/.dispatch_headers.json` (by name, size and date) and follows the
file when it moves, so a second plan reads nothing again. Folders are
listed once and listed again only when they change. On an external disk,
the first read of each file waits on the disk: a first plan of thousands of
files takes minutes, the next ones seconds.

## The shared models, and the systems that refuse links

Each instrument folder's `models/` is a **link** to the root's `models/`,
so that LBL downloads its models once for every instrument. A `models/` an
instrument folder already had is merged into the root's first (a file that
differs from the root's keeps the folder as it is, and is said).

- macOS, and Linux on its usual disks: a symbolic link.
- Windows: a symbolic link in developer mode (Settings, For developers) or
  for an administrator; otherwise a junction (NTFS), which needs no right.
- A disk or a system that takes no link (FAT, exFAT outside macOS, some
  network shares): nothing is moved, the folder stays the instrument
  folder's own (LBL downloads its models there), and the dispatch says why.

The templates of another star a run uses are linked into its `templates/`
the same way, and copied where no link can be made.

## The old layout

A root that is itself a data folder (a `science/` of its own, from before
the instrument folders) is still read: the menu of instrument folders
offers it as "science/ of the root (old layout)".

## The fields of the Folders tab

{{help data_dir | Data root}}

{{help dispatch | Dispatch}}

{{help pick | Add files and dispatch}}

{{help data_folder | Instrument folder (DATA_DIR)}}
