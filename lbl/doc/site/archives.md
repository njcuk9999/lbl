# Public spectra: the archive search

The **Archive search** of the Targets tab finds the public spectra LBL
reads of one star or several, and fetches those you tick into the root.

## How

1. **Targets** tab, then **Archive search**.
2. The star, any name SIMBAD knows (`GJ 581`); several, separated by
   commas.
3. Tick the instruments to search.
4. **Search the archives**: each archive is asked in turn (about a minute);
   one row per LBL reader, with its number of spectra, those already here,
   the size to fetch, and the folder they go to.
5. Tick the rows, **Fetch the ticked**: the spectra go straight into their
   instrument folder, `ROOT/<INSTRUMENT>_<SOURCE>/science/<star>/` (into
   the folder the star already has there, found by its place, if any), and
   the star appears in **My targets**.

The log under the table says each archive's answer.

## An example: GJ 581 with HARPS

GJ 581, an M3 dwarf observed by HARPS from 2004 to 2024, has a planet
visible in a first run: GJ 581 b, P = 5.37 d, K = 13 m/s ([Bonfils et al.
2005](https://ui.adsabs.harvard.edu/abs/2005A%26A...443L..15B/abstract)).
The search found 321 public HARPS spectra at ESO (482 MB, on 2026-11-09):
ESO's HARPS pipeline `e2ds` (about 1.5 MB each), read as `HARPS ORIG`,
into `HARPS_ORIG/science/GJ581/`, with a nightly blaze made from the
spectra in `calib/` (ESO does not give it). HARPS is optical: LBL's
telluric cleaning is on by default (off for NIRPS and SPIRou).

## Spectra downloaded by hand

Spectra from the archives' own pages, from a colleague or from a disk go
into `Downloads/`, then **Dispatch** (Folders tab): see [Your
data](data_root.md).

## The window's help

{{help archive | Archive search}}
