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

The log under the table says each archive's answer. A root that is not
there (a link to a disk not mounted) is said at the search, and the fetch
does not start; a fetch or a search that stops on an error says it in red,
with where it happened, instead of waiting.

## An example: GJ 581 with HARPS

GJ 581, an M3 dwarf observed by HARPS from 2004 to 2024, has a planet
visible in a first run: GJ 581 b, P = 5.37 d, K = 13 m/s ([Bonfils et al.
2005](https://ui.adsabs.harvard.edu/abs/2005A%26A...443L..15B/abstract)).
The search found 321 public HARPS spectra at ESO (482 MB, on 2026-11-09):
ESO's HARPS pipeline `e2ds` (about 1.5 MB each), read as `HARPS ORIG`,
into `HARPS_ORIG/science/GJ581/`, with a nightly blaze made from the
spectra in `calib/` (ESO does not give it for HARPS; for NIRPS it gives the
blaze of each night, fetched beside the spectra). Every blaze is written
under a name any disk takes, Windows included ([Your data](data_root.md)).
HARPS is optical: LBL's telluric cleaning is on by default (off for NIRPS
and SPIRou).

## CORALIE, from DACE

CORALIE's spectra come from DACE (Geneva), which serves since 2026 every
era reduced again by espdr (DRS-3.3.12): the `S2D_A`, blaze corrected on a
barycentric vacuum grid, 5.8 MB each, read as `CORALIE DACE` into
`CORALIE_DACE/science/<star>_C98/` (and `_C07`, `_C14`, `_C24`): an era is
an object of its own, since the instrument changed in 2007, 2014 and 2024
(CORALIE98 has 68 orders, the later eras 69). DACE finds the star by its
catalogue name (HD, GJ, HIP...), and is asked anonymously: public spectra
only. HD 23079 (on 2026-11-19): 41, 20 and 75 spectra in C98, C07 and C14;
the delivered velocities agree with DACE's CCF to 2.5-3.7 m/s rms, their
errors 5 to 13% smaller.

## From a script, or from koloa

The same search and fetch run without the window:

```
python -m lbl.gui.archives list  "HD 23079" --root ROOT --instruments HARPS CORALIE
python -m lbl.gui.archives fetch "HD 23079" --root ROOT --naming apero --rows "HD 23079|HARPS:"
```

`list` says what the archives have (a row per LBL reader and setting: its
spectra, those already in the root, the size to fetch, its folder);
`fetch` fetches the rows named by their ids (all of them when none). A
star the root already has keeps its folder (found by where it is); a new
one is named as `--naming` says, as the dispatch names new stars
([Naming the stars](naming.md)): `apero` (APERO's name, by one of the
star's names or by where it is, else SIMBAD's), `simbad`, or `headers`
(the name asked: no header is read before the files are there), its
setting after it (`HD_23079_C98`). `--ident` gives the star as its caller
resolved it (main, aliases, ra, dec, pmra, pmdec): SIMBAD is not asked
again. The log goes to stderr; the last line of stdout is `LBL-ARCHIVES `
and the state of the search and of the fetch as JSON. koloa
(github.com/eartigau/koloa) proposes the spectra of the star it analyses
this way, into the root it is given.

## Spectra downloaded by hand

Spectra from the archives' own pages, from a colleague or from a disk go
into `Downloads/`, then **Dispatch** (Folders tab): see [Your
data](data_root.md).

## The window's help

{{help archive | Archive search}}
