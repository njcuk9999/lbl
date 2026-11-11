# Naming the stars

Observers name the same star many ways: HR1084, HD 22049, eps Eri, K3_01
(a programme's nickname) are one star. The dispatch puts its files in one
folder whatever they say (see [Your data](data_root.md)); the **Naming of
new stars** menu of the Folders tab says what that folder is called.

## Three ways

- **APERO's database, else SIMBAD** (the default): the name APERO's
  astrometric database gives the star (GL699, EPSILONERI, TOI206,
  AN_SEX...). The star is found there by one of the names in its headers,
  or by where it is, so that a HARPS file named Gl699 or a nickname such as
  YO17 gets APERO's name too; APERO's own files carry it already
  (`DRSOBJN`). A star APERO does not know is named as SIMBAD names it.
- **SIMBAD's main name**: SIMBAD's main identifier of the star, without its
  prefixes (`* eps Eri` gives `eps_Eri`, `NAME Barnard's star` gives
  `Barnard_s_star`). It is found by a name of the headers that SIMBAD places
  there, else by asking SIMBAD for the brightest star (in Gaia's G) where
  the files point.
- **The headers' name**: the name the files' headers give the target most
  often (the observer's `OBJECT` before APERO's `DRSOBJN`), made a folder's
  name.

When one way gives no name, the next one does, and the headers come last.

What each way gave, on a few spectra of the demonstration data (a plan, not
an example made up):

| the files say | APERO, else SIMBAD | SIMBAD | headers |
|---|---|---|---|
| `YO17` (AD Leo, HARPS-N) | `GL388` | `AD_Leo` | `YO17` |
| `M112` (Gl 849, HARPS-N) | `GL849` | `BD-05_5715` | `M112` |
| `HR1084` and `K3_01` (eps Eri) | `EPSILONERI` | `eps_Eri` | `K3_01` |
| `Gl1` | `GJ1` | `HD_225213` | `Gl1` |
| `GJ3622` | `GJ3622` | `LP_731-58` | `GJ3622` |
| `HD10700` (EXPRES) | `HD10700` | `tau_Cet` | `HD10700` |

A star **already in the root** keeps the name it has there, in every
instrument folder (one star, one name), whatever the menu says: rename it
in the tree if you want another.

## APERO's database

APERO, the pipeline of SPIRou and NIRPS, keeps one entry per star it knows
(about 1 900): its APERO name, its SIMBAD name, its aliases and its place
(Gaia's, in J2000, with the proper motion). The window copies it once from
APERO's assets server (a few MB) into `~/.cache/lbl/gui/apero_astrometrics/`
(the folder `LBL_GUI_CACHE` points to, if set). A name is looked up as
APERO looks it up: spaces, signs and case aside (`GJ 699`, `Gl699`,
`GL699`). `LBL_APERO_ASTROMETRICS` can point to a folder of APERO's YAML
files instead. When the server does not answer and there is no copy yet,
the plan says so and the new stars are named as SIMBAD names them.

## Renaming by hand

- **In the plan**, before anything moves: the name of each new folder is a
  field; typed, it is checked, and the star's new folders take it in every
  instrument folder they go to (a setting keeps its suffix, `_EGGS`).
- **In the tree**, for a folder already there: the pencil beside it opens a
  field. Ticked, "also its folders of the same name" renames the star's
  folders of the other instrument folders with it. What would be renamed is
  said before you press **Rename**.

A name is checked as it is typed, and refused with why:

- letters, digits, `+`, `-` and `_` only (a space or a `*` is refused, and
  the name with `_` in their place is proposed);
- a letter or a digit first and last (no space at the start or the end);
- not ending as LBL's own folders do (`_pca2d`, `_tc`, `_SPIROU`...);
- not the name of another folder already there;
- 64 characters at most.

A rename is written in `Downloads/.dispatched.jsonl`, like a move. It is
refused while a run works in the root.

**What LBL made under the old name** (templates, masks, per-line velocities,
tables, reports, a corrected `<star>_pca2d/`) is not renamed: their names
hold the target's in many ways. They are counted and named before you
rename; the next run makes them again under the new name, and the
[Cleanup](tab_cleanup.md) tab can delete the old ones.

## The window's help

{{help naming | Naming of new stars}}
