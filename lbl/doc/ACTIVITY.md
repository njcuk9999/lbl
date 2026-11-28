# Activity lines: equivalent widths

Every science spectrum LBL measures gets the equivalent width of the
activity lines of `lbl/resources/activity_lines.yaml`
(`lbl/science/activity_lines.py`), in `lbl_compute` once its velocity has
converged. Nothing to set: the lines, their windows and their continuum are
fixed in the catalog. This page defines the measurement.

## The definition

The formalism is that of Schöfer et al. (2019, A&A 623, A44, section 3.2),
with the star's own template in place of their inactive reference star.

Each line has, in the catalog, in vacuum, in the star's rest frame:

- **a window** `[lo, hi]`: the range its equivalent width is integrated
  over (a boxcar). It holds the whole line, wide enough for the broadest
  emission seen in M dwarfs, as Schöfer et al. chose theirs;
- **two pseudo-continuum ranges**, one blueward and one redward of the
  window.

In the star's frame (the template's, at the velocity the RVs converged
to), with T the template, S the spectrum (before LBL's sigma clipping, which
would cut a flare out) and B the blaze:

- **PC(λ)**, the template's pseudo-continuum: the 90th percentile of T in
  each range, joined by a straight line between the ranges' centres. The
  90th percentile is robust to the noise and to the lines within the ranges
  (Schöfer et al. 2019, eq. 2). It is taken on the template itself, its own
  pixels merged over the orders: published ranges can be farther apart than
  an order holds valid pixels (Pa γ's are 6.5 nm apart, a NIRPS order holds
  about 11 nm), the template holds them all.
- **q(λ)**, the spectrum's level against the template: the median of
  S / (T B) in each range, a straight line between the two, on the order's
  pixels. Where the order does not hold a range, q (only q) is taken on as
  wide a stretch of the order's valid pixels nearest to it, on that side of
  the window and half the window's width from it at least. S / (q B) is the
  spectrum at the template's level, as Schöfer et al. normalise both spectra
  to their pseudo-continuum ranges. (On TOI-904's NIRPS spectra, taking q
  from Pa γ's one range the order held left all of them 24 mÅ below the
  template, 3σ; with the nearest stretch, 0.2 mÅ.)

Then, with a_i the length of pixel i inside the window:

| quantity | definition | what it is |
|---|---|---|
| `EWT` | Σ a_i (1 − T_i / PC_i), on the template's own pixels | the template's pseudo-equivalent width: the same for every spectrum, whatever its orders |
| `dEW` | Σ a_i (T_i − S_i / (q_i B_i)) / PC_i, on the order's pixels | what the spectrum absorbs more than the template (Schöfer et al.'s pEW′, the template as the reference) |
| `EW` | EWT + dEW ≈ Σ a_i (1 − S_i / (q_i B_i PC_i)) | the spectrum's own pseudo-equivalent width |

All in mÅ, positive for absorption: a line filling in goes down, emission is
negative.

## Missing pixels

A pixel of the window can be missing in a spectrum (a bad pixel, a telluric
line masked, the order's edge). The pixels that are there are made up for
by the template's line: with d_i = |1 − T_i / PC_i| its depth on the order's
pixels,

    FRACLINE = Σ_valid a_i d_i / Σ_window a_i d_i

is the share of the line the valid pixels hold, and dEW (and its error) is
the sum over the valid pixels divided by FRACLINE. A spectrum is measured
when at least 75 % of the window is valid (`FRACVALID`) and the valid pixels
hold at least 75 % of the line (`FRACLINE`).

This is exact when the change of the line follows the template's profile
(the line deepens or fills in as a whole). Losing a wing then costs little,
losing the core much, as it should: the core is where the change is. A
uniform correction (dividing by `FRACVALID`) assumes the change is spread
evenly over the window, which it is not. For a Gaussian line of 0.020 nm
FWHM in He I 10833's 0.050 nm window:

| pixels missing | counted as the template (before) | divided by FRACVALID | divided by FRACLINE |
|---|---|---|---|
| a quarter of the window, on one side (a wing) | 93 % | 124 % | 100 % |
| a quarter of the window, at the centre (the core) | 46 % | 61 % | 100 % |
| a tenth of the window, on one side | 99 % | 110 % | 100 % |

(the share of the true dEW each gives). A change that does not follow the
profile (an emission core narrower than the absorption) is not made up for
exactly: what is left is much smaller than with either of the other two.

A line flat in the template's window (its mean depth under 1 %, or the
template not defined over all of the window) has no profile to weight by:
no correction, `FRACLINE` is NaN, and 90 % of the window must be valid.

Where the pixels missing change with the BERV (a telluric line crossing the
window), so does `FRACLINE`: plot dEW against it to check that the
correction leaves no trend.

## The errors

| column | what it is |
|---|---|
| `sEW` (= `sdEW`) | the error of EW and of dEW: the noise of LBL's model (its rms) over the window, and that of q's level |
| `EWTSYS` | the systematic error of EWT from the choice of continuum: how far EWT moves with another one, the 95th percentile of T within 2.5 nm of the line's centre (LBL's continuum before 2026-11-27). The same for every spectrum: it does not average down, and the nightly table does not divide it. EW carries it, dEW does not |
| `FRACVALID` | the fraction of the window with valid pixels; under 0.75, no measure |
| `FRACLINE` | the share of the template's line in the valid pixels, which dEW is divided by (see Missing pixels); under 0.75, no measure; NaN for a line flat in the template |
| `CHI2` | the residuals of the window against LBL's rms, per pixel: the pixels LBL's sigma clipping would have rejected are kept (a flare would be cut out), CHI2 shows them |

The continuum is the largest uncertainty of an absolute equivalent width: in
a window of width w, a relative error ε on the continuum moves EWT by about
w ε (2.5 mÅ per percent in a 0.025 nm window). It moves dEW by a factor
1 + ε only: the variability of a line is little affected by the continuum,
its absolute level is.

## The order

The order is the one where the line, without the BERV, falls closest to its
centre: the same order for every spectrum, as long as the window is inside
it and enough of it is valid (`FRACVALID`, `FRACLINE`); the next closest
otherwise.

## The catalog

`lbl/resources/activity_lines.yaml`, one entry per line: `wave` (its centre,
NIST, for the labels), `window`, `continuum` (two ranges), `ref` (where they
come from), `band`, `kind` (chromospheric, or flare: lines seen mostly in
flares). Its `version` is written in each lblrv file (`ACTVERS` of the
`ACTIVITY` extension): a file measured with another version is measured
again by `lbl_compute`, and its rdb compiled again.

The windows and ranges come, line by line, from:

1. **Schöfer et al. 2019, Table 1** (He I D3, Na I D1 and D2, Hα, the three
   Ca II infrared lines, He I 10833, Paβ). Their Δλ is a HALF width: the
   window is λ0 ± Δλ, each range λ_l/r ± Δλ_PC.
2. Where they have none, **another high-resolution study** that defines a
   window and continuum for the line; the catalog names it.
3. Where none does, **a window and ranges derived** by the rule written in
   the catalog's header, named "derived" there.

The windows and ranges are in vacuum; a definition published in air is
brought to vacuum (Morton 2000) and says so.

## The columns of the rdb

`lbl_compile` adds, for each line measured at least once, `<column>_<line>`
with the line's space as an underscore: `EW_He_1.08333`, `sEW_He_1.08333`,
`dEW_He_1.08333`, `sdEW_He_1.08333`, `EWT_He_1.08333`, `EWTSYS_He_1.08333`,
`FRACVALID_He_1.08333`, `FRACLINE_He_1.08333`, `CHI2_He_1.08333`. The rdb per night (`lbl2_`)
bins EW and dEW weighted by their errors.

The report draws EW in time, the template's EWT dotted, with its systematic
error in each panel's title.

## Comparing instruments

An equivalent width depends on the resolution where a window does not hold
the whole line, and a pseudo-continuum does too (lines blend into it at a
lower resolution). The variability of a line is comparable from one
instrument to another; its absolute level only as an indication. The
catalog's windows are fixed, the same for every instrument.

## Changes

- 2026-11-28: missing pixels made up for by the template's line
  (`FRACLINE`); a spectrum is measured with 75 % of the window and of the
  line valid (90 % of the window before, the missing pixels counted as the
  template's). The lblrv files and rdb tables written before are measured
  and compiled again.

- 2026-11-27: Schöfer et al.'s formalism. Until then the windows were the
  core of each line (the windows taken from Schöfer et al. were half
  theirs, their Δλ having been read as a full width), the continuum the
  95th percentile of the template within 2.5 nm, the level q from side bands
  scaled to each window, and only dEW was given.
