#!/usr/bin/env python
"""Rebuild or correct an exposure from a two-frame fit, on its own wavelengths.

The `correct` stage of `pca2d-preclean`. Reads `twoframe_components.fits` and
puts the model back where the data came from: order by order, on the t.fits
file's native wavelengths, with each order taking the means of *its own
parity*.

    python -m lbl.pca2d.reconstruct --fits outputs/TOI2120/1-3v/twoframe_components.fits \
        --file data/TOI2120/2811170t.fits --out model.fits --plot model.pdf
    python -m lbl.pca2d.reconstruct --correct --all --cube cache/cube_tfits_<key> \
        --fits outputs/TOI2120/1-3v/twoframe_components.fits \
        --source-dir data/TOI2120 --corrected-dir outputs/TOI2120/1-3v/corrected

WHAT THE NUMBERS ARE. The cube was high-passed in the log, so every quantity
here is `ln f - savgol(ln f)`: zero in the continuum, negative in a line. The
Savitzky-Golay continuum was never stored, by anyone, so this cannot be turned
back into a flux. It is the right quantity to subtract from a fresh spectrum
that has been through the same high-pass, and the wrong one to multiply
anything by.

THE MODEL, per exposure n and order parity p:

    y_n = S_n (T_p + P^T a_n)  +  mu[p]  +  Q^T b_n

`S_n` carries the star block from the stellar rest frame into the observer
frame, a translation of -pixel_shift(BERV) samples because the grid is
log-uniform. `T_p` is the star-frame mean: zero by default, a median template
with the fit's --template, one per parity with --mean iterate. The observer
block and its mean `mu[p]` do not move.

WHAT A CORRECTED FILE HOLDS is panel 3 of the sequence figure, exactly: the
flux with the observer block and its per-parity mean divided out, and NaN
wherever the fit gave the sample no weight (fit_weights_mask). The star block
stays, T_p included. tests/test_panel3_is_the_correction.py holds them together.

THE PARITY WRAPPING is the part that is easy to get wrong. A t.fits row set is
split into even and odd orders, and `mu_earth` was fitted once per parity, so
order k must take `mean_even` if k is even and `mean_odd` if it is odd. Using
one mean for both puts a static even-minus-odd offset into every reconstructed
order, which is exactly the artefact the two-frame fit was fixed to avoid. The
star and Earth *components* have no parity: they are single vectors over the
whole grid, and only the means are per parity.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import CubicSpline

from .grids import pixel_shift
from .logger import log
from . import readers as _readers
from . import tfits as sptf
from . import twoframe as _bcd
from .progress import bar as _bar

#: the same treatment twoframe.load_cube gives the cube, so a refitted
#: coefficient is comparable with the fit's own rather than to its own scale
LN_CLIP_LOW = -0.5
RAMP_ZERO = 0.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits", required=True,
                   help="the twoframe_components.fits the fit stage wrote")
    p.add_argument("--file", default=None,
                   help="the t.fits to act on; matched to COEFFS by basename")
    p.add_argument("--all", action="store_true",
                   help="every exposure in the fit, not just --file")
    p.add_argument("--out", default=None, help="write the model to this FITS")
    p.add_argument("--plot", default=None, help="write a check figure here")
    p.add_argument("--kernel-halfwidth", type=int, default=8,
                   help="Lanczos support, must match the fit (default 8)")

    p.add_argument("--correct", action="store_true",
                   help="divide the components out of the flux and write a new"
                        " t.fits named <stem>t_M-N.fits, M star components and N"
                        " Earth components removed. The template and the"
                        " observer means stay in: the filename names components,"
                        " so components are what comes out")
    p.add_argument("--n-star", type=int, default=None,
                   help="M, star components to remove (default: all in the fit)")
    p.add_argument("--n-earth", type=int, default=None,
                   help="N, Earth components to remove (default: all in the fit)")
    p.add_argument("--max-sky-ratio", type=float, default=None,
                   help="set to NaN any sample where the subtracted sky"
                        " emission exceeded this many times the stellar flux."
                        " Must match quality.max_sky_ratio of the fit, or the"
                        " corrected files carry samples the model never saw")
    p.add_argument("--corrected-dir", default="corrected",
                   help="where the t_M-N.fits files go")
    p.add_argument("--keep-name", action="store_true",
                   help="write each corrected file under its source's own name"
                        " rather than <stem>t_M-N.fits: what LBL needs when the"
                        " corrected folder is one of its science objects"
                        " (lbl.recipes.lbl_pca2d), since its glob is the"
                        " instrument's and the folder already says what they are")
    p.add_argument("--source-dir", default=None,
                   help="where to find the input t.fits when using --all or"
                        " --by-night")
    p.add_argument("--refit", action="store_true",
                   help="fit each exposure's OWN coefficients against the saved"
                        " basis instead of reusing its night's. The basis is the"
                        " campaign's and stays fixed; only the amplitudes are"
                        " re-estimated. This is what --by-night needs to be"
                        " honest: the sky changes within a night, so the"
                        " airglow amplitude of a 3-exposure visit is not one"
                        " number. Needs --config, the same YAML the cube was"
                        " built from, because the exposure has to be resampled"
                        " and high-passed exactly as the fit saw it")
    p.add_argument("--config", default=None,
                   help="the cube's YAML, required by --refit")
    p.add_argument("--by-night", action="store_true",
                   help="correct EVERY t.fits in --source-dir, not only the one"
                        " file per night that carries coefficients. With a"
                        " nightly-stacked cube the fit sees one row per night"
                        " but the night was built from several exposures; this"
                        " applies that night's coefficients to each of them,"
                        " using each exposure's own BERV to place the star"
                        " basis. Nights absent from the fit are skipped")
    p.add_argument("--nsig-cut", type=float, default=None,
                   help="also NaN every sample whose residual, panel 5 of the"
                        " sequence figure, is beyond this many running robust"
                        " sigmas (outliers.residual_outliers); needs --cube")
    p.add_argument("--clip-window", type=int, default=151,
                   help="the running sigma's box in grid samples; the high"
                        " pass's own window is what the pipeline passes")
    p.add_argument("--excursion-nsig", type=float, default=None,
                   help="with --excursion-samples: flag every window of the"
                        " residual up to that many samples wide whose aggregate"
                        " significance is beyond this, the whole window"
                        " (outliers.excursions). The variance of a window's sum"
                        " is measured from the residual's own autocorrelation")
    p.add_argument("--excursion-samples", type=int, default=None,
                   help="how wide those windows go, in grid samples; the"
                        " pipeline passes the instrument's resolution element"
                        " times correct.excursion_elements")
    p.add_argument("--weight-floor", type=float, default=None,
                   help="NaN la ou la BOUCLE a deprecie l'echantillon sous"
                        " cette fraction de son poids nominal: le poids du fit"
                        " est le masque (clipped_weights_mask). 0.5 = |z| >"
                        " 4.24 sigma avec le clip a 3")
    p.add_argument("--bias-nsig", type=float, default=None,
                   help="avec --bias-samples: retirer de toutes les poses les"
                        " colonnes dont le residu MOYEN sur les poses depasse"
                        " ce nombre de sigmas (outliers.coherent_bias)")
    p.add_argument("--bias-samples", type=int, default=None,
                   help="la largeur sur laquelle cette moyenne est sommee, en"
                        " echantillons de grille")
    p.add_argument("--excess-nsig", type=float, default=None,
                   help="with --excess-samples: drop from every exposure the"
                        " observer columns whose chi2 over the exposures is this"
                        " many sigmas above the noise (outliers.excess_rms)")
    p.add_argument("--excess-chi2", type=float, default=None,
                   help="and whose windowed chi2 is above this")
    p.add_argument("--excess-samples", type=int, default=None,
                   help="the width that chi2 is averaged over, in grid samples")
    p.add_argument("--column-frac", type=float, default=None,
                   help="with --column-chi2: drop an observer column from EVERY"
                        " exposure when more than this fraction of them is"
                        " clipped there and the survivors are still noisy")
    p.add_argument("--column-chi2", type=float, default=None,
                   help="the survivors' reduced chi2 above which such a column"
                        " goes; pure noise gives 0.97 after a 3 sigma clip"
                        " (outliers.expected_chi2)")
    p.add_argument("--shrink", action="store_true",
                   help="keep each observer component at a column only where it"
                        " is significant there, all exposures together: Q_ji"
                        " times max(0, 1 - 1/z^2), z the significance of Q_ji"
                        " from every row's own weight, the"
                        " weights scaled by the fit's reduced chi2 (pca2d.shrink)."
                        " Needs --cube and the fit's archive")
    p.add_argument("--shrink-smooth", action="store_true",
                   help="with --shrink: average each component's z^2 over one"
                        " resolution element before its factor is taken, so a"
                        " telluric line is judged over its width. Needs --resolution")
    p.add_argument("--smooth-components", default=None,
                   help="observer components, 1-based and comma-separated, to"
                        " smooth by LBL's template filter over one resolution"
                        " element before they are shrunk and divided out; for the"
                        " components that are mostly noise. Needs --resolution")
    p.add_argument("--resolution", type=float, default=None,
                   help="the instrument's resolving power, the unit of the two"
                        " smoothings above")
    p.add_argument("--weight", choices=("flux", "velocity"), default="flux",
                   help="the metric the REFITTED amplitudes are measured in."
                        " 'flux', every sample as the fit saw it. 'velocity',"
                        " every sample weighted by the star's own derivative"
                        " there, (dT/dv)^2: what a contaminant does to a radial"
                        " velocity is set by its overlap with that derivative,"
                        " and a contaminant that is flat where the star has"
                        " structure moves no line. What is divided out of the"
                        " flux does not change, nor does which samples are"
                        " blanked; only how b is measured. Implies --refit, and"
                        " needs --resolution for the derivative's width")
    p.add_argument("--velocity-floor", type=float, default=VELOCITY_FLOOR,
                   help="with --weight velocity, the weight a sample with no"
                        " velocity information keeps, as a fraction of the"
                        " median. Zero would leave a component that lives only"
                        " where the star is flat with no amplitude at all")
    p.add_argument("--star-group", type=int, default=0,
                   help="the first star-spectrum group of this object in a joint"
                        " fit, 2 x its index among the objects (pca2d.joint);"
                        " the order parity is added to it. 0 for one object")
    p.add_argument("--mask", choices=("exposure", "common", "none"),
                   default="exposure",
                   help="which samples a corrected file blanks: the ones its own"
                        " exposure's fit gave no weight (exposure), the ones any"
                        " exposure's did (common, so every epoch carries the same"
                        " set of lines), or none, keeping the delivered flux there")
    p.add_argument("--mask-share", type=float, default=1.0,
                   help="with --mask common, the fraction of exposures that"
                        " must have weighted a sample for it to be kept. 1.0,"
                        " the default, is the strict intersection; below it the"
                        " line set is still ONE for the whole campaign, decided"
                        " by a majority rather than unanimously")
    p.add_argument("--permute-b", type=int, default=None, metavar="SEED",
                   help="TEST NUL (pca2d.diagnostics): chaque nuit recoit les"
                        " amplitudes observateur d'une AUTRE nuit, tiree avec"
                        " cette graine. Ce n'est pas une correction")
    p.add_argument("--q-support", type=float, default=None, metavar="T",
                   help="ne diviser la base observateur que la ou la"
                        " transmission tellurique du cube descend sous T sur au"
                        " moins une pose (pca2d.diagnostics). Demande --cube")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cube", default=None,
                   help="the cube the fit was made from. With it, every sample"
                        " the fit gave no weight to is NaN in the corrected"
                        " file, exactly the samples panel 3 of the sequence"
                        " figure hides")
    args = p.parse_args(argv)
    if not args.file and not args.all and not args.by_night:
        p.error("give --file, or --all for every exposure in the fit, or"
                " --by-night for every exposure in --source-dir")
    return args


def load_model(path):
    """The basis, the means and the coefficient table, as plain arrays."""
    with fits.open(path) as h:
        head = h[0].header
        basis = h["BASIS"].data
        coeffs = Table(h["COEFFS"].data)
        n_star, n_earth = int(head["NSTAR"]), int(head["NEARTH"])
        model = {
            "grid": np.asarray(basis["wavelength"], dtype=np.float64),
            "template": np.asarray(basis["template"], dtype=np.float64),
            # (n_star, M) even with no star component at all: an empty list
            # gives a one-dimensional array, and everything downstream asks the
            # basis how many columns it has. The refit of every exposure died
            # on that, twice, on 2026-09-12.
            "P": (np.array([basis["star_pc%d" % (k + 1)] for k in range(n_star)],
                           dtype=np.float64) if n_star else
                  np.zeros((0, len(basis["wavelength"])), dtype=np.float64)),
            "Q": np.array([basis["earth_pc%d" % (j + 1)] for j in range(n_earth)],
                          dtype=np.float64),
            "dv": float(head["DV"]),
            "n_star": n_star, "n_earth": n_earth,
            "content": head.get("CONTENT", "?"),
            "coeffs": coeffs,
        }
        names = [c for c in basis.columns.names if c.startswith("mean_")]
        # ordered even, odd -- the same order parity % 2 indexes
        model["means"] = {nm.split("_", 1)[1]: np.asarray(basis[nm], dtype=np.float64)
                          for nm in names}
        # the star-frame mean of each parity, from a --mean iterate fit; None
        # from an older one, which has the single `template` instead
        names = [c for c in basis.columns.names if c.startswith("template_")]
        model["templates"] = ({nm.split("_", 1)[1]:
                               np.asarray(basis[nm], dtype=np.float64)
                               for nm in names} or None)
        # where the fit had data, from its own weight sums per parity: the
        # correction may touch a grid sample only there. It used to be read off
        # the observer mean (mean != 0), which a fit that keeps no observer
        # mean (--mean star) has at zero everywhere.
        sums = [c for c in basis.columns.names if c.startswith("weight_sum_")]
        model["support"] = (np.any([np.asarray(basis[c], dtype=np.float64) > 0
                                    for c in sums], axis=0) if sums else None)
    return model


def group_of(model, parity, groups):
    """Which star-spectrum group an order of this parity belongs to.

    One object has two, the even orders and the odd ones. A joint fit of
    several objects has two per object, labelled 2 * index + parity, since
    every star has its own spectrum while a parity means the same thing for
    all of them. The correction of one object's files is told which pair is
    its own, in model["star_group"] (reconstruct --star-group).
    """
    return (int(model.get("star_group", 0)) + int(parity) % 2) % max(int(groups), 1)


def template_for(model, parity):
    """The star-frame mean for an order of this parity (0 even, 1 odd)."""
    per = model.get("templates")
    if per:
        names = list(per.keys())
        return per[names[group_of(model, parity, len(names))]]
    return model["template"]


def exposure_row(coeffs, filename):
    """The COEFFS row for this file, matched on basename."""
    want = os.path.basename(filename)
    names = [os.path.basename(str(v)) for v in coeffs["filename"]]
    if want not in names:
        stem = want.replace("t.fits", "")
        near = [n for n in names if stem[:20] in n][:3]
        raise SystemExit("%s is not in the fit%s"
                         % (want, ("; nearest: %s" % ", ".join(near)) if near else ""))
    row = coeffs[names.index(want)]
    if bool(row["rejected"]):
        log("WARNING: this exposure was rejected by the MAD cut. Its"
              " coefficients are zeros and the model below is the template"
              " plus the mean, nothing else.")
    return row


def model_on_grid(model, row):
    """The full model on the magic grid, split into what moves and what does not.

    Returned separately because only the star half is shifted, and keeping them
    apart is what lets the caller add the right parity's means afterwards:
    the star-frame one (template_for) before the carry, the observer one after.
    """
    n_star, n_earth = model["n_star"], model["n_earth"]
    a = np.array([row["a%d" % (k + 1)] for k in range(n_star)])
    b = np.array([row["b%d" % (j + 1)] for j in range(n_earth)])
    star_rest = a @ model["P"]           # the parity's template goes on per order
    earth = b @ model["Q"]
    return star_rest, earth, a, b


def carry_star(star_rest, berv, dv, halfwidth=8):
    """S_n applied to the star-frame vector: translate by -pixel_shift(BERV).

    Exactly the operator the fit used, taken from twoframe_bcd so the two can
    never drift apart. The sign is the one NOTES 11.11 settles: a stellar
    feature sits at lambda_bary = lambda_obs D(BERV), D the relativistic
    Doppler factor, so star -> observer is minus grids.pixel_shift(BERV, dv).
    """
    delta = np.array([-float(pixel_shift(float(berv), dv))])
    shifter = _bcd.SHIFTERS["lanczos"](
        star_rest.size, a=halfwidth,
        max_shift=int(np.ceil(abs(delta[0]))) + 2)
    prepared = shifter.prepare(star_rest[None, :])
    return shifter.carry(prepared, delta)[0, 0]


def reconstruct(model, row, halfwidth=8, path=None):
    """The model resampled onto every order of the file's own wavelength grid.

    Returns (wave, recon, parity): each (n_orders, n_pixels), parity per order.
    Samples the basis does not cover come back NaN rather than zero: zero is a
    meaningful value in a high-passed spectrum, "continuum", and handing back a
    flat continuum where the model has nothing to say would be a lie.
    """
    payload = _readers.read_any(path, config)
    wave = payload["wave"]
    n_orders, n_pixels = wave.shape
    grid = model["grid"]

    star_rest, earth, _, _ = model_on_grid(model, row)
    star_obs = [carry_star(template_for(model, p) + star_rest, row["berv"],
                           model["dv"], halfwidth) for p in (0, 1)]

    names = list(model["means"].keys())          # even, odd
    recon = np.full((n_orders, n_pixels), np.nan)
    parity = np.arange(n_orders) % 2
    for order in range(n_orders):
        mean = model["means"][names[group_of(model, parity[order], len(names))]]
        total = star_obs[parity[order]] + earth + mean
        # the basis has no support where the mean is exactly zero, and that is
        # per parity, so the mask has to be rebuilt for each order's parity
        support = mean != 0.0
        if model.get("support") is not None:
            support = support | model["support"]     # see order_correction
        w = wave[order]
        ok = np.isfinite(w)
        if not ok.any():
            continue
        lo = max(np.searchsorted(grid, np.nanmin(w)) - 4, 0)
        hi = min(np.searchsorted(grid, np.nanmax(w)) + 4, grid.size)
        if hi - lo < 4:
            continue
        seg_g, seg_v, seg_s = grid[lo:hi], total[lo:hi], support[lo:hi]
        spline = CubicSpline(seg_g, np.where(seg_s, seg_v, 0.0), extrapolate=False)
        values = spline(w)
        # drop any sample whose neighbourhood on the grid was not supported
        nearest = np.clip(np.searchsorted(seg_g, w) - 1, 0, seg_s.size - 2)
        covered = seg_s[nearest] & seg_s[nearest + 1]
        recon[order] = np.where(ok & covered, values, np.nan)
    return wave, recon, parity


def correction_on_grid(model, row, n_star=None, n_earth=None, halfwidth=8):
    """The part of the model a correction removes, on the magic grid.

    The components only: M star and N Earth, the pair `t_M-N.fits` is named
    after. Never the template, the star's mean spectrum, whose removal would
    not be a correction but a different product. The observer-frame mean the
    fit took out of each parity is not here either, only because which one
    applies depends on the order: order_correction adds it, order by order,
    together with the observer components.

    M and N may be smaller than the fit's, which is the point of naming the file
    after them: t_5-5, t_0-5 and t_5-0 are three different corrections of the
    same exposure and the filename is what tells them apart.
    """
    k = model["n_star"] if n_star is None else int(n_star)
    j = model["n_earth"] if n_earth is None else int(n_earth)
    if not 0 <= k <= model["n_star"] or not 0 <= j <= model["n_earth"]:
        raise SystemExit("the fit has %d star and %d Earth components; asked for %d and %d"
                         % (model["n_star"], model["n_earth"], k, j))
    star_rest = np.zeros(model["grid"].size)
    if k:
        a = np.array([row["a%d" % (i + 1)] for i in range(k)])
        star_rest = a @ model["P"][:k]
    earth = np.zeros(model["grid"].size)
    if j:
        b = np.array([row["b%d" % (i + 1)] for i in range(j)])
        # the observer basis as the correct stage divides it out: shrunk by its
        # significance, some components smoothed first (--shrink,
        # --smooth-components); the fit's own Q otherwise
        Q = model.get("Q_correct")
        Q = (model["Q"] if Q is None else Q)[:j]
        earth = b @ Q
    star_obs = (carry_star(star_rest, row["berv"], model["dv"], halfwidth)
                if k else star_rest)
    return star_obs + earth, k, j


def _field(row, name):
    """row[name] as a float, None when the row has no such field.

    Not `name in row`: on a FITS or astropy table row that asks whether the
    name is one of the row's VALUES, which it never is, and the PCASTR_V card
    it used to guard was never written to a corrected file.
    """
    try:
        return float(row[name])
    except (KeyError, ValueError, IndexError, TypeError):
        return None


def coefficient_cards(model, row, k, j):
    """The fit's coefficients for one exposure, as (keyword, value, comment).

    Every component the fit HAS, not only the ones divided out. With
    correct.n_star = 0 the star coefficients are exactly what stays in the
    corrected flux, and they are the thing a velocity is worth correlating
    against afterwards; nothing downstream carries them otherwise, since an rdb
    knows only what LBL measured.

    Every keyword here is PCASTR or PCAOBS and then either a two-digit index or
    _N and _D, so one prefix names the frame and the suffix names the number.
    The PCA2xxxx keywords beside them are the ones that belong to the run as a
    whole rather than to one frame: PCA2BERV, PCA2NPIX, PCA2REJ.

    Two digits and not three. A FITS keyword is eight characters: PCASTR001
    would be nine, and astropy would write it as a HIERARCH card, which is not
    what a file whose whole point is to stay an ordinary t.fits should carry.
    Ninety-nine components is already twice what the validator calls sane.
    """
    cards = []
    # The velocity the fit took out of the observer block's reach. Reported and
    # NOT applied: the corrected flux is the delivered flux with the observer
    # components divided out, and this number is what would have been written
    # into it had the term not been fitted.
    vrad = _field(row, "vrad_fit")
    if vrad is not None and np.isfinite(vrad):
        cards.append(("PCASTR_V", vrad, "m/s, star shift the fit absorbed"))
    for prefix, letter, count, removed, frame in (
            ("PCASTR", "a", model["n_star"], k, "star"),
            ("PCAOBS", "b", model["n_earth"], j, "observer")):
        if count > 99:
            log("%d %s components is more than the %s01..99 keywords can name;"
                " the rest are not written to the header"
                % (count, frame, prefix), "warn")
        listed = min(int(count), 99)
        # Two counts, and they are different numbers: _N is how many amplitude
        # cards follow, so a reader loops without guessing, and _D is how many
        # of those components were actually divided out of the flux. With
        # correct.n_star = 0 the first is 2 and the second is 0.
        cards.append(("%s_N" % prefix, listed,
                      "%s-frame comps listed, %s01..%02d"
                      % (frame, prefix, listed)))
        cards.append(("%s_D" % prefix, int(removed),
                      "%s-frame comps divided out of the flux" % frame))
        for i in range(listed):
            key = "%s%02d" % (prefix, i + 1)
            value = float(row["%s%d" % (letter, i + 1)])
            cards.append((key, value, "%s-frame comp %d amplitude, %s"
                          % (frame, i + 1, "divided out" if i < removed
                             else "left in the flux")))
    return cards


def mean_divided(model, n_earth):
    """Whether an observer-frame mean was actually taken out of this file.

    Not whether one would have been ADDED: the nominal, `mean: star`, keeps no
    observer mean at all, so its means are zero everywhere and order_correction
    adds nothing. The card used to say bool(n_earth), which read True on every
    nominal file while no mean of the kind existed. A provenance card that has
    to be interpreted is worse than no card, because it is read by whoever was
    not there when the run was made.
    """
    if not n_earth:
        return False
    return any(np.any(np.asarray(m) != 0.0)
               for m in (model.get("means") or {}).values())


def order_correction(model, correction, n_earth, order, wave):
    """What correct_file divides out of one order, on that order's own pixels.

    Returns (values, live): the model in ln f at each pixel, and where it may
    be applied, or None when the order misses the grid. `correction` is
    correction_on_grid's, the components. To it goes the mean the fit
    subtracted from this order's parity before solving for them, whenever any
    observer component is divided out: panel 3 of the sequence figure, "what a
    corrected file holds", is the data minus that mean minus the observer
    block, and a file that kept the mean kept a static observer-frame pattern
    the panel shows removed (2026-09-10, the stripes at 1267 nm). It belongs
    to the observer block, so a file that divides none of it out, t_k-0,
    keeps it.
    """
    names = list(model["means"].keys())
    grid = model["grid"]
    mean = model["means"][names[group_of(model, order, len(names))]]
    # where the fit had data: its own weight sums, and, as before, wherever the
    # observer mean is not zero. A fit that keeps no observer mean (--mean
    # star) has one that is zero everywhere, and the mean alone would then
    # leave nothing to correct.
    support = mean != 0.0
    if model.get("support") is not None:
        support = support | model["support"]
    ok = np.isfinite(wave)
    if not ok.any():
        return None
    lo = max(np.searchsorted(grid, np.nanmin(wave)) - 4, 0)
    hi = min(np.searchsorted(grid, np.nanmax(wave)) + 4, grid.size)
    if hi - lo < 4:
        return None
    total = correction + mean if n_earth else correction
    seg_g, seg_s = grid[lo:hi], support[lo:hi]
    spline = CubicSpline(seg_g, np.where(seg_s, total[lo:hi], 0.0),
                         extrapolate=False)
    values = spline(wave)
    nearest = np.clip(np.searchsorted(seg_g, wave) - 1, 0, seg_s.size - 2)
    live = ok & seg_s[nearest] & seg_s[nearest + 1] & np.isfinite(values)
    return values, live


def mask_for(mode, own_alive, alive_by_file, name):
    """Which samples a corrected file blanks, under correct --mask.

    `none` blanks nothing. `common` blanks every sample any exposure left
    unweighted, the same set for all of them, and it says so even in the refit
    path: that path computes a mask from the exposure's own weights, and
    leaving it in charge made the option do nothing at all on a nightly-stacked
    target, where every file goes through it (2026-09-12). `exposure` blanks
    what that exposure lost, the refit's own mask when it has one.
    """
    if mode == "none":
        return None
    if mode == "common" and alive_by_file:
        return next(iter(alive_by_file.values()))
    if own_alive is not None:
        return own_alive
    return alive_by_file.get(name) if alive_by_file is not None else None


def star_support(model):
    """The grid samples the star basis constrains, or None when the fit has no
    star component at all.

    A fit with n_star 0 stores an empty P, and np.any over the first axis of an
    empty array gives a scalar, which broke the refit of every exposure on
    2026-09-12. There is nothing to guard then: gap_guard falls back on the
    weights' own support.
    """
    # not atleast_2d: an empty P of one dimension becomes (1, 0) under it, and
    # the answer would be an empty vector rather than "there is no support"
    P = np.asarray(model["P"])
    if P.ndim != 2 or not P.size:
        return None
    return np.any(P != 0, axis=0)


def clipped_weights_mask(cube, fit_path, floor=0.5):
    """Les echantillons que la BOUCLE a deprecies sous `floor`: nom -> (2, grille).

    Le poids de la boucle EST le masque (2026-09-25): un echantillon ecoute a
    moins de la moitie de son poids nominal n'est pas une demi-mesure, c'est
    une mesure que le modele n'a pas decrite, et la livrer a LBL corrigee par
    un modele qu'elle n'a pas aide a construire n'a pas de sens.

    PAR POSE, et non commun a toutes, parce que c'est ce que la chose est: sur
    TOI-2120 le facteur tombe sous 0.5 pour 0.388% des echantillons, 2415 par
    pose, et au-dela de 20% des poses il ne reste que 13 colonnes sur 577 000.
    Ce sont des accidents locaux, pas des regions, et les masquer partout
    reviendrait a jeter 316 fois trop. Le jeu de raies commun, lui, continue de
    venir de fit_weights_mask.

    Un facteur de 0.5 est |z| > clip * sqrt(2), soit 4.24 sigma a clip = 3.
    """
    fit = np.load(fit_path)
    if "clip_factor" not in getattr(fit, "files", []):
        return None
    factor = np.asarray(fit["clip_factor"], dtype=float)
    if not factor.size:
        return {}
    rows = np.asarray(fit["clip_rows"], dtype=np.int64)
    cols = np.asarray(fit["clip_cols"], dtype=np.int64)
    below = factor < float(floor)
    rows, cols = rows[below], cols[below]
    _, _, w, meta = _bcd.load_cube(cube, dtype=np.float32,
                                   columns=np.arange(0, 1))
    n_rows = w.shape[0]
    grid = np.asarray(fit["grid"]) if "grid" in fit.files else None
    n_cols = int(grid.size) if grid is not None else int(cols.max() + 1)
    parity = _bcd.row_parity(meta, n_rows)
    names = [os.path.basename(str(v)) for v in meta["filename"]]
    out = {}
    for i, name in enumerate(names):
        out.setdefault(name, np.zeros((2, n_cols), dtype=bool))
    for r, c in zip(rows, cols):
        if r < n_rows and c < n_cols:
            out[names[r]][int(parity[r]) % 2][c] = True
    return out


def fit_weights_mask(cube, mode="exposure", share=1.0):
    """The grid samples the fit weighted, per exposure: file name -> (2, grid).

    With mode "common" every exposure gets the same mask, the samples EVERY
    exposure weighted: a file otherwise blanks its own, so each epoch is
    measured on its own set of lines and the set moves from one to the next.
    On TOI-4552 that alone cost 1.8 m/s of rms with no correction applied at
    all, while the error per exposure did not move: a line set that moves is
    not lost information, it is scatter. The common mask costs 3.6 points more
    of each exposure there, 6.7 on Proxima.

    Exactly what panel 3 of the sequence figure shows and nothing else: the
    weights the fit read from the cube (twoframe.load_cube, with the figure's
    own defaults) and, as the figure does, not a sample stranded between gaps
    (plotting.live_mask). Row 0 is the even orders, row 1 the odd ones.
    """
    from .plotting import live_mask
    _, _, w, meta = _bcd.load_cube(cube, dtype=np.float32)
    parity = _bcd.row_parity(meta, w.shape[0])
    names = [os.path.basename(str(v)) for v in meta["filename"]]
    out = {}
    for i, name in enumerate(names):
        out.setdefault(name, np.zeros((2, w.shape[1]), dtype=bool))
        out[name][int(parity[i]) % 2] = live_mask(w[i:i + 1])[0]
    if str(mode) == "common":
        # one set of lines per CAMPAIGN, and a joint cube holds several of them.
        # Intersected over all of them instead, a sample lost by one night of the
        # faintest star was blanked in every exposure of the brightest: on
        # PROXIMA+GJ1+GJ3090 that took the H band of a corrected Proxima file
        # from 9.9% NaN to 17.1% (2026-09-13). Each object keeps its own.
        names_of = {}
        # an astropy Table or a FITS record array, depending on the caller
        columns = (list(getattr(meta, "colnames", None) or [])
                   or list(getattr(getattr(meta, "dtype", None), "names", None) or []))
        objects = (np.asarray(meta["object"]) if "object" in columns
                   else np.zeros(w.shape[0], dtype=int))
        for i, name in enumerate(names):
            names_of.setdefault(str(objects[i]), set()).add(name)
        # `share` EST LA FRACTION DE POSES QU'IL FAUT POUR GARDER UN
        # ECHANTILLON, et 1.0 est l'intersection stricte, celle d'avant.
        #
        # Pourquoi en faire un reglage: l'intersection exige l'accord de
        # TOUTES les poses, ce qui est peu couteux quand elles couvrent le
        # meme intervalle, et tres couteux quand elles ne le couvrent pas.
        # Mesure le 2026-10-15 sur tau Ceti: 21.7% de pixels a NaN avec la
        # grille livree, 43.0% avec input.to_observer, ou chaque pose est
        # decalee de son propre BERV et ou l'intersection de 85 intervalles
        # decales est bien plus petite que chacun d'eux.
        #
        # Baisser le seuil ne rend pas un masque par pose (mask: exposure,
        # essaye: 693 m/s, parce que chaque pose garde alors un AUTRE jeu de
        # raies): le jeu reste UNIQUE pour la campagne, il est seulement
        # decide a la majorite plutot qu'a l'unanimite.
        fraction = float(share)
        for group in names_of.values():
            if fraction >= 1.0:
                shared = np.ones((2, w.shape[1]), dtype=bool)
                for name in group:
                    shared &= out[name]
            else:
                votes = np.zeros((2, w.shape[1]), dtype=np.int32)
                for name in group:
                    votes += out[name]
                shared = votes >= max(1, int(np.ceil(fraction * len(group))))
            for name in group:
                out[name] = shared.copy()
    return out


def weighted_on_pixels(alive, wave, grid):
    """Which of an order's pixels fall on grid samples the fit weighted.

    A pixel counts as weighted when both grid samples around it were, the rule
    order_correction uses for the basis support. Pixels outside the grid are
    outside the fit altogether and come back True, since there is no panel 3
    there to match.
    """
    inside = np.isfinite(wave) & (wave >= grid[0]) & (wave <= grid[-1])
    near = np.clip(np.searchsorted(grid, np.where(inside, wave, grid[0])) - 1,
                   0, grid.size - 2)
    return ~inside | (alive[near] & alive[near + 1])


def correct_file(model, row, path, outdir, n_star=None, n_earth=None,
                 halfwidth=8, overwrite=False, max_sky=None, alive=None,
                 clipped=None, clip_nsig=None, refit=False, weight="flux",
                 clip_columns=None, config=None, keep_name=False):
    """Write a t.fits with the model divided out. Returns the new path.

    The model lives in `ln f - savgol(ln f)`, so removing it from the flux is a
    division, not a subtraction:

        f_corrected = f * exp(-model)

    with the model the components and, with any observer component, the parity
    mean the fit removed first (order_correction). That leaves the
    Savitzky-Golay continuum exactly where it was. That matters:
    the continuum was never part of the fit and never stored, so anything that
    claimed to reconstruct it would be inventing it.

    Every other extension is copied untouched, including Recon and the blaze, so
    the result is still a t.fits and still reducible by anything that reads one.
    Samples outside the basis support keep their original flux rather than being
    blanked: the correction has nothing to say there, which is not the same as
    the flux being unknown.
    """
    correction, k, j = correction_on_grid(model, row, n_star, n_earth, halfwidth)

    with sptf.robust_open(path) as hdulist:
        out = fits.HDUList([h.copy() for h in hdulist])
    # LE FICHIER CORRIGE EST DU MEME FORMAT QUE L'ORIGINAL. C'est une copie
    # dont on remplace le flux, et rien d'autre: un t.fits reste un t.fits,
    # un S2D de l'ESO reste un S2D, lisible par ce qui lisait le premier.
    # Reste a savoir OU est le flux, et le nom change d'un instrument a
    # l'autre: FluxAB sur SPIRou, SCIDATA sur HARPS. La table d'extensions ne
    # connait qu'APERO, donc pour les instruments lus par le pont c'est le
    # pont qui repond, et la grille vient de la classe de LBL plutot que
    # d'une extension dont le nom n'existe pas ici (2026-10-14).
    bridged = config is not None and not _readers.reads_natively(config)
    if bridged:
        payload = _readers.read_any(path, config)
        e_flux = payload["meta"].get("flux_extension")
        if e_flux is None:
            raise SystemExit(
                "%s: impossible de retrouver l'extension du flux, donc rien"
                " n'est ecrit plutot que d'ecrire au mauvais endroit"
                % os.path.basename(path))
        flux = np.asarray(out[e_flux].data, dtype=np.float64)
        wave = np.atleast_2d(np.asarray(payload["wave"], dtype=np.float64))
    else:
        # the same instrument table the reader uses; a SPIRou file must have
        # its AB extensions corrected, not the single-fibre A that also exists
        e_flux, e_wave, _, _ = sptf.extensions_for(out)
        flux = np.asarray(out[e_flux].data, dtype=np.float64)
        wave = np.asarray(out[e_wave].data, dtype=np.float64)
    n_orders = flux.shape[0]
    touched = 0
    blanked = 0
    n_clipped = 0
    for order in range(n_orders):
        got = order_correction(model, correction, j, order, wave[order])
        if got is None:
            continue
        values, live = got
        flux[order] = np.where(live, flux[order] * np.exp(-values), flux[order])
        if alive is not None:
            # NaN wherever the fit gave the sample no weight: exactly the
            # samples panel 3 of the sequence figure hides. The model there was
            # constrained by nothing, and a flux corrected by it is not one.
            bad = ~weighted_on_pixels(alive[order % 2], wave[order], model["grid"])
            blanked += int((bad & np.isfinite(flux[order])).sum())
            flux[order][bad] = np.nan
        if clipped is not None:
            # NaN wherever the residual, panel 5, put a grid sample beside the
            # pixel beyond the clip: the model explains the rest of the order
            # and not that, whatever it is
            cut = ~weighted_on_pixels(~clipped[order % 2], wave[order], model["grid"])
            n_clipped += int((cut & np.isfinite(flux[order])).sum())
            flux[order][cut] = np.nan
        touched += int(live.sum())

    # Propagate the sky mask: samples the airglow drowned were excluded from
    # the fit, so leaving them in the corrected file would hand LBL data the
    # model never saw and could not correct. NaN, not zero: a zero is a
    # measurement of no flux and would be believed.
    masked = 0
    if max_sky and not bridged:
        # aucun spectrographe du visible ne livre l'emission du ciel par pose:
        # il n'y a pas d'extension a lire, et max_sky est sans objet
        e_sky = sptf.sky_extension_for(out)
        if e_sky:
            sky = np.asarray(out[e_sky].data, dtype=np.float64)
            if sky.shape == flux.shape:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.abs(sky) / np.where(flux > 0, flux, np.nan)
                drown = np.isfinite(ratio) & (ratio > float(max_sky))
                flux[drown] = np.nan
                masked = int(drown.sum())

    out[e_flux].data = flux
    head = out[0].header
    if max_sky:
        head["PCA2SKYR"] = (float(max_sky), "sky/flux above this was set to NaN")
        head["PCA2SKYN"] = (masked, "samples removed as sky-dominated")
    head["PCA2REF"] = (True, "two-frame PCA correction applied")
    head["PCA2BERV"] = (float(row["berv"]), "km/s used to carry the star basis")
    if "vstar" in getattr(row, "colnames", row):
        # the companions' part of it (registration.orbits); row is a Table
        #   Row or, after a refit, a plain dict
        head["PCA2VKEP"] = (float(row["vstar"]), "km/s, Keplerian of the star")
    head["PCA2NPIX"] = (touched, "samples corrected")
    head["PCA2MEAN"] = (mean_divided(model, j),
                        "observer-frame parity mean divided out")
    if alive is not None:
        head["PCA2WNAN"] = (blanked, "samples the fit gave no weight, set to NaN")
    if clipped is not None and clip_columns and clip_columns.get("excess"):
        head["PCA2ESIG"] = (float(clip_columns["excess"]),
                            "observer column dropped above this excess sigma")
        head["PCA2ECHI"] = (float(clip_columns.get("excess_chi2") or 0),
                            "and above this windowed chi2 of the column")
        head["PCA2EWID"] = (int(clip_columns.get("excess_samples") or 0),
                            "that chi2 averaged over this many samples")
        head["PCA2ECOL"] = (int(clip_columns.get("excess_columns") or 0),
                            "columns standing still in the observer frame")
    if clipped is not None and clip_columns and clip_columns.get("excursion"):
        head["PCA2XSIG"] = (float(clip_columns["excursion"]),
                            "excursion flagged beyond this aggregate sigma")
        head["PCA2XWID"] = (int(clip_columns.get("samples") or 0),
                            "widest excursion window, in grid samples")
        head["PCA2XRHO"] = (float(clip_columns.get("rho1") or 0),
                            "the residual's autocorrelation at lag 1")
    if clip_columns and clip_columns.get("floor"):
        head["PCA2WFLR"] = (float(clip_columns["floor"]),
                            "NaN below this fraction of the fit's own weight")
    if clipped is not None:
        head["PCA2RSIG"] = (float(clip_nsig or 0), "residual clip, in running robust sigma")
        head["PCA2RNAN"] = (n_clipped, "samples beyond it in panel 5, set to NaN")
    # the columns every exposure lost together, and the two thresholds that
    # sent them: a file says by itself whether it was cleaned this way
    if clip_columns and clip_columns.get("frac"):
        head["PCA2CFRC"] = (float(clip_columns.get("frac") or 0),
                            "column goes above this clipped fraction")
        head["PCA2CCHI"] = (float(clip_columns.get("chi2") or 0),
                            "and above this reduced chi2 of the survivors")
        head["PCA2CCOL"] = (int(clip_columns.get("columns") or 0),
                            "observer columns dropped from every exposure")
    head["PCA2REJ"] = (bool(row["rejected"]), "exposure was MAD-rejected")
    head["PCA2RFIT"] = (bool(refit), "coefficients solved for this file, basis fixed")
    head["PCA2WGHT"] = (str(weight), "metric the amplitudes were measured in")
    head["PCA2SHRK"] = (model.get("shrink") is not None,
                        "observer comps kept only where significant")
    head["PCA2SSIG"] = (bool(model.get("shrink_smooth")),
                        "significance averaged over a resolution element")
    head["PCA2SMTH"] = (",".join(str(j) for j in model.get("smoothed_components") or []),
                        "observer comps smoothed before division")
    from .provenance import stamp as code_stamp
    head["PCA2GIT"] = (code_stamp(), "pca2d commit that wrote it, + if modified")
    for key, value, comment in coefficient_cards(model, row, k, j):
        head[key] = (value, comment)
    head.add_history("two-frame PCA: %d star-frame + %d observer-frame"
                     " components%s divided out"
                     % (k, j, " and the parity mean" if j else ""))
    head.add_history("f_corrected = f * exp(-model), model in ln f - savgol(ln f)")

    stem = os.path.basename(path)
    stem = stem[:-len("t.fits")] if stem.endswith("t.fits") else stem.rsplit(".", 1)[0]
    new = os.path.join(outdir, "%st_%d-%d.fits" % (stem, k, j))
    if bridged:
        # LBL retrouve ses petits par le nom du fichier: sa classe HARPS_ESO
        # cherche des *_S2D_A.fits, et un fichier appele ..._S2D_At_0-7.fits
        # ne serait plus lu par personne. On garde donc le suffixe que
        # l'instrument attend et on met la signature avant lui.
        base, _, tail = stem.rpartition("_")
        if base and tail:
            new = os.path.join(outdir, "%s_%d-%d_%s.fits" % (base, k, j, tail))
    if keep_name:
        # LE NOM DE LA SOURCE, TEL QUEL: le dossier science/<objet>_pca2d dit
        # deja ce que sont ces fichiers, et LBL les trouve avec le motif de
        # l'instrument, comme ceux de science/<objet>_tc
        new = os.path.join(outdir, os.path.basename(path))
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(new) and not overwrite:
        raise SystemExit("%s exists; pass --overwrite" % new)
    out.writeto(new, overwrite=True)
    out.close()
    return new, touched, flux.size


def write_fits(path, wave, recon, parity, row, model, source):
    primary = fits.PrimaryHDU()
    primary.header["SOURCE"] = (os.path.basename(source), "t.fits rebuilt")
    primary.header["CONTENT"] = (model["content"], "ln f - savgol(ln f)")
    primary.header["BERV"] = (float(row["berv"]), "km/s, used for the star shift")
    primary.header["BJD"] = (float(row["bjd"]), "")
    primary.header["REJECTED"] = (bool(row["rejected"]), "MAD cut")
    hdus = [primary,
            fits.ImageHDU(recon, name="MODEL"),
            fits.ImageHDU(wave, name="WAVE"),
            fits.ImageHDU(parity.astype(np.int16), name="PARITY")]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    log("wrote %s" % path)


def plot(path, wave, recon, row, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    finite = np.isfinite(recon)
    if not finite.any():
        log("nothing to plot: the model covers none of this file")
        return
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(wave[finite], recon[finite], ".", ms=0.6, alpha=0.4, color="tab:blue")
    axes[0].set_xlabel("wavelength (nm)")
    axes[0].set_ylabel("model, ln f - savgol(ln f)")
    axes[0].set_title("%s   BERV %.3f km/s" % (os.path.basename(source), row["berv"]))
    # one order of each parity, side by side, to show the wrapping is right
    orders = [o for o in range(recon.shape[0]) if np.isfinite(recon[o]).sum() > 100]
    for o in orders[len(orders) // 2: len(orders) // 2 + 2]:
        axes[1].plot(wave[o], recon[o], lw=0.7,
                     label="order %d (%s)" % (o, "even" if o % 2 == 0 else "odd"))
    axes[1].set_xlabel("wavelength (nm)")
    axes[1].set_ylabel("model")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log("wrote %s" % path)


def night_index(coeffs):
    """Map a night number to its COEFFS row.

    The night is floor(BJD - 0.5), the same convention the cube builder uses, so
    an exposure taken after midnight belongs to the night it started in.
    """
    nights = {}
    for i, bjd in enumerate(np.asarray(coeffs["bjd"], dtype=float)):
        nights[int(np.floor(bjd - 0.5))] = i
    return nights


def night_row_names(cube):
    """Le nom que le cube donne a chaque nuit: {nuit: nom de sa ligne}.

    Sur un cube empile par nuit, la ligne d'une nuit porte le nom de sa
    PREMIERE pose (meta.fits, colonne filename), et tout ce qui est calcule par
    ligne (les coupures du residu, le plancher de la boucle) est range sous ce
    nom. La nuit est floor(BJD - 0.5), la convention de night_index et du
    constructeur du cube. Les deux parites d'une nuit partagent le meme nom.
    """
    from astropy.table import Table
    meta = Table.read(os.path.join(cube, "meta.fits"))
    names = {}
    for name, bjd in zip(meta["filename"], np.asarray(meta["bjd"], dtype=float)):
        names.setdefault(int(np.floor(bjd - 0.5)), os.path.basename(str(name)))
    return names


def member_row_names(cube):
    """{exposure: the name of its cube row}, from the cube's members.csv, or
    None for a cube written before there was one.

    The exact membership, as the coadder recorded it: what an exposure is
    matched by once a night is cut into bundles (nights.split_nights), where
    its night alone no longer says which row it went into. On a cube of whole
    nights it says what night_row_names says.
    """
    from .cube import MEMBERS_FILE
    path = os.path.join(cube, MEMBERS_FILE) if cube else None
    if not path or not os.path.exists(path):
        return None
    members = Table.read(path, format="ascii.csv")
    return {str(f): str(r) for f, r in zip(members["filename"], members["row"])}


def cut_key(path, row, names_by_night, members=None):
    """Sous quel nom chercher les coupures d'une pose.

    Sans cube empile (names_by_night None), le nom du fichier lui-meme: chaque
    pose a sa ligne. Avec, le nom de la ligne de SA nuit, retrouvee par le BJD
    de la ligne de coefficients que rows_by_night lui a donnee. Avant le
    2026-10-17, c'etait toujours le nom du fichier, si bien que seule la
    premiere pose de chaque nuit recevait les coupures (mesure: GJ 1, 2658
    echantillons coupes sur la premiere pose et 0 sur les autres; Proxima
    7790 contre 0).
    """
    own = os.path.basename(path)
    if members:
        return members.get(own, own)
    if not names_by_night:
        return own
    night = int(np.floor(float(row["bjd"]) - 0.5))
    return names_by_night.get(night, own)


def rows_by_night(model, args, config=None):
    """One (path, row) pair per t.fits in --source-dir, matched on its night.

    The row's coefficients are the night's; the BERV is the exposure's own,
    read from its header, because the star basis has to be placed where that
    exposure actually saw the star and exposures within a night differ by a
    fraction of a km/s. With orbits in the configuration it is the star's
    frame of that exposure, the BERV composed with the companions' velocity
    (readers.star_frame), as everywhere else.
    """
    import glob
    coeffs = model["coeffs"]
    index = night_index(coeffs)
    # BY MEMBERSHIP when the cube says which row each exposure went into:
    # with bundles a night has several rows, and its night is not enough
    members = member_row_names(getattr(args, "cube", None))
    by_row = {}
    if members:
        for i, name in enumerate(coeffs["filename"]):
            by_row[os.path.basename(str(name))] = i
    out, missing = [], 0
    for path in sorted(glob.glob(os.path.join(args.source_dir, "*t.fits"))):
        try:
            payload = _readers.star_frame(sptf.read_tfits(path), config)
        except Exception:                                     # noqa: BLE001
            missing += 1
            continue
        meta = payload["meta"]
        if members:
            name = members.get(os.path.basename(path))
            if name is None or name not in by_row:
                missing += 1
                continue
            row = Table(coeffs)[by_row[name]]
            row["berv"] = float(meta["berv"])  # this exposure, not the row
            if "vstar" in row.colnames:
                # 0 for a star with no orbits: a joint member beside one that
                #   has some (TOI904 beside GJ_1028, 2026-10-21)
                row["vstar"] = float(meta.get("vstar", 0.0))
            out.append((path, row))
            continue
        night = int(np.floor(float(meta["bjd"]) - 0.5))
        if night not in index:
            missing += 1
            continue
        row = Table(coeffs)[index[night]]
        row["berv"] = float(meta["berv"])      # this exposure, not the night
        if "vstar" in row.colnames:
            row["vstar"] = float(meta.get("vstar", 0.0))
        out.append((path, row))
    return out, missing


#: The weight a sample keeps in the amplitude fit when it carries no velocity
#: information at all, as a fraction of the median velocity weight. Not zero: a
#: component living only where the star is flat would have no amplitude at all,
#: and an unconstrained amplitude is divided out of the flux like any other.
VELOCITY_FLOOR = 0.05


def velocity_weights(w, derivative, floor=VELOCITY_FLOOR):
    """`w` weighted by (dT/dv)^2: where the measurement of velocity lives.

    What a contaminant does to a radial velocity is set by its overlap with the
    DERIVATIVE of the star, since that is what LBL projects a residual on: a
    contaminant that is flat where the star has structure moves no line. So the
    amplitudes of the observer components are measured in that metric rather
    than on the flux, where a broad residual with no velocity content weighs as
    much as a line wing.

    The weights are normalised by the WEIGHTED MEAN of (dT/dv)^2, so their
    total is the total they had and the covariance stays the size it was, and
    floored, so that a column the star says nothing about still constrains a
    component a little. The mean and not the median: most of a spectrum is
    between lines, so the median derivative is nearly zero and dividing by it
    turns the handful of samples that carry the information into infinities.
    Samples with no weight keep none: this changes what the fit BELIEVES, never
    which samples exist.
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(derivative, dtype=float) ** 2
    live = w > 0
    if not live.any():
        return w
    total = float(w[live].sum())
    scale = float((w[live] * g[live]).sum()) / total if total > 0 else 0.0
    if not np.isfinite(scale) or scale <= 0:
        return w                       # a star with no structure: nothing to say
    out = w * (g / scale + float(floor))
    out[~live] = 0.0
    return out


def star_derivative(carried, fwhm):
    """d/d(sample) of the star as this exposure sees it, row by row.

    The Savitzky-Golay derivative of the template-making filter, never
    np.gradient: on noisy data the difference is not cosmetic.
    """
    from .resolution import smooth

    out = np.zeros_like(np.asarray(carried, dtype=float))
    for row in range(out.shape[0]):
        if np.any(carried[row]):
            out[row] = smooth(np.asarray(carried[row], dtype=float),
                              float(fwhm), deriv=1)
    return out


def refit_row(model, path, config, shifter, base_row, weight="flux",
              floor=VELOCITY_FLOOR, fwhm=None):
    """This exposure's own coefficients, against the campaign's fixed basis.

    The alternative, which this replaces, was to hand every exposure of a night
    the night's coefficients and change only its BERV. That is defensible for
    the star, which does not vary over three hours, and wrong for everything
    the observer block describes: OH and O2 airglow rise and fall through a
    night by factors, and the water column moves with the airmass. A visit of
    four exposures was being corrected with one sky amplitude.

    The basis P and Q is NOT refitted, only the amplitudes a and b. Refitting
    the basis per exposure would be a different method, and a single spectrum
    does not constrain five components of anything.

    Returns a copy of `base_row` with fresh a1..ak, b1..bj and this exposure's
    BERV, or None if the file cannot be resampled onto the grid.
    """
    grid = model["grid"]
    payload = _readers.read_any(path, config)
    values, sigma, trans, good, n_written, segments = sptf.resample_exposure(
        payload, grid, config)
    if not n_written or not good.any():
        return None, None

    # LE MEME PASSE-HAUT QUE LE CUBE, et pas un autre. Avec
    # highpass.against_template, resample_exposure rend du ln(flux) brut et
    # laisse le filtre a la seconde passe; il faut donc la refaire ici, avec
    # LE MEME template, sans quoi on retranche des composantes ajustees sur un
    # residu passe-haut a un spectre qui ne l'est pas. Vu le 2026-10-16: les
    # valeurs sortaient de mediane +10.14 au lieu de +0.017, et le bruit des
    # spectres corriges doublait (0.0148 contre 0.0075) pendant que les river
    # plots, eux, restaient propres.
    if (config.get("highpass") or {}).get("against_template"):
        from . import cube as _cube
        from . import preprocess as _prep

        template = _cube.load_template(config)
        if template is None:
            raise SystemExit(
                "highpass.against_template is on but the cube kept no"
                " template: rebuild the cube (--rebuild-cube) so that"
                " `correct` filters the spectra the same way the fit did")
        reg = config["registration"]
        berv = float(payload["meta"]["berv"])
        here = np.empty_like(template)
        for parity in range(template.shape[0]):
            finite = np.isfinite(template[parity])
            shifted, shifted_good = _prep.register(
                grid, np.where(finite, template[parity], 0.0), finite, 0.0,
                berv, grid, spline_order=reg["spline_order"],
                mask_threshold=reg["mask_threshold"])
            here[parity] = np.where(shifted_good, shifted, np.nan)
        hip = config["highpass"]
        out = np.zeros_like(values)
        keep = np.zeros(good.shape, dtype=bool)
        for parity, start, stop in segments:
            seg, seg_good = _prep.highpass_against_template(
                values[parity, start:stop], good[parity, start:stop],
                here[parity, start:stop], hip["window"], hip["polyorder"],
                already_log=True)
            out[parity, start:stop] = seg
            keep[parity, start:stop] = seg_good
        values, good = np.where(keep, out, 0.0), keep

    # The SAME weights the fit used, not merely 1/sigma^2. twoframe.load_cube
    # also zeroes anything below ln_clip_low, ramps by the telluric
    # transmission and blanks the grid edges, and the transmission ramp in
    # particular matters here: it is near zero exactly where the atmosphere is
    # opaque, which is where the observer block does most of its work. Leaving
    # it out shifted the refitted amplitude by tens of per cent against the
    # fit's own, which is a method difference masquerading as a measurement.
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(good & np.isfinite(sigma) & (sigma > 0),
                     1.0 / np.where(sigma > 0, sigma, 1.0) ** 2, 0.0)
    w[~np.isfinite(values)] = 0.0
    w[values < LN_CLIP_LOW] = 0.0
    if trans is not None:
        w *= np.clip((trans - RAMP_ZERO) / (1.0 - RAMP_ZERO), 0.0, 1.0)
    w[:, :_bcd.EDGE] = 0.0
    w[:, -_bcd.EDGE:] = 0.0
    # and, as the fit does, nothing whose Lanczos taps reach a hole in the star
    # basis: the model there is built partly from samples the fit never
    # constrained (twoframe.gap_guard, against the support the basis has)
    berv = float(payload["meta"]["berv"])
    delta = np.full(2, -float(pixel_shift(berv, model["dv"])))
    _bcd.gap_guard(w, delta, 8, verbose=False, live=star_support(model))
    data = np.where(w > 0, values, 0.0)

    # take out exactly what the fit took out before it solved for coefficients:
    # the mean it stored per order parity, and the template carried into this
    # exposure's frame
    names = list(model["means"].keys())
    for parity in (0, 1):
        data[parity] -= model["means"][names[group_of(model, parity, len(names))]]
    berv = float(payload["meta"]["berv"])
    delta = np.full(2, -float(pixel_shift(berv, model["dv"])))
    carried = np.zeros_like(data)
    for parity in (0, 1):
        template = template_for(model, parity)
        if template is not None and np.any(template):
            carried[parity] = _bcd.carry_template(template, shifter,
                                                  delta[parity:parity + 1])[0]
    data -= carried
    data[w <= 0] = 0.0

    # The metric the amplitudes are measured in. `flux` is every sample as the
    # fit saw it; `velocity` weighs each sample by the star's own derivative
    # there, which is what decides whether a contaminant moves a line at all.
    # The correction itself is unchanged: this is how b is MEASURED, not what
    # is divided out, and the mask below is still the flux weights'.
    solve_w = w
    if str(weight) == "velocity" and fwhm:
        solve_w = velocity_weights(w, star_derivative(carried, fwhm), floor)

    tie = np.zeros(2, dtype=int)          # the two parities are one exposure
    # No velocity column here on purpose: this path re-solves ONE exposure
    # against a basis the fit already fixed, and the shift it would find has
    # nowhere to go. What the fit found is in the COEFFS row, and that is what
    # the header reports.
    a, b, _, _ = _bcd.joint_coeffs(data, solve_w, model["P"], model["Q"],
                                   shifter, delta, exposure=tie)
    if base_row is None:
        return None, None
    # this exposure's own no-weight mask, by the rule fit_weights_mask applies
    # to the cube's rows: on a cube of nights those rows are not this file
    from .plotting import live_mask
    alive = np.vstack([live_mask(w[parity:parity + 1])[0] for parity in (0, 1)])
    # a plain dict, not a Table Row: a Row is a view into its table, so writing
    # coefficients into it would edit the fit's own coefficient table and the
    # next exposure would inherit them
    try:
        row = {name: base_row[name] for name in base_row.colnames}
    except AttributeError:
        row = dict(base_row)
    for k in range(model["n_star"]):
        row["a%d" % (k + 1)] = float(a[0, k])
    for j in range(model["n_earth"]):
        row["b%d" % (j + 1)] = float(b[0, j])
    row["berv"] = berv
    if "vstar" in payload["meta"] or "vstar" in row:
        # 0 for a star with no orbits, whatever a joint table holds there
        row["vstar"] = float(payload["meta"].get("vstar", 0.0))
    return row, alive


def shrink_model(model, fits_path, cube, shrink=True, shrink_smooth=False,
                 smooth_which=(), resolution=None):
    """Give the model the observer basis the correct stage divides out.

    Each observer component kept at a column only where it is significant
    there, all exposures together (pca2d.shrink): not where one exposure's
    share of it is below its noise, since a pattern at half a sigma in every
    exposure is well detected in the basis. The components asked for smoothed
    over a resolution
    element first, their significance taken after. Held apart from Q, which a
    refit still solves against, and applied where the correction is built
    (correction_on_grid). The same function the sequence figure's panels 3 and
    6 call. Returns (the chi2 scale, the FWHM or None, the factors).
    """
    from .resolution import fwhm_samples
    from .shrink import correction_basis
    fit_path = os.path.join(os.path.dirname(os.path.abspath(fits_path)), "fit.npz")
    if not cube or not os.path.exists(fit_path):
        raise SystemExit("--shrink and --smooth-components need --cube and the"
                         " fit's archive beside %s: %s" % (fits_path, fit_path))
    b_rows = np.asarray(np.load(fit_path)["b"], dtype=float)
    _, _, w_cube, _ = _bcd.load_cube(cube, dtype=np.float32)
    if b_rows.shape[0] != w_cube.shape[0]:
        raise SystemExit("the fit has %d rows and the cube %d"
                         % (b_rows.shape[0], w_cube.shape[0]))
    chi2 = np.asarray(model["coeffs"]["chi2_red"], dtype=float)
    good = np.isfinite(chi2) & (chi2 > 0)
    scale = float(np.median(chi2[good])) if good.any() else 1.0
    fwhm = fwhm_samples(resolution, model["dv"]) if resolution else None
    Q = np.asarray(model["Q"], dtype=float)
    Q_correct, factors = correction_basis(Q, b_rows, w_cube, scale, shrink,
                                          shrink_smooth, smooth_which, fwhm)
    del w_cube
    model["Q_correct"] = Q_correct
    model["shrink"] = factors if shrink else None
    model["shrink_smooth"] = shrink_smooth
    model["smoothed_components"] = [j + 1 for j in smooth_which]
    return scale, fwhm, factors


def correct_many(model, args):
    """--correct over one file or over every exposure in the fit."""
    if (args.all or args.by_night) and not args.source_dir:
        raise SystemExit("--source-dir is required with --all or --by-night:"
                         " where the delivered t.fits are")
    # the configuration first: rows_by_night places each exposure in the
    #   star's frame it describes (its orbits)
    config = None
    if args.config:
        from .config import load_config
        config = load_config(args.config)
    if args.by_night:
        pairs, missing = rows_by_night(model, args, config)
        log("  %d exposures matched to a fitted row, %d skipped"
              % (len(pairs), missing))
    elif args.all:
        names = [os.path.basename(str(v)) for v in model["coeffs"]["filename"]]
        pairs = [(os.path.join(args.source_dir, n), None) for n in names]
    else:
        pairs = [(args.file, None)]
    covered = []
    shifter = None
    # LA CONFIGURATION SERT AUSSI A ECRIRE, PAS SEULEMENT A REAJUSTER. Elle dit
    # par quel lecteur le spectre a ete lu, donc quelle extension porte le flux
    # a remplacer et dans quel repere sont ses longueurs d'onde. Elle n'etait
    # chargee que sous --refit, si bien qu'une correction sans reajustement
    # (correct.weight: flux) retombait sur la table d'extensions d'APERO et
    # s'arretait sur « unsupported INSTRUME 'HARPS' » (2026-10-14).
    if getattr(args, "weight", "flux") == "velocity":
        # measuring b in another metric means measuring it again
        args.refit = True
    if args.refit:
        if not args.config:
            raise SystemExit("--refit needs --config, the YAML the cube was"
                             " built from: the exposure has to be resampled and"
                             " high-passed exactly as the fit saw it")
        m = model["grid"].size
        top = int(np.ceil(np.abs(np.asarray(model["coeffs"]["berv"],
                                            dtype=float)).max()
                          / model["dv"])) + 2
        shifter = _bcd.LanczosShifter(m, a=8, max_shift=top)
        log("  refitting coefficients per exposure against the fixed basis")
    elif args.by_night:
        log("  WARNING: every exposure of a night gets that night's"
              " coefficients. The sky is not constant over a night; --refit"
              " solves for each exposure's own amplitudes")

    # which of the fit's star spectra are this object's, when several were
    # fitted together against one observer basis (pca2d.joint)
    model["star_group"] = int(getattr(args, "star_group", 0) or 0)
    alive_by_file = None
    mask_mode = str(getattr(args, "mask", None) or "exposure")
    if mask_mode == "none":
        log("  --mask none: a sample the fit gave no weight keeps its delivered"
            " flux rather than being blanked, so nothing is lost and nothing"
            " there is corrected either", "warn")
    elif getattr(args, "cube", None):
        share = float(getattr(args, "mask_share", 1.0) or 1.0)
        if mask_mode == "common" and share < 1.0:
            what = ("every sample fewer than %.0f%% of the exposures weighted,"
                    " one line set for them all" % (100 * share))
        elif mask_mode == "common":
            what = "every sample no exposure weighted, one line set for them all"
        else:
            what = "every sample the fit gave no weight, as panel 3 hides it"
        log("  reading the fit's weights from %s: %s will be NaN"
            % (args.cube, what))
        alive_by_file = fit_weights_mask(args.cube, mask_mode,
                                         getattr(args, "mask_share", 1.0) or 1.0)
    else:
        log("  no --cube: samples the fit gave no weight to keep their flux,"
            " which panel 3 of the sequence figure does not show", "warn")
    # ce que la BOUCLE a deprecie sous le plancher: un masque par pose, lu du
    # fit lui-meme et non recalcule ici
    floor = getattr(args, "weight_floor", None)
    floored_by_file = None
    if floor:
        fit_path = os.path.join(os.path.dirname(os.path.abspath(args.fits)),
                                "fit.npz")
        if not getattr(args, "cube", None) or not os.path.exists(fit_path):
            raise SystemExit("--weight-floor a besoin de --cube et du fit"
                             " a cote de %s" % args.fits)
        floored_by_file = clipped_weights_mask(args.cube, fit_path, floor)
        if floored_by_file is None:
            log("  le fit ne porte pas le facteur de la boucle (fit d'avant"
                " 2026-09-26): --weight-floor ne peut rien masquer", "warn")
        else:
            total = sum(int(v.sum()) for v in floored_by_file.values())
            log("  la boucle a deprecie %d echantillons sous %.2f de leur poids"
                " nominal, soit %.0f par pose: ils deviennent NaN"
                % (total, float(floor), total / max(len(floored_by_file), 1)),
                "value")
    clipped_by_file = None
    clip_report = None
    excursion = getattr(args, "excursion_nsig", None)
    samples = getattr(args, "excursion_samples", None)
    excess = getattr(args, "excess_nsig", None)
    wide = getattr(args, "excess_samples", None)
    bias = getattr(args, "bias_nsig", None)
    bias_wide = getattr(args, "bias_samples", None)
    if (getattr(args, "nsig_cut", None) or (excursion and samples)
            or (excess and wide) or (bias and bias_wide)):
        if not getattr(args, "cube", None):
            raise SystemExit("--nsig-cut and --excursion-nsig need --cube: the"
                             " residual they read is the cube less the model")
        from .outliers import residual_outliers
        fit_path = os.path.join(os.path.dirname(os.path.abspath(args.fits)), "fit.npz")
        if not os.path.exists(fit_path):
            raise SystemExit("--nsig-cut needs the fit's archive beside %s: %s"
                             % (args.fits, fit_path))
        if excursion and samples:
            log("  flagging every excursion of the residual, panel 5, from 1 to"
                " %d samples wide, beyond %.1f sigma of aggregate significance"
                % (int(samples), float(excursion)))
        if getattr(args, "nsig_cut", None):
            log("  clipping the residual, panel 5, beyond %.1f running robust"
                " sigmas over %d samples" % (args.nsig_cut, args.clip_window))
        frac = getattr(args, "column_frac", None)
        chi2 = getattr(args, "column_chi2", None)
        if frac and chi2:
            from .outliers import expected_chi2
            # what noise itself leaves behind: a per-sample clip takes the
            # tails with it and the survivors carry less than one, while
            # flagging whole excursions barely touches the noise at all
            null = (expected_chi2(args.nsig_cut) if getattr(args, "nsig_cut", None)
                    else 1.0)
            log("  and dropping from every exposure each observer column where"
                " more than %.0f%% of them is flagged and the survivors'"
                " reduced chi2 is still above %.2f (noise gives %.3f)"
                % (100 * float(frac), float(chi2), null))
        if excess and wide:
            log("  and looking for what stands still in the observer's frame:"
                " the columns whose scatter over the exposures, averaged over"
                " %d samples, is %.0f sigmas above the noise%s"
                % (int(wide), float(excess),
                   " and above a chi2 of %.2f" % float(args.excess_chi2)
                   if getattr(args, "excess_chi2", None) else ""))
        clipped_by_file, clip_report = residual_outliers(
            args.cube, np.load(fit_path), args.nsig_cut, args.clip_window,
            column_frac=frac, column_chi2=chi2,
            excursion_nsig=excursion, excursion_samples=samples,
            excess_nsig=excess, excess_chi2=getattr(args, "excess_chi2", None),
            excess_samples=wide, bias_nsig=bias, bias_samples=bias_wide)
        if clip_report["rho"]:
            rho = clip_report["rho"]
            log("  the residual's own autocorrelation: rho_1 = %.3f, rho_2 ="
                " %.3f, rho_4 = %.3f, a correlation length of %.1f samples"
                % (rho[0], rho[1] if len(rho) > 1 else float("nan"),
                   rho[3] if len(rho) > 3 else float("nan"),
                   1 + 2 * sum(rho)), "value")
            if clip_report["variance"]:
                wide = clip_report["variance"][-1]
                log("    so a window of %d samples has variance %.1f, i.e. %.1f"
                    " independent samples, and a flat excursion needs %.2f sigma"
                    " of depth to reach %.1f"
                    % (int(samples), wide, int(samples) ** 2 / wide,
                       float(excursion) * wide ** 0.5 / int(samples),
                       float(excursion)), "value")
        log("  %d grid samples flagged, over %d exposures"
            % (sum(int(v.sum()) for v in clipped_by_file.values())
               - clip_report["added"]
               - (clip_report["excess"] or {}).get("added", 0),
               len(clipped_by_file)), "value")
        drift = clip_report.get("bias")
        if drift:
            log("  %d des %d colonnes mesurees portent un residu MOYEN"
                " significatif sur les poses, %.2f%% du domaine; c'est %d"
                " echantillons de plus sur les %d poses"
                % (drift["n_columns"], drift["measured"],
                   100.0 * drift["n_columns"] / max(drift["measured"], 1),
                   drift["added"], len(clipped_by_file)), "value")
        got = clip_report["excess"]
        if got:
            log("  %d of the %d measured columns stand still in the observer's"
                " frame, %.2f%% of them, over a correlation length of %.1f"
                " samples; that is %d more samples over the %d exposures"
                % (got["n_columns"], got["measured"],
                   100.0 * got["n_columns"] / max(got["measured"], 1),
                   got["length"], got["added"], len(clipped_by_file)), "value")
        if frac and chi2:
            seen = clip_report["seen"]
            measured = int((seen > 0).sum())
            log("  %d of the %d measured columns go whole, %.2f%% of them; that"
                " is %d more samples over the %d exposures"
                % (clip_report["n_columns"], measured,
                   100.0 * clip_report["n_columns"] / max(measured, 1),
                   clip_report["added"], len(clipped_by_file)), "value")
    # what the header will say about the column test and the weight floor,
    # once per run
    column_cards = None
    if floor:
        column_cards = dict(column_cards or {}, floor=float(floor))
    if clip_report:
        column_cards = dict(column_cards or {}, excursion=excursion,
                            samples=samples, rho1=(clip_report["rho"] or [0])[0])
        if clip_report.get("excess"):
            column_cards.update(
                excess=clip_report["excess"]["thresholds"][0],
                excess_chi2=clip_report["excess"]["thresholds"][1],
                excess_columns=clip_report["excess"]["n_columns"],
                excess_samples=clip_report["excess"]["samples"])
        if clip_report.get("thresholds"):
            column_cards.update(frac=clip_report["thresholds"][0],
                                chi2=clip_report["thresholds"][1],
                                columns=clip_report["n_columns"])
    smooth_which = [int(v) - 1 for v in
                    str(getattr(args, "smooth_components", None) or "").split(",")
                    if v.strip()]
    shrink = bool(getattr(args, "shrink", False))
    shrink_smooth = bool(getattr(args, "shrink_smooth", False))
    if (shrink_smooth or smooth_which) and not getattr(args, "resolution", None):
        raise SystemExit("--shrink-smooth and --smooth-components are measured in"
                         " resolution elements: pass --resolution")
    if shrink_smooth and not shrink:
        raise SystemExit("--shrink-smooth smooths the significance --shrink uses:"
                         " pass --shrink")
    if any(j < 0 or j >= model["n_earth"] for j in smooth_which):
        raise SystemExit("--smooth-components: the fit has %d observer components"
                         % model["n_earth"])
    if shrink or smooth_which:
        # the observer basis as the files will have it divided out
        # (shrink_model), by the function the sequence figure's panels 3 and 6
        # call
        scale, fwhm, factors = shrink_model(model, args.fits,
                                            getattr(args, "cube", None), shrink,
                                            shrink_smooth, smooth_which,
                                            getattr(args, "resolution", None))
        Q = np.asarray(model["Q"], dtype=float)
        live = np.any(Q != 0, axis=0)
        log("  observer components shrunk by their significance at each column,"
            " the weights scaled by the fit's median reduced chi2, %.2f%s%s"
            % (scale, "; significance averaged over %d samples" % fwhm
               if shrink_smooth else "",
               "; components %s smoothed over %d samples first"
               % (",".join(str(j + 1) for j in smooth_which), fwhm)
               if smooth_which else ""))
        for j, factor in enumerate(factors):
            kept = factor[live]
            log("    component %d%s: kept in full (factor > 0.9) at %.1f%% of the"
                " columns, dropped (0) at %.1f%%, mean factor %.2f"
                % (j + 1, " (smoothed)" if j in smooth_which else "",
                   100 * np.mean(kept > 0.9), 100 * np.mean(kept == 0),
                   float(kept.mean())), "value")
    # LE SUPPORT TELLURIQUE (diagnostic). Applique a la base que l'on DIVISE,
    # pas a celle contre laquelle le reajustement resout: comme le
    # retrecissement, il change ce qui sort des fichiers, pas la mesure des
    # amplitudes.
    q_support = getattr(args, "q_support", None)
    if q_support:
        if not getattr(args, "cube", None):
            raise SystemExit("--q-support needs --cube: the transmission it"
                             " reads is the cube's trans.npy")
        from .diagnostics import support_columns
        trans = np.load(os.path.join(args.cube, "trans.npy"), mmap_mode="r")
        base = model.get("Q_correct")
        base = np.asarray(model["Q"] if base is None else base, dtype=float)
        if trans.shape[1] != base.shape[1]:
            raise SystemExit("the cube's transmission has %d columns and the"
                             " basis %d" % (trans.shape[1], base.shape[1]))
        keep = support_columns(trans, float(q_support))
        model["Q_correct"] = base * keep[None, :]
        live = np.any(base != 0, axis=0)
        log("  --q-support %.4f: the observer basis is divided out only where"
            " the transmission falls below it on some exposure, %.1f%% of the"
            " columns the basis covers" % (float(q_support),
                                           100.0 * keep[live].mean()
                                           if live.any() else 0.0), "value")
        del trans
    # the derivative's width, one resolution element of this instrument on this
    # grid, exactly as every other Savitzky-Golay here is measured
    velocity_fwhm = None
    if getattr(args, "weight", "flux") == "velocity":
        from .resolution import fwhm_samples
        if not args.resolution:
            raise SystemExit("--weight velocity needs --resolution: the"
                             " derivative is taken over one resolution element,"
                             " and c/R is what says how many samples that is")
        velocity_fwhm = fwhm_samples(float(args.resolution), float(model["dv"]))
        log("amplitudes measured on (dT/dv)^2 weights, the derivative over %d"
            " samples of %.2f km/s, with a floor of %.2f: what a contaminant"
            " does to a velocity is its overlap with the star's derivative"
            % (velocity_fwhm, float(model["dv"]),
               float(getattr(args, "velocity_floor", VELOCITY_FLOOR))), "info")
    # sur un cube empile par nuit, les coupures sont rangees par nuit
    names_by_night = None
    members = None
    if args.by_night and getattr(args, "cube", None) and (
            clipped_by_file is not None or floored_by_file is not None):
        members = member_row_names(args.cube)
        if members:
            log("  the cube's rows are coadds: each exposure gets the cuts of"
                " the row it went into, %d rows for %d exposures"
                % (len(set(members.values())), len(members)), "info")
        else:
            names_by_night = night_row_names(args.cube)
            log("  the cube's rows are nights: each exposure gets the cuts of"
                " its night's row, %d nights" % len(names_by_night), "info")
    written = skipped = refitted = 0
    permute = getattr(args, "permute_b", None)
    prepared = []
    progress = _bar(pairs, desc="correcting", unit="file")
    for path, preset in progress:
        if not os.path.exists(path):
            log("  missing, skipped: %s" % path)
            skipped += 1
            continue
        row = preset if preset is not None else exposure_row(model["coeffs"], path)
        own_alive = None
        if args.refit:
            fresh, own_alive = refit_row(
                model, path, config, shifter, row,
                weight=getattr(args, "weight", "flux"),
                floor=getattr(args, "velocity_floor", VELOCITY_FLOOR),
                fwhm=velocity_fwhm)
            if fresh is None:
                log("  could not resample, skipped: %s" % os.path.basename(path))
                skipped += 1
                continue
            row = fresh
            refitted += 1
        # a refitted exposure has its own weights and so its own no-weight
        # mask; on a cube of nights the cube's rows are not this file
        alive = mask_for(mask_mode, own_alive, alive_by_file,
                         os.path.basename(path))
        # LES COUPURES DE LA NUIT, PAS CELLES DU FICHIER: sur un cube empile
        # elles sont rangees sous le nom de la ligne de la nuit (cut_key)
        key = cut_key(path, row, names_by_night, members)
        clipped = (clipped_by_file.get(key)
                   if clipped_by_file is not None else None)
        if floored_by_file is not None:
            floored = floored_by_file.get(key)
            if floored is not None:
                clipped = floored if clipped is None else (clipped | floored)
        if permute is not None:
            # les amplitudes d'une autre nuit ne sont connues qu'une fois
            # toutes les poses reajustees: on garde, on ecrit apres
            prepared.append((path, row, alive, clipped))
            continue
        new, touched, total = correct_file(
            model, row, path, args.corrected_dir, args.n_star, args.n_earth,
            args.kernel_halfwidth, args.overwrite, max_sky=args.max_sky_ratio,
            alive=alive, clipped=clipped, clip_nsig=getattr(args, "nsig_cut", None),
            refit=bool(args.refit), weight=getattr(args, "weight", "flux"),
            clip_columns=column_cards, config=config,
            keep_name=getattr(args, "keep_name", False))
        written += 1
        # onto the bar, not onto its own line: three hundred of these scroll
        # the narration off the screen and say nothing a total cannot
        covered.append(100.0 * touched / max(total, 1))
        if hasattr(progress, "set_postfix_str"):
            progress.set_postfix_str("%s  %.0f%% covered"
                                     % (os.path.basename(new), covered[-1]))
    if permute is not None and prepared:
        from .diagnostics import night_derangement
        nights = [int(np.floor(float(r["bjd"]) - 0.5)) for _, r, _, _ in prepared]
        source = night_derangement(nights, permute)
        log("  --permute-b %d: every exposure gets the observer amplitudes of"
            " an exposure from ANOTHER night, %d exposures over %d nights. A"
            " null test, not a correction" % (permute, len(prepared),
                                              len(set(nights))), "warn")
        n_earth_fit = int(model["n_earth"])
        # LES AMPLITUDES D'ORIGINE, FIGEES AVANT D'EN ECRIRE UNE SEULE: sinon
        # une pose servie apres son donneur recevrait ce que le donneur a deja
        # recu d'un troisieme
        original = np.array([[float(r["b%d" % (j + 1)]) for j in range(n_earth_fit)]
                             for _, r, _, _ in prepared])
        for i, (path, row, alive, clipped) in enumerate(prepared):
            for j in range(n_earth_fit):
                row["b%d" % (j + 1)] = float(original[int(source[i]), j])
            new, touched, total = correct_file(
                model, row, path, args.corrected_dir, args.n_star, args.n_earth,
                args.kernel_halfwidth, args.overwrite,
                max_sky=args.max_sky_ratio, alive=alive, clipped=clipped,
                clip_nsig=getattr(args, "nsig_cut", None),
                refit=bool(args.refit), weight=getattr(args, "weight", "flux"),
                clip_columns=column_cards, config=config,
            keep_name=getattr(args, "keep_name", False))
            written += 1
            covered.append(100.0 * touched / max(total, 1))
    if covered:
        import numpy as _np
        log("  samples corrected per file: median %.1f%%, worst %.1f%%"
              % (float(_np.median(covered)), float(_np.min(covered))))
    log("wrote %d file%s to %s%s%s"
          % (written, "" if written == 1 else "s", args.corrected_dir,
             ", %d missing" % skipped if skipped else "",
             ", %d refitted individually" % refitted if refitted else ""))
    return written


def main(argv=None):
    args = parse_args(argv)
    model = load_model(args.fits)
    log("%s: %d star + %d earth components, dv = %.4f km/s, content = %s"
          % (os.path.basename(args.fits), model["n_star"], model["n_earth"],
             model["dv"], model["content"]))
    if args.correct:
        k = model["n_star"] if args.n_star is None else args.n_star
        j = model["n_earth"] if args.n_earth is None else args.n_earth
        log("correcting: removing %d star + %d Earth components -> t_%d-%d.fits"
              % (k, j, k, j))
        # not `return correct_many(...)`: this is a console_script entry
        # point, so the count of files written would become the exit status.
        # 316 corrected spectra exited 316.
        correct_many(model, args)
        return None
    row = exposure_row(model["coeffs"], args.file)
    log("  BJD %.5f   BERV %+.4f km/s   star shift %+.2f samples"
          % (row["bjd"], row["berv"], -row["berv"] / model["dv"]))
    wave, recon, parity = reconstruct(model, row, args.kernel_halfwidth, args.file)
    good = np.isfinite(recon)
    log("  rebuilt %d of %d samples (%.1f%%) over %d orders"
          % (good.sum(), recon.size, 100 * good.mean(), recon.shape[0]))
    log("  model rms %.5f in ln flux" % np.nanstd(recon[good]))
    if args.out:
        write_fits(args.out, wave, recon, parity, row, model, args.file)
    if args.plot:
        plot(args.plot, wave, recon, row, args.file)
    return None          # exit status, not a value; see above


if __name__ == "__main__":
    main()
