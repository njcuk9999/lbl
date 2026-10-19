"""The LBL template and variability basis, from the star block of the fit.

The first star component, at its mean amplitude, is a high-passed template of
the star: fitted to every exposure at once, in the barycentric frame, with the
observer block already describing the atmosphere. That is what LBL's template
step spends its time estimating. Each order parity gets its own: the star block
plus that parity's weighted mean of what the fit left (star_coverage), since
LBL measures an order against the template of its parity. Written by LBL's own writer, in LBL's template
format and under the name LBL looks for, it takes the place of that step for
the corrected object, which is what makes pca2d a front end to LBL rather than
a stage in front of it.

The further star components, when the fit has any, are directions in which the
star's spectrum varies. LBL already projects every line's residual on vectors
of that kind, its RESPROJ tables: DTEMP3000 and the rest are temperature
gradients of model spectra. STRPCA2..N are the same object measured on the
star itself, and LBL reports their amplitude per exposure the way it reports
DTEMP, as a column of the rdb and its error.

FRAMES. The template is written in the barycentric frame, as LBL's own is; LBL
takes it to the star's rest frame itself, by the systemic velocity its mask
step measures (SYSTVELO in the mask header). It does not do that to a RESPROJ
table, which it evaluates as it stands, in the rest frame. So the STRPCA
vectors are written shifted by that same velocity, and can only be written
once the mask exists.

WHERE IT GOES. The template is made once per fit, beside the run's other
outputs (star_template.fits), and copied from there to where LBL looks for the
corrected object's template. A template LBL built itself is never replaced; one
of ours is, when the fit it came from has changed, and PCA2FIT in its header
(the fit's time and size) is how that is known.
"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .grids import doppler, pixel_shift
from .plotting import live_mask
from .twoframe import (LanczosShifter, carried_means, cube_grid, fit_means,
                       fit_templates, load_cube, row_parity, star_model)

#: what changes a template beyond the fit it came from, in its PCA2FIT stamp:
#: 2 adds each parity's residual mean (star_coverage)
TEMPLATE_VERSION = 2

#: the header keyword of a template written here. A template without it was
#: written by LBL, and is never overwritten from here.
PROVENANCE = "PCA2TPL"

#: the TEMPLATE columns LBL's own writer reads out of its `props`
COLUMNS = ("flux", "eflux", "rms", "flux_odd", "eflux_odd", "rms_odd",
           "flux_even", "eflux_even", "rms_even")


def mean_star(fit):
    """ln f of the star per order parity (even, odd), and what it is made of.

    Each star component at its mean amplitude over the exposures the fit kept,
    plus the star-frame mean of each parity the fit subtracted, if it had one
    (zero unless --mean iterate or the one-shot template). Returns
    (ln_star (2, M), mean amplitudes (K,), components P (K, M)).
    """
    files = list(getattr(fit, "files", fit.keys()))
    a = np.asarray(fit["a"], dtype=float)
    P = np.asarray(fit["P"], dtype=float)
    keep = (~np.asarray(fit["rejected"], dtype=bool) if "rejected" in files
            else np.ones(len(a), dtype=bool))
    abar = a[keep].mean(axis=0)
    if "templates" in files:
        T = np.atleast_2d(np.asarray(fit["templates"], dtype=float))
    elif "template" in files:
        T = np.atleast_2d(np.asarray(fit["template"], dtype=float))
    else:
        T = np.zeros((1, P.shape[1]))
    if T.shape[0] == 1:
        T = np.repeat(T, 2, axis=0)
    return (abar @ P)[None, :] + T[:2], abar, P


def star_coverage(cube, fit, chunk=16, block=40000):
    """How well the fit saw each star-frame sample per parity, and what it left.

    The weights the fit read from the cube (twoframe.load_cube, its defaults),
    carried into the star's frame by each exposure's own shift and kept only
    where plotting.live_mask keeps them, the rule the report's panels use;
    rows the MAD cut rejected are left out. Beside them, each parity's
    weighted mean of what the whole model leaves (the data less the star
    block, its star-frame means, the observer block and the parity offset),
    carried the same way.

    That mean is the part of the star each parity sees and the star block does
    not. Under the default --mean offset the block is one spectrum for both
    parities, while the two sample a line at different places on the order and
    so at different resolutions, and LBL measures each order against the
    template of its own parity. On TOI-2120 LBL's own template differs between
    parities at line scales by 0.014 in ln f, a third of the line structure,
    and against a template whose parities were one spectrum LBL's velocity
    errors came out 20% larger. The velocity term is not in the model
    subtracted: its mean over the exposures moves both parities by the same
    velocity, which LBL takes as an offset.

    The cube is read `block` columns at a time, each with a margin wider than
    the largest shift and the Lanczos kernel, so that the columns kept from it
    are computed exactly as from the whole cube. The whole of it in float32 is
    2.9 GB on TOI-2120, and the lbl stage makes this template while the next
    run's fit holds its own copy: the machine swapped until one sweep of that
    fit took ten times as long. `block=None` reads it whole.

    Returns (grid, rows that saw each sample (2, M), their summed weight
    (2, M), rows per parity (2,), residual mean (2, M), NaN where unseen).
    """
    grid = np.asarray(cube_grid(cube))
    m = grid.size
    _, _, w1, meta = load_cube(cube, dtype=np.float32, columns=np.arange(0, 1))
    n = w1.shape[0]
    if len(fit["berv"]) != n:
        raise SystemExit("the fit has %d rows and the cube %d: it was not made on"
                         " this cube" % (len(fit["berv"]), n))
    parity = row_parity(meta, n) % 2
    files = list(getattr(fit, "files", fit.keys()))
    keep = (~np.asarray(fit["rejected"], dtype=bool) if "rejected" in files
            else np.ones(n, dtype=bool))
    delta = -pixel_shift(np.asarray(fit["berv"], dtype=float), float(fit["dv"]))
    reach = int(np.ceil(np.abs(delta).max())) + 2
    margin = reach + 16                     # the shift, and the kernel's 8 on either side
    P, a = np.asarray(fit["P"], dtype=float), np.asarray(fit["a"], dtype=float)
    Q, b = np.asarray(fit["Q"], dtype=float), np.asarray(fit["b"], dtype=float)
    means, group = fit_means(fit, meta, n, m)
    T, tgroup = fit_templates(fit, meta, n, m)
    count, wsum, rsum = np.zeros((2, m)), np.zeros((2, m)), np.zeros((2, m))
    step = m if block is None else int(block)
    for c0 in range(0, m, step):
        c1 = min(c0 + step, m)
        a0, b0 = max(0, c0 - margin), min(m, c1 + margin)
        _, data, w, _ = load_cube(cube, dtype=np.float32, columns=np.arange(a0, b0))
        shifter = LanczosShifter(b0 - a0, a=8, max_shift=reach)
        Tf = shifter.prepare(T[:, a0:b0]) if np.any(T[:, a0:b0]) else None
        inner = slice(c0 - a0, c1 - a0)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            rows = slice(start, stop)
            model = star_model(P[:, a0:b0], a[rows], shifter, delta[rows],
                               stop - start, b0 - a0)
            if Tf is not None:
                model += carried_means(Tf, tgroup, shifter, delta, start, stop)
            model += b[rows] @ Q[:, a0:b0]
            model += means[:, a0:b0][group[rows]]
            weighted = np.where(w[rows] > 0, w[rows] * (data[rows] - model), 0.0)
            del model
            carried = shifter.rows(w[rows], -delta[rows])
            carried_r = shifter.rows(weighted, -delta[rows])
            live = live_mask(carried) & keep[rows, None]
            carried = np.where(live, carried, 0.0)[:, inner]
            carried_r = np.where(live, carried_r, 0.0)[:, inner]
            live = live[:, inner]
            for p in (0, 1):
                sel = parity[rows] == p
                if sel.any():
                    count[p, c0:c1] += live[sel].sum(axis=0)
                    wsum[p, c0:c1] += carried[sel].sum(axis=0)
                    rsum[p, c0:c1] += carried_r[sel].sum(axis=0)
        del data, w
    per_parity = np.array([np.sum(keep & (parity == p)) for p in (0, 1)])
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = np.where(wsum > 0, rsum / np.where(wsum > 0, wsum, 1.0), np.nan)
    return grid, count, wsum, per_parity, resid


def template_columns(ln_star, count, wsum, rows, min_fraction=0.5):
    """The TEMPLATE table's flux, eflux and rms, overall and per parity.

    flux is exp(ln f) of the star: a high-passed template with its continuum at
    one, NaN wherever fewer than `min_fraction` of a parity's exposures saw the
    sample. rms is what LBL means by it, the spread of one exposure about the
    template, here from the weights the fit used: flux over the square root of
    the mean inverse variance of the exposures that saw the sample. eflux is
    rms, as LBL writes it. The overall columns take whichever parity saw a
    sample, and the two weighted by their weight where both did.
    """
    out = {}
    seen = []
    for p, name in ((0, "even"), (1, "odd")):
        ok = (count[p] >= min_fraction * max(rows[p], 1)) & (wsum[p] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma = np.where(ok, 1.0 / np.sqrt(wsum[p] / np.maximum(count[p], 1)), np.nan)
        flux = np.where(ok, np.exp(ln_star[p]), np.nan)
        out["flux_" + name] = flux
        out["rms_" + name] = flux * sigma
        out["eflux_" + name] = out["rms_" + name].copy()
        seen.append(ok)
    wt = [np.where(seen[p], wsum[p], 0.0) for p in (0, 1)]
    nt = [np.where(seen[p], count[p], 0.0) for p in (0, 1)]
    total = wt[0] + wt[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ln = np.where(total > 0, (ln_star[0] * wt[0] + ln_star[1] * wt[1])
                      / np.where(total > 0, total, 1.0), np.nan)
        flux = np.where(total > 0, np.exp(ln), np.nan)
        out["flux"] = flux
        out["rms"] = flux / np.sqrt(total / np.maximum(nt[0] + nt[1], 1))
    out["eflux"] = out["rms"].copy()
    return out


def lbl_instrument(config_file, object_name):
    """LBL's reader for this spectrograph, set on one object.

    The same call lbl.check_profile makes, so the template is written by the
    very class that will read it back.
    """
    from lbl.instruments import select
    args = select.parse_args(
        ["INSTRUMENT", "DATA_DIR", "DATA_SOURCE", "DATA_TYPE", "INPUT_FILE"],
        dict(config_file=os.path.abspath(config_file)), __name__, parse=False)
    inst = select.load_instrument(args, plogger=None)
    for key in ("OBJECT_SCIENCE", "OBJECT_COMPARISON"):
        inst.params[key] = object_name
    return inst


def lbl_paths(inst):
    """(template, mask, models folder): where LBL keeps them for the object."""
    from lbl.instruments import select
    dirs = select.make_all_directories(inst)
    return (inst.template_file(dirs["TEMPLATE_DIR"], "science", required=False),
            inst.mask_file(dirs["MODEL_DIR"], dirs["MASK_DIR"], required=False),
            dirs["MODEL_DIR"])


def write_template(inst, path, grid, columns, science_files, dv_ms, provenance):
    """The star template, in LBL's format and by LBL's own writer.

    Every column and header keyword LBL writes is here, and made the way LBL
    makes it: the Savitzky-Golay columns by its calculate_savgol_template (its
    velocity code reads those, not `flux`), the science table row by row by its
    populate_sci_table, the primary header from the last science file. VSYS is
    zero for every exposure: LBL puts there the offset its template step used
    to register each spectrum, and the fit registers none.
    """
    savgol = inst.calculate_savgol_template(
        dv_grid=float(dv_ms),
        flux_dict={key: columns[key] for key in ("flux", "flux_odd", "flux_even")})
    sci_table, bervs = {}, []
    for filename in science_files:
        header = inst.load_header(filename)
        berv = inst.get_berv(header)
        bervs.append(berv)
        sci_table = inst.populate_sci_table(filename, sci_table, header, berv=berv)
    sci_table["VSYS"] = np.zeros(len(science_files))
    _, refhdr = inst.load_science_file(science_files[-1])
    for key, (value, comment) in provenance.items():
        refhdr.__setitem__(key, value, comment)
    coverage = len(np.unique(np.asarray(bervs) // 1000))
    props = dict(wavelength=np.asarray(grid, dtype=float),
                 template_coverage=coverage, total_nobs_berv=coverage,
                 template_nobs=len(science_files), savgol_fluxes=savgol,
                 template_type="LBL_SAVGOL" if savgol else "LBL_NON_SAVGOL",
                 **{key: columns[key] for key in COLUMNS})
    inst.write_template(path, props, refhdr, sci_table)
    return path


def is_ours(path):
    """Whether a template on disk was written here rather than by LBL."""
    try:
        return bool(fits.getheader(path, 0).get(PROVENANCE))
    except (OSError, KeyError):
        return False


def mask_systemic(mask_file):
    """The systemic velocity, m/s, that LBL's mask step measured (SYSTVELO)."""
    return float(fits.getheader(mask_file, 0)["SYSTVELO"])


def write_strpca(models_dir, prefix, grid, P, flux, systemic_ms, run=""):
    """The star's variability directions as LBL RESPROJ tables, STRPCA2..N.

    One table per star component past the first, with LBL's two columns.
    `wavelength` is the fit's barycentric grid taken to the star's rest frame
    by the mask's systemic velocity, exactly as LBL takes the template there
    (lbl.core.math.doppler_shift, which is grid / grids.doppler(v)).
    `fractional_gradient` is the change of the normalised flux per unit of that
    component's amplitude, flux * P_k: LBL's projection of a line's residual on
    it is then that component's amplitude in the exposure less its mean, the
    fit's own a_k measured line by line. `flux` is written beside it, as the
    DTEMP tables carry one. A sample the fit did not see has a gradient of
    zero, not NaN, since LBL would interpolate a NaN straight across the gap.
    Returns RESPROJ_TABLES, name -> file name in `models_dir`.
    """
    rest = np.asarray(grid, dtype=float) / doppler(float(systemic_ms) / 1000.0)
    seen = np.isfinite(flux)
    tables = {}
    for k in range(1, P.shape[0]):
        name = "STRPCA%d" % (k + 1)
        filename = "%s_%s.fits" % (name, prefix)
        table = Table()
        table["wavelength"] = rest
        table["flux"] = flux
        table["fractional_gradient"] = np.where(seen, flux * P[k], 0.0)
        table.meta.update({"PCA2COMP": k + 1, "SYSTVELO": float(systemic_ms),
                           "PCA2RUN": str(run)[:60]})
        table.write(os.path.join(models_dir, filename), overwrite=True)
        tables[name] = filename
    return tables


def _aliased(source):
    """Whether LBL source divides the residual through an alias of it."""
    import re
    return bool(re.search(r"^\s*frac_diff_seg\s*=\s*diff_seg\s*$", source, re.M)
                and re.search(r"^\s*frac_diff_seg\s*/=", source, re.M))


def _divided_in_place(source):
    """Whether a numba kernel divides the residual itself, element by element.

    LBL's speed branch moved the line loop into lbl/core/fastmath.py and kept
    the division exactly (its commits are bit-identical): `diff_seg[i] /=`
    inside the loop over the tables, with no alias left to find.
    """
    import re
    return bool(re.search(r"^\s*diff_seg\[\w+\]\s*/=", source, re.M))


def resproj_divides_in_place(package_dir=None):
    """Whether an LBL projects every RESPROJ table after the first on a
    residual it has already divided.

    lbl.science.general writes `frac_diff_seg = diff_seg` and then
    `frac_diff_seg /= (b_ratio_seg * norm_seg)`: an alias and not a copy, so
    each table divides the residual once more. The first table is projected
    right and every later one is not; the per-line RMSRATIO and CHI2 it then
    writes to lblrv use the divided residual too (nothing in LBL reads those
    back, so the velocities are untouched). Read from LBL's own source, so a
    fixed LBL stops being warned about.

    `package_dir` is the folder of the LBL package to read, the one that will
    run (lbl.chosen); None is the one this process imports. Its speed branch
    does the same division in lbl/core/fastmath.py, which is read too.
    """
    import inspect
    try:
        if package_dir is None:
            from lbl.science import general
            general_file = inspect.getsourcefile(general)
            package_dir = os.path.dirname(os.path.dirname(general_file))
        found = False
        for rel, test in ((os.path.join("science", "general.py"), _aliased),
                          (os.path.join("core", "fastmath.py"),
                           _divided_in_place)):
            path = os.path.join(package_dir, rel)
            if os.path.exists(path):
                with open(path) as handle:
                    found = found or test(handle.read())
        return found
    except Exception:                                          # noqa: BLE001
        return False


def fit_stamp(fit_path):
    """What identifies a fit on disk: its modification time and its size."""
    stat = os.stat(fit_path)
    return "%s %d" % (time.strftime("%Y-%m-%dT%H:%M:%S",
                                    time.localtime(stat.st_mtime)), stat.st_size)


def stamp(path):
    """The PCA2FIT a template carries, None if it has none."""
    try:
        return fits.getheader(path, 0).get("PCA2FIT")
    except OSError:
        return None


def star_fwhm(fit):
    """The FWHM, in samples, the fit smoothed its star side to (fit.npz
    star_fwhm, from twoframe.star_smooth); 0 when it did not, or when the fit
    predates the key."""
    files = list(getattr(fit, "files", fit.keys()))
    return int(fit["star_fwhm"]) if "star_fwhm" in files else 0


def template_stamp(fit_path):
    """PCA2FIT: the fit's stamp and the template rule, so that a template made
    from the same fit by an older rule is made again."""
    # a template made from a smoothed star carries the width in its stamp, so
    # one is never taken for the other and the smoothed one is made afresh
    fwhm = star_fwhm(np.load(fit_path))
    return "%s t%d%s" % (fit_stamp(fit_path), TEMPLATE_VERSION,
                         " s%d" % fwhm if fwhm else "")


def parity_split(columns):
    """rms of ln(flux_even / flux_odd) wherever the template has both parities:
    how differently the two see the star, as written for LBL."""
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.log(columns["flux_even"]) - np.log(columns["flux_odd"])
    both = np.isfinite(d)
    return float(np.std(d[both])) if both.any() else float("nan")


def build(cube, fit_path, config_file, object_name, science_files, path, run=""):
    """The fit's star template, written to `path` by LBL's writer.

    Each parity is the star block at its mean amplitude plus that parity's
    residual mean (star_coverage). Returns (fraction of the grid the template
    covers, star components, rms of the even-minus-odd residual means).
    """
    fit = np.load(fit_path)
    ln_star, _, P = mean_star(fit)
    grid, count, wsum, rows, resid = star_coverage(cube, fit)
    # Where the fit smoothed its star side (twoframe.star_smooth), the part of
    # the template it did not fit, each parity's residual mean, goes through
    # the same filter at the same width: added raw, it would hand LBL back
    # exactly the structure finer than the resolution the smoothing took out
    fwhm = star_fwhm(fit)
    if fwhm:
        from .resolution import smooth
        resid = np.array([np.where(np.isfinite(r),
                                   smooth(np.where(np.isfinite(r), r, 0.0), fwhm),
                                   np.nan) for r in resid])
    ln_star = ln_star + np.where(np.isfinite(resid), resid, 0.0)
    columns = template_columns(ln_star, count, wsum, rows)
    inst = lbl_instrument(config_file, object_name)
    from .provenance import stamp as code_stamp
    provenance = {
        PROVENANCE: (str(run)[:60], "LBL template from the pca2d star block"),
        "PCA2FIT": (template_stamp(fit_path), "the fit.npz it came from, and rule"),
        "PCA2NSTR": (int(P.shape[0]), "star components in that fit"),
        "PCA2GIT": (code_stamp(), "pca2d commit that wrote it, + if modified"),
    }
    if os.path.exists(path):
        os.remove(path)
    write_template(inst, path, grid, columns, science_files,
                   1000.0 * float(fit["dv"]), provenance)
    return (float(np.isfinite(columns["flux"]).mean()), int(P.shape[0]),
            parity_split(columns))


def place(made, slot):
    """Put the template made for a run where LBL looks for it.

    Returns what was done: "copied" into an empty slot, "same" when the slot
    already holds this very template, "replaced" when it held one of ours from
    another fit, "theirs" when LBL made the one there, which is left alone.
    """
    if os.path.exists(slot):
        if not is_ours(slot):
            return "theirs"
        if stamp(slot) == stamp(made):
            return "same"
        done = "replaced"
    else:
        done = "copied"
        os.makedirs(os.path.dirname(os.path.abspath(slot)), exist_ok=True)
    part = slot + ".part"
    shutil.copyfile(made, part)
    os.replace(part, slot)
    return done


def strpca_from(fit, template, mask, models_dir, prefix, run=""):
    """write_strpca from the files a run leaves: its fit, its template, LBL's mask.

    What run_lbl.py calls between LBL's mask step and its velocities, since the
    rest frame the tables are written in is the one that mask measured.
    """
    if not os.path.exists(mask):
        raise SystemExit("no LBL mask at %s, and the STRPCA tables are written in"
                         " the rest frame it measures: LBL's mask step has to"
                         " run first" % mask)
    P = np.asarray(np.load(fit)["P"], dtype=float)
    table = Table.read(template, hdu="TEMPLATE")
    grid = np.asarray(table["wavelength"], dtype=float)
    if grid.size != P.shape[1]:
        raise SystemExit("%s has %d samples and the fit's components %d: they are"
                         " not from the same run" % (template, grid.size, P.shape[1]))
    return write_strpca(models_dir, prefix, grid, P,
                        np.asarray(table["flux"], dtype=float),
                        mask_systemic(mask), run)
