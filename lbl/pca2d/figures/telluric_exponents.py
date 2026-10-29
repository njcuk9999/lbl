"""The telluric exponents of LBL: water, the dry component, and the dry
component against the airmass.

`lbl_telluclean` fits two exponents per exposure and writes them in the
header of the cleaned spectrum: TLPEH2O for water, TLPEOTR for everything
else (O2, CO2, CH4 together, and not per chemical species). This page shows
them.

    page 1  the two time series, water and the dry component, and the
            airmass below them for comparison
    page 2  the dry component AGAINST the airmass, and the residual below

WHAT PAGE 2 CHECKS. The dry species are well mixed in the atmosphere, so
their column follows the airmass and the cloud of points must fall on a
line. BUT NOT NECESSARILY ON THE y = x LINE: TAPAS is computed for a
reference column, and an observatory at altitude has less of it (La Silla
is at 2400 m), which moves the intercept without touching the slope. The
fitted line is therefore drawn with ITS slope and ITS offset, both free,
and the slope 1 through zero is recalled beside it for the eye. The
residual is taken against the fitted line, which is the model being
tested.

If the dry component was FIXED to the airmass in the fit, which the
shipped configuration does, the cloud is exactly on the line and the
residual is zero: the page then says so rather than suggesting a
measurement.

    python telluric_exponents.py --science DIR --out PAGE.pdf
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

RJD0 = 2400000.0
#: the deviation below which the dry component is taken as not fitted at all
FIXED_TOL = 1e-9


def read_exponents(directory, modes=None):
    """(rjd, water, dry, airmass, mode) of each cleaned spectrum in the
    directory.

    `modes` filters on HIERARCH ESO INS MODE: two modes of the same
    spectrograph are two instruments, and mixing them on one figure suggests
    a dispersion that is only the difference between the two.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.fits"))):
        try:
            header = fits.getheader(path)
        except Exception:                                     # noqa: BLE001
            continue
        if "TLPEH2O" not in header:
            continue
        mode = str(header.get("HIERARCH ESO INS MODE", "")).strip()
        if modes and mode not in modes:
            continue
        bjd = header.get("HIERARCH ESO QC BJD")
        rows.append((
            float(bjd) - RJD0 if bjd is not None else np.nan,
            float(header["TLPEH2O"]),
            float(header.get("TLPEOTR", np.nan)),
            float(header.get("HIERARCH ESO TEL AIRM START", np.nan)),
            mode,
        ))
    if not rows:
        return None
    return dict(
        rjd=np.array([r[0] for r in rows]),
        water=np.array([r[1] for r in rows]),
        dry=np.array([r[2] for r in rows]),
        airmass=np.array([r[3] for r in rows]),
        mode=np.array([r[4] for r in rows]),
    )


def robust_sigma(values):
    """The ROBUST standard deviation: 1.4826 times the median absolute
    deviation.

    Used everywhere rather than the standard deviation: one outlying
    exposure is enough to make a dispersion unreadable, and the factor makes
    the two coincide on a Gaussian, so the number reads as a sigma.
    """
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    if good.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(good - np.median(good))))


def dry_is_fixed(data):
    """Was the dry component fixed to the airmass rather than fitted?"""
    good = np.isfinite(data["dry"]) & np.isfinite(data["airmass"])
    if not good.any():
        return False
    return bool(np.max(np.abs(data["dry"][good] - data["airmass"][good]))
                < FIXED_TOL)


def fit_line(x, y):
    """(slope, intercept) of the least squares, BOTH free.

    The intercept is free because an observatory at altitude sees less dry
    column than the TAPAS reference: this offset is physical and must be
    able to show up instead of being absorbed by the slope.
    """
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan, np.nan
    design = np.vstack([x[good], np.ones(int(good.sum()))]).T
    slope, intercept = np.linalg.lstsq(design, y[good], rcond=None)[0]
    return float(slope), float(intercept)


def draw(data, out):
    fixed = dry_is_fixed(data)
    with PdfPages(out) as pdf:
        # ---- page 1: the time series
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(data["rjd"], data["water"], "o", ms=4, color="C0")
        axes[0].set_ylabel("H2O exponent")
        axes[0].set_title("the water column, fitted exposure by exposure"
                          " (TLPEH2O)", fontsize=10)
        axes[1].plot(data["rjd"], data["dry"], "o", ms=4, color="C1")
        axes[1].set_ylabel("dry exponent")
        axes[1].set_title("everything else as one block, not per chemical"
                          " species (TLPEOTR)%s"
                          % ("   FIXED to the airmass" if fixed else ""),
                          fontsize=10)
        axes[2].plot(data["rjd"], data["airmass"], "o", ms=4, color="0.4")
        axes[2].set_ylabel("airmass")
        axes[2].set_xlabel("rjd")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle("The telluric exponents of lbl_telluclean, %d exposures"
                     % len(data["rjd"]))
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- page 2: the dry exponent against the airmass, the residual,
        # the histogram
        fig = plt.figure(figsize=(11, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1],
                              hspace=0.05, wspace=0.25)
        axes = [fig.add_subplot(gs[0, 0])]
        axes.append(fig.add_subplot(gs[1, 0], sharex=axes[0]))
        ax_hist = fig.add_subplot(gs[:, 1])
        x, y = data["airmass"], data["dry"]
        axes[0].plot(x, y, "o", ms=5, color="C1", label="the exposures")
        span = np.linspace(np.nanmin(x) * 0.98, np.nanmax(x) * 1.02, 10)
        axes[0].plot(span, span, "k--", lw=1.0,
                     label="slope 1 through the origin")
        slope, intercept = fit_line(x, y)
        if np.isfinite(slope):
            axes[0].plot(span, slope * span + intercept, "-", lw=1.4,
                         color="C3",
                         label="fitted: %.3f x airmass %+.3f"
                               % (slope, intercept))
            resid = y - (slope * x + intercept)
        else:
            resid = np.full_like(y, np.nan)
        axes[0].set_ylabel("dry exponent")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].plot(x, resid, "o", ms=5, color="C3")
        axes[1].axhline(0.0, color="k", lw=1.0, ls=":")
        axes[1].set_xlabel("airmass")
        axes[1].set_ylabel("residual")
        axes[1].grid(alpha=0.3)

        # THE HISTOGRAM OF THE RESIDUAL, to see the dispersion at a glance.
        # Measured on 2026-10-16 on the 92 exposures of the HAM mode, the dry
        # component fitted freely: the dry/airmass ratio is 0.961 with a MAD
        # of 0.026, that is, the dry column follows the airmass to within
        # 2.6%. The width of this histogram IS that dispersion.
        good = np.isfinite(resid)
        if good.sum() > 3:
            ax_hist.hist(resid[good], bins=max(8, int(np.sqrt(good.sum()))),
                         orientation="horizontal", color="C3", alpha=0.75)
            spread = robust_sigma(resid[good])
            for sign in (-1, 1):
                ax_hist.axhline(sign * spread, color="k", lw=0.9, ls="--")
            ax_hist.axhline(float(np.median(resid[good])), color="k", lw=1.0)
            ax_hist.set_title("MAD %.4f" % spread, fontsize=9)
        ax_hist.set_xlabel("exposures")
        ax_hist.grid(alpha=0.3)
        ax_hist.set_ylim(axes[1].get_ylim())
        plt.setp(axes[0].get_xticklabels(), visible=False)
        if fixed:
            note = ("THE DRY EXPONENT IS FIXED to the airmass in the fit: the"
                    " cloud is on the line by CONSTRUCTION\nand the residual"
                    " is zero. This page checks the constraint, it does not"
                    " measure it.")
        else:
            note = ("the dry exponent is FITTED freely; residual against the"
                    " fitted line: MAD %.4f (raw rms %.4f)\na nonzero"
                    " intercept is expected, the observatory being at"
                    " altitude"
                    % (robust_sigma(resid), float(np.nanstd(resid, ddof=1))))
        fig.suptitle("The dry exponent against the airmass\n%s" % note,
                     fontsize=10)
        # no tight_layout: the gridspec of this page already carries its
        # spacings, and tight_layout overrides them while warning that it
        # does so
        pdf.savefig(fig)
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science", required=True,
                        help="the directory of the CLEANED spectra"
                             " (<OBJECT>_tc)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", default="HARPS",
                        help="the modes kept, separated by commas;"
                             " empty to keep them all")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    modes = [m for m in args.modes.split(",") if m] if args.modes else None
    data = read_exponents(args.science, modes)
    if data is None:
        raise SystemExit("no spectrum carrying TLPEH2O in %s" % args.science)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    draw(data, args.out)
    print("%d exposures -> %s" % (len(data["rjd"]), args.out))


if __name__ == "__main__":
    main()
