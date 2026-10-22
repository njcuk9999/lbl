#!/usr/bin/env python
"""Two-frame (star + observer) weighted PCA, by block coordinate descent.

The `fit` stage of `pca2d-preclean`. The blocks are updated one at a time, with
the coefficients of both solved *jointly* per spectrum, so the fit decides which
frame a feature belongs to rather than the analyst choosing a registration. The
design history is section 11 of NOTES.md in spectropca_per_obj.

    python -m lbl.pca2d.twoframe --cube cache/cube_tfits_<key> \
        --outdir outputs/<object>/<M>-<N> --config <resolved_config.yaml>
    python -m lbl.pca2d.twoframe --replot --outdir outputs/<object>/<M>-<N>

Working frame is the OBSERVER frame, so the weights (photon noise, telluric
ramp) never move. The star basis is carried into each spectrum's frame by an
exact Lanczos translation: on the log-uniform grid a Doppler shift is a
translation of atanh(v/c) / (dv/c) samples (grids.pixel_shift), and the
operator's adjoint is the same taps scattered rather than gathered (NOTES.md
11.3).
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .logger import log
from .progress import bar as _bar
from .progress import set_label as _set_label
from .grids import pixel_shift, shift_velocity
from .nights import row_unit
import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft, next_fast_len, rfft

C_KMS = 299792.458
PARITY_NAMES = ("even", "odd")
EDGE = 64          # samples zeroed at each end; the FFT wraps, the shifts are <=43


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="YAML whose `twoframe:` section supplies every option"
                        " below. Anything given on the command line wins over"
                        " it, and anything in neither falls back to"
                        " pca2d/config.py:DEFAULTS")
    p.add_argument("--cube", default=None,
                   help="observer-frame cube, log_sub high-pass (build with registration.frame: observer)")
    p.add_argument("--order", choices=["star_first", "earth_first"], default=None,
                   help="which block is updated first; star_first for a reason, see 11.7")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="sweeps in a row allowed to be worse than the best before"
                        " the fit stops; 2 by default")
    p.add_argument("--keep", choices=("best", "last"), default=None,
                   help="which iterate the fit keeps: the one of lowest chi2"
                        " (default) or the last one run. The soft clip changes"
                        " the weights every sweep, so successive chi2 are not"
                        " strictly comparable, and the velocity term was still"
                        " converging when the best chi2 was reached")
    p.add_argument("-k", "--n-star", type=int, default=None,
                   help="M, components in the STELLAR rest frame")
    p.add_argument("-j", "--n-earth", type=int, default=None,
                   help="N, components in the OBSERVER frame")
    p.add_argument("--no-observer-first", dest="observer_first",
                   action="store_false", default=None,
                   help="with --earth-basis: estimate the star's spectrum from"
                        " the data as they are, as without a frozen basis,"
                        " instead of after taking the basis out")
    p.add_argument("--earth-basis", default=None,
                   help="a saved observer basis (pca2d.basis): its components"
                        " are taken as they are and never updated; only the"
                        " star's block and the amplitudes are fitted")
    p.add_argument("--leakage", action="store_true", default=None,
                   help="report the cross-block leakage matrix at the end (slow)")
    p.add_argument("--velocity-min-transmission", type=float, default=None,
                   help="measure the shift only where the telluric transmission"
                        " stays above this; 0 keeps the band cut alone")
    p.add_argument("--star-basis", choices=("grid", "spline"), default=None,
                   help="how the star-side vectors are carried and updated:"
                        " 'spline', one cubic B-spline evaluated at each"
                        " exposure's shifted positions and updated exactly"
                        " (pca2d.splinestar), which is the default and the only"
                        " one the window offers; 'grid', samples carried by the"
                        " Lanczos kernel and updated from the normal diagonal,"
                        " kept for redoing the runs made on it")
    p.add_argument("--resolution", type=float, default=None,
                   help="the instrument's resolving power, lambda/dlambda: the"
                        " unit --star-smooth is measured in")
    p.add_argument("--star-smooth", type=float, default=None,
                   help="smooth the star's spectra and components to this FWHM, in"
                        " resolution elements, by LBL's template filter; the"
                        " observer components are not. Off by default")
    p.add_argument("--star-resolution", type=float, default=None,
                   help="the older spelling: smooth the star to one resolution"
                        " element of this resolving power")
    p.add_argument("--velocity-term", dest="velocity_term",
                   action="store_true", default=None,
                   help="fit one velocity per exposure beside the components;"
                        " off by default since 2026-09-11 (twoframe.velocity_term)")
    p.add_argument("--no-velocity-term", dest="velocity_term",
                   action="store_false", default=None,
                   help="do not fit one velocity per exposure. The observer"
                        " block then describes the star's own motion and the"
                        " correction divides it out of the flux; kept only so"
                        " the two can be compared")
    p.add_argument("--shift", choices=["lanczos", "fft"], default=None,
                   help="shift operator; fft is kept only for comparison (NOTES 11.3)")
    p.add_argument("-a", "--kernel-halfwidth", type=int, default=None,
                   help="Lanczos support in samples")
    p.add_argument("--chunk", type=int, default=None,
                   help="spectra per carry block; default keeps the (chunk, K, M)"
                        " buffer near 128 MB, which is what stops the K=10 run"
                        " from swapping")
    p.add_argument("--no-gap-guard", dest="gap_guard", action="store_false", default=None,
                   help="keep samples whose tap window reaches into a hole in the"
                        " basis support (see gap_guard); only useful to measure"
                        " what the guard is buying")
    p.add_argument("--min-snr-frac", type=float, default=None,
                   help="drop spectra below this fraction of the median band SNR")
    p.add_argument("--clip", type=float, default=None,
                   help="soft-clip threshold in sigma; 0 disables")
    p.add_argument("--mean", choices=("iterate", "offset", "full", "star"), default=None,
                   help="the static part of the model. 'offset' (default) and"
                        " 'full' are observer-frame means taken out once"
                        " before the fit, and put back into the correction:"
                        " 'offset' the even-minus-odd half-difference, 'full'"
                        " the whole mean per parity. 'iterate', experimental:"
                        " one mean per order parity in EACH frame, the star's"
                        " and the observer's, re-estimated at every sweep from"
                        " the residual of everything else, the components"
                        " centred so static content can live only in the"
                        " means. It converges on synthetic data and on a"
                        " 70 nm slice, and did not on a whole TOI-2120 cube")
    p.add_argument("--no-template", dest="template", action="store_false", default=None,
                   help="skip the star-frame median template. The star block"
                        " then spends its first component rebuilding the mean"
                        " spectrum instead of describing variability")
    p.add_argument("--outdir", default=None,
                   help="where to write coefficients.csv and coefficients_vs_time.pdf")
    p.add_argument("--dtype", choices=("float64", "float32"), default=None,
                   help="storage for the two (rows, pixels) arrays the fit"
                        " lives in. float32 halves them, and their two model"
                        " temporaries with them, which is what lets every"
                        " exposure be fitted rather than nightly means. The"
                        " weighted sums accumulate in float64 either way")
    p.add_argument("--no-tie-parities", dest="tie_parities", action="store_false", default=None,
                   help="solve a separate coefficient vector for each parity"
                        " row. The two rows of an exposure are one measurement,"
                        " so this is only meaningful as a diagnostic")
    # --max-mad is kept as an alias so every existing command, config and log
    # line still means exactly what it meant. The name was wrong, not the
    # number: the cut is in robust sigmas, 1.4826 x MAD, so the default of 10
    # has always been 14.83 plain MADs. Renaming without changing the value is
    # deliberate; changing the value would make every fit already on disk
    # incomparable with the next one.
    # LBL's own values, so the two steps of the chain bin the same way; see
    # lbl/recipes/lbl_template.py, "bin cube by BERV (to give equal weighting
    # to epochs)". 0 restores the plain median over all spectra.
    p.add_argument("--template-berv-bin", type=float, default=None,
                   help="BERV bin width in m/s for the hierarchical template;"
                        " 0 or unset gives the plain median")
    p.add_argument("--template-berv-min-entries", type=int, default=3,
                   help="exposures a BERV bin needs before it gets a vote")
    p.add_argument("--max-sigma", "--max-mad", dest="max_mad", type=float,
                   default=None,
                   help="drop a spectrum whose coefficient on ANY of the K+J"
                        " components sits more than this many robust sigmas"
                        " (1.4826 x MAD) from that component's median, then"
                        " refit; 0 disables the cut. 10 sigma = 14.83 MAD")
    p.add_argument("--max-mad-rounds", type=int, default=None,
                   help="how many times to reject and refit at most")
    p.add_argument("--holdout", choices=("even", "odd"), default=None,
                   help="VALIDATION CROISEE: tenir hors du fit les nuits de"
                        " rang pair ou impair (pca2d.diagnostics). Elles"
                        " gardent leur place, sans poids, et l'etape correct"
                        " les corrige par le reajustement de leurs amplitudes"
                        " contre une base qui ne les a jamais vues")
    p.add_argument("--replot", action="store_true",
                   help="regenerate the figures from <outdir>/fit.npz and exit,"
                        " without loading a cube or refitting")
    args = p.parse_args(argv)
    return _resolve(args)


# every option the YAML can supply; the CLI still wins over it
CONFIGURABLE = ("n_star", "n_earth", "iters", "order", "tie_parities", "max_mad",
                "template_berv_bin", "template_berv_min_entries",
                "max_mad_rounds", "clip", "min_snr_frac", "template", "shift",
                "kernel_halfwidth", "gap_guard", "leakage", "chunk", "dtype",
                "velocity_term", "velocity_min_transmission", "mean",
                "patience", "keep", "star_resolution", "star_basis",
                "resolution", "star_smooth", "holdout", "earth_basis",
                "observer_first")


def _resolve(args):
    """Fill anything the command line left unset from the YAML, then DEFAULTS.

    Three layers, and the order matters: an explicit flag must beat the config
    file, or a one-off experiment silently inherits the nominal settings and the
    run is not what its command line says it is. Options default to None rather
    than to a value so "not given" is distinguishable from "given the default".
    """
    from .config import DEFAULTS, load_config

    section = dict(DEFAULTS["twoframe"])
    if args.config:
        section.update(load_config(args.config).get("twoframe", {}) or {})
        log("options from %s: %s" % (args.config, ", ".join(
            "%s=%s" % (k, section[k]) for k in ("n_star", "n_earth", "iters",
                                                "tie_parities", "max_mad"))))
        from .config import describe_star
        log("star side: %s" % describe_star(section["n_star"]))
    for name in CONFIGURABLE:
        if getattr(args, name, None) is None:
            setattr(args, name, section[name])
    return args


# --------------------------------------------------------------------------
# the shift operator
# --------------------------------------------------------------------------
class FourierShifter:
    """Translation by a fractional number of samples, via a Fourier phase ramp.

    KEPT FOR COMPARISON ONLY -- see NOTES.md 11.3, amendment of 2026-08-24, and
    `--shift fft`. A phase ramp is a valid translation only for a band-limited,
    gap-free signal. The spectra qualify (7.5 samples per FWHM); the weight maps
    do not, being a mask with a step edge at each of ~22 gaps per row. Shifting w
    this way drives 0.63% of samples negative, and the exact diag(S^T W S) under
    a global operator is flat straight through a gap rather than gapped.
    """

    banded = False

    def __init__(self, n_pixels, pad=256, **_):
        self.n = n_pixels
        self.n_pad = next_fast_len(n_pixels + pad)
        self.freq = np.fft.rfftfreq(self.n_pad)

    def rows(self, arr, pix, chunk=64, desc=None):
        out = np.empty_like(arr)
        for start in range(0, arr.shape[0], chunk):
            stop = min(start + chunk, arr.shape[0])
            buf = np.zeros((stop - start, self.n_pad))
            buf[:, :self.n] = arr[start:stop]
            spec = rfft(buf, axis=1)
            spec *= np.exp(-2j * np.pi * self.freq[None, :] * pix[start:stop, None])
            out[start:stop] = irfft(spec, n=self.n_pad, axis=1)[:, :self.n]
        return out

    def prepare(self, basis):
        buf = np.zeros((basis.shape[0], self.n_pad))
        buf[:, :self.n] = basis
        return rfft(buf, axis=1)

    def carry(self, prepared, pix):
        ramp = np.exp(-2j * np.pi * self.freq[None, None, :] * pix[:, None, None])
        return irfft(prepared[None] * ramp, n=self.n_pad, axis=2)[:, :, :self.n]

    def adjoint(self, arr, pix, desc=None):
        """S^T x. Only equal to shift(-pix) because the operator is (nearly) unitary."""
        return self.rows(arr, -pix)

    def diag_normal(self, w, pix, desc=None):
        """diag(S^T W S), approximated by the shifted weight map. This is the
        step that is wrong; the clip exists only to hide the ringing."""
        return np.clip(self.rows(w, -pix), 0.0, None)


class LanczosShifter:
    """Translation by a compactly supported windowed-sinc kernel.

    out[m] = sum_t c_t y[m - base + t],  base = floor(p), t in [-a, a)

    so S[m, j] = c_{j - m + base} and the adjoint is a scatter with the *same*
    taps, exact by construction. Nothing here relies on unitarity, which is what
    11.3 originally wanted the FFT for and which the FFT does not actually
    deliver on real input at fractional shift anyway.

    Two properties the FFT does not have, both measured in NOTES.md 11.3:

      * a gap contaminates only +-a samples, exactly zero beyond, against a 61 px
        span with a 1/k tail for the FFT;
      * S^T W S is banded with bandwidth 2a, so its diagonal -- which is what the
        M-step uses -- is a good approximation rather than a crude one, and is
        computable exactly as sum_t c_t^2 w[j + base - t]. It is non-negative by
        construction, so no clipping is needed anywhere.

    Accuracy costs nothing at this sampling: a = 8 gives an rms error 1.2% of the
    per-pixel photon noise.

    Everything is done on a zero-padded copy so that no sample wraps around; the
    FFT version needs apodisation for the same reason and gets it from EDGE.
    """

    banded = True

    #: worker threads for carry(); set to 1 to serialise, None to autodetect
    threads = None

    def __init__(self, n_pixels, a=8, max_shift=64, threads=None):
        self.n = n_pixels
        self.a = int(a)
        self.pad = int(max_shift) + self.a + 2
        if threads is None:
            threads = min(8, (os.cpu_count() or 1))
        self.threads = int(threads)

    def _taps(self, p):
        """(base, c) with c[i] the weight on y[m - base + t], t = i - a."""
        base = int(np.floor(p))
        frac = p - base
        t = np.arange(-self.a, self.a)
        x = -(frac + t)
        c = np.sinc(x) * np.sinc(x / self.a)
        return base, c / c.sum()

    def _padded(self, arr):
        return np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(self.pad, self.pad)])

    def _apply(self, padded, base, c, sign):
        """sum_t c_t padded[..., off + t], with off placing the shift.

        sign=+1 is the forward operator, sign=-1 the adjoint (same taps, the
        offset reflected), which is exactly the transpose of the same band.

        Written as one contraction over a sliding-window view rather than as 2a
        accumulating passes. The taps are a contiguous, overlapping band, so the
        window view costs nothing to build and the sum becomes a single matmul
        over the tap axis. Measured 1.7x faster on the 421782-column grid, and
        bit-identical: the same 16 products in the same order.
        """
        off = self.pad - sign * base
        taps = c if sign > 0 else c[::-1]
        lo = off - self.a if sign > 0 else off - self.a + 1
        segment = padded[..., lo:lo + self.n + 2 * self.a - 1]
        window = sliding_window_view(segment, 2 * self.a, axis=-1)
        return window @ taps

    # `desc`, where it appears below, is a progress bar and nothing else. These
    # three loops run once per spectrum and are where the seventy seconds of a
    # basis update actually go; a caller that wants the wait to be visible
    # names the step, and a caller that does not passes nothing and sees no bar.

    def rows(self, arr, pix, desc=None):
        padded = self._padded(arr)
        out = np.empty((arr.shape[0], self.n))
        for i in _bar(range(arr.shape[0]), desc=desc, unit="row") if desc \
                else range(arr.shape[0]):
            base, c = self._taps(pix[i])
            out[i] = self._apply(padded[i], base, c, +1)
        return out

    def adjoint(self, arr, pix, desc=None):
        """S^T x, exactly: the same taps scattered rather than gathered."""
        padded = self._padded(arr)
        out = np.empty((arr.shape[0], self.n))
        for i in _bar(range(arr.shape[0]), desc=desc, unit="row") if desc \
                else range(arr.shape[0]):
            base, c = self._taps(pix[i])
            out[i] = self._apply(padded[i], base, c, -1)
        return out

    def diag_normal(self, w, pix, desc=None):
        """diag(S^T W S)[j] = sum_t c_t^2 w[j + base - t]. Exact, never negative."""
        padded = self._padded(w)
        out = np.empty((w.shape[0], self.n))
        for i in _bar(range(w.shape[0]), desc=desc, unit="row") if desc \
                else range(w.shape[0]):
            base, c = self._taps(pix[i])
            out[i] = self._apply(padded[i], base, c ** 2, -1)
        return out

    def prepare(self, basis):
        return self._padded(basis)

    def carry(self, prepared, pix, threads=None):
        """S_n applied to a whole basis, for a chunk of spectra.

        The rows are independent, and after the sliding-window rewrite of
        `_apply` the work is a matmul that releases the GIL, so a thread pool
        helps: measured 5.6x on ten cores for a 421782-column grid, with an
        identical result since each row writes its own slice. Before that
        rewrite the same pool bought only 2.5x, the loop then being limited by
        allocating sixteen temporaries per row rather than by arithmetic.
        """
        out = np.empty((pix.size, prepared.shape[0], self.n))
        if not prepared.shape[0]:
            return out                      # an empty basis carries to nothing
        n_threads = self.threads if threads is None else threads
        if n_threads and n_threads > 1 and pix.size > 1:
            from concurrent.futures import ThreadPoolExecutor

            def one(i):
                base, c = self._taps(pix[i])
                out[i] = self._apply(prepared, base, c, +1)

            with ThreadPoolExecutor(min(n_threads, pix.size)) as pool:
                list(pool.map(one, range(pix.size)))
            return out
        for i in range(pix.size):
            base, c = self._taps(pix[i])
            out[i] = self._apply(prepared, base, c, +1)
        return out


SHIFTERS = {"fft": FourierShifter, "lanczos": LanczosShifter}


# --------------------------------------------------------------------------
# the three steps
# --------------------------------------------------------------------------
def clean_columns(path, threshold=0.95, quantile=0.1, chunk=20000):
    """Grid columns whose telluric transmission stays above `threshold`.

    Read from the cube's own trans.npy, which is the Recon extension carried
    through the identical resampling, so it is already on the magic grid and no
    t.fits has to be reopened.

    Reduced to one boolean vector at load time and never held per exposure: the
    full array is (rows x samples) and 2.9 GB on a campaign cube, which is the
    same reason load_cube folds it into the weights and drops it. Read in
    column blocks through a memory map so the reduction costs a few hundred MB
    rather than the whole thing.

    A column counts as clean when it is above the threshold in 90% of the
    exposures, not in the median one: telluric depth follows airmass and water,
    so a column that sits at 0.96 on a good night can be half absorbed on a bad
    one, and the median would call it clean. The 10th percentile asks the
    question the velocity term needs answered, which is whether the column is
    reliably stellar rather than usually stellar.
    """
    if not os.path.isdir(path):
        return None
    full = os.path.join(path, "trans.npy")
    if not os.path.exists(full):
        return None
    trans = np.load(full, mmap_mode="r")
    out = np.zeros(trans.shape[1], dtype=bool)
    for start in range(0, trans.shape[1], chunk):
        stop = min(start + chunk, trans.shape[1])
        block = np.asarray(trans[:, start:stop], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            low = np.nanquantile(block, quantile, axis=0)
        out[start:stop] = low > threshold
    return out


def equilibrated_solve(amat, bvec, ridge=1e-12):
    """Solve amat c = bvec with the columns put on the same scale first.

    The design's columns do not have comparable norms. P and Q are orthonormal,
    so theirs are 1; the velocity column is d/dpix of the reconstructed star,
    whose amplitude is the star's own divided by the width of a line, which on
    SPIRou is a factor of ten away. The normal matrix then has diagonal entries
    spread over two decades and a condition number to match, without a single
    pair of columns being collinear: it is a scaling problem, not a degeneracy.

    So each column is scaled to unit diagonal, the system is solved, and the
    scaling is undone on the coefficients. In exact arithmetic that is the same
    solution to the digit; in floating point it is the difference between a
    condition number of 400 and one of 8. It also makes the ridge mean the same
    thing for every column, which it did not when one column dominated the
    trace.
    """
    diag = np.diag(amat).copy()
    scale = np.sqrt(np.where(diag > 0, diag, 1.0))
    inv = 1.0 / scale
    scaled = amat * np.outer(inv, inv)
    # the diagonal is now 1 wherever the column is constrained, so the ridge is
    # a fixed fraction of every column rather than of the largest one
    scaled.flat[:: scaled.shape[0] + 1] += ridge
    try:
        y = np.linalg.solve(scaled, bvec * inv)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(scaled, bvec * inv, rcond=None)[0]
    return y * inv, float(np.linalg.cond(scaled))


def velocity_column(star_row, mask=None):
    """d(star model)/d(pixel): the shape a small shift of the star has.

    First order, and exact at first order: f(x + eps) = f + eps f'(x). On the
    magic grid a pixel IS a velocity, dv km/s of it, so the amplitude that
    multiplies this column is a shift in pixels and alpha * dv is a velocity in
    km/s. It is also why d/d(ln lambda) needs nothing but np.gradient: the grid
    is uniform in ln lambda by construction, which is the whole reason it was
    built that way.

    WHAT IS HANDED IN HERE IS ALWAYS A RECONSTRUCTION, the carried a.P of one
    exposure, and never a measured spectrum. The reconstruction is the sum of
    every exposure that went into the basis, so its derivative is as clean as
    the basis; np.gradient of a single observed spectrum would be a derivative
    of its noise, and at SPIRou SNR per sample that is most of what it would
    be. The column would then be noise, alpha would fit noise against noise,
    and the term would inject exactly the velocity error it exists to prevent.

    WHERE IT IS ALLOWED TO SPEAK. With `mask` given the column is zero outside
    it, which is how the shift is estimated from clean stellar lines only: the
    columns inside a photometric band whose telluric transmission stays high
    across the campaign. Estimating it anywhere else means fitting the star's
    velocity to a place where the flux is mostly atmosphere and the star model
    is least trustworthy, and it showed: against LBL the unmasked term came out
    1.8 times too large.

    Masked in the model too, and not only in the solve. A term fitted under one
    model and removed under another is how a residual acquires a shape nobody
    put there.
    """
    column = np.gradient(star_row)
    if mask is not None:
        column = column * mask
    return column


def joint_coeffs(data, w, P, Q, shifter, delta, chunk=64, exposure=None,
                 velocity_from=None, velocity_mask=None, desc=None,
                 star_mean=None):
    """Step 1 of 11.2: solve [a_n ; b_n ; alpha_n] together, per spectrum.

    B_n = [ S_n P^T , Q^T ] is M x (K+J); the off-diagonal blocks of
    B_n^T W_n B_n are the cross-frame overlaps and are the whole point.

    With `exposure` given, the rows sharing an id are solved as ONE measurement.
    For a t.fits cube those rows are the even and the odd orders of a single
    exposure: one spectrum, of one star, at one instant, split across the
    detector. It has one coefficient vector, and this is how that is imposed.

    THE CONSTRAINT. Per row r the normal equations are

        (B_r^T W_r B_r) c = B_r^T W_r y_r,   B_r = [ S_r P^T | Q^T ]

    with W_r the diagonal weights of that row. Rather than solving that twice
    per exposure, the two systems are SUMMED and solved once,

        (sum_r B_r^T W_r B_r) c_n = sum_r B_r^T W_r y_r

    and the single c_n is written back to both rows.

    WHY THAT IS EXACT, not an approximation. B_r depends on the row only through
    S_r, and S_r depends only on the BERV. Both rows are the same exposure, so
    they carry the same BERV to machine precision and B_even = B_odd. The two
    systems therefore differ only through their weights, which say which columns
    each parity covers, and those supports are disjoint outside the order
    overlaps. Summing them is then exactly the normal equation of the single
    least-squares problem in which every sample of the exposure, from either
    parity, constrains one coefficient vector. The sum reassembles the exposure;
    it does not average two estimates of it.

    NOTE what is and is not per parity. The components are not: P and Q are
    single vectors over the whole grid, and there is no such thing as an even
    component or an odd one. Only the means are per parity, both of them with
    --mean iterate, the star's and the observer's: the two parities see the
    same lines at different resolutions, a static property of the instrument,
    and only the weights know which columns a row covers.

    THE VELOCITY COLUMN. With `velocity_from` given the design gains one more
    column per spectrum, the derivative of the star model those coefficients
    describe, carried into this exposure's frame. It is there to keep a
    velocity OUT of the observer block.

    The star is never exactly where the BERV alone would put it: it has its own
    motion, the planet and the activity, and the carry operator leaves a
    residual of its own. Whatever the cause, that residual has one shape, f',
    and the observer block has seven free vectors and an amplitude per exposure
    with which to describe it. It does. Dividing that block out of the flux
    then moves the star's lines, which is a velocity written into the corrected
    spectrum by the correction itself; measured on TOI2120 it accounts for 41%
    of the variance of what the correction changed in LBL's velocities. And
    when the star has a real signal, the observer block absorbs the planet and
    the correction subtracts it: the thing being looked for, removed by the
    tool meant to clear the way to it.

    So the shift gets a name and a column of its own. It is FITTED and it is
    never divided out. The term exists to keep the velocity out of Q, not to
    take it out of the data; taking it out of the data would erase exactly the
    signal the pipeline is for.

    Tied by exposure like everything else here, because a velocity is a
    property of the exposure and not of a detector parity.

    With `star_mean` = (T, group), the derivative is of T_g + a_n P, the whole
    star: with --mean iterate the star's static part lives in its per-parity
    mean and the components hold only its variations, so a derivative of the
    components alone would be the derivative of almost nothing.
    """
    n_spectra = data.shape[0]
    n_star, n_earth = P.shape[0], Q.shape[0]
    n_vel = 1 if velocity_from is not None else 0
    n_tot = n_star + n_earth + n_vel
    coeffs = np.zeros((n_spectra, n_tot))
    cond = np.zeros(n_spectra)
    eye = np.eye(n_tot)
    # nothing to carry when the fit has no star component
    Pf = shifter.prepare(P) if n_star else None
    Tf = (shifter.prepare(star_mean[0])
          if n_vel and star_mean is not None else None)
    B = np.empty((n_tot, data.shape[1]))
    Bw = np.empty((n_tot, data.shape[1]))
    tied = exposure is not None
    amats = np.zeros((n_spectra, n_tot, n_tot)) if tied else None
    bvecs = np.zeros((n_spectra, n_tot)) if tied else None
    steps = range(0, n_spectra, chunk)
    for start in (_bar(steps, desc=desc, unit="chunk") if desc else steps):
        stop = min(start + chunk, n_spectra)
        SP = shifter.carry(Pf, delta[start:stop]) if Pf is not None else None
        TT = (carried_means(Tf, star_mean[1], shifter, delta, start, stop)
              if Tf is not None else None)
        for i in range(stop - start):
            n = start + i
            if n_star:
                B[:n_star] = SP[i]
            B[n_star:n_star + n_earth] = Q
            if n_vel:
                # linearised around the star model the previous solve found: a
                # lagged Jacobian, which converges with the sweeps like every
                # other block here
                star_row = (velocity_from[n] @ SP[i] if n_star
                            else np.zeros(data.shape[1]))
                if TT is not None:
                    star_row = star_row + TT[i]
                B[-1] = velocity_column(star_row, velocity_mask)
            # into a buffer allocated once: B * w[n] is (K+J, n_pixels), 34 MB
            # at these sizes, and allocating it per row is 11% of the loop
            np.multiply(B, w[n], out=Bw)
            amat = Bw @ B.T
            bvec = Bw @ data[n]
            if tied:
                amats[n], bvecs[n] = amat, bvec
                continue
            if np.trace(amat) <= 0:
                continue
            coeffs[n], cond[n] = equilibrated_solve(amat, bvec)
    if tied:
        exposure = np.asarray(exposure)
        for value in np.unique(exposure):
            rows = np.where(exposure == value)[0]
            amat = amats[rows].sum(axis=0)
            if np.trace(amat) <= 0:
                continue
            # the ridge is 1e-12 of each column after equilibration: far below
            # any real curvature, but enough to keep the solve finite when a
            # component is unconstrained for this exposure, which happens when
            # its support is fully masked
            solved, cond_value = equilibrated_solve(amat, bvecs[rows].sum(axis=0))
            coeffs[rows] = solved
            cond[rows] = cond_value
    alpha = coeffs[:, -1] if n_vel else np.zeros(n_spectra)
    return coeffs[:, :n_star], coeffs[:, n_star:n_star + n_earth], alpha, cond


def star_support(w, delta, shifter, chunk=64, floor_frac=1e-2):
    """The star-frame columns the star basis is constrained at.

    The star update's normal diagonal carried home and summed over the rows,
    above the floor `mstep` applies to it: where it is below, mstep sets the
    basis to zero, and that is what a hole in the basis support is. Summed a
    chunk of rows at a time, so no (rows x samples) array is ever held.
    """
    total = np.zeros(w.shape[1])
    for start in range(0, w.shape[0], chunk):
        stop = min(start + chunk, w.shape[0])
        total += np.asarray(shifter.diag_normal(w[start:stop], delta[start:stop])).sum(axis=0)
    positive = total[total > 0]
    floor = floor_frac * float(np.median(positive)) if positive.size else 0.0
    return total > floor


def gap_guard(w, delta, a, verbose=True, live=None):
    """Zero the weight wherever the carried star basis would reach into a hole.

    `live` is the STAR frame's support, star_support. Until 2026-09-10 it was
    taken as w.sum(axis=0) > 0, which on an observer-frame cube is the
    observer frame's support: an OH core masked in every exposure is a hole
    there and never one in the star frame, where the exposures at other BERVs
    see it. The guard then cut a hole 2a wide into every row at the star
    column equal to the OH line's observer column, mstep zeroed the basis
    there, and the first column past it, just above mstep's floor, grew a
    spike: the vertical lines in the report's panels 2, 4 and 5, each touching
    an OH river at the BERV where the two frames coincide. Without `live` the
    old rule is kept, for a cube registered in the star's frame, where the two
    supports are the same.

    This is what it takes to run the two-frame model over a domain that is
    mostly holes. The M-step side is already safe: `diag_normal` reports an
    honest near-zero star-frame weight inside a gap and `mstep` floors it away,
    so the basis is simply not defined there. The *forward* side is not. The
    model at sample m is sum_t c_t P[m - base + t], t in [-a, a), so a sample
    within a of a hole edge is built partly from basis samples that were never
    constrained and are therefore zero. The model comes out too small, the fit
    sees a residual it cannot explain in the star frame, and the Earth block
    obligingly absorbs a rim of star-frame structure around every gap.

    The exact fix is to stop using those samples: a sample is admissible only if
    its whole tap window lands on constrained basis samples. Note this acts on
    the weights, never on the operator, so the forward map, its adjoint and
    `diag_normal` stay the same object -- which matters, because an operator
    that is not its own adjoint is what NOTES.md 11.9 spent a day diagnosing.

    Apodising the taps at the hole edge, or renormalising them over the live
    subset, would retain those samples instead of dropping them. It is not worth
    it here: on the 965-1950 nm NIRPS domain the holes are a few dozen wide
    blocks, so the rim is a fraction of a percent of the data, and the price
    would be a shift operator whose adjoint has to be rederived.
    """
    if live is None:
        live = w.sum(axis=0) > 0
    n_pixels = live.size
    window = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([np.zeros(a, bool), live, np.zeros(a, bool)]), 2 * a)
    eroded = window.all(axis=1)[:n_pixels]      # eroded[j] = all(live[j-a:j+a])

    before = int(np.count_nonzero(w > 0))
    base = np.floor(delta).astype(int)
    index = np.arange(n_pixels)
    for value in np.unique(base):
        rows = base == value
        source = index - value                  # sample m needs eroded[m - base]
        ok = (source >= 0) & (source < n_pixels)
        admissible = np.zeros(n_pixels, dtype=bool)
        admissible[ok] = eroded[source[ok]]
        w[rows] *= admissible
    after = int(np.count_nonzero(w > 0))

    if verbose:
        edges = int(np.count_nonzero(np.diff(live.astype(np.int8)) != 0))
        log("gap guard: %d holes in the basis support (%.1f%% of the grid),"
              " %d distinct integer shifts" % (edges // 2 + 1,
                                               100 * np.mean(~live),
                                               np.unique(base).size))
        log("  dropped %d of %d weighted samples (%.3f%%) within %d of a hole"
              % (before - after, before, 100 * (before - after) / max(before, 1), a))
    return w


#: Where the sweep's own (rows, columns) array goes when it will not fit in
#: memory. A file, mapped: the kernel then pages it instead of killing the
#: process, and the arithmetic is the same arithmetic on the same numbers.
SPILL_NAME = "pca2d-spill"


def workspace(shape, dtype, spill=None, tag="tmp"):
    """An array of that shape: in memory, or mapped onto a file under `spill`.

    The fit holds three arrays of (rows, columns): the data, the weights, and
    one temporary per sweep. On a machine that cannot hold them, holding them
    anyway is how a run dies with exit -9 and nothing said. Mapped onto a file
    they page instead, which is slower and finishes.
    """
    if not spill:
        return np.zeros(shape, dtype=dtype)
    import shutil

    os.makedirs(spill, exist_ok=True)
    path = os.path.join(spill, "%s-%s-%s-%d.dat" % (SPILL_NAME, tag, _host(),
                                                    os.getpid()))
    # A new file reads as zeros already, and writing them was one more pass
    # over the whole of it, a slow one on a machine that is paging. What the
    # writing also did was claim the disk space; without it a disk that fills
    # during the fit kills the process at its next write, so the space is
    # checked here instead, where there is still something useful to say.
    need = int(np.prod(shape)) * np.dtype(dtype).itemsize
    free = shutil.disk_usage(spill).free
    if need > 0.95 * free:
        raise SystemExit(
            "the fit's %s array needs %.1f GB on disk and %s has %.1f GB free."
            " Free some space (the cleanup tab lists what can go) or fit less"
            " at once" % (tag, need / 1e9, spill, free / 1e9))
    return np.memmap(path, dtype=dtype, mode="w+", shape=tuple(shape))


def _host():
    """This machine's name as it can sit in a file name."""
    import socket

    name = socket.gethostname().split(".")[0] or "host"
    return "".join(c if c.isalnum() else "_" for c in name)


def sweep_spill(spill):
    """Remove the mapped files of fits on this machine that have ended.

    A fit removes its own files when it finishes, and one that is killed, by
    the kernel or by a Stop, cannot: on 2026-09-16 a stopped run had left
    11.5 GB in cache/spill. The file names carry the machine and the process,
    so only files of this machine whose process is gone are touched; a folder
    shared with another machine keeps that machine's files whatever their
    number says.
    """
    import re

    if not spill or not os.path.isdir(spill):
        return 0
    pattern = re.compile(r"^%s-[a-z0-9]+-(.+)-(\d+)\.dat$" % re.escape(SPILL_NAME))
    freed = 0
    for name in sorted(os.listdir(spill)):
        found = pattern.match(name)
        if not found or found.group(1) != _host():
            continue
        pid = int(found.group(2))
        if pid == os.getpid() or _alive(pid):
            continue
        path = os.path.join(spill, name)
        try:
            size = os.path.getsize(path)
            os.remove(path)
        except OSError:
            continue
        freed += size
    if freed:
        log("removed %.1f GB of mapped files left in %s by fits that had ended"
            % (freed / 1e9, spill), "value")
    return freed


def _alive(pid):
    """Whether a process of that number exists on this machine."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:                    # it exists and belongs to someone else
        return True
    return True


def drop_workspace(array):
    """Let a mapped workspace go, and take its file with it."""
    path = getattr(array, "filename", None)
    del array
    if path and SPILL_NAME in os.path.basename(path):
        try:
            os.remove(path)
        except OSError:
            pass


def star_model(P, a, shifter, delta, n_spectra, n_pixels, chunk=64, desc=None,
               alpha=None, velocity_mask=None, star_mean=None, out=None):
    """sum_k a_nk (S_n P_k), never materialising S_n P^T for all n at once.

    With `alpha` given, each row also gets alpha_n times its own derivative:
    the whole star-side model, shift included, which is what the observer block
    must be shown the residual of.

    With `star_mean` the derivative is of the whole star, its carried
    per-parity mean included; the mean itself is not in what comes back, which
    is the components' model.
    """
    # `out` lets the caller say where this goes: on a big cube it is a mapped
    # workspace rather than another (rows, columns) float64 array in memory
    if out is None:
        out = np.zeros((n_spectra, n_pixels))
    Pf = shifter.prepare(P)
    Tf = (shifter.prepare(star_mean[0])
          if alpha is not None and star_mean is not None else None)
    steps = range(0, n_spectra, chunk)
    for start in (_bar(steps, desc=desc, unit="chunk") if desc else steps):
        stop = min(start + chunk, n_spectra)
        block = np.einsum(
            "nk,nkm->nm", a[start:stop], shifter.carry(Pf, delta[start:stop])
        )
        if alpha is not None:
            TT = (carried_means(Tf, star_mean[1], shifter, delta, start, stop)
                  if Tf is not None else None)
            # row by row: np.gradient over a whole chunk would allocate a
            # second (chunk x samples) array beside the one we just built
            for i in range(stop - start):
                if alpha[start + i]:
                    whole = block[i] if TT is None else block[i] + TT[i]
                    block[i] += alpha[start + i] * velocity_column(whole,
                                                                  velocity_mask)
        out[start:stop] = block
    return out


def model_residual(data, P, Q, a, b, shifter, delta, chunk, alpha=None,
                  velocity_mask=None, star_mean=None, w=None, out=None):
    """data - (star model + b Q), a chunk of ROWS at a time, and its chi2.

    Returns (chi2, residual): the chi2 against `w` when weights are given, and
    the residual itself only when `out` is given to hold it. Spelled the
    obvious way,

        model = star_model(...) + b @ Q
        chi2 = np.sum(w * (data - model) ** 2)

    allocates five arrays of (rows, columns) for one number, and on the cube
    that the kernel killed on 2026-09-15 that is five times 19 GB. A chunk of
    rows at a time it is five times a chunk, and the sum is float64 either way.
    """
    rows, cols = data.shape
    step = max(1, int(chunk))
    Pf = shifter.prepare(P)
    chi2 = 0.0
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        block = star_model(P, a[start:stop], shifter, delta[start:stop],
                           stop - start, cols, chunk,
                           alpha=None if alpha is None else alpha[start:stop],
                           velocity_mask=velocity_mask,
                           star_mean=None if star_mean is None else
                           (star_mean[0], star_mean[1][start:stop]))
        block += b[start:stop] @ Q
        block *= -1.0
        block += data[start:stop]             # in place: data - model
        if w is not None:
            chi2 += float(np.sum(w[start:stop] * block ** 2, dtype=np.float64))
        if out is not None:
            out[start:stop] = block
    del Pf
    return chi2, out


def deflate_velocity(resid, P, a, alpha, shifter, delta, chunk=64, desc=None,
                    velocity_mask=None, star_mean=None):
    """resid -= alpha_n d/dpix(S_n P a_n), in place and a chunk at a time.

    In place for the same reason star_model is chunked: the residual is a
    (rows x samples) array and there is no room for a second one.

    Why the star basis update needs this at all: the velocity term lives in the
    star's frame, so a residual that still contains it teaches P the derivative
    of its own first component. P would then describe the shift, alpha would
    describe it too, and the two would trade amplitude from sweep to sweep.
    """
    if alpha is None or not np.any(alpha):
        return resid
    Pf = shifter.prepare(P)
    Tf = shifter.prepare(star_mean[0]) if star_mean is not None else None
    steps = range(0, resid.shape[0], chunk)
    for start in (_bar(steps, desc=desc, unit="chunk") if desc else steps):
        stop = min(start + chunk, resid.shape[0])
        SP = shifter.carry(Pf, delta[start:stop])
        TT = (carried_means(Tf, star_mean[1], shifter, delta, start, stop)
              if Tf is not None else None)
        for i in range(stop - start):
            n = start + i
            if alpha[n]:
                whole = a[n] @ SP[i] if TT is None else a[n] @ SP[i] + TT[i]
                resid[n] -= alpha[n] * velocity_column(whole, velocity_mask)
    return resid


def mstep(residual, weights, coeffs, basis, floor_frac=1e-2, chunk=64):
    """One EMPCA eigenvector sweep, in whatever frame the residual lives in.

    `denominator > 0` is not a sufficient guard. With an honest banded operator
    the star-frame weight really does fall to ~1e-15 of nominal inside a telluric
    gap (that is the whole point of NOTES 11.3), and dividing by it puts a huge
    spike in the eigenvector: it drove cond(A) from 6 to 5e10 the first time this
    ran, and a 1e-8 floor still left max|v|/rms at 196 against 17 for the input.
    The FFT operator never tripped this only because its smearing filled the gaps
    back in with fictitious weight.

    floor_frac = 1e-2 costs essentially nothing here, because the denominator is
    strongly bimodal: constrained samples sit within a factor of ~1.3 of the
    maximum (median 6.4e4 against max 8.7e4) and only **0.11%** of samples fall
    below 1% of it. Measured counts below a given fraction of the max, out of
    38696 samples: 1e-8 -> 29, 1e-4 -> 36, 1e-2 -> 44, 1e-1 -> 46. There is no
    population in between to lose; the spikes come entirely from the handful of
    samples sitting just above whatever floor is chosen.

    The floor is a fraction of the **median** of the positive denominators, not
    of their maximum. Over the 1500-1600 nm band those two differ by a factor
    1.3 and the choice is immaterial, which is why the original spelling was
    fine. Over 965-1950 nm they do not: the per-order extraction SNR runs from
    108 to 294, so the weight alone spans a factor of seven before the telluric
    ramp and the per-column exposure count are applied, and a floor set at 1% of
    the global maximum would silently delete legitimately constrained samples at
    the blue end of Y. What the floor has to separate is real data from
    numerical residue, and residue sits at ~1e-15 of nominal; the typical
    constrained weight is the right yardstick for that, the largest one is not.
    """
    basis = basis.copy()
    # NOTE: `residual` is consumed in place. Every caller passes a freshly built
    # temporary, and copying it costs an N x M float64 array we cannot spare.
    #
    # IN ROW CHUNKS, both sums. Spelled `(weights * residual).T @ ck` they are
    # one BLAS call and one more (rows, columns) array per component, which on
    # a cube that barely fits is the allocation that ends it. The chunk sums
    # are float64 whatever the arrays are, and they differ from the single call
    # only by the order of the additions, which is last-bit.
    rows = residual.shape[0]
    step = max(1, int(chunk))
    for k in range(basis.shape[0]):
        ck = coeffs[:, k]
        numerator = np.zeros(residual.shape[1], dtype=np.float64)
        denominator = np.zeros(residual.shape[1], dtype=np.float64)
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            wb = weights[start:stop]
            numerator += (wb * residual[start:stop]).T @ ck[start:stop]
            denominator += wb.T @ (ck[start:stop] ** 2)
        positive = denominator[denominator > 0]
        scale = float(np.median(positive)) if positive.size else 0.0
        floor = floor_frac * scale
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = np.where(denominator > floor,
                           numerator / np.maximum(denominator, floor), 0.0)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            basis[k] = vec / norm
        for start in range(0, rows, step):          # the deflation, in chunks too
            stop = min(start + step, rows)
            residual[start:stop] -= np.outer(ck[start:stop], basis[k])
    for i in range(basis.shape[0]):          # modified Gram-Schmidt
        for j in range(i):
            basis[i] -= np.dot(basis[i], basis[j]) * basis[j]
        basis[i] /= np.linalg.norm(basis[i])
    return basis


def update_star(data, w, P, Q, a, b, shifter, delta, chunk, alpha=None,
                velocity_mask=None, star_mean=None, star_fwhm=None, work=None):
    """Step 3: deflate the Earth model, carry the weighted residual home.

    The normal equation wants sum_n c^2 S_n^T W_n S_n; we use its diagonal. With
    a banded operator that matrix has bandwidth 2a and the diagonal is a fair
    approximation to it; with the FFT it is dense and the diagonal is not (11.3).
    Either way the diagonal is now computed exactly by `diag_normal` rather than
    approximated by a shifted weight map, and needs no clipping.

    With `star_fwhm`, one resolution element in samples, every updated vector is
    smoothed to it by LBL's template filter and made orthonormal again: the star
    has nothing finer (pca2d.resolution). update_earth has no such step, since an
    observer component may carry pixel-level detector structure.
    """
    # built in place: `data - b @ Q`, then `w * that`, then the adjoint. Spelled
    # the obvious way this holds four N x M float64 arrays at once. `work` is
    # the fit's one scratch array of that shape, in memory or mapped onto a
    # file: without it, every sweep allocates another one.
    resid = np.empty_like(data, dtype=np.float64) if work is None else work
    np.matmul(b, Q, out=resid)
    resid *= -1.0
    resid += data
    deflate_velocity(resid, P, a, alpha, shifter, delta, chunk,
                     desc="star basis, taking the shift out" if alpha is not None
                     and np.any(alpha) else None, velocity_mask=velocity_mask,
                     star_mean=star_mean)
    if getattr(shifter, "spline", False):
        # the star as one B-spline (--star-basis spline): the same residual,
        # and the exact solution of the banded normal equations in place of
        # their diagonal
        from .splinestar import solve_components
        P_new = solve_components(resid, w, a, delta, shifter.pad)
        if star_fwhm:
            from .resolution import smooth_rows
            P_new = smooth_rows(P_new, star_fwhm)
        return P_new
    # the grid star basis works on whole arrays: clipped weights stored
    # sparsely (ClippedWeights) are assembled for it
    w = np.asarray(w)
    resid *= w
    resid_star = shifter.adjoint(resid, delta,
                                 desc="star basis, carrying home")
    del resid
    w_star = shifter.diag_normal(w, delta,
                                 desc="star basis, normal diagonal")
    with np.errstate(invalid="ignore", divide="ignore"):
        resid_star /= np.where(w_star > 1e-12, w_star, 1.0)
    resid_star[w_star <= 1e-12] = 0.0
    P_new = mstep(resid_star, w_star, a, P, chunk=chunk)
    if star_fwhm:
        from .resolution import smooth_rows
        P_new = smooth_rows(P_new, star_fwhm)
    return P_new


def update_earth(data, w, P, Q, a, b, shifter, delta, chunk, alpha=None,
                 velocity_mask=None, star_mean=None, work=None):
    """Step 2: deflate the star model, shift included. No shifting of weights.

    `alpha` is what keeps the velocity out of Q: the residual this hands to the
    basis update no longer contains the star's shift, so there is nothing of
    that shape left for an observer component to describe.
    """
    model = star_model(P, a, shifter, delta, data.shape[0], data.shape[1], chunk,
                       desc="observer basis, carrying the star out", alpha=alpha,
                       velocity_mask=velocity_mask, star_mean=star_mean,
                       out=work)
    model *= -1.0
    model += data                       # in place: data - model
    return mstep(model, w, b, Q, chunk=chunk)


# --------------------------------------------------------------------------
#: how much of one column block to hold while the weights are built. 256 MB is
#: small beside any cube this reads and large enough that the loop is not the
#: cost: the whole point is that sigma and trans are never resident.
_LOAD_BLOCK_BYTES = 256 * 1024 * 1024


#: LE SEUIL PAR PIXEL, EN SNR. Un pixel dont le bruit est sous ce rapport
#: n'apporte rien au fit; au-dessus, il compte par son poids 1/sigma^2, et le
#: fond d'une raie profonde reste donc dans le jeu. 5 est bas exprès: le but
#: est d'ecarter le pixel mort, pas de trier par brillance. Mesure du
#: 2026-10-16 sur HD 10700 (voir docs/harps_tc_result.md): l'ancien seuil de
#: profondeur laissait 13.7% de NaN dans les spectres corriges et coutait a
#: LBL le tiers de ses raies.
DEFAULT_MIN_PIXEL_SNR = 5.0


def read_cube_files(path, dtype=np.float64, columns=None, lazy=False):
    """Read a cached cube, in either of the two formats `cube.py` has written.

    Until 2026-08-26 the cache was a single compressed `.npz`. It is now a
    directory holding `grid.npy`, `data.npy`, `sigma.npy`, an optional
    `trans.npy` and `meta.fits`: a full-domain t.fits cube is several GB, and
    `savez_compressed` on that spends minutes in zlib for a payload that is
    float32 noise and does not compress. Both are read here so the older caches
    on disk stay usable.
    """
    # With `columns`, only those grid columns are read, through a memory map:
    # a figure of one window then reads its few thousand columns rather than
    # the whole cube, 79 MB for the eight windows of a campaign against 4.4 GB.
    # `lazy` keeps sigma and trans as memory maps for a caller that reads them
    # in blocks; the data is materialised either way, since the fit works in it
    mmap = "r" if (columns is not None or lazy) else None
    if os.path.isdir(path):
        def _load(name, required=True):
            full = os.path.join(path, name)
            if os.path.exists(full):
                return np.load(full, mmap_mode=mmap)
            if required:
                raise FileNotFoundError("%s has no %s" % (path, name))
            return None
        grid = _load("grid.npy")
        data = _load("data.npy")
        sigma = _load("sigma.npy")
        trans = _load("trans.npy", required=False)
        meta = Table.read(os.path.join(path, "meta.fits"))
    else:
        blob = np.load(path, allow_pickle=True)
        grid = blob["grid"]
        data = blob["data"]
        sigma = blob["sigma"]
        trans = blob["trans"] if "trans" in blob.files else None
        meta = Table(blob["meta"])
    if columns is not None:
        columns = np.asarray(columns)
        grid = np.asarray(grid)[columns]
        data, sigma = data[:, columns], sigma[:, columns]
        if trans is not None:
            trans = trans[:, columns]
    grid = np.asarray(grid)
    data = np.asarray(data, dtype=dtype)
    if not lazy:
        sigma = np.asarray(sigma, dtype=dtype)
        if trans is not None:
            trans = np.asarray(trans, dtype=dtype)
    return grid, data, sigma, trans, meta


def cube_grid_size(path):
    """Columns of a cached cube's grid, without reading anything else."""
    if os.path.isdir(path):
        return int(np.load(os.path.join(path, "grid.npy"), mmap_mode="r").shape[0])
    return int(np.load(path, allow_pickle=True)["grid"].shape[0])


def cube_grid(path):
    """A cached cube's wavelength grid, and nothing else from it."""
    if os.path.isdir(path):
        return np.load(os.path.join(path, "grid.npy"))
    return np.asarray(np.load(path, allow_pickle=True)["grid"])


def group_names(n_groups):
    """A name per star-spectrum group, unique and readable.

    One object has two, the even orders and the odd ones. A joint fit of
    several objects has two per object, labelled 2 * index + parity
    (pca2d.joint), and they all need a column of their own: naming them all
    "all", as this did until 2026-09-12, wrote one mean and dropped the rest,
    or refused the file outright for using a name twice.
    """
    if n_groups <= 1:
        return ["all"]
    if n_groups == 2:
        return list(PARITY_NAMES)
    return ["g%d_%s" % (i // 2, PARITY_NAMES[i % 2]) for i in range(n_groups)]


def row_parity(meta, n_rows):
    """Which half of the echellogram each cube row came from.

    A t.fits cube carries two rows per exposure, the even orders and the odd
    ones (NOTES.md 13.2). An s1d cube, and any cube built before 2026-08-26,
    has one row per exposure and no `parity` column; those get a single group,
    so everything downstream can be written once.
    """
    if "parity" in getattr(meta, "colnames", []):
        return np.asarray(meta["parity"], dtype=int)
    return np.zeros(n_rows, dtype=int)


def row_exposure(meta, n_rows):
    """Which original exposure each cube row came from.

    The two parity rows of a t.fits exposure share this id, which is what ties
    them together in the coefficient solve. Without an `exposure` column every
    row is its own exposure and the tie is a no-op.
    """
    if "exposure" in getattr(meta, "colnames", []):
        return np.asarray(meta["exposure"], dtype=int)
    return np.arange(n_rows, dtype=int)


def parity_means(data, w, parity):
    """Weighted mean spectrum per parity, one row per group, in the cube frame.

    Not a refinement: with a t.fits cube a single global mean leaves a static
    even-minus-odd offset in the residual, which is the order overlaps not being
    at the same resolution, and it is the largest coherent structure in the
    cube. In the main pipeline it took the whole of PC1 (NOTES.md 13.9). It is
    fixed in the instrument frame, so the observer-frame mean is exactly where
    it belongs and removing it per parity kills it outright.
    """
    groups = np.unique(parity)
    means = np.zeros((groups.size, data.shape[1]))
    for i, value in enumerate(groups):
        # one row at a time, in numpy's order and dtype, so that the sums are
        # sum(axis=0)'s to the bit without three copies of half the cube
        total, weighted = column_sums(
            [(lambda r: w[r], data.shape[0]),
             (lambda r: w[r] * data[r], data.shape[0])],
            rows=np.flatnonzero(parity == value))
        with np.errstate(invalid="ignore", divide="ignore"):
            means[i] = np.where(total > 0,
                                weighted / np.where(total > 0, total, 1.0), 0.0)
    return means, groups


def fit_means(blob, meta, n_rows, n_pixels):
    """The mean spectrum a saved fit removed, as (means, group_per_row).

    `means[group]` is what was subtracted from each cube row. Since 2026-08-26
    a fit.npz stores one mean per order parity and the parity of every row;
    older files store a single `mean`, which is read back as one group so the
    caller needs no special case.
    """
    files = list(getattr(blob, "files", []))
    if "means" in files:
        means = np.atleast_2d(blob["means"])
        parity = np.asarray(blob["parity"], dtype=int) if "parity" in files else None
        if parity is None or parity.size != n_rows:
            # the fit and this cube disagree on which rows survived the SNR cut;
            # the parity of a row is a property of the cube, so take it from there
            parity = row_parity(meta, n_rows)
        groups = np.unique(parity)
        if groups.size != means.shape[0]:
            raise SystemExit(
                "the fit removed %d mean spectra but this cube has %d order"
                " parities: mismatched runs" % (means.shape[0], groups.size))
        return means, np.searchsorted(groups, parity)
    mean = blob["mean"] if "mean" in files else np.zeros(n_pixels)
    return np.atleast_2d(mean), np.zeros(n_rows, dtype=int)


def fit_templates(blob, meta, n_rows, n_pixels):
    """The star-frame means a saved fit holds, as (templates, group_per_row).

    One per parity from a --mean iterate fit (`templates`); from an older fit
    the single `template`, zero unless its one-shot median was used, as one
    group, so a caller needs no special case.
    """
    files = list(getattr(blob, "files", []))
    if "templates" in files:
        templates = np.atleast_2d(blob["templates"])
        parity = np.asarray(blob["parity"], dtype=int) if "parity" in files else None
        if parity is None or parity.size != n_rows:
            parity = row_parity(meta, n_rows)
        return templates, np.searchsorted(np.unique(parity), parity)
    template = blob["template"] if "template" in files else np.zeros(n_pixels)
    return np.atleast_2d(template), np.zeros(n_rows, dtype=int)


def mean_rows(means, group, n_rows):
    """`means[group]` as a broadcast view when there is only one group.

    With a single mean this is a zero-copy view; fancy-indexing it instead
    would allocate a full (N, M) array, which on the 764 x 92426 cube of
    section 11 is 565 MB of pure waste.
    """
    if means.shape[0] == 1:
        return np.broadcast_to(means[0], (n_rows, means.shape[1]))
    return means[group]


def subtract_means(data, means, group):
    """data -= means[group], in place and without an (N, M) temporary."""
    if not np.any(means):              # x - 0 is x: a pass over the cube saved
        return data
    if means.shape[0] == 1:
        data -= means[0][None, :]
        return data
    # a block of rows at a time: data[rows] on a mask is a copy of the rows,
    # half the cube at once, written back afterwards
    group = np.asarray(group)
    step = _row_step(data.shape[1])
    for start in range(0, data.shape[0], step):
        stop = min(start + step, data.shape[0])
        block = data[start:stop]
        here = group[start:stop]
        for i in np.unique(here):
            rows = here == i
            block[rows] -= means[i][None, :]
    return data


def carried_means(prepared, group, shifter, delta, start, stop):
    """S_n T_g for rows start..stop, each row carrying the mean of its parity.

    `prepared` is shifter.prepare(T) for the (groups, samples) array T, so a
    caller looping over chunks prepares it once.
    """
    out = shifter.carry(prepared, delta[start:stop])        # (rows, groups, M)
    return out[np.arange(stop - start), np.asarray(group)[start:stop]]


def subtract_carried(data, T, group, shifter, delta, chunk, w=None,
                     desc=None):
    """data -= S_n T_g, a chunk of rows at a time; zero again where w <= 0."""
    Tf = shifter.prepare(np.atleast_2d(T))
    starts = range(0, data.shape[0], chunk)
    for start in (_bar(starts, desc=desc, unit="chunk") if desc else starts):
        stop = min(start + chunk, data.shape[0])
        data[start:stop] -= carried_means(Tf, group, shifter, delta, start, stop)
        if w is not None:
            data[start:stop][w[start:stop] <= 0] = 0.0
    return data


#: alternations of the two means, before any component exists
MEAN_INIT_ROUNDS = 3
#: alternations of the two means per sweep, on that sweep's residual
MEAN_SWEEP_ROUNDS = 1
#: what the star-frame step divides by: "diag", the diagonal of
#: sum_n S_n^T W_n S_n as update_star uses it, or "lumped", its row sums,
#: sum_n S_n^T w_n, exact for a smooth feature and smaller than any step that
#: could overshoot for a sharp one
MEAN_STAR_NORMAL = "diag"
#: where the means start. "template": the star-frame median of each parity
#: first and the observer mean of what it leaves, component zero of each block
#: as NOTES 11.13 found it has to be; "zero": the observer mean alone. Measured
#: on the synthetic cube of tests/test_parity_means_iterate.py on 2026-09-10:
#: from "template" chi2 descends at every sweep, each frame's mean correlates
#: 1.00 with its truth, the smeared star stays out of the observer mean (-0.07)
#: and the observer component follows the telluric depth (-1.00). From "zero"
#: the fit turns over after one sweep, the observer component follows the BERV
#: (+0.97) and the smeared star is still in the observer mean (+0.22): the trap
#: of NOTES 11.12 again. "diag" against "lumped", and one against three rounds
#: per sweep, changed nothing from "template".
MEAN_INIT = "template"


def update_means(data, w, T, O, group, shifter, delta, chunk, resid=None,
                 desc=None):
    """One step of the two per-parity means: T in the star's frame, O in the
    observer's, each from the residual of everything else.

    `data` holds the cube minus S_n T_g minus O_g and is updated in place, as
    T and O are. `resid` is the residual of the components, data minus their
    model, and is consumed; None means nothing else is in the model yet and the
    residual is the data itself.

    The observer mean first, then the star mean on what the new observer mean
    left: block coordinate descent, each step the weighted least-squares one
    given the rest, up to the same diagonal of sum_n S_n^T W_n S_n that
    update_star uses in the star's frame. Re-estimated at every sweep, which is
    what stops the frames trading static content: a star smeared over the BERV
    that sits in the observer mean shows up in the residual as soon as the
    star-frame mean holds the star, and the next observer step takes it back
    out. One mean per parity in both frames, never shared: the even and the odd
    orders see the same star and the same sky at different resolutions, far
    more so on NIRPS, and a shared mean would hand that difference to whichever
    block could reach it.

    Returns the rms of the two steps, which is how convergence is judged.
    """
    same = resid is None
    if same:
        resid = data
    group = np.asarray(group)
    n_groups, n_rows = T.shape[0], data.shape[0]
    # ---- observer frame: a weighted mean per parity, column by column
    num = np.zeros_like(O)
    tot = np.zeros_like(O)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        wc = np.asarray(w[start:stop], dtype=np.float64)
        rc = wc * resid[start:stop]
        for g in range(n_groups):
            sel = group[start:stop] == g
            if sel.any():
                num[g] += rc[sel].sum(axis=0)
                tot[g] += wc[sel].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        step_o = np.where(tot > 0, num / np.where(tot > 0, tot, 1.0), 0.0)
    # ---- star frame, on what that left: the weighted residual carried home
    # with the adjoint, over the exact diagonal of the normal matrix
    num = np.zeros_like(T)
    den = np.zeros_like(T)
    steps = range(0, n_rows, chunk)
    for start in (_bar(steps, desc=desc, unit="chunk") if desc else steps):
        stop = min(start + chunk, n_rows)
        rows_g = group[start:stop]
        data[start:stop] -= step_o[rows_g]
        if not same:
            resid[start:stop] -= step_o[rows_g]
        wc = np.asarray(w[start:stop], dtype=np.float64)
        home = shifter.adjoint(wc * resid[start:stop], delta[start:stop])
        if MEAN_STAR_NORMAL == "lumped":
            norm = np.maximum(shifter.adjoint(wc, delta[start:stop]), 0.0)
        else:
            norm = shifter.diag_normal(wc, delta[start:stop])
        for g in range(n_groups):
            sel = rows_g == g
            if sel.any():
                num[g] += home[sel].sum(axis=0)
                den[g] += norm[sel].sum(axis=0)
    step_t = np.zeros_like(T)
    for g in range(n_groups):
        positive = den[g][den[g] > 0]
        # mstep's floor, for mstep's reason: inside a telluric gap the
        # star-frame weight is numerical residue, and dividing by it is a spike
        floor = 1e-2 * float(np.median(positive)) if positive.size else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            step_t[g] = np.where(den[g] > floor,
                                 num[g] / np.maximum(den[g], floor), 0.0)
    subtract_carried(data, step_t, group, shifter, delta, chunk, w=w)
    if not same:
        # so that another round can start from this one's residual
        subtract_carried(resid, step_t, group, shifter, delta, chunk)
    O += step_o
    T += step_t
    return float(np.sqrt(np.mean(step_t ** 2))), float(np.sqrt(np.mean(step_o ** 2)))


def center_blocks(data, a, b, P, Q, T, O, group, shifter, delta, chunk, rows,
                  w=None):
    """Move the components' average into the means. Returns the centred a, b.

    Exact: the model does not change, only which block holds what. It is what
    gives the means' re-estimation its grip. A component with a non-zero
    average carries static content, and a static pattern in a component can
    cancel a wrong assignment between the two means before the residual ever
    shows it. Centred, static content has nowhere to live but the means.

    One average over every good row, added to every parity's mean alike, not
    one per parity: the two parity rows of an exposure share one coefficient
    set (joint_coeffs, `exposure`), and per-parity averages would untie them.
    """
    if not np.any(rows):
        return a, b
    abar, bbar = a[rows].mean(axis=0), b[rows].mean(axis=0)
    star_part, obs_part = abar @ P, bbar @ Q
    T += star_part[None, :]
    O += obs_part[None, :]
    data -= obs_part[None, :]
    subtract_carried(data, star_part, np.zeros(data.shape[0], dtype=int),
                     shifter, delta, chunk, w=w)
    return a - abar, b - bbar


def restore_means(data, T, O, T_to, O_to, group, shifter, delta, chunk, w=None):
    """Put the means back to an earlier state, the data following them."""
    group = np.asarray(group)
    back_o = O - O_to
    for start in range(0, data.shape[0], chunk):
        stop = min(start + chunk, data.shape[0])
        data[start:stop] += back_o[group[start:stop]]
    subtract_carried(data, T_to - T, group, shifter, delta, chunk, w=w)
    T[:] = T_to
    O[:] = O_to


def _per_row(meta):
    """meta's n_exposures, or None when the rows are single exposures.

    A cube written before nightly stacking existed has no such column, and one
    that was not stacked has it all ones; both mean "the rows ARE exposures".
    """
    names = getattr(meta, "colnames", None) or getattr(meta, "files", None) or []
    if "n_exposures" not in names:
        return None
    per_row = np.asarray(meta["n_exposures"], dtype=float)
    return per_row if np.nanmax(per_row) > 1 else None


def count_exposures(meta, rows=None):
    """How many distinct exposures a set of rows belongs to.

    A t.fits cube has one row per order parity, so two rows per exposure, and a
    count of rows is not a count of spectra. Every message that tells a person
    how many were dropped, rejected or kept goes through here, so the number in
    the log is the number of exposures, which is what they will go and count,
    with the row count beside it.
    """
    names = np.asarray(meta["filename"])
    if rows is not None:
        names = names[np.asarray(rows)]
    return len({str(n).strip() for n in names})


def exposures_label(names, keep, per_row=None, unit="nights"):
    """'316 exposures' for the rows `keep` selects, counted as exposures.

    What a figure says about how many points went into it. The correlation
    matrix used to print keep.sum(), which on a t.fits cube is 632 for 316
    exposures, two rows each, and a reader rightly asks where the other half
    of the data came from. Without file names only rows can be counted, and
    the label says rows.

    On a nightly-stacked cube a "file name" is a NIGHT, not an exposure, and
    calling 458 of them exposures invited exactly the reading it was meant to
    prevent: GL699 has 1976 spectra and this label said 458, so the figure
    looked like it had thrown three quarters of the campaign away. `per_row`
    is meta's n_exposures; when it says the rows hold more spectra than there
    are rows, the label says so, in the words the sequence figures already use.
    """
    keep = np.asarray(keep, dtype=bool)
    if names is None:
        return "%d rows" % int(keep.sum())
    count = count_exposures({"filename": np.asarray(names)}, keep)
    if per_row is None:
        return "%d exposures" % count
    per_row = np.asarray(per_row, dtype=float)
    # one row per (night, parity): summing n_exposures over the kept rows would
    # count every night twice, so collapse to one entry per distinct name first
    seen, spectra = set(), 0.0
    for name, n, take in zip(np.asarray(names), per_row, keep):
        key = str(name).strip()
        if take and key not in seen:
            seen.add(key)
            spectra += n
    if not np.isfinite(spectra) or spectra <= count:
        return "%d exposures" % count
    return "%d %s of %d spectra" % (count, unit, int(round(spectra)))


def cube_shape(path):
    """(rows, columns) of a cached cube, read from the array header alone.

    The first thing the fit opens, so it is where a cube that is not there is
    said in words. cli.check_cubes looks for one before any stage announces
    itself and puts the cube stage back when it is gone, but this module is
    also run on its own with --cube, and then nothing has looked: a cache
    emptied between two runs used to arrive here as a bare FileNotFoundError
    out of np.load, three frames deep, after the fit had already timed itself.
    """
    missing = None
    if os.path.isdir(path):
        data = os.path.join(path, "data.npy")
        if not os.path.exists(data):
            missing = "the cube directory %s holds no data.npy" % path
        else:
            shape = np.load(data, mmap_mode="r").shape
            return int(shape[0]), int(shape[1])
    elif not os.path.exists(path):
        missing = "there is no cube at %s" % path
    if missing:
        raise SystemExit(
            "%s. That is the cube stage's output, and it is gone or was never"
            " built. Run this configuration with the cube stage as well: tick"
            " `cube` among the stages in the window, or pass --stages cube,fit"
            " on the command line. It reads every spectrum once and takes a few"
            " minutes." % missing)
    with np.load(path, allow_pickle=True) as handle:
        shape = handle["data"].shape
    return int(shape[0]), int(shape[1])


def memory_needed(rows, columns, dtype, margin=1.05):
    """GB a fit peaks at, and WHERE that peak is.

    Two arrays of (rows, columns) live from the read to the end: the data and
    the weights, at the storage dtype. On top of them, each sweep allocates ONE
    full float64 array of the same shape and frees it again: `b @ Q` in
    update_star, the carried star model in update_earth. The bases are
    float64, so those temporaries are float64 whatever the cube is stored as,
    and they are the reason a float32 cube still costs 16 bytes a sample.

    Measured: the read alone peaks at 3.2 GB on the TOI-2120 cube (642 x 577002
    float32) and this says 3.0 for it; the sweeps take it to 5.8, which is why
    that run fits in half of a 17 GB machine and the ten-object joint cube,
    37.9 GB by the same count, was killed by the kernel on 2026-09-15.

    NOT counted: the model is carried in chunks of rows (`chunk`, hundreds of
    MB) and the bases are (K, columns), tens of MB.

    Since 2026-09-16 that count is what the fit actually holds. Until then it
    also held a copy of the weights, the data copied back out of its mapped
    file by the gap guard, and in every sweep the clip's float64 weights with
    two more copies for their medians. A joint fit this count put at 11.7 GB,
    with 8.5 allowed, went to mapped files as it should and then held 11 GB
    of memory beside them, most of it in swap. The weights are no
    longer copied, the clip stores only the samples it changed
    (ClippedWeights), and the star spectra, the medians and the means are
    taken a block at a time (tests/test_low_memory.py).
    """
    item = np.dtype(dtype).itemsize
    return rows * columns * (2 * item + 8) * float(margin) / 1e9


def spill_dir(rows, columns, dtype, cube, outdir, fraction=None):
    """Where the fit's big arrays go: None for memory, a folder for a file.

    Half of the machine, never more, and the arithmetic does not change either
    way: the same numbers in the same order, either in memory or on a mapped
    file the kernel pages. A run that does not fit used to be killed with exit
    -9 and nothing said; it now runs, more slowly, and says why.
    """
    from .machine import DEFAULT_FRACTION, describe

    need = memory_needed(rows, columns, dtype)
    total, budget = describe(DEFAULT_FRACTION if fraction is None else fraction)
    where = os.path.join(os.path.dirname(os.path.abspath(cube)), "spill")
    sweep_spill(where)
    if total is None:
        log("this fit needs about %.1f GB; the machine's memory could not be"
            " read, so it is kept in memory and nothing is checked" % need,
            "warn")
        return None
    if need <= budget:
        log("memory: this fit needs about %.1f GB of the %.1f GB allowed (half"
            " of this machine's %.1f GB), so it is held in memory"
            % (need, budget, total), "value")
        return None
    log("memory: this fit needs about %.1f GB and may use %.1f, half of this"
        " machine's %.1f GB. Its arrays go to a mapped file in %s instead, and"
        " the kernel pages them: the same arithmetic, slower, and it finishes."
        % (need, budget, total, where), "warn")
    log("  (the ways to make it FIT in memory, cheapest first: coadd nights"
        " (input.nightly_stack: true), narrow the domain, fit fewer objects at"
        " once, or store the cube as float32)", "info")
    return where


def check_memory(rows, columns, dtype, fraction=None):
    """Refuse a fit that cannot fit, BEFORE it has read anything.

    The alternative is what happened on 2026-09-15: a joint cube of ten
    objects, 19 GB of arrays on a machine with 17, twenty-six minutes of cube
    and eleven of preparation, and then exit -9 from the kernel with nothing
    said. A number and a list of ways out cost a second.
    """
    from .machine import DEFAULT_FRACTION, describe

    need = memory_needed(rows, columns, dtype)
    total, budget = describe(DEFAULT_FRACTION if fraction is None else fraction)
    if total is None:
        log("this fit needs about %.1f GB; the machine's memory could not be"
            " read, so nothing is checked against it" % need, "warn")
        return need
    log("memory: this fit peaks at about %.1f GB while the cube is read, of"
        " %.1f GB allowed (half of the machine's %.1f GB)"
        % (need, budget, total), "value")
    if need > budget:
        raise MemoryError(
            "this fit peaks at about %.1f GB and may use %.1f, half of this"
            " machine's %.1f GB. Its arrays are %d rows x %d columns.\n"
            "Ways out, cheapest first: coadd nights (input.nightly_stack:"
            " true), narrow the domain (domain.wave_min/wave_max), fit fewer"
            " objects at once, or drop the storage to float32"
            " (twoframe.dtype) if it is not already."
            % (need, budget, total, rows, columns))
    return need


def load_cube(path, ln_clip_low=None, ramp_zero=0.5, min_snr_frac=0.5,
              dtype=np.float64, columns=None, spill=None,
              min_pixel_snr=DEFAULT_MIN_PIXEL_SNR):
    """The observer-frame cube plus the weights, condensed from cube.py.

    `min_snr_frac` drops spectra whose band SNR is below that fraction of the
    median. The pipeline's `quality.min_snr` is an *absolute* floor, which only
    catches outright extraction failures; this is the relative cut. A spectrum at
    a third of the usual SNR is not wrong, it is just nine times less
    informative, and the weights already know that -- but it still gets a full
    vote in the median template and contributes a row of mostly-noise
    coefficients, so it is cheaper to drop it than to carry it.

    `min_pixel_snr` est le seuil par PIXEL, en SNR, et il a remplace le
    2026-10-16 un seuil de PROFONDEUR (`ln_clip_low`, qui valait -0.5 par
    defaut ici et qu'aucun appelant ne passait). L'ancien tuait tout pixel
    sous exp(-0.5) = 0.607 du continu, donc le fond de toute raie plus
    profonde que 39%, c'est-a-dire l'endroit le plus riche en contenu de
    vitesse. Passer `ln_clip_low` et `min_pixel_snr=None` retrouve l'ancien
    comportement.
    """
    grid, data, sigma, trans, meta = read_cube_files(path, dtype, columns,
                                                    lazy=True)
    rows = None                         # which rows survive the relative cut

    if min_snr_frac:
        snr = np.asarray(meta["snr_band"], dtype=float)
        # PER OBJECT. The cut is relative by design: it drops the bad nights OF A
        # CAMPAIGN, a spectrum at a third of that campaign's usual SNR. Taken
        # over a joint cube it becomes "below half the BRIGHTEST stars' median"
        # and punishes the faintest star for the others' brightness: GJ 3090 lost
        # 56 of its 198 rows to Proxima's and GJ 1's median, where its own
        # threshold drops 12 (2026-09-13).
        objects = (np.asarray([str(v) for v in meta["object"]])
                   if "object" in getattr(meta, "colnames", [])
                   else np.zeros(len(snr), dtype=int))
        keep = np.isfinite(snr)
        thresholds = {}
        for name in np.unique(objects):
            here = objects == name
            thresholds[name] = float(min_snr_frac * np.nanmedian(snr[here]))
            keep &= ~here | (snr >= thresholds[name])
        if not keep.all():
            shown = ", ".join("%.1f" % t for t in thresholds.values())
            log("dropping %d exposures (%d rows, one per order parity) with band"
                " SNR below %.0f%% of the median of their own object (%s): %d of"
                " %d exposures left"
                % (count_exposures(meta, ~keep), int((~keep).sum()),
                   100 * min_snr_frac, shown,
                   count_exposures(meta, keep), count_exposures(meta)))
            rows = np.flatnonzero(keep)
            meta = meta[keep]

    # ONE PASS, IN COLUMN BLOCKS, and neither sigma nor the transmission ever
    # resident. Read whole, a cube costs four arrays of its own size at the
    # peak (data, sigma, transmission, weights), and that peak is what a kernel
    # kills on: a ten-object joint cube asked 38 GB of a 17 GB machine on
    # 2026-09-15 and died with nothing said. Block by block the peak is the two
    # arrays that survive the read, plus one block of each of the others. The
    # arithmetic per column is untouched, so the answer is the same to the bit
    # (tests/test_blocked_load.py holds the two side by side).
    n_rows = len(rows) if rows is not None else data.shape[0]
    n_cols = data.shape[1]
    # in memory, or mapped onto a file under `spill` when they will not fit
    out = workspace((n_rows, n_cols), dtype, spill, tag="data")
    w = workspace((n_rows, n_cols), dtype, spill, tag="weights")
    step = max(1, int(_LOAD_BLOCK_BYTES // max(1, n_rows * np.dtype(dtype).itemsize)))
    grid_size = cube_grid_size(path) if columns is not None else n_cols
    cols = np.asarray(columns) if columns is not None else None
    for first in range(0, n_cols, step):
        last = min(first + step, n_cols)
        sl = slice(first, last)
        # np.array and not np.asarray: a slice of a memory map of the same
        # dtype is a READ-ONLY view of it, and this block is written into
        take = (lambda arr: np.array(arr[np.ix_(rows, np.arange(first, last))],
                                     dtype=dtype)) if rows is not None else \
               (lambda arr: np.array(arr[:, sl], dtype=dtype))
        d_blk = take(data)
        s_blk = take(sigma)
        # over="ignore": le carre d'une sigma proche du maximum du type
        # deborde et rend un poids nul, ce qui est la bonne reponse (un
        # echantillon dont l'incertitude sature le type ne pese rien) mais
        # numpy le signalait 45 fois par campagne. L'ARITHMETIQUE EST LAISSEE
        # TELLE QUELLE: ecrire (1/s)^2 au lieu de 1/s^2 eviterait le
        # debordement mais changerait le dernier bit de tous les autres
        # poids, et tests/test_blocked_load.py compare les deux chemins de
        # chargement bit pour bit, a juste titre. Vu le 2026-10-14 sur
        # ESPRESSO, dont les recouvrements d'ordres laissent des echantillons
        # au bord mourant d'un ordre.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            wb = np.where(np.isfinite(s_blk) & (s_blk > 0), 1.0 / s_blk ** 2, 0.0)
        wb[~np.isfinite(d_blk)] = 0.0
        # LE SEUIL PAR PIXEL, et c'est ICI qu'il agit: le cube ne garde que
        # data et sigma bruts, jamais les poids, donc ils sont refaits a chaque
        # lecture. Il porte sur le SNR du pixel (1/sigma) et non sur sa
        # profondeur, pour ne pas jeter le fond des raies profondes, qui est
        # l'endroit le plus riche en contenu de vitesse. min_pixel_snr a None
        # rend la main a ln_clip_low, l'ancien seuil de profondeur.
        if min_pixel_snr is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                wb[~(1.0 / s_blk >= float(min_pixel_snr))] = 0.0
        elif ln_clip_low is not None:
            wb[d_blk < ln_clip_low] = 0.0
        if trans is not None:
            t_blk = take(trans)
            wb *= np.clip((t_blk - ramp_zero) / (1.0 - ramp_zero), 0.0, 1.0)
        # the EDGE columns of the GRID, not of whatever slice of it was read: a
        # window in the middle of the domain must not lose its own first and
        # last 64 columns because they happen to be the ends of the array it
        # came in
        here = np.arange(first, last) if cols is None else cols[first:last]
        wb[:, (here < EDGE) | (here >= grid_size - EDGE)] = 0.0
        d_blk[wb <= 0] = 0.0
        w[:, sl] = wb
        out[:, sl] = d_blk
    del sigma, trans, data
    return grid, out, w, meta


def leakage(data, w, P, Q, shifter, delta, chunk=64):
    """How much of each block can the *other* block reproduce, averaged over n."""
    n_spectra, n_pixels = data.shape
    n_star, n_earth = P.shape[0], Q.shape[0]
    to_earth = np.zeros(n_star)
    to_star = np.zeros(n_earth)
    Pf = shifter.prepare(P)
    for start in range(0, n_spectra, chunk):
        stop = min(start + chunk, n_spectra)
        SP = shifter.carry(Pf, delta[start:stop])
        for i in range(stop - start):
            wi = w[start + i]
            gram_q = (Q * wi) @ Q.T + 1e-10 * np.eye(n_earth)
            gram_p = (SP[i] * wi) @ SP[i].T + 1e-10 * np.eye(n_star)
            for k in range(n_star):
                vec = SP[i, k]
                norm = float(vec @ (wi * vec))
                if norm > 0:
                    rhs = (Q * wi) @ vec
                    to_earth[k] += float(np.linalg.solve(gram_q, rhs) @ rhs) / norm
            for j in range(n_earth):
                vec = Q[j]
                norm = float(vec @ (wi * vec))
                if norm > 0:
                    rhs = (SP[i] * wi) @ vec
                    to_star[j] += float(np.linalg.solve(gram_p, rhs) @ rhs) / norm
    return to_earth / n_spectra, to_star / n_spectra



# --------------------------------------------------------------------------
# post-processing and output
# --------------------------------------------------------------------------
def block_power(w, coeffs, carried, chunk=64):
    """Weighted power carried by each component of one block.

    sum_nm w_nm (c_nk V_nkm)^2, with V the block's vectors as they appear in the
    observer frame: fixed for the Earth block, shifted per spectrum for the
    star block. Same quantity `wpca.component_power` computes for the
    single-frame case, and the right thing to rank on for the same reason: the
    pixels a component acts on can carry very different weights from one
    component to the next.
    """
    n_spectra = w.shape[0]
    n_comp = coeffs.shape[1]
    power = np.zeros(n_comp)
    for start in range(0, n_spectra, chunk):
        stop = min(start + chunk, n_spectra)
        vecs = carried(start, stop)                      # (n, k, m)
        if vecs.ndim == 2:                               # fixed basis
            power += ((coeffs[start:stop] ** 2)
                      * (w[start:stop] @ (vecs ** 2).T)).sum(axis=0)
        else:
            power += ((coeffs[start:stop] ** 2)
                      * np.einsum("nm,nkm->nk", w[start:stop], vecs ** 2)).sum(axis=0)
    return power


def tidy_block(basis, coeffs, power):
    """Sort by decreasing weighted power and pin the sign of each vector.

    The sign of an eigenvector is arbitrary; fixing it (largest excursion
    positive) is what `wpca.sign_convention` does, and keeps runs comparable.
    """
    order = np.argsort(power)[::-1]
    basis, coeffs, power = basis[order].copy(), coeffs[:, order].copy(), power[order]
    for k in range(basis.shape[0]):
        if basis[k][np.argmax(np.abs(basis[k]))] < 0:
            basis[k] *= -1.0
            coeffs[:, k] *= -1.0
    return basis, coeffs, power


def nightly_bin(bjd, coeffs):
    night = np.floor(bjd - 0.5).astype(int)
    unique = np.unique(night)
    out_t = np.zeros(unique.size)
    out_c = np.zeros((unique.size, coeffs.shape[1]))
    for i, value in enumerate(unique):
        sel = night == value
        out_t[i] = np.mean(bjd[sel])
        out_c[i] = np.mean(coeffs[sel], axis=0)
    return out_t, out_c


def plot_coeffs(bjd, a, b, pa, pb, chi2_null, path, err_a=None, err_b=None):
    """Both blocks' coefficients against time, star on the left, Earth on the right.

    Side by side and on a shared time axis on purpose: the question this figure
    exists to answer is whether a feature in one block also appears in the other,
    which is what cross-frame leakage looks like from the coefficient side.
    """
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    n_rows = max(a.shape[1], b.shape[1])
    t = bjd - 2400000.5
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 1.7 * n_rows + 1.2),
                             sharex=True, squeeze=False)
    blocks = [("star rest frame", a, pa, "tab:blue", err_a),
              ("observer (Earth) rest frame", b, pb, "tab:red", err_b)]
    for col, (title, coeffs, power, colour, err) in enumerate(blocks):
        bt, bc = nightly_bin(bjd, coeffs)
        # the nightly error is the per-exposure error added in quadrature and
        # divided by the count, i.e. sqrt(sum e^2)/n -- not e/sqrt(n), because
        # the exposures in a night do not all carry the same weight
        if err is not None:
            night = np.floor(bjd - 0.5).astype(int)
            uniq = np.unique(night)
            be = np.zeros((uniq.size, coeffs.shape[1]))
            for i, v in enumerate(uniq):
                sel = night == v
                be[i] = np.sqrt(np.nansum(err[sel] ** 2, axis=0)) / sel.sum()
        for row in range(n_rows):
            ax = axes[row][col]
            if row >= coeffs.shape[1]:
                ax.axis("off")
                continue
            ax.plot(t, coeffs[:, row], ".", ms=2.5, alpha=0.35, color=colour)
            if err is not None:
                ax.errorbar(bt - 2400000.5, bc[:, row], yerr=be[:, row],
                            fmt="none", ecolor="0.35", elinewidth=0.6,
                            capsize=0, alpha=0.8, zorder=2)
            ax.plot(bt - 2400000.5, bc[:, row], "o", ms=3.2, mfc="none",
                    mew=0.8, color="k", label="nightly mean", zorder=3)
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
            ax.set_ylabel("%s%d" % ("a" if col == 0 else "b", row + 1))
            ax.text(0.995, 0.95, "%.1f%% of weighted variance"
                    % (100 * power[row] / chi2_null),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7",
                              lw=0.5, alpha=0.85))
            # the panel spans what this coefficient does, not the way to zero:
            # a coefficient sitting at -6 and moving by one reads as flat when
            # the axis has to reach zero, and the line at zero autoscales too
            span = np.concatenate([coeffs[:, row], bc[:, row]])
            span = span[np.isfinite(span)]
            if span.size:
                lo, hi = float(span.min()), float(span.max())
                pad = 0.05 * (hi - lo) if hi > lo else 0.05 * max(abs(lo), 1.0)
                ax.set_ylim(lo - pad, hi + pad)
            if row == 0:
                ax.set_title(title)
                ax.legend(fontsize=7, loc="lower left")
        axes[-1][col].set_xlabel("MJD (days)")
    fig.suptitle("Two-frame PCA: coefficients versus time\n"
                 "left = star-anchored block (fitted first), "
                 "right = Earth-anchored block, same fit", y=0.998, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log("  wrote %s" % path)


def plot_variance(power_star, power_earth, chi2_null, chi2_best, path):
    """Weighted variance carried by each component of both blocks.

    A single-frame PCA has one bar per component because it fits one basis.
    This model fits two, so the figure carries K + J bars: how
    the power divides between the star-anchored and the Earth-anchored block is
    the whole question the two-frame model exists to answer.

    Everything is quoted against `chi2_null`, the cube after the star-frame
    median template and the per-parity observer mean have been taken out. That
    is what the blocks actually fit and it is the only reference that means
    anything here: the raw pre-template variance is dominated by the stellar
    spectrum itself, which no component is trying to describe, so a share of it
    says nothing about the model. The headline number is the RMS of the residual
    against the RMS of that median-subtracted cube.

    DO NOT read these bars against a single-frame PCA's.
    Neither half of the fraction is the same quantity. The numerator here is
    `block_power`, a component's own weighted power, which that pipeline used
    only to *rank* components; the bars it plotted were the incremental chi2
    removed when a component is added on top of the previous ones. The two agree
    only when the coefficients are weighted-uncorrelated, which they are not.
    The denominator differs too: there, the mean-subtracted cube; here, the cube
    after a star-frame template that already takes out 95% of the variance.
    """
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    n_star, n_earth = power_star.size, power_earth.size
    power = np.concatenate([power_star, power_earth])
    share = 100 * power / chi2_null                  # of what the blocks fit
    # RMS of the residual over RMS of the median-subtracted cube
    rms_ratio = np.sqrt(max(chi2_best / chi2_null, 0.0)) if chi2_best else np.nan
    labels = (["a%d" % (k + 1) for k in range(n_star)]
              + ["b%d" % (j + 1) for j in range(n_earth)])
    colours = ["tab:blue"] * n_star + ["tab:red"] * n_earth
    x = np.arange(power.size)

    fig, ax = plt.subplots(figsize=(1.0 * power.size + 3.0, 4.2))
    # on a log scale: the first star component carries tens of per cent and
    # the last observer one a few thousandths, and on a linear axis the small
    # ones are bars nobody can see
    ax.bar(x, share, color=colours, alpha=0.85, log=True)
    for xi, s in zip(x, share):
        ax.text(xi, s, "%.3g" % s, ha="center", va="bottom",
                fontsize=6.5, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("component  (a = star rest frame, b = observer rest frame)")
    ax.set_ylabel("component's own weighted power\n(% of the median-subtracted cube)")
    ax.margins(y=0.16)

    twin = ax.twinx()
    twin.plot(x, np.cumsum(share), "k.-", lw=1)
    twin.set_ylabel("cumulative (%)")
    twin.grid(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:blue", alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color="tab:red", alpha=0.85)]
    ax.legend(handles, ["star block, %d components (%.2f%% together)"
                        % (n_star, share[:n_star].sum()),
                        "Earth block, %d components (%.2f%% together)"
                        % (n_earth, share[n_star:].sum())],
              fontsize=7.5, loc="upper right")
    fig.suptitle("Two-frame PCA: contribution to variance, %d + %d components"
                 % (n_star, n_earth), fontsize=10)
    ax.set_title("bars are each component's own weighted power, as %% of the"
                 " median-subtracted cube.\nThe %d together take the residual RMS"
                 " to %.4f of its starting value: a %.2f%% reduction."
                 % (power.size, rms_ratio, 100 * (1 - rms_ratio)),
                 fontsize=7.5, color="0.35")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log("  wrote %s" % path)


def write_components_fits(path, grid, P, Q, template, means, power_star,
                          power_earth, chi2_null, chi2_best, table, dv,
                          tied=True, max_mad=0.0, frame="observer",
                          highpass="log_sub", weights=None, parity=None,
                          min_fraction=0.2, templates=None, mean_mode=None):
    """Everything needed to rebuild a spectrum, in one file.

    Three extensions. BASIS carries the magic grid, the star-frame template,
    one observer-frame mean per parity and the K + J basis vectors, all on the
    same grid and all in the same units as the cube. COEFFS carries one row per
    exposure: the coefficients, their errors, the BERV needed to place the star
    basis, and the MAD-cut flag. VARIANCE carries the weighted power of each
    component, which is what "eigenvalue" means for a weighted PCA where the
    vectors are orthonormal in wavelength but their coefficients need not be.

    The quantity is the *high-passed* log flux, ln f - savgol(ln f), because
    that is what the cube holds and therefore all the fit ever saw. The
    Savitzky-Golay continuum is not stored anywhere, so nothing here can be
    turned back into a flux; see pca2d/reconstruct.py.
    """
    n_star, n_earth = P.shape[0], Q.shape[0]
    primary = fits.PrimaryHDU()
    h = primary.header
    h["INFORMAT"] = ("tfits", "input format the cube was built from")
    h["CONTENT"] = ("hp-log-flux", "ln f - savgol(ln f); no continuum stored")
    h["NSTAR"] = (n_star, "star-frame components")
    h["NEARTH"] = (n_earth, "observer-frame components")
    h["NPARITY"] = (means.shape[0], "means stored, one per order parity")
    h["MEANMODE"] = (str(mean_mode or "offset"), "static part: iterate, offset or full")
    h["FRAME"] = (frame, "frame the cube was registered in")
    h["HIGHPASS"] = (highpass, "high-pass mode")
    h["WAVE0"] = (float(grid[0]), "nm, first sample of the magic grid")
    h["DV"] = (float(dv), "km/s per sample; a shift is a translation")
    h["TIED"] = (bool(tied), "parity rows share one coefficient set")
    h["MAXMAD"] = (float(max_mad), "MAD cut applied to the coefficients")
    h["CHI2NULL"] = (float(chi2_null), "weighted var of the median-subtracted cube")
    h["CHI2BEST"] = (float(chi2_best), "weighted var of the residual")
    ratio = float(np.sqrt(max(chi2_best / chi2_null, 0.0)))
    h["RMSRATIO"] = (ratio, "residual RMS / median-subtracted RMS")
    h["CONSTRAI"] = ("n_spectra/n_rows >= %.2f" % min_fraction,
                     "definition of the `constrained` flag")
    h["R2"] = (1.0 - chi2_best / chi2_null, "on the median-subtracted cube")

    cols = [fits.Column(name="wavelength", format="D", unit="nm", array=grid),
            fits.Column(name="template", format="D", array=template)]
    names = group_names(means.shape[0])
    for i, nm in enumerate(names[:means.shape[0]]):
        cols.append(fits.Column(name="mean_%s" % nm, format="D", array=means[i]))
    if templates is not None:
        # the star-frame mean of each parity, from --mean iterate; `template`
        # above is then zero and a reader takes these (reconstruct.template_for)
        for i, nm in enumerate(names[:templates.shape[0]]):
            cols.append(fits.Column(name="template_%s" % nm, format="D",
                                    array=templates[i]))
    for k in range(n_star):
        cols.append(fits.Column(name="star_pc%d" % (k + 1), format="D", array=P[k]))
    for j in range(n_earth):
        cols.append(fits.Column(name="earth_pc%d" % (j + 1), format="D", array=Q[j]))

    # How well each column is determined, and by how much of the data. Without
    # this a reader cannot tell a column carried by 250 spectra from one carried
    # by three, and the vectors look equally trustworthy everywhere. Counted per
    # parity because half the domain is reached by the even orders only.
    if weights is not None:
        # a row at a time: `weights > 0` and `weights[rows]` were a boolean of
        # the cube's size and a copy of half of it, for two numbers a column
        n_rows = weights.shape[0]
        groups = ([0] if parity is None
                  else list(np.unique(np.asarray(parity))))
        total = 0
        for i, value in enumerate(groups):
            rows = (np.arange(n_rows) if parity is None
                    else np.flatnonzero(np.asarray(parity) == value))
            name = group_names(len(groups))[i]
            count, wsum = column_sums(
                [(lambda r: (np.asarray(weights[r]) > 0).astype(np.int64), n_rows),
                 (lambda r: np.asarray(weights[r]), n_rows)], rows=rows)
            count = count.astype(np.int32)
            cols.append(fits.Column(name="n_spectra_%s" % name, format="J",
                                    array=count))
            cols.append(fits.Column(name="weight_sum_%s" % name, format="D",
                                    array=wsum))
            total += count
        fraction = total / float(n_rows)
        cols.append(fits.Column(name="constrained", format="L",
                                array=fraction >= min_fraction))
    basis = fits.BinTableHDU.from_columns(cols, name="BASIS")

    # one row per exposure, not per cube row: with the parities tied the two
    # rows of an exposure carry byte-identical coefficients, errors and chi2,
    # so keeping both would only invite someone to average a number with itself
    tab = Table(table)
    if tied and "filename" in tab.colnames:
        _, first = np.unique(np.asarray(tab["filename"]), return_index=True)
        tab = tab[np.sort(first)]
    coeffs = fits.BinTableHDU(tab, name="COEFFS")

    power = np.concatenate([power_star, power_earth])
    var = Table()
    var["component"] = (["a%d" % (k + 1) for k in range(n_star)]
                        + ["b%d" % (j + 1) for j in range(n_earth)])
    var["block"] = ["star"] * n_star + ["earth"] * n_earth
    var["weighted_power"] = power
    var["percent_of_median_subtracted"] = 100 * power / chi2_null
    variance = fits.BinTableHDU(var, name="VARIANCE")

    fits.HDUList([primary, basis, coeffs, variance]).writeto(path, overwrite=True)
    log("  wrote %s" % path)


def plot_components(grid, P, Q, power_star, power_earth, chi2_null, path):
    """The basis vectors themselves, star block left, Earth block right.

    The coefficients say when something happened; only this says what. Reading
    the two columns against each other is how a component is identified: a star
    vector should carry stellar lines, an Earth vector should carry telluric
    residuals, and a vector that looks like the other block's is leakage.

    Samples no row covers are stored as exact zeros. They are masked to NaN
    before plotting so the line breaks at a gap instead of diving to zero and
    drawing 622 spurious spikes across the domain.
    """
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    n_rows = max(P.shape[0], Q.shape[0])
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 1.7 * n_rows + 1.2),
                             sharex=True, squeeze=False)
    blocks = [("star rest frame", P, power_star, "tab:blue", "a"),
              ("observer (Earth) rest frame", Q, power_earth, "tab:red", "b")]
    for col, (title, basis, power, colour, letter) in enumerate(blocks):
        for row in range(n_rows):
            ax = axes[row][col]
            if row >= basis.shape[0]:
                ax.axis("off")
                continue
            v = np.where(basis[row] != 0.0, basis[row], np.nan)
            ax.plot(grid, v, lw=0.25, color=colour)
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
            ax.set_ylabel("%s%d" % (letter, row + 1))
            ax.text(0.995, 0.95, "%.2f%% of the median-subtracted cube"
                    % (100 * power[row] / chi2_null),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7",
                              lw=0.5, alpha=0.85))
            if row == 0:
                ax.set_title(title)
        axes[-1][col].set_xlabel("wavelength (nm)")
    fig.suptitle("Two-frame PCA: the basis vectors", y=0.998, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log("  wrote %s" % path)


def write_variance_table(power_star, power_earth, chi2_null, chi2_best, path):
    """The same numbers as plot_variance, machine-readable."""
    n_star = power_star.size
    power = np.concatenate([power_star, power_earth])
    share = 100 * power / chi2_null
    table = Table()
    table["component"] = (["a%d" % (k + 1) for k in range(n_star)]
                          + ["b%d" % (j + 1) for j in range(power_earth.size)])
    table["block"] = ["star"] * n_star + ["earth"] * power_earth.size
    table["weighted_power"] = power
    table["percent_of_median_subtracted"] = share
    table["cumulative_percent"] = np.cumsum(share)
    table.meta["residual_rms_ratio"] = float(np.sqrt(max(chi2_best / chi2_null, 0.0)))
    table.write(path, format="csv", overwrite=True)
    log("  wrote %s" % path)



def _hierarchical_median(home, berv, bin_kms, min_entries, desc=None):
    """Median within each BERV bin, then median across bins.

    A plain median over spectra is robust to a minority, and that is exactly
    the assumption that fails here. Anything fixed in the OBSERVER frame, an OH
    airglow residual above all, lands at the same STAR-frame column for every
    exposure sharing a barycentric velocity. On TOI-2120, 23 of 80 exposures sit
    within one resolution element of each other, so 29 per cent of the sample
    votes for the same wrong value and the median is pulled toward it.

    Giving each BERV bin one vote removes that. Measured on TOI-2120 against a
    template built without the dominant group, the excess deviation on
    OH-affected columns falls from 1.46 to 1.20 times the deviation on control
    columns, and the template's own noise rises by 5 per cent.

    This is LBL's approach, not ours: see lbl/recipes/lbl_template.py, "bin cube
    by BERV (to give equal weighting to epochs)", whose BERVBIN_SIZE of 3000 m/s
    and BERVBIN_MIN_ENTRIES of 3 are the defaults used here so that the two
    steps of the chain bin the same way.
    """
    labels = np.floor((berv - np.nanmin(berv)) / bin_kms).astype(int)
    groups = []
    for value in np.unique(labels):
        rows = np.flatnonzero(labels == value)
        if rows.size >= min_entries:
            groups.append(rows)
    if len(groups) < 2:                        # not enough bins to be worth it
        return _nanmedian_over_rows(home, desc=desc)
    # a block of columns at a time: each column's medians are its own, and a
    # bin's rows copied whole would be another array of the carried cube's size
    n_cols = home.shape[1]
    out = np.empty(n_cols)
    step = _col_step(home.shape[0])
    starts = range(0, n_cols, step)
    for first in (_bar(starts, desc=desc, unit="block") if desc else starts):
        last = min(first + step, n_cols)
        part = home[:, first:last]
        per_bin = np.vstack([_nanmedian_over_rows(part[rows], step=last - first)
                             for rows in groups])
        out[first:last] = _nanmedian_over_rows(per_bin, step=last - first)
    return out

def star_frame_template(data, w, shifter, delta, min_spectra=20, berv=None,
                        berv_bin=None, berv_min_entries=3, rows=None,
                        spill=None, desc=None):
    """Component zero of the star block: the median spectrum in the star frame.

    NOTES.md 11.12: with only an observer-frame mean removed, the star block
    spends its first component (32% of the weighted variance, correlation -0.85)
    turning that mean into this template. Removing it up front hands that
    component back to the fit.

    Median rather than mean so flares and cosmics do not set the template, and
    samples a spectrum does not constrain are masked before the median rather
    than contributing a zero, which would pull the template toward zero exactly
    under the tellurics.

    `rows`, a mask or a list of indices, takes the template over those rows
    only; `data`, `w`, `delta` and `berv` are then the whole cube's. The rows
    are read and carried a block at a time, so neither the selection nor its
    weights are ever held whole, and with `spill` the carried copy, the one
    array of that size this needs, is a mapped file like the fit's own. The
    medians are the same to the bit (tests/test_low_memory.py).
    """
    index = np.arange(data.shape[0]) if rows is None else np.asarray(rows)
    if index.dtype == bool:
        index = np.flatnonzero(index)
    n_cols = data.shape[1]
    delta = np.asarray(delta)
    home = workspace((index.size, n_cols), np.float64, spill, tag="home")
    # two carried blocks, the data's and the weights', and their padded copies
    step = max(1, _row_step(n_cols) // 2)
    starts = range(0, index.size, step)
    for start in (_bar(starts, desc=desc and desc + ", to the star's frame",
                       unit="block") if desc else starts):
        pick = index[start:start + step]
        shift = -delta[pick]
        carried = shifter.rows(np.asarray(data[pick]), shift)  # observer -> star
        carried[shifter.rows(np.asarray(w[pick]), shift) <= 0] = np.nan
        home[start:start + pick.size] = carried
        del carried
    with warnings.catch_warnings():
        # a t.fits cube has columns no row covers at all (dead orders, and the
        # half of the domain the other parity owns); those are all-NaN by
        # construction and are zeroed by the min_spectra cut below
        warnings.simplefilter("ignore", RuntimeWarning)
        median_desc = desc and desc + ", median"
        if berv_bin and berv is not None:
            template = _hierarchical_median(
                home, np.asarray(berv, dtype=float)[index],
                float(berv_bin) / 1000.0, int(berv_min_entries),
                desc=median_desc)
        else:
            template = _nanmedian_over_rows(home, desc=median_desc)
    count = np.zeros(n_cols, dtype=np.int64)
    for start in starts:
        count += np.sum(np.isfinite(home[start:start + step]), axis=0)
    drop_workspace(home)
    del home
    template = np.where(np.isfinite(template), template, 0.0)
    template[count < min_spectra] = 0.0
    return template


def carry_template(template, shifter, delta):
    """S_n T for every spectrum: the template in each observer frame."""
    return shifter.carry(shifter.prepare(template[None, :]), delta)[:, 0, :]



#: samples in one block when a (rows, columns) array is worked through a piece
#: at a time: 256 MB at float64. The blocks below exist so that no step of the
#: fit holds a second array of the cube's size, and the arithmetic in each is
#: the whole-array arithmetic restricted to the block, so the numbers do not
#: change (tests/test_low_memory.py holds them side by side).
_BLOCK_SAMPLES = 32 * 1024 * 1024


def _row_step(n_columns):
    """Rows per block for arrays of `n_columns` columns."""
    return max(1, int(_BLOCK_SAMPLES // max(1, int(n_columns))))


def _col_step(n_rows):
    """Columns per block for arrays of `n_rows` rows."""
    return max(1, int(_BLOCK_SAMPLES // max(1, int(n_rows))))


def zero_unweighted(data, w, step=None):
    """data = 0 wherever w is not positive, in place and a block at a time.

    `data[w <= 0] = 0` spelled on the whole array holds a boolean of the
    cube's size, and `np.where(w > 0, data, 0)` a copy of it, which on a
    mapped cube is the copy that puts it back in memory.
    """
    step = step or _row_step(data.shape[1])
    for start in range(0, data.shape[0], step):
        stop = min(start + step, data.shape[0])
        block = data[start:stop]
        block[~(np.asarray(w[start:stop]) > 0)] = 0.0
    return data


def weighted_power_rows(data, w, step=None):
    """sum_m w data^2 for each row, in float64, a block of rows at a time."""
    step = step or _row_step(data.shape[1])
    out = np.empty(data.shape[0], dtype=np.float64)
    for start in range(0, data.shape[0], step):
        stop = min(start + step, data.shape[0])
        out[start:stop] = np.sum(w[start:stop] * data[start:stop] ** 2,
                                 axis=1, dtype=np.float64)
    return out


def weighted_power(data, w, chunk):
    """sum w data^2 over the cube, in float64, `chunk` rows at a time.

    The blocks are the fit's own chunks and are summed in that order, so the
    total is the one the fit has always quoted, to the bit.
    """
    total = 0.0
    step = max(1, int(chunk))
    for start in range(0, data.shape[0], step):
        stop = min(start + step, data.shape[0])
        total += float(np.sum(w[start:stop] * data[start:stop] ** 2,
                              dtype=np.float64))
    return total


def column_sums(arrays, rows=None):
    """Sum over rows of each array, per column, one row at a time.

    numpy's sum(axis=0) adds the rows in order into an accumulator of the
    array's own dtype; so does this, so a float32 total comes out as numpy's
    to the bit, without a copy of the rows it is taken over. `arrays` are
    callables of a row index giving that row, so a product of two arrays is
    formed one row at a time too.
    """
    rows = range(arrays[0][1]) if rows is None else rows
    totals = [None] * len(arrays)
    for r in rows:
        for i, (get, _n) in enumerate(arrays):
            value = get(r)
            if totals[i] is None:
                totals[i] = np.zeros(value.shape, dtype=value.dtype)
            totals[i] += value
    return totals


try:                                    # optional, and worth having
    import bottleneck as _bn
except ImportError:                     # pragma: no cover
    _bn = None


def _nanmedian_over_rows(z, step=None, desc=None):
    """Median over spectra, per wavelength column, ignoring NaN.

    This is the single most expensive operation in an iteration after the carry:
    two calls per clip step over an (n_spectra, 421782) array, and numpy's
    nanmedian sorts every column. Bottleneck's does the same job five to six
    times faster, but only reduces over the LAST axis, so the array is
    transposed first. That copy costs 0.05 s against the 1.9 s it saves, and it
    is exact: measured difference from numpy, 0.

    A block of columns at a time: the transposed copy, and bottleneck's own
    copy inside the call, are each the size of the array, and on a cube that
    barely fits two more of those is what sends the machine into swap (a
    three-object joint fit, 2026-09-16: 11 GB resident for a 2.8 GB cube). A
    column's median does not depend on its neighbours, so the result is the
    same to the bit.

    Falls back to numpy when bottleneck is absent, so the package keeps working
    without it and merely runs slower.
    """
    n_rows, n_cols = z.shape
    step = step or _col_step(n_rows)
    out = None
    starts = range(0, n_cols, step)
    for first in (_bar(starts, desc=desc, unit="block") if desc else starts):
        last = min(first + step, n_cols)
        part = z[:, first:last]
        if _bn is None:
            got = np.nanmedian(part, axis=0)
        else:
            got = _bn.nanmedian(np.ascontiguousarray(part.T), axis=1)
        if out is None:
            out = np.empty(n_cols, dtype=got.dtype)
        out[first:last] = got
    return out if out is not None else np.empty(0)


class ClippedWeights:
    """The clipped weights, stored as the original ones plus what changed.

    The clip lowers a few per cent of the samples at most (the run log's
    "clipped" column), and a float64 array of the cube's size to hold them is
    the largest thing a sweep allocated: 5.6 GB on a 2.8 GB float32 cube. Here
    only the samples whose factor is not exactly 1 are stored, as (row,
    column, weight), and a row or a block of rows is assembled when it is
    read: the original weights in float64, with the stored ones written over
    them. Those are the numbers the dense array held, to the bit, since a
    factor of exactly 1 times w0 is w0.

    Read by row (`w[n]`) or by block of rows (`w[start:stop]`), which is how
    every step of a sweep reads its weights; anything else assembles the whole
    array, as np.asarray does.
    """

    ndim = 2
    dtype = np.dtype(np.float64)

    def __init__(self, w0, rows, cols, values):
        self.w0 = w0
        self.shape = tuple(w0.shape)
        order = np.argsort(rows, kind="stable")
        self.cols = np.asarray(cols, dtype=np.int32)[order]
        self.values = np.asarray(values, dtype=np.float64)[order]
        counts = np.bincount(np.asarray(rows), minlength=self.shape[0])
        self.ptr = np.concatenate([[0], np.cumsum(counts)])

    def __len__(self):
        return self.shape[0]

    @property
    def nbytes(self):
        return int(self.cols.nbytes + self.values.nbytes + self.ptr.nbytes)

    def _block(self, start, stop):
        out = np.array(self.w0[start:stop], dtype=np.float64)
        lo, hi = self.ptr[start], self.ptr[stop]
        if hi > lo:
            local = np.repeat(np.arange(stop - start),
                              np.diff(self.ptr[start:stop + 1]))
            out[local, self.cols[lo:hi]] = self.values[lo:hi]
        return out

    def __getitem__(self, key):
        n = self.shape[0]
        if isinstance(key, (int, np.integer)):
            row = int(key) + n if key < 0 else int(key)
            if not 0 <= row < n:
                raise IndexError(key)
            return self._block(row, row + 1)[0]
        if isinstance(key, slice) and key.step in (None, 1):
            start, stop, _ = key.indices(n)
            return self._block(start, max(start, stop))
        return np.asarray(self)[key]

    def __array__(self, dtype=None, copy=None):
        out = self._block(0, self.shape[0])
        return out if dtype is None else out.astype(dtype, copy=False)


def clip_weights(w0, residual, clip=3.0, min_spectra=20, sparse=False,
                 desc=None):
    """Iterative soft down-weighting of local >clip-sigma outliers.

    Returns w0 scaled by a factor that is 1 inside the clip and falls as
    (clip/|z|)^2 outside it. That exponent is not arbitrary: it is what you get
    by inflating the variance of an outlying sample until it *is* a clip-sigma
    sample, so a 6-sigma point is kept with a quarter of its weight rather than
    thrown away. Nothing is ever hard-rejected, which matters because a flare is
    real data and a hard cut on residual would sculpt the very variability the
    PCA is here to find.

    "Local" is per wavelength column: z is the residual in units of the
    per-pixel photon sigma, rescaled by a robust MAD over spectra *at that
    wavelength*. So a region the model fits badly does not have its whole
    column clipped, only the spectra that are outliers within it.

    Always applied to the original weights, never compounded, so this is
    iterative re-weighting and not a ratchet that slowly deletes the data.

    A block of columns at a time, twice: the column medians first, then the
    factors, since the fallback scale of a poorly covered column is a median
    over every column. Each column's arithmetic is the whole-array arithmetic,
    so the weights are the same to the bit; what is gone is the z array of the
    cube's size and the two copies of it the medians made. With `sparse` the
    weights come back as a ClippedWeights, which stores only the samples the
    clip changed; otherwise as an array.
    """
    n_rows, n_cols = w0.shape
    step = _col_step(n_rows)

    def zblock(first, last):
        # z = residual * sqrt(w0), NaN where there is no weight
        wb = np.asarray(w0[:, first:last])
        good = wb > 0
        z = np.sqrt(wb, dtype=np.float64)
        z *= residual[:, first:last]
        z[~good] = np.nan
        return z, good, wb

    centre = np.empty(n_cols)
    scale = np.empty(n_cols)
    live = np.zeros(n_cols, dtype=np.int64)
    starts = range(0, n_cols, step)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # columns no row constrains are all-NaN; the min_spectra test below is
        # what handles them, so the median's complaint is noise
        warnings.simplefilter("ignore", RuntimeWarning)
        for first in (_bar(starts, desc=desc, unit="block") if desc else starts):
            last = min(first + step, n_cols)
            z, good, _ = zblock(first, last)
            centre[first:last] = _nanmedian_over_rows(z, step=last - first)
            z -= centre[None, first:last]
            np.abs(z, out=z)
            scale[first:last] = 1.4826 * _nanmedian_over_rows(z, step=last - first)
            live[first:last] = np.sum(good, axis=0)
        enough = live >= min_spectra
        fallback = np.nanmedian(scale[enough]) if np.any(enough) else 1.0
    scale = np.where(enough & np.isfinite(scale) & (scale > 0), scale, fallback)
    if not np.isfinite(fallback) or fallback <= 0:
        scale = np.ones_like(scale)

    # Stored sparsely only where a factor inside the clip is exactly 1, as it
    # is at clip = 3 (1/9 * 9 rounds to 1). Where it is not, every live sample
    # differs from w0 and a sparse copy would be larger than the array.
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.maximum(np.full(4, float(clip)), clip) ** -2.0 * clip ** 2
    sparse = sparse and bool(np.all(unit == 1.0))
    dense = None
    if not sparse:
        dense = np.empty((n_rows, n_cols), dtype=np.float64)
    rows, cols, values = [], [], []
    beyond = 0
    for first in starts:
        last = min(first + step, n_cols)
        z, good, wb = zblock(first, last)
        with np.errstate(invalid="ignore", divide="ignore"):
            z -= centre[None, first:last]
            np.abs(z, out=z)
            z /= scale[None, first:last]
        beyond += int(np.count_nonzero(z[good] > clip))
        with np.errstate(invalid="ignore", divide="ignore"):
            np.maximum(z, clip, out=z)      # inside the clip -> exactly clip
            z **= -2.0
            z *= clip ** 2                  # -> 1 inside, (clip/|z|)^2 outside
        z[~np.isfinite(z)] = 1.0
        if dense is not None:
            z *= wb
            dense[:, first:last] = z
            continue
        r, c = np.nonzero(z != 1.0)
        values.append(z[r, c] * wb[r, c])
        rows.append(r.astype(np.int32))
        cols.append((c + first).astype(np.int32))
    hit = float(beyond / max(int(live.sum()), 1))
    if dense is not None:
        return dense, hit
    joined = [np.concatenate(x) if x else np.empty(0, dtype=t)
              for x, t in ((rows, np.int32), (cols, np.int32),
                           (values, np.float64))]
    return ClippedWeights(w0, *joined), hit


def _clip_triplet(w):
    """(rangees, colonnes, facteur) de ce que le dernier balayage a deprecie.

    Le facteur est le rapport du poids clipe au poids d'origine: 1 la ou la
    boucle n'a rien change, moins la ou elle a doute. Vide quand le fit n'a pas
    clipe du tout, et vide aussi pour des poids denses, que seuls les petits
    cubes des tests produisent.
    """
    if w is None or not hasattr(w, "ptr"):
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float32))
    rows = np.repeat(np.arange(len(w.ptr) - 1, dtype=np.int32),
                     np.diff(w.ptr)).astype(np.int32)
    base = np.asarray(w.w0)[rows, w.cols].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = np.where(base > 0, w.values / base, 1.0)
    return rows, np.asarray(w.cols, np.int32), factor.astype(np.float32)


def coefficient_errors(data, w, P, Q, shifter, delta, a, b, chunk=64,
                       exposure=None, alpha=None, velocity_mask=None,
                       star_mean=None):
    """Per-spectrum 1-sigma errors on the coefficients, two flavours.

    The coefficient step is a weighted least squares, so under "the weights are
    inverse variances" the covariance of c_n is just the inverse of the normal
    matrix already formed in joint_coeffs:

        Cov(c_n) = (B_n^T W_n B_n)^-1 = A_n^-1

    That is `sigma_formal`. It is optimistic for two reasons, so a second column
    is reported alongside it:

    1. The weights are NOT purely inverse variances. The telluric ramp of 2.3 and
       the soft clip of 11.14 both multiply w by a distrust factor. The clip is
       self-consistent (it *is* a variance inflation, by construction), the ramp
       is not.
    2. The noise model of 2.2 is reconstructed rather than read -- APERO's eflux
       column is all zeros -- so its absolute normalisation is only as good as
       the header SNR it was tied to. Measured residual/sigma is 1.07, not 1.00.

    `sigma_scaled` multiplies by sqrt(chi2_red) per spectrum, the standard fix,
    which absorbs both. Use it.

    With `exposure` given the covariance is formed on the summed normal matrix
    of the group, and chi2_red on the group's pooled residual and pooled degrees
    of freedom, matching the tied solve in joint_coeffs. Inverting one row's
    normal matrix while the coefficient was fitted to two would overstate the
    error by roughly sqrt(2).

    **Neither includes the uncertainty in P and Q.** The bases are estimated from
    the same 764 spectra, and these errors are conditional on them being exactly
    right. Getting that honestly needs a bootstrap over spectra, refitting the
    bases each time. Expect these to be underestimates.
    """
    n_spectra, n_pixels = data.shape
    n_star, n_earth = P.shape[0], Q.shape[0]
    # the velocity column is part of the design that was fitted, so it is part
    # of the design the covariance is taken from; leaving it out would report
    # the errors of a model that was not the one solved, and its own diagonal
    # is a 1-sigma on the shift, which is a formal RV uncertainty for free
    n_vel = 1 if alpha is not None else 0
    n_tot = n_star + n_earth + n_vel
    eye = np.eye(n_tot)
    var = np.full((n_spectra, n_tot), np.nan)
    chi2_red = np.full(n_spectra, np.nan)
    coeffs = np.hstack([a, b] + ([alpha[:, None]] if n_vel else []))
    Pf = shifter.prepare(P)
    Tf = (shifter.prepare(star_mean[0])
          if n_vel and star_mean is not None else None)
    B = np.empty((n_tot, n_pixels))
    tied = exposure is not None
    amats = np.zeros((n_spectra, n_tot, n_tot)) if tied else None
    chi2_rows = np.zeros(n_spectra)
    dof_rows = np.zeros(n_spectra)
    for start in range(0, n_spectra, chunk):
        stop = min(start + chunk, n_spectra)
        SP = shifter.carry(Pf, delta[start:stop])
        TT = (carried_means(Tf, star_mean[1], shifter, delta, start, stop)
              if Tf is not None else None)
        for i in range(stop - start):
            n = start + i
            B[:n_star] = SP[i]
            B[n_star:n_star + n_earth] = Q
            if n_vel:
                whole = a[n] @ SP[i] if TT is None else a[n] @ SP[i] + TT[i]
                B[-1] = velocity_column(whole, velocity_mask)
            wn = w[n]
            amat = (B * wn) @ B.T
            resid = data[n] - coeffs[n] @ B
            chi2_rows[n] = float(np.sum(wn * resid ** 2, dtype=np.float64))
            dof_rows[n] = int(np.count_nonzero(wn))
            if tied:
                amats[n] = amat
                continue
            trace = np.trace(amat)
            if trace <= 0:
                continue
            try:
                cov = np.linalg.inv(amat + 1e-12 * trace * eye)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(amat)
            var[n] = np.clip(np.diag(cov), 0.0, None)
            dof = dof_rows[n] - n_tot
            if dof > 0:
                chi2_red[n] = chi2_rows[n] / dof
    if tied:
        exposure = np.asarray(exposure)
        for value in np.unique(exposure):
            rows = np.where(exposure == value)[0]
            amat = amats[rows].sum(axis=0)
            trace = np.trace(amat)
            if trace <= 0:
                continue
            try:
                cov = np.linalg.inv(amat + 1e-12 * trace * eye)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(amat)
            var[rows] = np.clip(np.diag(cov), 0.0, None)
            dof = dof_rows[rows].sum() - n_tot
            if dof > 0:
                chi2_red[rows] = chi2_rows[rows].sum() / dof
    sigma_formal = np.sqrt(var)
    sigma_scaled = sigma_formal * np.sqrt(chi2_red)[:, None]
    # The velocity sigma travels on its own and NOT as one more column of the
    # matrix. Every caller slices [:, n_star:] for the observer block, and an
    # extra column on the end silently becomes an eighth observer error bar:
    # it did, and the run died two hours in with "could not broadcast (8,) into
    # (7,)" from the plotting, after the fit had already been paid for.
    sigma_vel = sigma_scaled[:, -1] if n_vel else None
    if n_vel:
        sigma_formal = sigma_formal[:, :-1]
        sigma_scaled = sigma_scaled[:, :-1]
    return sigma_formal, sigma_scaled, chi2_red, sigma_vel


def mad_outliers(coeffs, parity, threshold, already=None):
    """Rows whose coefficient on ANY component sits too far from the others.

    Robust z-score, |c - median| / (1.4826 MAD), component by component; a row
    is flagged if any single component exceeds the threshold.

    THE THRESHOLD IS IN SIGMAS, NOT IN MADS. The 1.4826 puts the MAD on the
    scale of a standard deviation for Gaussian data, so `threshold=10` rejects
    beyond 10 robust sigmas, which is 14.83 plain MADs. The option was called
    --max-mad for a while and that name was simply wrong; it survives as an
    alias for --max-sigma so that existing configs keep their exact behaviour.
    Nothing here reports a "MAD" that carries the factor: where a MAD is
    reported, in the RV compilation and the run summaries, it is
    median(|x - median x|) with no scaling.

    Computed within each parity group rather than over the pooled rows. The two
    parities are different populations here -- the star block's coefficients are
    very nearly pure even-minus-odd -- and pooling them makes each component's
    distribution bimodal, which inflates the MAD by roughly the separation of
    the two modes and hides every real outlier inside it. With an s1d cube there
    is one group and this reduces to the plain cut.
    """
    n_rows = coeffs.shape[0]
    flag = np.zeros(n_rows, dtype=bool)
    if threshold <= 0:
        return flag
    live = np.ones(n_rows, dtype=bool) if already is None else ~np.asarray(already)
    for value in np.unique(parity):
        rows = (parity == value) & live
        if rows.sum() < 5:                 # too few for a robust scale
            continue
        block = coeffs[rows]
        centre = np.median(block, axis=0)
        mad = 1.4826 * np.median(np.abs(block - centre), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(block - centre) / np.where(mad > 0, mad, np.inf)
        flag[np.where(rows)[0][z.max(axis=1) > threshold]] = True
    return flag


def correlate_and_plot(outdir, a, b, labels, values, title=""):
    """Write correlations.pdf and correlations.csv for one fit.

    This runs on every fit rather than living in a diagnostic script somebody
    has to remember. A component carries no label: that a1 varies on 83 days is
    a result only for as long as it is not simply tracking the water column, and
    the cheapest way to keep that honest is to draw the matrix every time.
    """
    from .plotting import (plot_correlations, rank_correlations,
                           report_correlations, write_correlation_table)

    comps = [("a%d" % (k + 1), a[:, k]) for k in range(a.shape[1])]
    comps += [("b%d" % (j + 1), b[:, j]) for j in range(b.shape[1])]
    if not labels:
        log("  no ancillary quantity available, skipping correlations.pdf")
        return
    rho = rank_correlations(comps, np.asarray(values, dtype=float))
    plot_correlations(rho, comps, list(labels),
                      os.path.join(outdir, "correlations.pdf"), title=title)
    write_correlation_table(rho, comps, list(labels),
                            os.path.join(outdir, "correlations.csv"))
    report_correlations(rho, comps, list(labels))


def replot(outdir, cube=None):
    """Regenerate every figure from a saved fit.npz, without touching the cube.

    fit.npz has always been advertised as "re-plot without refitting"; this is
    the switch that makes good on it. It buys more than convenience: the figures
    are written only after the last iteration, so until now a half-hour fit that
    produced a plot nobody liked had to be run again from the top to change a
    label. Everything plot_coeffs and plot_variance need is in the archive.
    """
    path = os.path.join(outdir, "fit.npz")
    fit = np.load(path)
    n_star = fit["a"].shape[1]
    log("re-plotting %s: %d star + %d Earth components, %d rows, iterate %d"
          % (path, n_star, fit["b"].shape[1], fit["bjd"].size, fit["best_iter"]))
    sigma = fit["sigma_scaled"]
    chi2_null = float(fit["chi2_null"])
    chi2_best = float(fit["chi2_best"]) if "chi2_best" in fit.files else np.nan
    keep = ~fit["rejected"] if "rejected" in fit.files else np.ones(
        fit["bjd"].size, dtype=bool)
    if not keep.all():
        log("  %d of %d rows were rejected by the MAD cut and are not"
              " plotted" % (int((~keep).sum()), keep.size))
    plot_coeffs(fit["bjd"][keep], fit["a"][keep], fit["b"][keep],
                fit["power_star"], fit["power_earth"], chi2_null,
                os.path.join(outdir, "coefficients_vs_time.pdf"),
                err_a=sigma[keep][:, :n_star], err_b=sigma[keep][:, n_star:])
    plot_variance(fit["power_star"], fit["power_earth"], chi2_null, chi2_best,
                  os.path.join(outdir, "variance.pdf"))
    write_variance_table(fit["power_star"], fit["power_earth"], chi2_null,
                         chi2_best, os.path.join(outdir, "variance.csv"))
    if "grid" in fit.files:
        plot_components(fit["grid"], fit["P"], fit["Q"], fit["power_star"],
                        fit["power_earth"], chi2_null,
                        os.path.join(outdir, "components.pdf"))
    else:
        log("  no grid in the archive, skipping components.pdf")
    # the ancillary quantities are stored in the archive, so the correlation
    # matrix redraws without the cube or the source spectra being present
    if "anc_values" in fit.files and fit["anc_values"].size:
        correlate_and_plot(outdir, fit["a"][keep], fit["b"][keep],
                           [str(x) for x in fit["anc_labels"]],
                           fit["anc_values"][:, keep],
                           title=exposures_label(fit["filename"] if "filename"
                                                 in fit.files else None, keep,
                                                 _per_row(fit),
                                                 row_unit(cube) if cube
                                                 else "nights"))
    elif cube and "filename" in fit.files:
        # an archive written before the ancillary table was stored: rebuild it
        # from the cube, matching on filename so a rejected row cannot shift the
        # columns against the coefficients
        from astropy.table import Table
        from .plotting import ancillary_table, source_directory
        meta_path = os.path.join(cube, "meta.fits")
        if not os.path.exists(meta_path):
            log("  no %s, skipping correlations.pdf" % meta_path)
            return
        names = [os.path.basename(str(v)) for v in fit["filename"]]
        meta = Table.read(meta_path)
        labels, values = ancillary_table(meta, names,
                                         source_dir=source_directory(cube))
        # the cube's rows are not the fit's rows once a quality cut has dropped
        # one, so n_exposures is matched on the file name, the way the
        # ancillary columns just were, and never by position
        per_row = _per_row(fit)
        if per_row is None and _per_row(meta) is not None:
            by_name = {os.path.basename(str(f)).strip(): n for f, n
                       in zip(meta["filename"], _per_row(meta))}
            per_row = np.array([by_name.get(str(n).strip(), 1.0) for n in names])
        correlate_and_plot(outdir, fit["a"][keep], fit["b"][keep], labels,
                           values[:, keep] if len(labels) else values,
                           title=exposures_label(names, keep, per_row,
                                                 row_unit(meta)))
    else:
        log("  no ancillary quantities in the archive and no --cube given,"
              " skipping correlations.pdf")


def main(argv=None):
    args = parse_args(argv)
    if not args.outdir:
        raise SystemExit("--outdir is required: where the fit's products go")
    if args.replot:
        replot(args.outdir, args.cube)
        return None                  # exit status, not a value; see below
    if not args.cube:
        raise SystemExit("--cube is required: the cache/cube_<format>_<key>"
                         " directory the cube stage wrote")
    # float32 halves the two (rows, pixels) arrays the fit lives in, and the
    # two temporaries the model costs with them. On 321 individual exposures of
    # TOI-2120 that is the difference between 11.9 GB and 6 GB, which is the
    # difference between fitting every file and coadding nights. Every weighted
    # sum still accumulates in float64: a float32 accumulator over 3.7e8 terms
    # loses the low bits of the very totals the fit is judged by.
    rows, columns = cube_shape(args.cube)
    spill = spill_dir(rows, columns, args.dtype, args.cube, args.outdir)
    grid, data, w, meta = load_cube(args.cube, min_snr_frac=args.min_snr_frac,
                                    dtype=np.dtype(args.dtype), spill=spill)
    # LA VALIDATION CROISEE, AVANT TOUTE MOYENNE. Les nuits tenues a l'ecart
    # perdent leur poids ici, donc avant les moyennes par parite, l'etoile
    # moyenne et le premier balayage: la base ne doit rien en apprendre, pas
    # meme leur contribution au spectre moyen. Elles gardent leur place dans
    # chaque tableau, comme une ligne rejetee par la coupure MAD, et sortent
    # marquees `rejected` dans fit.npz.
    held = None
    if getattr(args, "holdout", None):
        from .diagnostics import holdout_rows
        held = holdout_rows(np.asarray(meta["bjd"], dtype=float), args.holdout)
        w[held] = 0.0
        data[held] = 0.0
        log("HOLDOUT: %d of %d rows (the %s-ranked nights) held out of the fit;"
            " the correct stage refits their amplitudes against a basis that"
            " never saw them" % (int(held.sum()), held.size, args.holdout),
            "warn")
    log("fit arrays in %s: %.1f GB for the data and the weights together"
          % (data.dtype.name, 2 * data.nbytes / 1e9))
    n_spectra, n_pixels = data.shape
    parity = row_parity(meta, n_spectra)
    n_parities = np.unique(parity).size
    if n_parities > 1:
        log("t.fits cube: %d rows = %d exposures x %d order parities, two"
              " measurements of the same photons at two places on the detector,"
              " carrying the same BERV." % (n_spectra, n_spectra // n_parities,
                                            n_parities))
    tie = row_exposure(meta, n_spectra) if args.tie_parities else None
    if tie is not None and n_parities > 1:
        log("tying the %d parity rows of each exposure to one set of"
              " coefficients (--no-tie-parities to undo)" % n_parities)
    dv = float(np.median(np.diff(np.log(grid))) * C_KMS)
    # delta is the STAR -> OBSERVER shift in samples, i.e. what S_n applies.
    #
    # APERO's s1d_v wavelengths are in the observer frame, and a stellar feature
    # sits at lambda_bary = lambda_obs * D(BERV), D the relativistic Doppler
    # factor (grids.doppler). On a log-uniform grid that is a translation of
    # +atanh(BERV/c) / (dv/c) samples (grids.pixel_shift), so observer -> star
    # is +pixel_shift(BERV) and the inverse, star -> observer, is **minus** it.
    #
    # This had the wrong sign until 2026-08-24 and the error was silent: the fit
    # still converged, to a "star" block anchored to a mirror frame moving at
    # -BERV, which corresponds to nothing physical. See NOTES.md 11.11. Verified
    # against the pipeline's own barycentric cube: rows(observer, +BERV/dv)
    # reproduces it to 1.4% of the data rms, rows(observer, -BERV/dv) does not.
    delta = -pixel_shift(np.asarray(meta["berv"], dtype=np.float64), dv)
    shifter = SHIFTERS[args.shift](
        n_pixels, a=args.kernel_halfwidth,
        max_shift=int(np.ceil(np.abs(delta).max())) + 2,
    )
    # spline unless something asks for the older grid: the default lives in
    # config.DEFAULTS, and this fallback is for a caller that built its own args
    if getattr(args, "star_basis", None) in (None, "spline"):
        # the star side as one cubic B-spline, carried by evaluating it at each
        # row's shifted positions and updated exactly; data and weights still
        # go through the Lanczos kernel (pca2d.splinestar)
        from .splinestar import SplineStar
        shifter = SplineStar(shifter)
        log("the star as one cubic B-spline with a knot per sample, evaluated at"
            " each exposure's shifted positions and updated by the exact banded"
            " normal equations; the data and weights are still carried by the"
            " Lanczos kernel", "value")
    if args.gap_guard and getattr(shifter, "banded", False):
        # the holes are the star frame's: columns the basis cannot be
        # constrained at once every row is carried home. Twice, since dropping
        # the samples beside a hole can take a column next to it under the floor
        for _ in range(2):
            gap_guard(w, delta, args.kernel_halfwidth,
                      live=star_support(w, delta, shifter))
        zero_unweighted(data, w)           # in place: the data may be mapped
    elif args.gap_guard:
        log("gap guard skipped: it is defined for a compactly supported"
              " kernel, and %s is not one" % args.shift)
    n_max = max(args.n_star, args.n_earth)
    # The carried basis is (chunk, K, n_pixels) float64 and is the single largest
    # transient in the run: at K=5 and 421782 columns one spectrum costs 17 MB,
    # so the chunk is sized to keep that buffer near 128 MB rather than to any
    # round number. Bigger is not faster; the loop is bandwidth-bound, which is
    # also why threading carry() helps and why the tap loop was rewritten as a
    # single contraction over a sliding-window view.
    chunk = args.chunk or max(4, int(128e6 / (8 * n_max * n_pixels)))
    log("carry chunk %d rows -> %.0f MB per (chunk, K, M) buffer"
          % (chunk, 8 * chunk * n_max * n_pixels / 1e6))
    log("cube %s: N=%d M=%d dv=%.3f km/s  shifts %.1f..%.1f pix"
          % (os.path.basename(args.cube), n_spectra, n_pixels, dv,
             delta.min(), delta.max()))
    log("shift operator: %s%s"
          % (args.shift, " (a=%d)" % args.kernel_halfwidth
             if args.shift == "lanczos" else ""))

    # Everything is scored against the raw high-passed cube so that the numbers
    # are comparable across runs and against the template baseline of 11.12.
    tick = time.time()
    log("scoring the cube as it is read: its weighted variance, row by row")
    chi2_raw_rows = weighted_power_rows(data, w)
    chi2_raw = float(chi2_raw_rows.sum())
    log("  raw weighted variance %.6g, in %.1f s"
        % (chi2_raw, time.time() - tick), "value")
    # The weights as read, which the clip starts from at every sweep and the
    # scores are taken against. The SAME array, not a copy: the clip writes
    # its weights elsewhere (ClippedWeights), and the MAD cut, the only thing
    # that changes these, changes them for both. A copy was one more array of
    # the cube's size in memory, a mapped cube's included.
    w0 = w

    iterate = args.mean == "iterate"
    if args.template and iterate:
        log("the one-shot star-frame template is not used with --mean iterate:"
            " a star-frame mean per parity replaces it, re-estimated at every"
            " sweep", "warn")
    if args.template and not iterate:
        template = star_frame_template(
            data, w, shifter, delta,
            berv=np.asarray(meta["berv"], dtype=float),
            berv_bin=args.template_berv_bin,
            berv_min_entries=args.template_berv_min_entries)
        template_model = carry_template(template, shifter, delta)
        data = data - template_model
        zero_unweighted(data, w)
        log("subtracted the star-frame median template: %.4f of the raw"
              " weighted variance left" % (float(np.sum(w * data ** 2, dtype=np.float64)) / chi2_raw))
    else:
        template = np.zeros(n_pixels)
        template_model = None

    # component zero of the *Earth* block, on whatever the template left behind.
    # One mean per parity, not one overall: a t.fits cube holds two rows per
    # exposure and they carry a static even-minus-odd offset, fixed in the
    # instrument frame, which a single mean would leave in the residual for a
    # component to spend itself on (NOTES.md 13.9). With an s1d cube there is
    # one group and this is the old code path exactly.
    if args.mean == "star" and not iterate:
        # zeroed below before anything reads them: the pass over the cube that
        # would compute them is saved
        parity_groups = np.unique(parity)
        means = np.zeros((parity_groups.size, n_pixels))
    else:
        tick = time.time()
        log("the observer-frame mean of each order parity, row by row")
        means, parity_groups = parity_means(data, w, parity)
        log("  in %.1f s" % (time.time() - tick))
    mean = means[0] if means.shape[0] == 1 else np.average(
        means, axis=0, weights=[np.sum(parity == g) for g in parity_groups])
    group = np.searchsorted(parity_groups, parity)
    # What that mean holds splits in two, and only one half should come out
    # here. The even-minus-odd offset is instrumental and has to go, for the
    # reason above. The part the two parities SHARE is whatever is static in
    # the observer's frame, which is the atmosphere: the airglow emission that
    # sits at the same wavelength every night lands in it at full amplitude.
    # Subtracting that before the fit hid it from the components, and while the
    # correction removed the components and never the means it was then never
    # removed from anything: hence leaving it in, at the cost of one observer
    # component. Since 2026-09-10 the correction divides out the parity mean as
    # well (reconstruct.order_correction), so what is subtracted here comes out
    # of the corrected files too.
    # the star's spectra and components smoothed to star_smooth resolution
    # elements, as LBL smooths its templates (pca2d.resolution); the observer
    # block is not. star_resolution is the older spelling: one element of it
    star_fwhm = None
    resolution = getattr(args, "resolution", None)
    fraction = getattr(args, "star_smooth", None)
    if getattr(args, "star_resolution", None) and not fraction:
        resolution, fraction = args.star_resolution, 1.0
    if resolution and fraction:
        from .resolution import fwhm_samples
        star_fwhm = fwhm_samples(resolution, dv, fraction)
        log("the star side smoothed to %.2f of a resolution element of R = %.0f:"
            " %d samples of %.2f km/s FWHM, LBL's template filter; the observer"
            " components are not smoothed"
            % (float(fraction), float(resolution), star_fwhm, dv), "value")
    # A FROZEN OBSERVER BASIS (pca2d.basis): the components a fit learned on
    # stars whose two frames separate, taken as they are and never updated,
    # for a star whose BERV barely moves and cannot separate them itself.
    # Only the star's block and the amplitudes are fitted. Read here, before
    # the star's spectrum, so that it can come first (observer_first)
    frozen = None
    if getattr(args, "earth_basis", None):
        from .basis import Basis
        frozen = Basis(args.earth_basis)
        Q_frozen = frozen.on_grid(grid, args.n_earth)
        log("observer basis FROZEN, %d components of %s"
            % (Q_frozen.shape[0], frozen.describe()), "info")
        outside = frozen.check_dates(
            np.asarray(meta["bjd"], dtype=float) - 2400000.5)
        if outside:
            log("%d of %d rows are more than 60 days outside the dates the"
                " basis was trained on" % (outside, n_spectra), "warn")
    observer_first = frozen is not None and bool(
        getattr(args, "observer_first", True))
    templates = np.zeros_like(means)
    star_mean = None
    if args.mean == "star":
        # ONE STAR SPECTRUM PER ORDER PARITY, in the star's frame, and no mean
        # in the observer's (2026-09-11). Where the orders overlap the two
        # parities see the same lines at different depths (Proxima, 1201.1 nm:
        # every line shallower in the odd orders), a difference that moves with
        # the star and that no observer-frame mean can hold. Outside the
        # overlaps the 'offset' mean is half of the observer-frame mean, a
        # BERV-smeared copy of the star the observer block then has to give
        # back; dividing out either one alone cost Proxima 46 and 48 m/s.
        # Estimated once, before any component, from each parity's own rows
        # (the BERV-binned median that also starts --mean iterate), carried to
        # every row and taken out. Nothing is re-estimated afterwards.
        berv_rows = np.asarray(meta["berv"], dtype=float)
        names = group_names(templates.shape[0])
        if observer_first:
            # THE OBSERVER FIRST (2026-10-22). With little BERV, what is
            # fixed on the detector is nearly fixed on the star too, and a
            # star spectrum estimated from the data as they are takes its
            # MEAN in: the components then describe only the variations
            # around it, the mean is never divided out, and LBL finds it in
            # its template (TOI-904: 11% of the BERV in the velocities;
            # KIC 12060710, the OH left by the sky: 78%). With the observer
            # components known beforehand, their amplitudes are fitted on the
            # data as they are, the star's lines being noise to components
            # that never saw them at this velocity, taken out, and the star's
            # spectrum estimated only then; they are put back for the sweeps
            tick = time.time()
            _, b_first, _, _ = joint_coeffs(
                data, w, np.zeros((0, n_pixels)), Q_frozen, shifter, delta,
                chunk=chunk, exposure=tie, desc="observer first, amplitudes")
            for s0 in range(0, n_spectra, chunk):
                s1 = min(s0 + chunk, n_spectra)
                data[s0:s1] -= (b_first[s0:s1] @ Q_frozen).astype(data.dtype)
            log("the OBSERVER FIRST: the %d frozen components fitted on the"
                " data as they are and taken out (amplitudes %s rms) before"
                " the star's spectrum is estimated, in %.1f s"
                % (Q_frozen.shape[0], np.array2string(
                    np.std(b_first, axis=0), precision=3),
                   time.time() - tick), "info")
        for g in range(templates.shape[0]):
            rows_g = group == g
            tick = time.time()
            log("star spectrum of the %s rows (%d of %d): %d rows carried into"
                " the star's frame, then %s"
                % (names[g], g + 1, templates.shape[0], int(rows_g.sum()),
                   "a median per %.1f km/s of BERV and a median of those"
                   % (float(args.template_berv_bin) / 1000.0)
                   if args.template_berv_bin else "their median"))
            templates[g] = star_frame_template(
                data, w, shifter, delta, rows=rows_g,
                berv=berv_rows, berv_bin=args.template_berv_bin,
                berv_min_entries=args.template_berv_min_entries,
                spill=spill, desc="star spectrum, %s rows" % names[g])
            if star_fwhm:
                from .resolution import smooth
                templates[g] = smooth(templates[g], star_fwhm)
            log("  %s rows done in %.1f s, %d columns defined"
                % (names[g], time.time() - tick,
                   int(np.count_nonzero(templates[g]))), "value")
        if observer_first:
            # the observer components back in: the sweeps fit them with the
            #   star's spectrum, which no longer holds their mean
            for s0 in range(0, n_spectra, chunk):
                s1 = min(s0 + chunk, n_spectra)
                data[s0:s1] += (b_first[s0:s1] @ Q_frozen).astype(data.dtype)
            zero_unweighted(data, w)
        tick = time.time()
        log("carrying the %d star spectra to each of the %d rows and"
            " subtracting them" % (templates.shape[0], n_spectra))
        subtract_carried(data, templates, group, shifter, delta, chunk, w=w,
                         desc="star spectra, out of every row")
        zero_unweighted(data, w)
        log("  in %.1f s" % (time.time() - tick))
        means = np.zeros_like(means)
        star_mean = (templates, group)
        both = np.all(templates != 0.0, axis=0)
        log("one star spectrum per order parity, in the star's frame, and no"
            " observer-frame mean; where both parities have one they differ by"
            " %.4g rms" % (float(np.std((templates[0] - templates[-1])[both]))
                           if both.any() else np.nan), "value")
    if iterate:
        # Component zero of each block before any component exists (NOTES
        # 11.13). With MEAN_INIT "template", the star-frame median of each
        # parity first and the observer mean of what it leaves; with "zero",
        # the observer mean alone. Then alternations between the two frames,
        # so the components start on data whose static content each frame
        # already holds.
        if MEAN_INIT == "template":
            berv_rows = np.asarray(meta["berv"], dtype=float)
            for g in range(templates.shape[0]):
                rows_g = group == g
                templates[g] = star_frame_template(
                    data, w, shifter, delta, rows=rows_g,
                    berv=berv_rows, berv_bin=args.template_berv_bin,
                    berv_min_entries=args.template_berv_min_entries,
                    spill=spill, desc="star-frame mean %d" % (g + 1))
            subtract_carried(data, templates, group, shifter, delta, chunk, w=w)
            means, _ = parity_means(data, w, parity)
        subtract_means(data, means, group)
        zero_unweighted(data, w)
        for _ in range(MEAN_INIT_ROUNDS):
            step_t, step_o = update_means(data, w, templates, means, group,
                                          shifter, delta, chunk,
                                          desc="per-parity means, both frames")
        star_mean = (templates, group)
        log("one mean per order parity in EACH frame, re-estimated at every"
            " sweep with the components centred; after %d alternations the"
            " last moved the star-frame mean by %.2e and the observer-frame one"
            " by %.2e rms" % (MEAN_INIT_ROUNDS, step_t, step_o), "value")
    common = means.mean(axis=0)
    if args.mean == "offset":
        means = means - common
    if not iterate:
        subtract_means(data, means, group)
        if np.any(means):
            zero_unweighted(data, w)
    if args.mean == "offset":
        live_any = w.sum(axis=0) > 0
        log("left the shared part of the observer-frame mean IN the data,"
              " %.4g rms, for the Earth block to describe"
              % float(np.std(common[live_any])))
    if means.shape[0] > 1 and not iterate and args.mean != "star":
        both = np.ones(n_pixels, dtype=bool)
        for i, value in enumerate(parity_groups):
            both &= w[parity == value].sum(axis=0) > 0
        log("subtracted one mean per parity; the even/odd offset it removed is"
              " %.4g rms over the %d columns both cover"
              % (float(np.std((means[0] - means[1])[both])) if both.any() else np.nan,
                 int(both.sum())))
    # ONE scratch array of (rows, columns), made here and reused by every step
    # that needs one: the two basis updates, the re-weighting and the scoring.
    # Each of them used to allocate its own, and each of those is the size of
    # the data itself.
    log("the sweeps' scratch array, %d x %d float64: %.1f GB %s"
        % (n_spectra, n_pixels, 8 * n_spectra * n_pixels / 1e9,
           "in a mapped file in %s" % spill if spill else "in memory"))
    work = workspace((n_spectra, n_pixels), np.float64, spill, tag="work")
    tick = time.time()
    chi2_null = weighted_power(data, w, chunk)
    log("after the %s as well: %.4f of the raw weighted variance left"
        " (scored in %.1f s)"
        % ("per-parity means of both frames" if iterate
           else "star's spectrum per parity" if args.mean == "star"
           else "observer-frame mean", chi2_null / chi2_raw,
           time.time() - tick))
    if args.clip > 0:
        log("soft clip at %.1f sigma, local per wavelength column" % args.clip)

    # Random orthonormal start. Orthonormal because the update steps
    # re-orthonormalise anyway and starting inside the constraint set avoids a
    # first iteration spent getting there; random because the fit converges in a
    # dozen iterations from anywhere, so a cleverer seed buys nothing and would
    # tie the result to whatever produced it. The seed is fixed, so a rerun with
    # the same cube and options reproduces the same basis.
    rng = np.random.default_rng(0)
    P = np.linalg.qr(rng.normal(size=(n_pixels, args.n_star)))[0].T.copy()
    Q = np.linalg.qr(rng.normal(size=(n_pixels, args.n_earth)))[0].T.copy()
    log("starting from a random orthonormal basis, seed fixed so a rerun on the"
        " same cube reproduces it")
    if frozen is not None:
        Q = Q_frozen

    # The velocity column of joint_coeffs. `a_lin` is the star model it is
    # linearised around, so there is none to build on the very first solve and
    # the term joins from the second one on, which is also when there is
    # anything for it to describe. With the term off it stays None throughout
    # and every solve is the one this package had before.
    velocity_term = bool(getattr(args, "velocity_term", False))
    a_lin = None
    alpha = np.zeros(n_spectra)
    alpha_prev = alpha
    velocity_mask = None
    if velocity_term:
        # Where the shift is allowed to be measured: inside a photometric band
        # and clear of tellurics. Both halves matter. The bands are per
        # instrument, so NIRPS drops K by simply not defining it rather than by
        # anyone remembering to; and a column whose transmission collapses on a
        # bad night is a column where the flux is atmosphere, the star model is
        # least trustworthy, and a velocity fitted there is fitted to the wrong
        # thing.
        from . import grids as _grids
        from .config import DEFAULTS as _D
        cfg_dom = {}
        if args.config:
            from .config import load_config as _load
            cfg_dom = (_load(args.config).get("domain") or {})
        bands = cfg_dom.get("bands") or _D["domain"]["bands"]
        band_ok, used = _grids.band_mask(grid, bands)
        threshold = float(getattr(args, "velocity_min_transmission", 0.95) or 0)
        clean = clean_columns(args.cube, threshold) if threshold > 0 else None
        velocity_mask = band_ok if clean is None else (band_ok & clean)
        live = w0.sum(axis=0) > 0
        log("shift measured on %d of %d columns: bands %s%s, and %.0f%% of the"
            " grid survives both"
            % (int((velocity_mask & live).sum()), int(live.sum()),
               "+".join(used),
               ", transmission above %.2f in 90%% of exposures" % threshold
               if clean is not None else " (no transmission in the cube)",
               100.0 * (velocity_mask & live).sum() / max(int(live.sum()), 1)),
            "value")
        velocity_mask = velocity_mask.astype(np.float64)
    if velocity_term:
        log("fitting one velocity per exposure alongside the components, as the"
            " derivative of the reconstructed star: it keeps the star's own"
            " motion out of the observer block, which would otherwise describe"
            " it and the correction would then divide it back out of the flux",
            "info")
    else:
        log("velocity term OFF: whatever the star's own motion leaves in the"
            " residual is free to end up in the observer block, and to be"
            " divided out of the corrected flux with it", "warn")
    log("each sweep solves the coefficients of both blocks jointly, then"
        " updates each basis; on %d rows x %d samples that is minutes, and one"
        " line is printed at the end of each" % (n_spectra, n_pixels))

    # This alternation is NOT guaranteed to decrease chi2, and on this dataset it
    # does not: it peaks around iteration 6 and then degrades while the condition
    # number runs away (see NOTES.md 11.9). Two approximations are responsible --
    # the diagonal stand-in for S_n^T W_n S_n in update_star, and the
    # re-orthonormalisation after each M-step. Until the star update is made an
    # exact minimiser, keep the best iterate rather than the last one, and stop
    # once it has clearly turned over.
    rejected = (held.copy() if held is not None
                else np.zeros(n_spectra, dtype=bool))
    for mad_round in range(int(args.max_mad_rounds) + 1):
        best = None
        best_chi2 = best_at = None
        worse = 0
        prev_chi2, flat = None, 0
        hit = 0.0
        a_prev = b_prev = None      # carried between iterations, see the clip step
        # ce que le dernier balayage aura deprecie, pour la correction: None
        # tant que la boucle n'a pas clipe (balayage 0, ou clip desactive)
        last_clip = None
        # A sweep on a few hundred exposures takes minutes. There is no bar
        # over the sweeps: each phase INSIDE a sweep has its own, labelled
        # "sweep N: <phase>", which vanishes when the phase ends, and the
        # sweep's one-line summary is printed once they are all gone, so
        # nothing draws over it. A slow sweep and a wedged one still look
        # different, which is what the bars are for.
        for iteration in range(args.iters):
            _set_label("sweep %d" % iteration)
            # Re-weight before fitting, using the model as it stands. Iteration 0 has
            # no model yet, so the residual is the data itself, which is the right
            # thing: it catches cosmics and flares against the template before the
            # components ever see them.
            if args.clip > 0:
                # The coefficients this needs are the ones the previous
                # iteration ended on: same P, same Q, same w, so solving again
                # returns the same numbers. Reusing them removes one of the four
                # coefficient solves per iteration, which is the single most
                # expensive call in the loop. Only iteration 0 has nothing to
                # reuse.
                if a_prev is None:
                    a_prev, b_prev, alpha_prev, _ = joint_coeffs(
                        data, w, P, Q, shifter, delta, chunk=chunk,
                        exposure=tie, velocity_from=a_lin,
                        velocity_mask=velocity_mask, star_mean=star_mean,
                        desc="re-weighting, coefficients")
                # the residual straight into the workspace: `star_model(...)
                # + b @ Q` and then `data - model` is three arrays of this
                # shape for one that is used once
                _chi2, resid_prev = model_residual(
                    data, P, Q, a_prev, b_prev, shifter, delta, chunk,
                    alpha=alpha_prev, velocity_mask=velocity_mask,
                    star_mean=star_mean, out=work)
                # stored as w0 and the samples the clip changed: a float64
                # array of the cube's size otherwise, the largest one a sweep
                # used to hold
                w, hit = clip_weights(w0, resid_prev, clip=args.clip,
                                      sparse=True, desc="re-weighting, clip")
                # CE QUE LA BOUCLE A CONCLU, garde pour la correction. Le
                # facteur etait recalcule et jete a chaque balayage, donc les
                # fichiers corriges ne savaient rien de ce que l'ajustement
                # avait renonce a decrire: un echantillon juge indigne de
                # contraindre le modele etait quand meme livre a LBL, corrige
                # par un modele qu'il n'avait pas aide a construire
                # (2026-09-25). Creux: 763 041 echantillons sur 197 millions
                # sur TOI-2120.
                last_clip = w

            # ---- E-step: coefficients for both blocks at fixed bases --------
            # Solved JOINTLY, never one block then the other. The off-diagonal
            # part of the normal matrix is the overlap between the carried star
            # basis and the fixed Earth basis, and it is exactly what lets the
            # fit attribute a feature to one frame rather than the other. Solve
            # the blocks separately and each one claims whatever it can reach.
            tick = time.time()
            a, b, alpha, cond = joint_coeffs(data, w, P, Q, shifter, delta,
                                             chunk=chunk, exposure=tie,
                                             velocity_from=a_lin,
                                             velocity_mask=velocity_mask, star_mean=star_mean,
                                             desc="coefficients")
            a_lin = a if velocity_term else None
            t_coeff = time.time() - tick

            # ---- M-step: one basis, then re-solve, then the other -----------
            # The re-solve in the middle is not optional. update_earth needs the
            # coefficients that go with the basis update_star has just produced;
            # feeding it the stale ones makes the second update fit a residual
            # that no longer exists, and the alternation stops descending.
            tick = time.time()
            if args.order == "star_first":
                P = update_star(data, w, P, Q, a, b, shifter, delta, chunk,
                                work=work,
                                alpha=alpha, velocity_mask=velocity_mask,
                                star_mean=star_mean, star_fwhm=star_fwhm)
                a, b, alpha, _ = joint_coeffs(data, w, P, Q, shifter, delta,
                                              chunk=chunk, exposure=tie,
                                              velocity_from=a_lin,
                                             velocity_mask=velocity_mask, star_mean=star_mean,
                                             desc="coefficients")
                a_lin = a if velocity_term else None
                if frozen is None:
                    Q = update_earth(data, w, P, Q, a, b, shifter, delta,
                                     chunk, work=work, alpha=alpha,
                                     velocity_mask=velocity_mask,
                                     star_mean=star_mean)
            else:
                if frozen is None:
                    Q = update_earth(data, w, P, Q, a, b, shifter, delta,
                                     chunk, work=work, alpha=alpha,
                                     velocity_mask=velocity_mask,
                                     star_mean=star_mean)
                a, b, alpha, _ = joint_coeffs(data, w, P, Q, shifter, delta,
                                              chunk=chunk, exposure=tie,
                                              velocity_from=a_lin,
                                             velocity_mask=velocity_mask, star_mean=star_mean,
                                             desc="coefficients")
                a_lin = a if velocity_term else None
                P = update_star(data, w, P, Q, a, b, shifter, delta, chunk,
                                work=work,
                                alpha=alpha, velocity_mask=velocity_mask,
                                star_mean=star_mean, star_fwhm=star_fwhm)
            t_basis = time.time() - tick

            a, b, alpha, cond = joint_coeffs(data, w, P, Q, shifter, delta,
                                             chunk=chunk, exposure=tie,
                                             velocity_from=a_lin,
                                             velocity_mask=velocity_mask, star_mean=star_mean,
                                             desc="coefficients")
            a_lin = a if velocity_term else None
            # these are what the next iteration's clip step would recompute
            a_prev, b_prev = a, b
            # scored on w0: the clip weights change every iteration, so scoring on
            # them would be a moving target and the monotonicity check meaningless.
            # A chunk of rows at a time, and the residual kept only when the
            # means step below is going to want it: spelled as one expression
            # this is five arrays of (rows, columns) for one number.
            chi2, model = model_residual(
                data, P, Q, a, b, shifter, delta, chunk, alpha=alpha,
                velocity_mask=velocity_mask, star_mean=star_mean, w=w0,
                out=work if iterate else None)
            if iterate:
                # the components' residual, which centring leaves as it is;
                # centred before the state is kept, so that the kept state has
                # its static content in the means, where the next step of the
                # means has to find it. model_residual already handed back
                # data - model, so `model` here IS that residual.
                good_rows = (~rejected) & (w0.sum(axis=1) > 0)
                a, b = center_blocks(data, a, b, P, Q, templates, means, group,
                                     shifter, delta, chunk, good_rows, w=w0)
                a_prev, b_prev = a, b
                a_lin = a if velocity_term else None
            flag = ""
            if best_chi2 is None or chi2 < best_chi2:
                best_chi2, best_at = chi2, iteration
                worse = 0
            else:
                worse += 1
                flag = "   <- worse than iter %d" % best_at
            # what is kept: the iterate of lowest chi2, or, with --keep last,
            # whichever ran last; the stopping rule counts against the best
            # either way
            if best_at == iteration or args.keep == "last":
                best = (chi2, P.copy(), Q.copy(), iteration,
                        templates.copy(), means.copy())
            step = ""
            if iterate:
                tick = time.time()
                for _ in range(MEAN_SWEEP_ROUNDS):
                    step_t, step_o = update_means(
                        data, w, templates, means, group, shifter, delta,
                        chunk, resid=model, desc="per-parity means, both frames")
                step = "  means moved %.1e/%.1e" % (step_t, step_o)
                t_basis += time.time() - tick
            del model
            # `data` here is already template- and mean-subtracted, so this chi2 is
            # the residual of the *full* model and can be quoted against chi2_raw.
            shift = ""
            if velocity_term:
                v = shift_velocity(alpha[~rejected], dv) * 1000.0        # m/s
                shift = "  shift %.1f m/s rms" % (
                    1.4826 * np.median(np.abs(v - np.median(v))) if v.size else 0.0)
            # a digit more than seems necessary on both: they are what
            # convergence is read off, and on a fit that settles (TOI-2120,
            # 800 x 421782) four decimals held left/raw at 0.0423 for five
            # sweeps in a row, which says "stopped" for something still moving
            log("  iter %d  R2=%.7f  left/raw=%.5f  clipped=%.3f%%"
                  "  cond(A) med/max %.3f/%.3f%s%s  [%.1fs coeff, %.1fs bases]%s"
                  % (iteration, 1 - chi2 / chi2_null, chi2 / chi2_raw, 100 * hit,
                     np.median(cond), cond.max(), shift, step, t_coeff, t_basis,
                     flag))
            # With the means iterated chi2 descends and settles, rather than
            # turning over, and a settled fit gains nothing from more sweeps
            if iterate and prev_chi2 is not None and 0 <= prev_chi2 - chi2 < 1e-4 * chi2:
                flat += 1
            else:
                flat = 0
            prev_chi2 = chi2
            if flat >= 2:
                log("  chi2 has settled, less than 0.01%% per sweep twice in a"
                    " row; stopping at iteration %d" % iteration)
                break
            if worse >= int(args.patience):
                log("  chi2 has turned over; stopping and keeping iteration %d%s"
                    % (best[3], " (the last, --keep last)" if args.keep == "last" else ""))
                break
        _set_label(None)

        chi2, P, Q, best_iter = best[:4]
        if iterate:
            # back to the means of the kept iterate, the data following them
            restore_means(data, templates, means, best[4], best[5], group,
                          shifter, delta, chunk, w=w0)
        log("  best iterate: %d, R2 = %.6f, %.4f of the raw weighted variance"
              " left" % (best_iter, 1 - chi2 / chi2_null, chi2 / chi2_raw))

        # ---- MAD cut: retire a spectrum the components disagree about ------
        # After the fit, not before: an outlier is only definable against the
        # other spectra's coefficients, which is what the fit just produced.
        # The rejected rows keep their place in every array and simply lose
        # their weight, so nothing downstream has to be re-indexed and
        # joint_coeffs leaves their coefficients at zero of its own accord.
        if args.max_mad <= 0 or mad_round == int(args.max_mad_rounds):
            break
        a_now, b_now, _, _ = joint_coeffs(data, w, P, Q, shifter, delta,
                                          chunk=chunk, exposure=tie,
                                          velocity_from=a_lin,
                                          velocity_mask=velocity_mask,
                                          star_mean=star_mean)
        fresh = mad_outliers(np.hstack([a_now, b_now]), parity, args.max_mad,
                             rejected)
        if not fresh.any():
            log("  MAD cut at %.1f sigma: nothing left to reject" % args.max_mad)
            break
        rejected |= fresh
        log("  MAD cut at %.1f sigma: rejecting %d rows (%d exposures) this"
            " round, %d of %d rows in total (%.1f%%); refitting"
            % (args.max_mad, int(fresh.sum()), count_exposures(meta, fresh),
               int(rejected.sum()), n_spectra, 100 * rejected.mean()))
        w0[fresh] = 0.0
        w = w0
        data[fresh] = 0.0
        # the per-parity mean moved when those rows left; take out the
        # difference, and in offset mode only the part of that difference that
        # separates the parities, so the shared term stays in as before. With
        # --mean iterate the next round's sweeps re-estimate both means anyway.
        # --mean star keeps no observer mean to move, and its star spectra are
        # a median, which a handful of retired rows does not shift
        if not iterate and args.mean != "star":
            residual_means, _ = parity_means(data, w0, parity)
            if args.mean == "offset":
                residual_means = residual_means - residual_means.mean(axis=0)
            subtract_means(data, residual_means, group)
            zero_unweighted(data, w0)
            means = means + residual_means
        chi2_raw = float(chi2_raw_rows[~rejected].sum())
        # in the fit's chunks, as the first chi2_null was, rather than a
        # product and a square of the cube's size each
        chi2_null = weighted_power(data, w0, chunk)

    chi2, P, Q, best_iter = best[:4]
    if rejected.any():
        log("  MAD cut kept %d of %d rows; %d rows from %d exposures rejected"
            " at %.1f sigma"
            % (n_spectra - int(rejected.sum()), n_spectra, int(rejected.sum()),
               count_exposures(meta, rejected), args.max_mad))

    if args.leakage:
        to_earth, to_star = leakage(data, w, P, Q, shifter, delta, chunk)
        log("  leakage star -> Earth block: %s"
            % np.array2string(np.round(to_earth, 4)), "value")
        log("  leakage Earth -> star block: %s"
            % np.array2string(np.round(to_star, 4)), "value")

    # ------------------------------------------------------- outputs -------
    a, b, alpha, _ = joint_coeffs(data, w, P, Q, shifter, delta, chunk=chunk,
                                  exposure=tie, velocity_from=a_lin,
                                  velocity_mask=velocity_mask,
                                  star_mean=star_mean)
    Pf = shifter.prepare(P)
    power_star = block_power(w, a, lambda s0, s1: shifter.carry(Pf, delta[s0:s1]))
    power_earth = block_power(w, b, lambda s0, s1: Q)
    P, a, power_star = tidy_block(P, a, power_star)
    # a frozen basis keeps its order and signs: its amplitudes are then
    #   those of the basis' components, comparable to the training's
    if frozen is None:
        Q, b, power_earth = tidy_block(Q, b, power_earth)

    sig_formal, sig_scaled, chi2_red, sig_vel = coefficient_errors(
        data, w0, P, Q, shifter, delta, a, b, chunk, exposure=tie,
        alpha=alpha if velocity_term else None,
        velocity_mask=velocity_mask, star_mean=star_mean)
    n_star = a.shape[1]
    log("  median reduced chi2 per spectrum: %.3f   (1.0 would mean the noise"
          " model of 2.2 is exactly right)" % np.nanmedian(chi2_red))
    log("  median error scaling sqrt(chi2_red): %.3f"
          % np.nanmedian(np.sqrt(chi2_red)))
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_a = np.nanmedian(np.abs(a) / sig_scaled[:, :n_star], axis=0)
        snr_b = np.nanmedian(np.abs(b) / sig_scaled[:, n_star:], axis=0)
    log("  median |coefficient| / sigma, star block : %s"
          % np.array2string(snr_a, precision=1))
    log("  median |coefficient| / sigma, Earth block: %s"
          % np.array2string(snr_b, precision=1))

    os.makedirs(args.outdir, exist_ok=True)
    bjd = np.asarray(meta["bjd"], dtype=float)
    table = Table()
    table["filename"] = meta["filename"]
    table["bjd"] = bjd
    table["berv"] = np.asarray(meta["berv"], dtype=float)
    # with orbits, berv is the star's frame; the BERV itself and the
    #   companions' velocity beside it (readers.star_frame)
    for key in ("berv_bary", "vstar"):
        if key in getattr(meta, "colnames", meta):
            table[key] = np.asarray(meta[key], dtype=float)
    table["airmass"] = np.asarray(meta["airmass"], dtype=float)
    table["snr_band"] = np.asarray(meta["snr_band"], dtype=float)
    table["chi2_red"] = chi2_red
    table["rejected"] = rejected      # MAD cut; their coefficients are zeros
    for k in range(a.shape[1]):
        table["a%d" % (k + 1)] = a[:, k]              # star-frame block
        table["ea%d" % (k + 1)] = sig_scaled[:, k]    # 1-sigma, chi2-scaled
    for j in range(b.shape[1]):
        table["b%d" % (j + 1)] = b[:, j]              # observer-frame block
        table["eb%d" % (j + 1)] = sig_scaled[:, n_star + j]
    if velocity_term:
        # the shift the fit absorbed rather than let the observer block have.
        # In pixels because that is what the solve works in, and in m/s because
        # that is what it means, read back through the relativistic Doppler
        # (grids.shift_velocity). Its error is the derivative there: dv km/s
        # per pixel, to 1e-13 at these few pixels.
        table["alpha"] = alpha
        table["vrad_fit"] = shift_velocity(alpha, dv) * 1000.0
        table["evrad_fit"] = sig_vel * dv * 1000.0
        v = shift_velocity(alpha[~rejected], dv) * 1000.0
        log("  velocity absorbed by the shift term: %.1f m/s rms, median"
            " %+.1f, worst %+.1f"
            % (1.4826 * np.median(np.abs(v - np.median(v))), np.median(v),
               v[np.argmax(np.abs(v))] if v.size else 0.0), "value")
    csv_path = os.path.join(args.outdir, "coefficients.csv")
    table.write(csv_path, format="csv", overwrite=True)
    log("  wrote %s" % csv_path)

    log("  star  block weighted variance share: %s"
          % np.array2string(100 * power_star / chi2_null, precision=2))
    log("  Earth block weighted variance share: %s"
          % np.array2string(100 * power_earth / chi2_null, precision=2))
    # the ancillary quantities, gathered once and stored in the archive so that
    # --replot can redraw the correlation matrix on its own
    from .plotting import ancillary_table, source_directory
    anc_labels, anc_values = ancillary_table(
        meta, [os.path.basename(str(v)) for v in meta["filename"]],
        source_dir=source_directory(args.cube))

    mean = means[0] if means.shape[0] == 1 else np.average(
        means, axis=0, weights=[np.sum(parity == g) for g in parity_groups])
    npz_path = os.path.join(args.outdir, "fit.npz")
    from .provenance import stamp as code_stamp
    # le facteur du dernier balayage, en (rangee, colonne, facteur): ce que la
    # correction lit pour masquer sous correct.weight_floor
    clip_rows, clip_cols, clip_factor = _clip_triplet(last_clip)
    np.savez_compressed(npz_path, bjd=bjd, a=a, b=b, P=P, Q=Q, alpha=alpha,
                        clip_rows=clip_rows, clip_cols=clip_cols,
                        clip_factor=clip_factor, clip_nsig=float(args.clip),
                        velocity_term=bool(velocity_term),
                        anc_labels=np.asarray(anc_labels, dtype="U32"),
                        anc_values=np.asarray(anc_values, dtype=float),
                        power_star=power_star, power_earth=power_earth,
                        chi2_null=chi2_null, chi2_raw=chi2_raw,
                        best_iter=best_iter, template=template, mean=mean,
                        means=means, parity=parity,
                        used_template=bool(args.template),
                        sigma_formal=sig_formal, sigma_scaled=sig_scaled,
                        chi2_red=chi2_red, rejected=rejected,
                        max_mad=float(args.max_mad), chi2_best=chi2, grid=grid,
                        tied=bool(args.tie_parities), dv=dv,
                        filename=np.asarray(meta["filename"], dtype="U64"),
                        berv=np.asarray(meta["berv"], dtype=float),
                        # how many spectra each row holds: 1 unless the night
                        # was coadded. Without it a --replot cannot tell a
                        # nightly-stacked fit from a fit of single exposures,
                        # and labelled 458 nights as 458 exposures
                        n_exposures=np.asarray(
                            meta["n_exposures"] if "n_exposures" in meta.colnames
                            else np.ones(len(meta)), dtype=float),
                        mean_mode=str(args.mean),
                        # R the star side was smoothed to; 0 when it was not
                        star_resolution=float(getattr(args, "star_resolution", None) or 0),
                        # the FWHM, in samples, the star side was smoothed to
                        star_fwhm=int(star_fwhm or 0),
                        # how the star side was carried and updated
                        star_basis=str(getattr(args, "star_basis", None)
                                       or "spline"),
                        # the pca2d commit that made the fit, + if modified
                        pca2d_code=code_stamp(),
                        # the frozen observer basis, if there was one
                        earth_basis=str(frozen.name if frozen else ""),
                        earth_basis_id=str(frozen.ident if frozen else ""),
                        # the star-frame spectra per parity, from --mean iterate
                        # or --mean star; every reader takes them from here
                        **({"templates": templates} if star_mean is not None else {}))
    log("  wrote %s (re-plot without refitting)" % npz_path)

    # the rejected rows carry zero weight, so their coefficients are zeros and
    # plotting them would draw a spurious line on the axis
    keep = ~rejected
    plot_coeffs(bjd[keep], a[keep], b[keep], power_star, power_earth, chi2_null,
                os.path.join(args.outdir, "coefficients_vs_time.pdf"),
                err_a=sig_scaled[keep][:, :n_star],
                err_b=sig_scaled[keep][:, n_star:])
    plot_variance(power_star, power_earth, chi2_null, chi2,
                  os.path.join(args.outdir, "variance.pdf"))
    write_variance_table(power_star, power_earth, chi2_null, chi2,
                         os.path.join(args.outdir, "variance.csv"))
    plot_components(grid, P, Q, power_star, power_earth, chi2_null,
                    os.path.join(args.outdir, "components.pdf"))
    correlate_and_plot(args.outdir, a[keep], b[keep], anc_labels,
                       anc_values[:, keep] if len(anc_labels) else anc_values,
                       title=exposures_label(table["filename"], keep,
                                             _per_row(table), row_unit(meta)))
    write_components_fits(os.path.join(args.outdir, "twoframe_components.fits"),
                          grid, P, Q, template, means, power_star, power_earth,
                          chi2_null, chi2, table, dv, tied=bool(args.tie_parities),
                          max_mad=float(args.max_mad),
                          frame="observer", highpass="log_sub",
                          weights=w0, parity=parity,
                          templates=templates if star_mean is not None else None,
                          mean_mode=str(args.mean))
    # the mapped arrays go with the run that made them: they are scratch, they
    # are the size of the cube, and a folder of them left behind fills a disk
    if spill:
        for array in (work, data, w0):
            try:
                drop_workspace(array)
            except Exception:                                 # noqa: BLE001
                pass
        log("removed the mapped workspace in %s" % spill)

    # Nothing, on purpose. This is a console_script entry point, so whatever it
    # returns is handed to sys.exit: returning the two bases printed a pair of
    # arrays and exited 1 on a successful fit, which is why nothing could be
    # chained after it and why a `set -e` script died at the first one.
    return None


if __name__ == "__main__":
    main()
