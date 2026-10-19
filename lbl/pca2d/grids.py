"""The destination wavelength grid, and the nominal photometric bands.

Two things live here because both are properties of the *output* of the
pipeline rather than of any particular input file:

  * the "magic grid": a log-uniform wavelength grid with a constant velocity
    step, defined analytically by an anchor wavelength and that step. APERO's
    NIRPS s1d_v products are sampled on exactly this grid with wave0 = 965 nm
    and dv = 0.5 km/s, so building it from those two numbers reproduces the s1d
    sampling to 5e-13 nm and needs no file on disk. That matters for the t.fits
    path, where there is no s1d grid to recycle;

  * the nominal photometric bands. The PCA is fitted on those and only those:
    between the bands the atmosphere is opaque, the blaze has collapsed, or
    both, and columns like that contribute nothing but noise and telluric
    residuals to the eigenvectors.
"""

from __future__ import annotations

import numpy as np

from .logger import log

C_KMS = 299792.458

# MKO half-power points, in nm. K is included on purpose even though NIRPS
# stops at 1.95 um: the same configuration has to serve an instrument that
# does reach K, and a band that falls outside the grid is simply dropped.
DEFAULT_BANDS = {
    "Y": [970.0, 1070.0],
    "J": [1170.0, 1330.0],
    "H": [1490.0, 1780.0],
    "K": [2030.0, 2370.0],
}


def magic_grid(wave0: float, dv: float, wave_min: float, wave_max: float) -> np.ndarray:
    """Log-uniform grid  lambda_i = wave0 * exp(i * dv / c), cropped to the band.

    The anchor matters as much as the step: it is what makes two runs, two
    instruments' worth of reduction, or a t.fits and an s1d of the same
    exposure land on identical wavelengths, so that products can be compared
    sample by sample instead of being re-splined into each other.
    """
    wave0 = float(wave0)
    step = float(dv) / C_KMS
    if step <= 0:
        raise ValueError("domain.dv must be positive, got %r" % dv)
    first = int(np.ceil(np.log(float(wave_min) / wave0) / step))
    last = int(np.floor(np.log(float(wave_max) / wave0) / step))
    if last < first:
        raise ValueError(
            "empty grid: %.3f-%.3f nm at %.4f km/s from an anchor of %.3f nm"
            % (wave_min, wave_max, dv, wave0)
        )
    return wave0 * np.exp(step * np.arange(first, last + 1))


def grid_dv(grid: np.ndarray) -> float:
    """Velocity step of a log-uniform grid, in km/s."""
    return float(np.median(np.diff(np.log(np.asarray(grid, dtype=np.float64)))) * C_KMS)


# --------------------------------------------------------------------------
# Doppler
# --------------------------------------------------------------------------
# Every velocity that becomes a wavelength ratio, a log-wavelength shift or a
# pixel shift goes through these, and always relativistically. The first-order
# 1 + v/c is off by (v/c)**2 / 2, 1.5 m/s at a 30 km/s BERV: the size of what
# the corrected spectra are measured for. The grids are left as they are: the
# dv of a log-uniform grid is, by definition, c times its ln step (the magic
# grid convention shared with APERO), and a Doppler shift on it is atanh(v/c)
# in ln lambda.

def doppler(v_kms):
    """Wavelength ratio, received over emitted, for a velocity in km/s.

    sqrt((1 + v/c) / (1 - v/c)), positive v receding. Two shifts compose by
    multiplying their factors, and undoing one is dividing by it.
    """
    beta = np.asarray(v_kms, dtype=np.float64) / C_KMS
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def log_shift(v_kms):
    """ln(doppler(v)) = atanh(v/c): the shift in ln lambda, additive on a log grid."""
    return np.arctanh(np.asarray(v_kms, dtype=np.float64) / C_KMS)


def velocity(log_ratio):
    """Inverse of log_shift: the velocity, in km/s, of a ln wavelength ratio."""
    return C_KMS * np.tanh(log_ratio)


def pixel_shift(v_kms, dv):
    """The shift, in samples, of a velocity in km/s on a log grid of step dv."""
    return log_shift(v_kms) * C_KMS / float(dv)


def shift_velocity(pixels, dv):
    """Inverse of pixel_shift: the velocity, in km/s, of a shift in samples."""
    return velocity(np.asarray(pixels, dtype=np.float64) * float(dv) / C_KMS)


def band_mask(grid: np.ndarray, bands: dict, names=None):
    """Boolean mask of the grid columns inside the requested bands.

    Returns (mask, used) where `used` lists the bands that actually intersect
    the grid. A band that misses the grid entirely -- K on NIRPS -- is reported
    and ignored rather than treated as an error, because the same config is
    meant to be reusable across instruments.
    """
    grid = np.asarray(grid, dtype=np.float64)
    mask = np.zeros(grid.size, dtype=bool)
    used, missing = [], []
    for name in (names if names else list(bands)):
        if name not in bands:
            log("band '%s' is requested but not defined; ignored" % name, "warn")
            continue
        low, high = [float(v) for v in bands[name]]
        inside = (grid >= low) & (grid <= high)
        if not np.any(inside):
            missing.append(name)
            continue
        mask |= inside
        used.append(name)
        log("band %s: %.1f-%.1f nm, %d grid columns" % (name, low, high, inside.sum()),
            "value")
    if missing:
        log("bands outside the grid, skipped: %s" % ", ".join(missing), "warn")
    if not np.any(mask):
        log("no band overlaps the grid; fitting on the whole domain instead", "warn")
        mask[:] = True
        used = ["<all>"]
    return mask, used


#: The largest barycentric velocity the Earth can give a star, in km/s: orbit
#: 29.8 plus rotation 0.47 at the equator, rounded up. It bounds how far any
#: exposure's spectrum can move on the grid, which is what a window's margin
#: has to cover for a shift computed on the window alone to be exact inside it.
MAX_BERV_KMS = 31.0


def parse_window(spec):
    """(centre, width) in nm from 'centre:width'; a bare centre gets 2 nm."""
    centre, _, width = str(spec).partition(":")
    return float(centre), float(width or 2.0)


def window_block(grid, centre, width, dv, kernel_halfwidth=8, spare=16):
    """(a0, b0): the contiguous grid columns a figure of this window needs.

    The window plus a margin wide enough that carrying the block by any BERV
    with a Lanczos kernel of `kernel_halfwidth` is exact inside the window:
    every sample the carry reads to make a window sample lies in the block.
    The margin comes from physics, not from the exposures at hand, so the cube
    build, which saves the raw flux of exactly these columns, and the figures,
    which read exactly these columns back, always agree on them. None when the
    window misses the grid.
    """
    grid = np.asarray(grid)
    inside = np.where((grid >= centre - 0.5 * width)
                      & (grid <= centre + 0.5 * width))[0]
    if inside.size == 0:
        return None
    margin = (int(np.ceil(MAX_BERV_KMS / float(dv))) + int(kernel_halfwidth)
              + int(spare))
    return (max(0, int(inside[0]) - margin),
            min(grid.size, int(inside[-1]) + 1 + margin))

