"""The star as a cubic B-spline, evaluated at every exposure's Doppler-shifted positions.

The two-frame fit carries its star-side vectors into each exposure's frame with
a Lanczos kernel and updates the star basis from the diagonal of the normal
equations only: an approximation, and the reason chi2 can rise from one sweep
to the next. Here the star-side vectors (the per-parity star spectra and the
star components P) are cubic B-splines with a knot at every grid sample. Taking
one to exposure n is evaluating it at the grid positions moved by that
exposure's shift, four taps per sample. That is linear in the spline's
coefficients, so the star update is a least-squares problem whose normal matrix
has seven diagonals, and it is solved exactly. Wobble (Bedell et al. 2019) fits
a star and the tellurics this way; the velocity here stays LBL's to measure.

The data and the weights are still carried into the star's frame by the
Lanczos kernel, as before: the median star spectra, the gap guard and the
figures do not change. Only the star's own representation and its update do.
"""

import numpy as np
from scipy.linalg import LinAlgError, solve_banded, solveh_banded
from scipy.ndimage import spline_filter1d


def bspline3(u):
    """The cubic B-spline kernel: 2/3 at 0, 1/6 at +-1, zero from +-2."""
    u = np.abs(np.asarray(u, dtype=float))
    return np.where(u < 1.0, (4.0 - 6.0 * u ** 2 + 3.0 * u ** 3) / 6.0,
                    np.where(u < 2.0, (2.0 - u) ** 3 / 6.0, 0.0))


def taps(p):
    """(base, h) for a shift of p samples: out[m] = f(m - p), the convention of
    LanczosShifter, written sum_t h[t + 2] c[m - base + t] over t = -2..1."""
    base = int(np.floor(p))
    return base, bspline3(np.arange(-2, 2) + (p - base))


def coefficients(values):
    """B-spline coefficients whose spline passes through `values` at every sample."""
    return spline_filter1d(np.atleast_2d(np.asarray(values, dtype=float)),
                           order=3, axis=-1, mode="mirror")


def values(coeffs):
    """The spline at the samples themselves: (c[j-1] + 4 c[j] + c[j+1]) / 6."""
    c = np.atleast_2d(np.asarray(coeffs, dtype=float))
    out = 4.0 * c
    out[:, 1:] += c[:, :-1]
    out[:, :-1] += c[:, 1:]
    return out / 6.0


def evaluate(padded, pix, n, pad):
    """Each row's spline at m - p for every shift p in `pix`: (len(pix), K, n).
    `padded` is coefficients with `pad` zeros on either side."""
    pix = np.atleast_1d(pix)
    out = np.zeros((pix.size, padded.shape[0], n))
    for i, p in enumerate(pix):
        base, h = taps(p)
        lo = pad - base - 2
        for k in range(4):
            out[i] += h[k] * padded[:, lo + k:lo + k + n]
    return out


class SplineStar:
    """A shifter whose star side is a B-spline and whose data side is Lanczos.

    prepare and carry, which every star-side vector goes through (the components
    in the coefficient solve and in the models, the per-parity star spectra
    carried to each row), evaluate the spline. rows, adjoint and diag_normal,
    which carry data and weights into the star's frame (the median star spectra,
    the gap guard), are the Lanczos shifter's, unchanged.
    """

    banded = True
    spline = True

    def __init__(self, lanczos):
        self.inner = lanczos
        self.n = lanczos.n
        self.pad = lanczos.pad

    def rows(self, arr, pix, desc=None):
        return self.inner.rows(arr, pix, desc=desc)

    def adjoint(self, arr, pix, desc=None):
        return self.inner.adjoint(arr, pix, desc=desc)

    def diag_normal(self, w, pix, desc=None):
        return self.inner.diag_normal(w, pix, desc=desc)

    def prepare(self, basis):
        return np.pad(coefficients(basis), [(0, 0), (self.pad, self.pad)])

    def carry(self, prepared, pix, threads=None):
        return evaluate(prepared, pix, self.n, self.pad)


def _solve(A, rhs, floor_frac):
    """The banded normal equations, A[d][j] = matrix[j, j + d], solved exactly.
    A knot whose diagonal is below floor_frac of the median is not constrained
    and comes back zero, as mstep's floor makes it."""
    n = rhs.size
    positive = A[0][A[0] > 0]
    if not positive.size:
        return np.zeros(n)
    scale = float(np.median(positive))
    dead = A[0] <= floor_frac * scale
    A = A.copy()
    rhs = np.where(dead, 0.0, rhs)
    for d in range(1, 4):
        A[d][:n - d][dead[:n - d] | dead[d:]] = 0.0
        A[d][n - d:] = 0.0
    A[0][dead] = scale
    A[0] += 1e-9 * scale
    ab = np.zeros((4, n))
    ab[3] = A[0]
    for d in range(1, 4):
        ab[3 - d, d:] = A[d][:n - d]
    try:
        c = solveh_banded(ab, rhs, check_finite=False)
    except LinAlgError:
        full = np.zeros((7, n))
        full[3] = A[0]
        for d in range(1, 4):
            full[3 - d, d:] = A[d][:n - d]
            full[3 + d, :n - d] = A[d][:n - d]
        c = solve_banded((3, 3), full, rhs, check_finite=False)
    c[dead] = 0.0
    return c


def solve_components(resid, w, a, delta, pad, floor_frac=1e-2):
    """The star components that minimise chi2 exactly, given their amplitudes.

    `resid` is the data less everything but the star components, in each row's
    frame; `w` its weights; `a` the (rows, K) amplitudes. Component k solves

        (sum_n a_nk^2 E_n^T W_n E_n) c_k = sum_n a_nk E_n^T W_n r_n

    with E_n the spline evaluated at row n's shift, a matrix of seven diagonals,
    and r_n the residual less the components already solved. Returns the
    components as values on the grid, made orthonormal as mstep leaves them.
    `resid` is consumed.
    """
    n_rows, n = resid.shape
    n_comp = a.shape[1]
    out = np.zeros((n_comp, n))
    for k in range(n_comp):
        A = np.zeros((4, n))
        rhs = np.zeros(n)
        for i in range(n_rows):
            aik = float(a[i, k])
            if aik == 0.0 or not np.any(w[i]):
                continue
            base, h = taps(delta[i])
            wp = np.pad(w[i], pad)
            wr = np.pad(w[i] * resid[i], pad)
            for t1 in range(4):
                s = pad + base - t1 + 2          # j + base - t, t = t1 - 2
                rhs += aik * h[t1] * wr[s:s + n]
                for d in range(4 - t1):
                    A[d] += aik * aik * h[t1] * h[t1 + d] * wp[s:s + n]
        c = _solve(A, rhs, floor_frac)
        out[k] = values(c)[0]
        if k < n_comp - 1:
            # the next component sees the residual without this one
            padded = np.pad(c[None, :], [(0, 0), (pad, pad)])
            for i in range(n_rows):
                resid[i] -= a[i, k] * evaluate(padded, delta[i:i + 1], n, pad)[0, 0]
    for i in range(n_comp):
        for j in range(i):
            out[i] -= np.dot(out[i], out[j]) * out[j]
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out
