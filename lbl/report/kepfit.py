"""The joint fit of the K of a star's confirmed planets, at their published
ephemerides, to the velocities of a run, for the report (PLANETS).

THE MODEL. Each planet of lbl/resources/planets/ephemerides.yaml on a
circular orbit, with the period and the time of inferior conjunction (the
transit, for a transiting planet) of the catalog, both fixed: K_j b_j(t),
with b_j = -sin(2 pi (t - T0_j) / P_j), the velocity of a circular orbit of
K = 1 (lbl.science.planets.orbit_velocity), 0 at the conjunction and falling
through it. Plus one offset, plus the known orbits the template was made on
(PLANETS, with their K), fixed.

THE FIT. The offset and the K are linear in this model: weighted least
squares, with a jitter s added in quadrature to every error bar, the s that
maximises the likelihood (it absorbs what the error bars miss: activity,
instrument). The errors of the K are those of the least squares at that
jitter. K is not forced positive: a K consistent with zero, or negative,
says that the velocities do not see the planet at that period and phase.
"""
import numpy as np
from scipy.optimize import minimize_scalar

from lbl.science import planets as kepler


def basis(bjd, planet):
    """The velocity of a circular orbit of K = 1 at the planet's period and
    time of inferior conjunction, at these times [BJD]"""
    orbit = dict(period=float(planet['period']), t0=float(planet['t0']),
                 k=1.0, e=0.0, omega=90.0)
    return kepler.orbit_velocity(bjd, orbit)


def _solve(design, values, variance):
    """Weighted least squares: (coefficients, their covariance)"""
    root = 1.0 / np.sqrt(variance)
    matrix = design * root[:, None]
    cov = np.linalg.inv(matrix.T @ matrix)
    return cov @ (matrix.T @ (values * root)), cov


def fit(bjd, vrad, svrad, planets, known=()):
    """
    Fit the offset and the K of the planets, with a jitter

    :param bjd: np.ndarray, the times [BJD]
    :param vrad: np.ndarray, the velocities [m/s]
    :param svrad: np.ndarray, their errors [m/s]
    :param planets: list of dict, each with period and t0 (ephemerides.yaml)
    :param known: list of dict, complete orbits held fixed (PLANETS)

    :return: dict: offset, soffset, k, sk (one per planet), jitter,
             planets, known, rms_before and rms_after (the residuals
             without and with the planets, the known orbits taken out of
             both), chi2_before and chi2_after (the same residuals as reduced
             chi2 on the error bars alone, over nfit - 1 and dof degrees of
             freedom), chi2_jitter (after, with the jitter added to the error
             bars), dof, nfit (velocities used)
    """
    bjd, vrad, svrad = (np.asarray(x, dtype=float) for x in (bjd, vrad, svrad))
    good = (np.isfinite(bjd) & np.isfinite(vrad) & np.isfinite(svrad)
            & (svrad > 0))
    time, values, error = bjd[good], vrad[good], svrad[good]
    known = list(known)
    if known:
        values = values - kepler.star_velocity(time, known)
    design = np.column_stack([np.ones_like(time)] +
                             [basis(time, planet) for planet in planets])
    if len(time) <= design.shape[1] + 1:
        raise ValueError('%d velocities for %d parameters'
                         % (len(time), design.shape[1] + 1))

    def minus_two_log_like(jitter):
        variance = error ** 2 + jitter ** 2
        coef, _ = _solve(design, values, variance)
        resid = values - design @ coef
        return float(np.sum(resid ** 2 / variance + np.log(variance)))

    upper = max(3.0 * float(np.std(values)), 1.0)
    jitter = float(minimize_scalar(minus_two_log_like, bounds=(0.0, upper),
                                   method='bounded',
                                   options=dict(xatol=1e-3)).x)
    coef, cov = _solve(design, values, error ** 2 + jitter ** 2)
    resid = values - design @ coef
    weights = 1.0 / error ** 2
    before = values - np.sum(weights * values) / np.sum(weights)
    # REDUCED CHI2, on the error bars alone: how far the scatter is from what
    #   the error bars promise, before the planets (one offset) and after
    #   them. With the jitter it is near 1 by construction, since the jitter
    #   is the excess the likelihood finds; it is given to show that it is
    dof = len(time) - design.shape[1]
    return dict(offset=float(coef[0]), soffset=float(np.sqrt(cov[0, 0])),
                k=coef[1:].astype(float),
                sk=np.sqrt(np.diag(cov)[1:]).astype(float),
                jitter=jitter, planets=list(planets), known=known,
                rms_before=float(np.std(before, ddof=1)),
                rms_after=float(np.std(resid, ddof=1)),
                chi2_before=float(np.sum(before ** 2 * weights)
                                  / (len(time) - 1)),
                chi2_after=float(np.sum(resid ** 2 * weights) / dof),
                chi2_jitter=float(np.sum(resid ** 2
                                         / (error ** 2 + jitter ** 2)) / dof),
                dof=int(dof), nfit=int(len(time)))


def model(result, bjd, offset=True, known=True, skip=None):
    """
    The velocity of the fitted model at these times [BJD]

    :param result: dict, from fit
    :param bjd: np.ndarray, the times
    :param offset: bool, include the offset
    :param known: bool, include the known orbits
    :param skip: int or None, the planet left out (for its phase fold)

    :return: np.ndarray [m/s]
    """
    bjd = np.asarray(bjd, dtype=float)
    out = np.full_like(bjd, result['offset'] if offset else 0.0)
    if known and result['known']:
        out = out + kepler.star_velocity(bjd, result['known'])
    for index, planet in enumerate(result['planets']):
        if index != skip:
            out = out + result['k'][index] * basis(bjd, planet)
    return out
