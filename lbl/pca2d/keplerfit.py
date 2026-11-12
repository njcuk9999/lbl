"""
The known planets of a star fitted together: every planet at once, so that
none is read through another's signal, twice, as circular orbits and as
full Keplerians, whatever their number.

Each fit has an offset, a straight line in time when every period is under
a third of the campaign (a longer period, the line would take the planet
with it), one orbit per planet at its published period, and a free jitter
added to the error bars in quadrature; its likelihood is maximised.

    circular    v = K cos(lam + 2 pi (t - t_ref) / P): K and lam a planet,
                linear (A cos + B sin), solved exactly
    Keplerian   v = K [cos(nu + omega) + e cos omega]: K, e, omega and the
                mean longitude lam = M + omega at t_ref a planet, from the
                circular solution (e cos omega, e sin omega rather than e
                and omega: smooth through e = 0, where omega means nothing)

The periods are held at the published ones (transit periods are known to
1e-5 d, better than any velocity series here) so that each planet folds on
one phase. What each fit says (Delta BIC, positive when the data prefer
the richer model, on Kass and Raftery's scale as bervbias' is):

    fit_bic[kind]       BIC(no planet: offset, line, jitter) - BIC(the fit)
    planet_bic[i]       BIC(circular fit without planet i) - BIC(circular)
    kep_planet_bic[i]   BIC(Keplerian fit without planet i) - BIC(Keplerian)
    ecc_bic[i]          BIC(Keplerian fit, planet i circular) - BIC(Keplerian)

Created on 2026-11-12

@author: artigau
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .bervbias import JITTER_MAX, JITTER_MIN

#: the highest eccentricity a fit reaches
E_MAX = 0.95
#: a Gaussian of this width on e cos omega and e sin omega (a Rayleigh law
#:   on e, mean 0.38): without it, a few points far off the others (a
#:   series with a large jitter) are fitted by an orbit of e near 1 whose
#:   peaks go through them. Wide enough to leave a well sampled eccentric
#:   orbit alone. None: the likelihood alone
E_SIGMA = 0.3
#: the starts of an eccentric fit, (e cos omega, e sin omega) of every
#:   eccentric planet, the circular solution first
STARTS = ((0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (-0.3, 0.0), (0.0, -0.3))


def solve_kepler(mean, ecc: float) -> np.ndarray:
    """E such that E - e sin E = M (Newton, vectorised)"""
    mean = np.asarray(mean, float)
    if ecc <= 0:
        return mean.copy()
    ecc_anom = mean + 0.85 * ecc * np.sign(np.sin(mean))
    for _ in range(60):
        step = ((ecc_anom - ecc * np.sin(ecc_anom) - mean)
                / (1.0 - ecc * np.cos(ecc_anom)))
        ecc_anom = ecc_anom - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return ecc_anom


def keplerian(t, period: float, amp: float, ecosw: float, esinw: float,
              lam: float, t_ref: float) -> np.ndarray:
    """The star's velocity [m/s] from one planet: K [cos(nu + omega)
    + e cos omega], with lam the mean longitude at t_ref (e = 0 gives
    K cos(lam + 2 pi (t - t_ref) / P))"""
    t = np.asarray(t, float)
    ecc = float(np.hypot(ecosw, esinw))
    if ecc > E_MAX:
        ecosw, esinw, ecc = (ecosw * E_MAX / ecc, esinw * E_MAX / ecc,
                             E_MAX)
    omega = float(np.arctan2(esinw, ecosw)) if ecc > 0 else 0.0
    mean = lam - omega + 2.0 * np.pi * (t - t_ref) / period
    if ecc == 0:
        return amp * np.cos(mean + omega)
    ecc_anom = solve_kepler(mean, ecc)
    true = 2.0 * np.arctan2(np.sqrt(1.0 + ecc) * np.sin(ecc_anom / 2.0),
                            np.sqrt(1.0 - ecc) * np.cos(ecc_anom / 2.0))
    return amp * (np.cos(true + omega) + ecc * np.cos(omega))


def _lnl(resid, err, jitter: float) -> float:
    """ln L of residuals with error bars and a jitter (the 2 pi left out,
    as bervbias counts it)"""
    var = err ** 2 + jitter ** 2
    return -0.5 * float(np.sum(resid ** 2 / var + np.log(var)))


def _best_jitter(resid, err) -> float:
    """The jitter that maximises ln L of these residuals"""
    from scipy.optimize import minimize_scalar
    found = minimize_scalar(lambda lj: -_lnl(resid, err, np.exp(lj)),
                            bounds=(np.log(JITTER_MIN), np.log(JITTER_MAX)),
                            method="bounded", options={"xatol": 1e-5})
    # the bounded search can miss a maximum at its edge: both tried
    return max((float(np.exp(found.x)), JITTER_MIN, JITTER_MAX),
               key=lambda j: _lnl(resid, err, j))


def _linear(design, v, err):
    """(coefficients, their covariance, jitter, ln L) of a linear model, the
    jitter free: the weighted least squares at each jitter, and the jitter
    that does best"""
    from scipy.optimize import minimize_scalar

    def solve(jitter):
        root = 1.0 / np.sqrt(err ** 2 + jitter ** 2)
        coef, *_ = np.linalg.lstsq(design * root[:, None], v * root,
                                   rcond=None)
        return coef, v - design @ coef

    found = minimize_scalar(lambda lj: -_lnl(solve(np.exp(lj))[1], err,
                                             np.exp(lj)),
                            bounds=(np.log(JITTER_MIN), np.log(JITTER_MAX)),
                            method="bounded", options={"xatol": 1e-5})
    jitter = max((float(np.exp(found.x)), JITTER_MIN, JITTER_MAX),
                 key=lambda j: _lnl(solve(j)[1], err, j))
    coef, resid = solve(jitter)
    weight = 1.0 / (err ** 2 + jitter ** 2)
    cov = np.linalg.pinv(design.T @ (design * weight[:, None]))
    return coef, cov, jitter, _lnl(resid, err, jitter)


def _clean(t, v, err):
    t, v, err = (np.asarray(x, float) for x in (t, v, err))
    keep = np.isfinite(t) & np.isfinite(v) & np.isfinite(err) & (err > 0)
    return t[keep], v[keep], err[keep]


def _base(t, t_ref, drift):
    cols = [np.ones_like(t)]
    if drift:
        # in years about the middle: a slope in m/s per year
        cols.append((t - t_ref) / 365.25)
    return np.column_stack(cols)


def _angles(t, periods, t_ref):
    return [2.0 * np.pi * (t - t_ref) / float(p) for p in periods]


def _bic(lnl, n_par, n):
    return float(n_par * np.log(n) - 2.0 * lnl)


class Fit:
    """One fit: its planets (period, K, e, omega... with their errors),
    offset, slope, jitter, ln L, BIC, and the velocities it models"""

    def __init__(self, kind, periods, t_ref, drift, n):
        self.kind = kind
        self.periods = [float(p) for p in periods]
        self.t_ref = float(t_ref)
        self.drift = bool(drift)
        self.n = int(n)
        self.planets: List[Dict] = []
        self.offset = np.nan
        self.slope = 0.0
        self.jitter = np.nan
        self.lnl = -np.inf
        self.n_par = 0

    @property
    def bic(self) -> float:
        return _bic(self.lnl, self.n_par, self.n)

    def planet(self, index: int, t) -> np.ndarray:
        """Planet `index`'s velocity at t [m/s]"""
        p = self.planets[index]
        return keplerian(t, p["period"], p["k"], p["ecosw"], p["esinw"],
                         p["lam"], self.t_ref)

    def trend(self, t) -> np.ndarray:
        """The offset and the straight line at t [m/s]"""
        t = np.asarray(t, float)
        return self.offset + self.slope * (t - self.t_ref) / 365.25

    def model(self, t) -> np.ndarray:
        """Everything at t [m/s]: the trend and every planet"""
        out = self.trend(t)
        for index in range(len(self.planets)):
            out = out + self.planet(index, t)
        return out

    def others(self, index: int, t) -> np.ndarray:
        """The model less planet `index`: what is taken out of the data to
        show that planet alone"""
        return self.model(t) - self.planet(index, t)


def circular(t, v, err, periods, drift: bool, t_ref: float) -> Fit:
    """The circular orbits of every planet at once: linear, solved exactly"""
    angles = _angles(t, periods, t_ref)
    design = np.column_stack([_base(t, t_ref, drift)]
                             + [np.column_stack([np.cos(a), np.sin(a)])
                                for a in angles])
    coef, cov, jitter, lnl = _linear(design, v, err)
    out = Fit("circular", periods, t_ref, drift, len(t))
    nb = 2 if drift else 1
    out.offset = float(coef[0])
    out.slope = float(coef[1]) if drift else 0.0
    out.jitter, out.lnl = jitter, lnl
    out.n_par = design.shape[1] + 1
    for i, period in enumerate(periods):
        a, b = coef[nb + 2 * i], coef[nb + 2 * i + 1]
        c = cov[nb + 2 * i:nb + 2 * i + 2, nb + 2 * i:nb + 2 * i + 2]
        amp = float(np.hypot(a, b))
        var = a * a * c[0, 0] + b * b * c[1, 1] + 2 * a * b * c[0, 1]
        out.planets.append(dict(
            period=float(period), k=amp,
            k_err=float(np.sqrt(max(var, 0.0)) / max(amp, 1e-12)),
            # A cos x + B sin x = K cos(x + lam)
            lam=float(np.arctan2(-b, a)), ecosw=0.0, esinw=0.0,
            e=0.0, e_err=np.nan, omega=np.nan, omega_err=np.nan,
            eccentric=False))
    return out


def _unpack(x, eccentric, drift):
    offset, pos = x[0], 1
    slope = 0.0
    if drift:
        slope, pos = x[1], 2
    planets = []
    for ecc in eccentric:
        if ecc:
            amp, ecosw, esinw, lam = x[pos:pos + 4]
            pos += 4
        else:
            (amp, lam), ecosw, esinw = x[pos:pos + 2], 0.0, 0.0
            pos += 2
        planets.append((amp, ecosw, esinw, lam))
    return offset, slope, planets


def _prior(x, eccentric, drift):
    """The terms E_SIGMA adds to the least squares: e cos omega / E_SIGMA
    and e sin omega / E_SIGMA of each eccentric planet (none without it)"""
    if not E_SIGMA:
        return np.zeros(0)
    _, _, planets = _unpack(x, eccentric, drift)
    return np.array([value / E_SIGMA for (_, ecosw, esinw, _), ecc
                     in zip(planets, eccentric) if ecc
                     for value in (ecosw, esinw)], float)


def _model(x, t, periods, eccentric, drift, t_ref):
    offset, slope, planets = _unpack(x, eccentric, drift)
    out = offset + slope * (t - t_ref) / 365.25
    for period, (amp, ecosw, esinw, lam) in zip(periods, planets):
        out = out + keplerian(t, period, amp, ecosw, esinw, lam, t_ref)
    return out


def _start(start: Fit, eccentric, hk):
    x, lower, upper = [start.offset], [-np.inf], [np.inf]
    if start.drift:
        x.append(start.slope)
        lower.append(-np.inf)
        upper.append(np.inf)
    for p, ecc in zip(start.planets, eccentric):
        if ecc:
            ecosw, esinw = hk if p.get("e", 0.0) == 0 else (p["ecosw"],
                                                             p["esinw"])
            x += [p["k"], ecosw, esinw, p["lam"]]
            lower += [0.0, -E_MAX, -E_MAX, -np.inf]
            upper += [np.inf, E_MAX, E_MAX, np.inf]
        else:
            x += [p["k"], p["lam"]]
            lower += [0.0, -np.inf]
            upper += [np.inf, np.inf]
    x = np.clip(np.array(x, float), np.array(lower) + 1e-12,
                np.array(upper) - 1e-12)
    return x, (np.array(lower), np.array(upper))


def eccentric(t, v, err, periods, start: Fit,
              which: Optional[Sequence[bool]] = None) -> Fit:
    """Keplerian orbits of every planet at once (those `which` says, every
    one by default; the others circular), the jitter free: least squares at
    a jitter, then the jitter, three times over, from the circular solution
    and from a few eccentric starts; the best kept"""
    from scipy.optimize import least_squares
    which = [True] * len(periods) if which is None else [bool(w)
                                                         for w in which]
    drift, t_ref = start.drift, start.t_ref
    best = None
    # from the circular solution, a few eccentric starts too; from a
    #   Keplerian one (a planet made circular in it), that one alone
    starts = STARTS if any(which) and start.kind == "circular" \
        else STARTS[:1]
    for hk in starts:
        x, bounds = _start(start, which, hk)
        jitter = start.jitter
        sol = None
        for _ in range(3):
            root = 1.0 / np.sqrt(err ** 2 + jitter ** 2)
            sol = least_squares(
                lambda p: np.concatenate([
                    (v - _model(p, t, periods, which, drift, t_ref)) * root,
                    _prior(p, which, drift)]),
                x, bounds=bounds, x_scale="jac", method="trf")
            x = sol.x
            jitter = _best_jitter(v - _model(x, t, periods, which, drift,
                                             t_ref), err)
        resid = v - _model(x, t, periods, which, drift, t_ref)
        lnl = _lnl(resid, err, jitter)
        if best is None or lnl > best[0]:
            best = (lnl, x, jitter, sol)
    # each step (the least squares at a jitter, the jitter) raises ln L, so
    #   the fit is never below the circular solution it starts from
    lnl, x, jitter, sol = best
    out = Fit("keplerian", periods, t_ref, drift, len(t))
    out.lnl, out.jitter = lnl, jitter
    out.n_par = len(x) + 1
    offset, slope, planets = _unpack(x, which, drift)
    out.offset, out.slope = float(offset), float(slope)
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    except (ValueError, np.linalg.LinAlgError):
        cov = np.full((len(x), len(x)), np.nan)
    pos = 2 if drift else 1
    for period, ecc_on, (amp, ecosw, esinw, lam) in zip(periods, which,
                                                        planets):
        width = 4 if ecc_on else 2
        c = cov[pos:pos + width, pos:pos + width]
        pos += width
        ecc = float(np.hypot(ecosw, esinw))
        row = dict(period=float(period), k=float(amp),
                   k_err=float(np.sqrt(max(c[0, 0], 0.0))),
                   lam=float(lam), ecosw=float(ecosw), esinw=float(esinw),
                   e=min(ecc, E_MAX), e_err=np.nan, omega=np.nan,
                   omega_err=np.nan, eccentric=ecc_on,
                   # at the bound, the orbit goes through a few points far
                   #   off the others: no error means anything there
                   at_bound=bool(ecc_on and ecc >= E_MAX - 0.005))
        if ecc_on and ecc > 0 and not row["at_bound"]:
            hk = c[1:3, 1:3]
            grad_e = np.array([ecosw, esinw]) / ecc
            grad_w = np.array([-esinw, ecosw]) / ecc ** 2
            row.update(e_err=float(np.sqrt(max(grad_e @ hk @ grad_e, 0.0))),
                       omega=float(np.degrees(np.arctan2(esinw, ecosw))),
                       omega_err=float(np.degrees(np.sqrt(max(
                           grad_w @ hk @ grad_w, 0.0)))))
        out.planets.append(row)
    return out


def drift_for(t, periods) -> bool:
    """A straight line in time is fitted when every period is under a third
    of the campaign; at a longer one the line would take the planet"""
    span = float(np.ptp(t)) if np.size(t) else 0.0
    return bool(len(periods) == 0 or max(periods) < span / 3.0)


def fit_planets(t, v, err, periods, drift: Optional[bool] = None,
                with_keplerian: bool = True) -> Optional[Dict]:
    """Both fits of a star's known planets, and what they say. None when
    there are no periods, or too few points to fit them.

    Returns dict(circular=Fit, keplerian=Fit, null_bic, fit_bic=dict(
    circular, keplerian), planet_bic=[...], kep_planet_bic=[...],
    ecc_bic=[...], n, drift); what is Keplerian is None without
    with_keplerian (the circular fit alone, which is exact and quick)"""
    periods = [float(p) for p in periods or [] if p and np.isfinite(p)]
    t, v, err = _clean(t, v, err)
    if not periods or t.size < 4 * len(periods) + 6:
        return None
    if drift is None:
        drift = drift_for(t, periods)
    t_ref = 0.5 * (t.min() + t.max())
    v = v - np.median(v)
    base = _base(t, t_ref, drift)
    null = _linear(base, v, err)[3]
    circ = circular(t, v, err, periods, drift, t_ref)
    n = t.size
    # each planet in the circular fit: the fit without it
    planet_bic = []
    for i in range(len(periods)):
        rest = [p for j, p in enumerate(periods) if j != i]
        lnl = (circular(t, v, err, rest, drift, t_ref).lnl if rest
               else null)
        planet_bic.append(_bic(lnl, circ.n_par - 2, n) - circ.bic)
    null_bic = _bic(null, base.shape[1] + 1, n)
    out = dict(circular=circ, keplerian=None, null_bic=null_bic,
               fit_bic=dict(circular=null_bic - circ.bic, keplerian=None),
               planet_bic=planet_bic, kep_planet_bic=None, ecc_bic=None,
               n=n, drift=drift)
    if not with_keplerian:
        return out
    kep = eccentric(t, v, err, periods, circ)
    # each planet in the Keplerian fit: the fit without it, from the
    #   Keplerian solution of the others
    kep_planet_bic = []
    for i in range(len(periods)):
        rest = [p for j, p in enumerate(periods) if j != i]
        bic = (eccentric(t, v, err, rest, _without(kep, i)).bic if rest
               else null_bic)
        kep_planet_bic.append(bic - kep.bic)
    # each planet's eccentricity: the Keplerian fit with that planet
    #   circular, from the Keplerian solution
    ecc_bic = []
    for i in range(len(periods)):
        which = [j != i for j in range(len(periods))]
        if any(which):
            part = eccentric(t, v, err, periods, kep, which)
        else:
            part = circ
        ecc_bic.append(part.bic - kep.bic)
    out.update(keplerian=kep, kep_planet_bic=kep_planet_bic, ecc_bic=ecc_bic)
    out["fit_bic"]["keplerian"] = null_bic - kep.bic
    return out


def _without(fit: Fit, index: int) -> Fit:
    """A fit less planet `index`, as the start of the fit without it"""
    out = Fit(fit.kind, [p for j, p in enumerate(fit.periods) if j != index],
              fit.t_ref, fit.drift, fit.n)
    out.planets = [p for j, p in enumerate(fit.planets) if j != index]
    out.offset, out.slope = fit.offset, fit.slope
    out.jitter, out.lnl = fit.jitter, fit.lnl
    out.n_par = fit.n_par - (4 if fit.planets[index]["eccentric"] else 2)
    return out
