"""The generalised Lomb-Scargle periodogram (Zechmeister & Kuerster 2009,
A&A 496, 577, https://doi.org/10.1051/0004-6361:200811296: floating mean,
weighted) and its frequency grid, for the report's periodogram page.

The power at each frequency is built from weighted sums over the points
(_gls_sums), all frequencies at once as matrix products, then combined as
in eq. 5 of the paper (_gls_power): numpy only.
"""
from typing import Dict, Optional

import numpy as np


def frequency_grid(time: np.ndarray, pmin: float = 1.1,
                   pmax: Optional[float] = None, oversample: int = 10
                   ) -> np.ndarray:
    """
    A grid regular in frequency, `oversample` points per peak width (1/T)

    :param time: np.ndarray, the time [days]
    :param pmin: float, the shortest period [days]
    :param pmax: float or None, the longest period (twice the baseline when
                 None) [days]
    :param oversample: int, points per 1/T

    :return: np.ndarray, the frequencies [1/day], increasing
    """
    baseline = float(np.ptp(time))
    if pmax is None:
        pmax = 2 * baseline
    step = 1.0 / (oversample * baseline)
    fmin, fmax = 1.0 / pmax, 1.0 / pmin
    return np.arange(fmin, fmax + 0.5 * step, step)


def _trig(time: np.ndarray, freq: np.ndarray, tref: float = 0.0):
    """
    cos and sin of 2 pi f (t - tref), (n x G)

    :param time: np.ndarray, the time [days]
    :param freq: np.ndarray, the frequencies [1/day]
    :param tref: float, the reference time [days]

    :return: tuple, the cos and sin matrices
    """
    phase = 2 * np.pi * np.outer(time - tref, freq)
    return np.cos(phase), np.sin(phase)


def _gls_sums(weight: np.ndarray, value: np.ndarray, cos: np.ndarray,
              sin: np.ndarray) -> Dict[str, np.ndarray]:
    """
    The weighted sums the GLS power is made of (additive over points)

    :return: dict, the sums
    """
    return dict(w=np.sum(weight), y=weight @ value, yy=weight @ value ** 2,
                c=weight @ cos, s=weight @ sin, yc=(weight * value) @ cos,
                ys=(weight * value) @ sin, cc=weight @ cos ** 2,
                ss=weight @ sin ** 2, cs=weight @ (cos * sin))


def _gls_power(sums: Dict[str, np.ndarray]) -> np.ndarray:
    """
    The GLS power from its sums (Zechmeister & Kurster 2009, eq. 5)

    :param sums: dict, the output of _gls_sums (or a difference of them)

    :return: np.ndarray, the power (0 to 1)
    """
    wsum = sums['w']
    ymean, cmean, smean = sums['y'] / wsum, sums['c'] / wsum, sums['s'] / wsum
    yy = sums['yy'] / wsum - ymean ** 2
    yc = sums['yc'] / wsum - ymean * cmean
    ys = sums['ys'] / wsum - ymean * smean
    cc = sums['cc'] / wsum - cmean ** 2
    ss = sums['ss'] / wsum - smean ** 2
    cs = sums['cs'] / wsum - cmean * smean
    det = cc * ss - cs ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        power = (ss * yc ** 2 + cc * ys ** 2 - 2 * cs * yc * ys) / (yy * det)
    return np.clip(np.nan_to_num(power), 0, 1)


def gls(time: np.ndarray, value: np.ndarray, err: np.ndarray,
        freq: np.ndarray) -> np.ndarray:
    """
    The generalised Lomb-Scargle periodogram (floating mean, weighted)

    :param time: np.ndarray, the time [days]
    :param value: np.ndarray, the values
    :param err: np.ndarray, the error bars
    :param freq: np.ndarray, the frequencies [1/day]

    :return: np.ndarray, the power (0 to 1)
    """
    cos, sin = _trig(time, freq, float(np.mean(time)))
    return _gls_power(_gls_sums(1.0 / err ** 2, value, cos, sin))
