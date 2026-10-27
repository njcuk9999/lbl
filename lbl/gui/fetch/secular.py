"""
The acceleration of a star, with its error.

"Secular acceleration" is used in two senses, and this module has both.

1. The acceleration of the star itself, dv_r/dt, from a companion too far
   out to complete an orbit during the series (or an instrument). It is
   measured, not known: the trend of RVModel is a polynomial in time, and
   acceleration() reads it in physical units, dv_r/dt in m/s/yr at the
   reference time and, with a trend of degree two, its change in
   m/s/yr^2, with their errors from the fit (outlier-aware with koloa's
   likelihood). A companion at a projected separation rho that makes an
   acceleration a has a mass of at least

       M >= (sqrt(27) / 2) |a| rho^2 / G

   whatever its orbit (Torres 1999; Liu et al. 2002: 5.34e-6 Msun times
   (d/pc rho/arcsec)^2 |a|/(m/s/yr) times a factor of at least sqrt(27)/2,
   reached when the companion lies rho / sqrt(2) behind or in front of the
   star); companion_min_mass() gives it, for draws of a too.

2. The perspective acceleration: a star that crosses the sky at a transverse
   velocity v_t = mu d sees its radial velocity grow even at constant space
   velocity, because the line of sight turns towards its motion:

       dv_r / dt = v_t^2 / d = mu^2 d

   (Kurster et al. 2003 for Barnard's star, 4.5 m/s/yr; also called the
   secular acceleration). It is geometry, known in advance from the proper
   motion and the parallax, so it enters a fit as a parameter with a
   gaussian prior from the astrometry (RVModel(perspective=(value,
   error))) rather than as a free trend: a trend fitted at the same time
   takes the acceleration of the star itself.

   With mu in mas/yr and the parallax varpi in mas,

       a [m/s/yr] = C mu^2 / varpi,   C = mas^2 pc 1000 / yr = 2.2983e-5,

   and its error follows from the covariance of (pmra, pmdec, parallax),
   correlations included (Gaia gives them). The first-order propagation is
   exact enough whenever the parallax is measured to better than a few per
   cent; perspective_acceleration also draws the astrometry to check it.
   The second-order terms (the change of the acceleration itself, of order
   3 a v_r / d) are below a cm/s over a decade for every star but a few,
   and are neglected.

   Check first that the velocities still hold it. A barycentric correction
   that is given the proper motion and the parallax of the star removes it
   already: barycorrpy does (Kanodia & Wright 2018, Eq. 28 of Wright &
   Eastman 2014), and APERO, which reduces SPIRou and NIRPS, passes it the
   Gaia astrometry of every target, so APERO velocities carry no
   perspective acceleration and adding this term to their fit would count
   it twice (the SPIRou series of Barnard's star drifts by 1.2 m/s/yr, not
   4.5). The term is for velocities whose barycentric correction ignores
   the motion of the star.

   gaia_astrometry fetches the astrometry of a star by name: SIMBAD gives
   its Gaia DR3 identifier, the Gaia archive its astrometry.

The first version of this module called the perspective functions
secular_acceleration, secular_from_astrometry, secular_drift and
remove_secular; those names still work.

Created on 2026-09-27

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module secular): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import csv
import io
import json
import os
import re
import time as _time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional
import numpy as np
from lbl.core import simbad
from .log import log
from .paths import cache


#: where the astrometry of the stars is kept (a file per star: a machine
#: without the network has it once it was asked)
ASTROMETRY_CACHE = cache('gaia')


#: the stars whose astrometry could not be asked, and when
_FAILED: Dict[str, float] = {}


#: the services (SIMBAD's through its Python client, lbl.core.simbad)
GAIA_TAP = 'https://gea.esac.esa.int/tap-server/tap/sync'


GAIA_COLUMNS = ['source_id', 'ra', 'dec', 'parallax', 'parallax_error',
                'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
                'parallax_pmra_corr', 'parallax_pmdec_corr',
                'pmra_pmdec_corr', 'radial_velocity',
                'radial_velocity_error', 'ruwe']


def _tap(url: str, query: str, timeout: float) -> list:
    """One synchronous ADQL query, as a list of dict"""
    body = urllib.parse.urlencode(dict(REQUEST='doQuery', LANG='ADQL',
                                       FORMAT='csv', QUERY=query)).encode()
    request = urllib.request.Request(url, data=body)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        text = resp.read().decode()
    return list(csv.DictReader(io.StringIO(text)))


def star_astrometry(ident: Dict[str, Any], timeout: float = 60.0,
                    refresh: bool = False) -> Optional[Dict[str, Any]]:
    """
    The astrometry of a star that was resolved (koloa.archive.resolve):
    that of its Gaia DR3 source, else SIMBAD's proper motion and parallax
    (a star too bright for Gaia). Kept in koloa's cache, a file per star,
    so that a machine without the network has it once it was asked

    :param ident: dict, the star: gaia_dr3 (its Gaia DR3 source) and main
                  (its SIMBAD name)
    :param timeout: float [s]
    :param refresh: bool, ask again

    :return: dict or None (no astrometry, or no network and nothing
             kept): pmra, pmdec [mas/yr], parallax [mas], source (in
             words), and Gaia's other columns when it is Gaia's
    """
    # the number of the source, not the 3 of the release
    found = re.findall(r'\d{6,}', str(ident.get('gaia_dr3') or ''))
    number = found[-1] if found else ''
    name = str(ident.get('main') or ident.get('name') or '')
    if not number and not name:
        return None
    key = number or 'name_' + re.sub(r'[^A-Za-z0-9]+', '_', name)
    path = os.path.join(ASTROMETRY_CACHE, f'{key}.json')
    if os.path.exists(path) and not refresh:
        with open(path) as handle:
            return json.load(handle) or None
    # not asked again at once where it could not be (no network)
    if _time.time() - _FAILED.get(key, 0.0) < 600.0:
        return None
    out: Optional[Dict[str, Any]] = None
    # the archive of Gaia stalls at times: SIMBAD's then, and Gaia's
    #   asked again another time (nothing is kept)
    stalled = False
    try:
        if number:
            try:
                rows = _tap(GAIA_TAP, f"SELECT {', '.join(GAIA_COLUMNS)} "
                                      f"FROM gaiadr3.gaia_source WHERE "
                                      f"source_id = {number}",
                            min(timeout, 20.0))
            except (OSError, ValueError) as err:
                rows, stalled = [], True
                log(f'astrometry of {name or number}: Gaia not asked '
                    f'({err}), SIMBAD instead', 'warn')
            if rows:
                out = dict(source_id=int(number),
                           source=f'Gaia DR3 {number}')
                for col in GAIA_COLUMNS[1:]:
                    val = rows[0].get(col, '')
                    out[col] = (float(val) if val not in ('', None)
                                else float('nan'))
        if (out is None or not np.isfinite(out['pmra'] + out['pmdec']
                                           + out['parallax'])) and name:
            rows = simbad.tap("SELECT pmra, pmdec, plx_value FROM basic "
                              "JOIN ident ON oidref = oid WHERE id = "
                              + simbad.quoted(name))
            vals = [rows[0].get(col) for col in
                    ('pmra', 'pmdec', 'plx_value')] if rows else []
            out = (dict(source='SIMBAD', pmra=float(vals[0]),
                        pmdec=float(vals[1]), parallax=float(vals[2]))
                   if vals and None not in vals else None)
    except (OSError, ValueError) as err:
        _FAILED[key] = _time.time()
        log(f'astrometry of {name or number}: not asked ({err})', 'warn')
        return None
    if stalled:
        return out
    os.makedirs(ASTROMETRY_CACHE, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(out or {}, handle)
    return out
