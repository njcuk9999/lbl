#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIMBAD, asked through its Python client (astroquery.simbad, on SIMBAD's TAP
service), for the window's archive search and Teff, the PCA2D report and
the astrometry of the secular acceleration. Not through a web page or CDS's
Sesame resolver: a service that takes a script for a robot can block it,
and on 2026-10-27 Sesame did not answer at all while SIMBAD's TAP service
did, in a fraction of a second.

A name is asked as it is written, then as SIMBAD writes it
(name_variants): TOI_1234, TOI 1234 and TOI1234 are TOI-1234, Gl_699 is
GJ 699, and LBL's suffixes (_tc, _pca2d...) are taken off.

@author: artigau
"""
import re
from typing import Any, Dict, List, Optional

# =============================================================================
# Define variables
# =============================================================================
#: seconds a request to SIMBAD may take, connection and answer
TIMEOUT = 30
#: what LBL and its users append to a star's name: the telluric cleaning
#:   (_tc), PCA2D's corrections (_pca2d...), an instrument
SUFFIXES = re.compile(r'(_tc|_pca2d[0-9A-Za-z._+-]*|_(SPIROU|NIRPS|NIRPS_HE|'
                      r'NIRPS_HA|HARPS|HARPSN|ESPRESSO))+$', re.I)
#: catalogues whose numbers SIMBAD writes in one way, and that way
FAMILIES = ((r'TOI', 'TOI-{0}'), (r'TIC', 'TIC {0}'),
            (r'GJ|GL|GLIESE', 'GJ {0}'), (r'HD', 'HD {0}'),
            (r'HIP', 'HIP {0}'), (r'LHS', 'LHS {0}'), (r'K2', 'K2-{0}'),
            (r'KEPLER', 'Kepler-{0}'), (r'WASP', 'WASP-{0}'))

_SIMBAD = None


# =============================================================================
# Define functions
# =============================================================================
def toi_number(name: Any) -> Optional[str]:
    """
    The TOI number of a name in any spelling: TOI-1234, TOI 1234, TOI_1234
    and TOI1234, with a planet's .01 or LBL's suffixes after it

    :param name: a name

    :return: str, the number without leading zeros, or None for a name that
             is not a TOI's
    """
    found = re.match(r'^\s*TOI[\s_-]*0*(\d+)(?:\.\d+)?(?![0-9])',
                     str(name or ''), re.I)
    return found.group(1) if found else None


def name_variants(name: Any) -> List[str]:
    """
    The spellings a name is asked of SIMBAD in, the name as written first:
    without LBL's suffixes, with spaces for underscores, and as SIMBAD writes
    the catalogues of FAMILIES (TOI_1234 -> TOI-1234, gl699 -> GJ 699,
    GJ_15A -> GJ 15 A)

    :param name: a name

    :return: list of str, without repeats
    """
    raw = str(name or '').strip()
    out = []
    for base in (raw, SUFFIXES.sub('', raw)):
        if not base:
            continue
        out += [base, re.sub(r'_+', ' ', base).strip()]
        for family, spelled in FAMILIES:
            found = re.match(r'^(?:%s)[\s_-]*0*(\d+)(?:[\s_-]*([A-Ca-c]))?$'
                             % family, base, re.I)
            if found:
                component = ' ' + found.group(2).upper() \
                    if found.group(2) else ''
                out.append(spelled.format(found.group(1)) + component)
        # a TOI's planet (TOI-1452.01, TOI_1452.01): its star
        if toi_number(base):
            out.append('TOI-' + toi_number(base))
    return [one for one in dict.fromkeys(out) if one]


def _simbad():
    """The client, made once, its requests given TIMEOUT (astroquery's own
    timeout is how long SIMBAD may work on a query, not how long the
    network may take: a service that does not answer would hold a report
    or the window for as long as the connection stays open)"""
    global _SIMBAD
    if _SIMBAD is None:
        from astroquery.simbad import Simbad
        simbad = Simbad()
        session = simbad._session
        request = session.request

        def with_timeout(method, url, **kwargs):
            kwargs.setdefault('timeout', TIMEOUT)
            return request(method, url, **kwargs)
        session.request = with_timeout
        _SIMBAD = simbad
    return _SIMBAD


def _value(value: Any) -> Any:
    """A table cell as Python: None where it is masked, str for text"""
    import numpy as np
    if value is np.ma.masked or value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def quoted(text: Any) -> str:
    """A string as an ADQL literal"""
    return "'%s'" % str(text).replace("'", "''")


def tap(query: str) -> List[Dict[str, Any]]:
    """
    The rows of an ADQL query of SIMBAD's tables

    :param query: str, the query

    :return: list of dict, column -> value (None where SIMBAD has none)

    :raises OSError: SIMBAD could not be asked, or did not answer
    """
    try:
        table = _simbad().query_tap(query)
    except ImportError as exc:
        raise OSError('astroquery is not installed (pip install astroquery):'
                      ' SIMBAD cannot be asked') from exc
    except Exception as exc:                              # noqa: BLE001
        raise OSError('SIMBAD: %s: %s' % (type(exc).__name__,
                                          str(exc)[:200])) from exc
    return [{col: _value(row[col]) for col in table.colnames}
            for row in table]


def resolve(name: Any) -> Optional[Dict[str, Any]]:
    """
    A star as SIMBAD knows it, from the first of its name_variants SIMBAD
    knows

    :param name: a name

    :return: dict(asked: the spelling SIMBAD knew, main: its main
             identifier, ra, dec: J2000 [deg], sptype, plx [mas], ids:
             every identifier, spaces collapsed), or None when SIMBAD
             knows none of the spellings

    :raises OSError: SIMBAD could not be asked, or did not answer
    """
    for one in name_variants(name):
        rows = tap('SELECT b.main_id, b.ra, b.dec, b.sp_type, b.plx_value,'
                   ' s.ids FROM basic AS b JOIN ident AS i ON i.oidref ='
                   ' b.oid JOIN ids AS s ON s.oidref = b.oid WHERE i.id = '
                   + quoted(one))
        if rows:
            row = rows[0]
            ids = [' '.join(str(ident).split())
                   for ident in str(row['ids'] or '').split('|')]
            return dict(asked=one, main=' '.join(str(row['main_id']).split()),
                        ra=row['ra'], dec=row['dec'],
                        sptype=(str(row['sp_type']).strip()
                                if row['sp_type'] else None),
                        plx=row['plx_value'], ids=[i for i in ids if i])
    return None


# =============================================================================
# End of code
# =============================================================================
