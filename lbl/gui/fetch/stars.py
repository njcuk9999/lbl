"""
The star of an archive search: who it is, where it is, how fast it moves
across the sky, and its effective temperature.

- resolve(name): SIMBAD's main identifier of the star, its other
  identifiers (the archives' exposures are matched against their
  catalogue numbers), its J2000 position, its proper motion (the archives
  are asked around where the star was when it was observed), its
  spectral type and its TIC number, kept on disk once asked;
- stellar_teff(ident): the best guess at its effective temperature, which
  a fetched folder keeps for the wrap (LBL's OBJECT_TEFF): the TESS Input
  Catalog's (v8, at MAST) when the star has a TIC number, else that of its
  spectral type;
- teff_from_spectral_type(sptype): the Teff of a dwarf of that spectral
  type, from the table of Pecaut & Mamajek (2013, ApJS 208, 9) as Mamajek
  keeps it (EEM_dwarf_UBVIJHK_colors_Teff.txt, version 2022.04.16),
  interpolated in the subclass. lbl.gui.simbad uses it too.

SIMBAD is asked through LBL's own client (lbl.core.simbad).
"""
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lbl.core import simbad
from .log import log
from .paths import USER_AGENT, cache


# =============================================================================
# Define variables
# =============================================================================
#: the effective temperature of a dwarf by spectral type [K] (Pecaut &
#:    Mamajek 2013)
SPT_TEFF: List[Tuple[str, int]] = [
    ('O3V', 44900), ('O4V', 42900), ('O5V', 41400), ('O5.5V', 40500),
    ('O6V', 39500), ('O6.5V', 38300), ('O7V', 37100), ('O7.5V', 36100),
    ('O8V', 35100), ('O8.5V', 34300), ('O9V', 33300), ('O9.5V', 31900),
    ('B0V', 31400), ('B0.5V', 29000), ('B1V', 26000), ('B1.5V', 24500),
    ('B2V', 20600), ('B2.5V', 18500), ('B3V', 17000), ('B4V', 16400),
    ('B5V', 15700), ('B6V', 14500), ('B7V', 14000), ('B8V', 12300),
    ('B9V', 10700), ('B9.5V', 10400), ('A0V', 9700), ('A1V', 9300),
    ('A2V', 8800), ('A3V', 8600), ('A4V', 8250), ('A5V', 8100),
    ('A6V', 7910), ('A7V', 7760), ('A8V', 7590), ('A9V', 7400),
    ('F0V', 7220), ('F1V', 7020), ('F2V', 6820), ('F3V', 6750),
    ('F4V', 6670), ('F5V', 6550), ('F6V', 6350), ('F7V', 6280),
    ('F8V', 6180), ('F9V', 6050), ('F9.5V', 5990), ('G0V', 5930),
    ('G1V', 5860), ('G2V', 5770), ('G3V', 5720), ('G4V', 5680),
    ('G5V', 5660), ('G6V', 5600), ('G7V', 5550), ('G8V', 5480),
    ('G9V', 5380), ('K0V', 5270), ('K1V', 5170), ('K2V', 5100),
    ('K3V', 4830), ('K4V', 4600), ('K5V', 4440), ('K6V', 4300),
    ('K7V', 4100), ('K8V', 3990), ('K9V', 3930), ('M0V', 3850),
    ('M0.5V', 3770), ('M1V', 3660), ('M1.5V', 3620), ('M2V', 3560),
    ('M2.5V', 3470), ('M3V', 3430), ('M3.5V', 3270), ('M4V', 3210),
    ('M4.5V', 3110), ('M5V', 3060), ('M5.5V', 2930), ('M6V', 2810),
    ('M6.5V', 2740), ('M7V', 2680), ('M7.5V', 2630), ('M8V', 2570),
    ('M8.5V', 2420), ('M9V', 2380), ('M9.5V', 2350), ('L0V', 2270),
    ('L1V', 2160), ('L2V', 2060)]


#: the letters of the spectral classes, in order
CLASSES = 'OBAFGKML'


#: where SIMBAD's answers about a star are kept, and the TIC's Teff
SIMBAD_CACHE = cache('simbad')


TIC_CACHE = cache('tic')


#: the keys a star kept on disk must have to be read back
_KEPT = ('main', 'aliases', 'ra', 'dec', 'pmra', 'pmdec', 'sptype', 'tic')


#: MAST's API, which serves the TESS Input Catalog
MAST = 'https://mast.stsci.edu/api/v0/invoke'


# =============================================================================
# Define functions
# =============================================================================
def _float(value: Any) -> Optional[float]:
    """a float, or None"""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def resolve(name: str, refresh: bool = False) -> Dict[str, Any]:
    """
    A star as SIMBAD knows it (lbl.core.simbad, which takes most spellings:
    Gl687, GJ 687, HD69830, TOI_1452...), kept on disk once asked
    (SIMBAD_CACHE), since it does not change

    :param name: str, a name
    :param refresh: bool, ask SIMBAD again

    :return: dict, name (as given), main (SIMBAD's main identifier),
             aliases (every identifier), ra, dec (J2000, deg), pmra, pmdec
             (its proper motion [mas/yr], None when SIMBAD has none or
             could not say), sptype (SIMBAD's spectral type), tic (its TIC
             identifier, or None)

    :raises ValueError: SIMBAD does not know the star
    :raises OSError: SIMBAD could not be asked
    """
    kept = os.path.join(SIMBAD_CACHE, re.sub(r'[^A-Za-z0-9+\-.]+', '_',
                                             name.strip()) + '.json')
    if os.path.exists(kept) and not refresh:
        try:
            with open(kept) as handle:
                out = json.load(handle)
            if all(key in out for key in _KEPT):
                return dict(out, name=name)
        except (OSError, ValueError):
            pass
    found = simbad.resolve(name)
    if found is None:
        tried = ', '.join(simbad.name_variants(name))
        raise ValueError(f'SIMBAD does not know {name} (asked: {tried})')
    aliases = found['ids']
    out = dict(name=name, main=found['main'], aliases=aliases,
               ra=_float(found['ra']), dec=_float(found['dec']),
               pmra=None, pmdec=None, sptype=found['sptype'],
               tic=next((alias for alias in aliases
                         if alias.startswith('TIC ')), None))
    # the proper motion: without it the star is looked for where it was in
    #   2000, and the exposures of a fast mover are missed
    try:
        rows = simbad.tap('SELECT b.pmra, b.pmdec FROM basic AS b JOIN '
                          'ident AS i ON i.oidref = b.oid WHERE i.id = '
                          + simbad.quoted(found['main']))
    except OSError as err:
        log(f'proper motion of {found["main"]}: not asked ({err}); the '
            f'star is looked for at its J2000 place', 'warn')
        return out
    if rows:
        out.update(pmra=_float(rows[0].get('pmra')),
                   pmdec=_float(rows[0].get('pmdec')))
    os.makedirs(os.path.dirname(kept), exist_ok=True)
    with open(kept, 'w') as handle:
        json.dump(out, handle)
    return out


def spectral_code(sptype: Any) -> Optional[float]:
    """
    A spectral type as a number (O0 = 0, B0 = 10, ... M0 = 60, L0 = 70; the
    subclass after), for a dwarf or a type with no luminosity class; None
    for a giant or a subgiant (I to IV), a white dwarf or what cannot be
    read ('M3.5Ve' 63.5, 'dM2' 62, 'K7V+M0V' 57)

    :param sptype: str, the spectral type (SIMBAD's)

    :return: float or None
    """
    if not sptype:
        return None
    text = str(sptype).strip()
    text = re.sub(r'^(sd|d)', '', text)
    match = re.match(r'([OBAFGKML])\s*([0-9](?:\.[0-9]+)?)?\s*'
                     r'((?:I{1,3}|IV|V|VI)?)', text)
    if not match or text.startswith(('D', 'W')):
        return None
    if match.group(3) and match.group(3) not in ('V', 'VI'):
        return None
    sub = float(match.group(2)) if match.group(2) else 5.0
    return CLASSES.index(match.group(1)) * 10 + sub


def teff_from_spectral_type(sptype: Any) -> Optional[float]:
    """the rough effective temperature of a dwarf from its spectral type
    (Pecaut & Mamajek 2013), interpolated in the subclass [K], or None"""
    code = spectral_code(sptype)
    if code is None:
        return None
    codes = np.array([spectral_code(spt) for spt, _ in SPT_TEFF])
    temps = np.array([float(teff) for _, teff in SPT_TEFF])
    if code < codes.min() - 1 or code > codes.max() + 1:
        return None
    return float(np.interp(code, codes, temps))


def tic_teff(tic: Any, timeout: float = 30.0) -> Optional[float]:
    """
    The effective temperature the TESS Input Catalog (v8, at MAST) gives a
    star, kept on disk once asked

    :param tic: str or int, its TIC number ('TIC 77612635' will do)
    :param timeout: float [s]

    :return: float [K], or None (none in the TIC, or MAST did not answer)
    """
    number = int(str(tic).replace('TIC', '').strip())
    path = os.path.join(TIC_CACHE, f'{number}.json')
    if os.path.exists(path):
        try:
            with open(path) as handle:
                return _float(json.load(handle).get('teff'))
        except (OSError, ValueError, AttributeError):
            pass
    request = dict(service='Mast.Catalogs.Filtered.Tic', format='json',
                   params=dict(columns='ID,Teff', filters=[dict(
                       paramName='ID', values=[str(number)])]))
    body = urllib.parse.urlencode(dict(request=json.dumps(request))).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                MAST, data=body, headers={'User-Agent': USER_AGENT}),
                timeout=timeout) as response:
            rows = json.loads(response.read()).get('data', [])
    except (OSError, ValueError):
        return None
    row = next((row for row in rows if row.get('ID') == number), None)
    if row is None:
        return None
    teff = _float(row.get('Teff'))
    os.makedirs(TIC_CACHE, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(dict(teff=teff), handle)
    return teff


def stellar_teff(ident: Dict[str, Any]) -> Dict[str, Any]:
    """
    The best guess at the effective temperature of a star: the TESS Input
    Catalog's, else its spectral type's

    :param ident: dict, the star (resolve): tic and sptype

    :return: dict, teff [K] and source (None where there is none)
    """
    if ident.get('tic'):
        teff = tic_teff(ident['tic'])
        if teff:
            return dict(teff=teff, source='TESS Input Catalog v8')
    teff = teff_from_spectral_type(ident.get('sptype'))
    if teff is not None:
        return dict(teff=teff,
                    source='its spectral type (Pecaut & Mamajek 2013)')
    return dict(teff=None, source=None)
