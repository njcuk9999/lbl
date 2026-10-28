"""
The star: its spectral type, a rough mass (for the minimum mass of a
companion) and a best-guess radius (for the depth of a transit)

The mass of a main-sequence star from its spectral type is that of the
table of Pecaut & Mamajek (2013, ApJS 208, 9), as Mamajek keeps it
(EEM_dwarf_UBVIJHK_colors_Teff.txt, version 2022.04.16), interpolated in
the subclass; it is rough (a dwarf of a type spans some 10 % in mass), and
the NASA Exoplanet Archive's mass of a planet host is used first when
there is one. The masses of the Earth and Jupiter are the IAU 2015 nominal
values; Neptune's, its GM from JPL Horizons (6835099.97 km^3/s^2, the planet
without Triton).

The radius and the effective temperature (for the equilibrium temperature
of a planet), best guess first: the NASA Exoplanet Archive's for a planet
host, the TESS Input Catalog's (v8, MAST: Gaia, and the relations of Mann
et al. for the cool dwarfs), APERO's (Mann et al. 2015 from M_Ks), the
same table's by spectral type (its R_Rsun), or the mass's (R ~ M^0.9).

Created on 2026-10-03

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module stars): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .paths import cache


#: the effective temperature of a dwarf by spectral type [K]: the same
#: table (its Teff), for the same types
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


#: where the TIC's answers are kept
TIC_CACHE = cache('tic')


#: the letters of the spectral classes, in order
CLASSES = 'OBAFGKML'


# =============================================================================
# Define functions
# =============================================================================
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


def stellar_teff(ident: Dict[str, Any],
                 archive_star: Optional[Dict[str, Any]] = None,
                 apero: Optional[Dict[str, Any]] = None,
                 tic: bool = True) -> Dict[str, Any]:
    """
    The best guess at the effective temperature of a star, in the order of
    the radius (stellar_radius): the NASA Exoplanet Archive's, the TESS
    Input Catalog's, APERO's, its spectral type's

    :return: dict, teff [K] and source (None where there is none)
    """
    archive_star = archive_star or {}
    if archive_star.get('teff'):
        return dict(teff=float(archive_star['teff']),
                    source='NASA Exoplanet Archive')
    if tic and ident.get('tic'):
        found = tic_star(ident['tic'])
        if found.get('teff'):
            return dict(teff=float(found['teff']),
                        source='TESS Input Catalog v8')
    if apero and apero.get('teff'):
        return dict(teff=float(apero['teff']), source='APERO')
    found = teff_from_spectral_type(ident.get('sptype'))
    if found is not None:
        return dict(teff=found,
                    source='its spectral type (Pecaut & Mamajek 2013)')
    return dict(teff=None, source=None)


def tic_star(tic: Any, timeout: float = 30.0) -> Dict[str, Any]:
    """
    What the TESS Input Catalog says of a star (version 8, MAST: its radius
    and mass from Gaia and the relations of Mann et al. for the cool
    dwarfs), kept on this machine

    :param tic: str or int, its TIC number ('TIC 77612635' will do)

    :return: dict, radius, radius_err [solar radii], mass, mass_err [solar
             masses], teff [K]; empty when MAST does not answer
    """
    number = int(str(tic).replace('TIC', '').strip())
    path = os.path.join(TIC_CACHE, f'{number}.json')
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    from .tess import _mast
    try:
        rows = _mast('Mast.Catalogs.Filtered.Tic', dict(
            columns='ID,rad,e_rad,mass,e_mass,Teff', filters=[dict(
                paramName='ID', values=[str(number)])]), timeout=timeout)
    except (OSError, ValueError):
        return {}
    row = next((row for row in rows if row.get('ID') == number), None)
    if row is None:
        return {}
    out = dict(radius=row.get('rad'), radius_err=row.get('e_rad'),
               mass=row.get('mass'), mass_err=row.get('e_mass'),
               teff=row.get('Teff'))
    os.makedirs(TIC_CACHE, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(out, handle)
    return out
