"""
Who the star is, and what is known of its planets.

- resolve(name): the names of a star, from CDS's Sesame (SIMBAD): its main
  identifier and aliases (GJ, HD, HIP, TIC, Gaia DR3...), whatever the
  spelling in a file header ('Gl687' is GJ 687);
- known_planets(name): its planets in the NASA Exoplanet Archive, found by
  their Gaia DR3 or TIC identifier (then by host name): the archive's
  default solution of each planet (pscomppars) and every published one
  that gives K (ps), with their references;
- compare(orbit, known): a fitted orbit against the planet of its period,
  its K against every published solution, the most recent first (the
  archive's default is often the discovery paper, from far fewer velocities
  than a star has today);
- conjunction(planet): the published time of conjunction of a planet, to
  carry its ephemeris to a new series; bibcode(url): the paper of a
  solution (koloa.literature finds its velocities on VizieR).

known_planets also keeps what the archive knows of the star (rotation
period, spectral type, Teff, radius, metallicity, v sin i, magnitudes).

A second opinion: encyclopaedia() keeps the catalogue of the Extrasolar
Planets Encyclopaedia (exoplanet.eu), and known_planets sets its answer
beside the archive's (eu of each planet: its P, K, e, time of transit...;
the planets it alone has are added). The two mostly agree; each has
planets and times of transit the other lacks.

The archive is fetched once and kept: its two tables (pscomppars, the
default solution of every planet, and the solutions of ps that give K, a
few MB in all) in ~/.cache/koloa/archive, where every star is then looked
up without the network. tables(refresh=True) (koloa --refresh-archive, or
--refresh with --detailed) fetches them again. known_planets also keeps
what it found for a star in a JSON file when given a path.

Created on 2026-09-29

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module archive): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict
from .paths import cache


# =============================================================================
# Define variables
# =============================================================================
SESAME = 'https://cds.unistra.fr/cgi-bin/nph-sesame/-oxI/S'


#: where the archive's tables are kept, once fetched
CACHE = cache('archive')


# =============================================================================
# Define functions
# =============================================================================
def resolve(name: str, timeout: float = 30.0, refresh: bool = False
            ) -> Dict[str, Any]:
    """
    The names of a star (CDS Sesame, SIMBAD's resolver, which takes most
    spellings: Gl687, GJ 687, HD69830, Proxima...), kept on disk once asked
    (CACHE/sesame), since they do not change

    :param name: str, a name
    :param timeout: float [s]
    :param refresh: bool, ask Sesame again

    :return: dict, name (as given), main (SIMBAD's main identifier), aliases
             (every identifier), and the ones koloa uses: gaia_dr3, tic,
             hip, hd, gj (None when SIMBAD has none), ra, dec (J2000,
             degrees), sptype (SIMBAD's spectral type) and plx [mas]
    """
    kept = os.path.join(CACHE, 'sesame', re.sub(r'[^A-Za-z0-9+\-.]+', '_',
                                                name.strip()) + '.json')
    if os.path.exists(kept) and not refresh:
        with open(kept) as handle:
            out = dict(json.load(handle), name=name)
        if 'sptype' in out:
            return out
        # kept before the spectral type was: asked once more, or as it is
        try:
            return resolve(name, timeout=timeout, refresh=True)
        except Exception:
            return out
    url = f'{SESAME}?{urllib.parse.quote(name)}'
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode('utf-8', 'replace')
    main = re.search(r'<oname>(.*?)</oname>', text)
    if main is None:
        raise ValueError(f'SIMBAD does not know {name}')
    aliases = [' '.join(alias.split())
               for alias in re.findall(r'<alias>(.*?)</alias>', text)]

    def first(prefix):
        return next((alias for alias in aliases
                     if alias.startswith(prefix + ' ')), None)
    # the position (J2000, degrees), for planning observations
    radeg = re.search(r'<jradeg>(.*?)</jradeg>', text)
    dedeg = re.search(r'<jdedeg>(.*?)</jdedeg>', text)
    sptype = re.search(r'<spType>(.*?)</spType>', text)
    plx = re.search(r'<plx><v>(.*?)</v>', text)
    out = dict(name=name, main=' '.join(main.group(1).split()),
               aliases=aliases, gaia_dr3=first('Gaia DR3'),
               tic=first('TIC'), hip=first('HIP'), hd=first('HD'),
               gj=first('GJ'),
               ra=float(radeg.group(1)) if radeg else None,
               dec=float(dedeg.group(1)) if dedeg else None,
               sptype=sptype.group(1).strip() if sptype else None,
               plx=float(plx.group(1)) if plx else None)
    os.makedirs(os.path.dirname(kept), exist_ok=True)
    with open(kept, 'w') as handle:
        json.dump(out, handle)
    return out
