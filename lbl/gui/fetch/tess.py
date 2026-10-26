"""
The TESS light curves of a star, and whether its brightness varies at the
period of a velocity signal.

Spots that rotate with the star modulate its brightness and its velocities
at the rotation period and its harmonics (P/2, P/3); a planet does not
change the brightness, unless it transits. A photometric peak at the period
of a velocity signal, at its half or third, or at its double, therefore
speaks against a planet.

- light_curves(name): the light curve of every sector at MAST (by the TIC
  number, from Sesame), one per sector: SPOC's 2-minute light curve when
  there is one, else TESS-SPOC's or QLP's from the full frames. The FITS
  files are kept in a folder and read back.
- periodicity(lcs, periods): each sector binned to 30 minutes (flares and
  bad points clipped first), its GLS periodogram, its strongest periods and
  their amplitudes, and at each velocity period and its harmonics the power
  there against the sector's highest peak.
  A match is a local maximum of the periodogram within one peak width
  (1/T) of the tested period that holds half the sector's highest power;
  the check flags it when it is the strongest peak of two sectors or more
  (or of the only one), unless it sits at the orbit of TESS (13.7 d) or its
  half, where the light curves keep systematics.
- figure(res): the periodograms of the sectors with the periods marked, and
  the light curve of the sector that best shows the strongest period.

What TESS cannot tell: a sector lasts 27 days and SPOC's PDCSAP flux
flattens slower changes, so periods longer than about half a sector (13 d)
are not tested; putting sectors side by side does not help, since each is
normalised on its own. The rotation of most M dwarfs (tens of days) is out
of reach; its harmonics may not be.

Created on 2026-09-30

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module tess): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List


# =============================================================================
# Define variables
# =============================================================================
MAST = 'https://mast.stsci.edu/api/v0/invoke'


# =============================================================================
# Define functions
# =============================================================================
def _mast(service: str, params: Dict[str, Any],
          timeout: float = 60.0) -> List[Dict[str, Any]]:
    """the rows of a MAST query"""
    req = dict(service=service, format='json', params=params)
    body = urllib.parse.urlencode(dict(request=json.dumps(req))).encode()
    with urllib.request.urlopen(urllib.request.Request(MAST, data=body),
                                timeout=timeout) as resp:
        return json.loads(resp.read()).get('data', [])
