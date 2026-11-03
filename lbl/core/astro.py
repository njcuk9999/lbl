#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2023-05-29 at 14:40

@author: cook
"""
from typing import List, Tuple


# =============================================================================
# Define variables
# =============================================================================


# =============================================================================
# Define class
# =============================================================================
# Define photometric bands
class Band:
    minimum: float = None
    maximum: float = None
    mean: float = None

    def __init__(self, label, minimum=None, maximum=None, mean=None, ref=None):
        self.name = label
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.ref = ref


# =============================================================================
# Define functions
# =============================================================================
def choose_bands(bandobjs: List[Band], wavemin: float, wavemax: float
                 ) -> Tuple[List[str], List[float], List[float], List[bool]]:
    """
    Choose bands to use for a given wavelength range

    :param bandobjs: list of bands
    :param wavemin: float, the minimum wavelength in nm
    :param wavemax: float, the maximum wavelength in nm
    :return:
    """
    bandnames = []
    blue_end = []
    red_end = []
    use_regions = []
    # loop around bands
    for band in bandobjs:
        cond = band.minimum > wavemin
        cond &= band.maximum < wavemax
        # only add if within limits
        if cond:
            bandnames.append(band.name)
            blue_end.append(band.minimum)
            red_end.append(band.maximum)
            use_regions.append(True)
    # return lists
    return bandnames, blue_end, red_end, use_regions


# =============================================================================
# Define bands using
#    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse
# =============================================================================
uband = Band('u', minimum=305.511, maximum=403.064, mean=3572.18,
             ref='SLOAN/SDSS')
gband = Band('g', minimum=379.764, maximum=555.304, mean=475.082,
             ref='SLOAN/SDSS')
rband = Band('r', minimum=541.823, maximum=699.442, mean=620.429,
             ref='SLOAN/SDSS')
iband = Band('i', minimum=669.241, maximum=840.032, mean=751.927,
             ref='SLOAN/SDSS')
zband = Band('z', minimum=796.470, maximum=1087.333, mean=899.226,
             ref='/SDSS')
yband = Band('y', minimum=938.600, maximum=1113.400, mean=1025.880,
             ref='CFHT/Wircam')
jband = Band('j', minimum=1148.178, maximum=1349.441, mean=1248.414,
             ref='MKO/NSFCam')
hband = Band('h', minimum=1450.980, maximum=1809.105, mean=1629.826,
             ref='MKO/NSFCam')
kband = Band('k', minimum=1985.930, maximum=2401.514, mean=2200.537,
             ref='MKO/NSFCam')

# add all bands
bands = [uband, gband, rband, iband, zband, yband, jband, hband, kband]


# =============================================================================
# Define good CCF regions
# =============================================================================
# We do not want to use full band pass for the ccf - we really need the
# cleanest regions (these are defined as dictionaries so the instrument can
# choose which to use and which not to use)
ccf_regions = dict()
# clean r band
ccf_regions['r'] = [500, 650]
# clean i band
ccf_regions['i'] = [750, 850]
# clean y band
ccf_regions['y'] = [985, 1113]
# clean h band
ccf_regions['h'] = [1500, 1700]
# clean k band
ccf_regions['k'] = [2100, 2200]


# =============================================================================
# Define observing nights
# =============================================================================
# ONE NIGHT RUNS FROM LOCAL NOON TO THE NEXT LOCAL NOON AT THE OBSERVATORY, for
# every part of LBL that groups exposures by night: the compile's epochs
# (Instrument.get_epoch_groups), PCA2D's coadds (lbl.pca2d.nights), its correct
# stage, its holdout and its null test, the reports and the window. Local
# noon is local MEAN solar time, UT + east longitude / 15 h, from the site the
# instrument names in EARTH_LOCATION: no time zone and no equation of time,
# so the boundary is the same every day of the year (12:00 local mean time is
# 16:43 UT at La Silla, 22:22 UT on Mauna Kea, 11:37 UT at the OHP). One fixed
# UT hour cannot do this: noon UT is daytime in Chile and on La Palma but
# 2 h HST on Mauna Kea, where it cut SPIRou's nights in two.
def wrap_longitude(longitude):
    """
    An east longitude brought into [-180, 180) degrees, so that the night
    numbers of night_number are the same whichever way a site's longitude
    is written (-155.47 or 204.53)

    :param longitude: float, east longitude [deg]

    :return: float, the same longitude in [-180, 180) [deg]
    """
    if longitude is None:
        raise ValueError('no longitude: nights are cut at local noon at the '
                         'observatory, and its longitude is needed (see '
                         'lbl.core.astro.observatory_longitude)')
    value = float(longitude)
    if value != value or value in (float('inf'), float('-inf')):
        raise ValueError('the longitude {0!r} is not a finite number'
                         ''.format(longitude))
    return (value + 180.0) % 360.0 - 180.0


def night_number(jd, longitude):
    """
    The night a Julian date belongs to at an observatory at east longitude
    `longitude`: floor(jd + longitude / 360), an integer that changes at
    local mean noon there. A JD's fraction starts at noon UT, and adding the
    longitude as a fraction of a day moves that start to noon of the local
    mean solar time.

    The integer is the Julian day number of the LOCAL date the night began
    on (its noon UT): the same for a whole night on either side of local
    midnight, and night_number + 1 for the next one. A JD, a BJD or an rjd
    (BJD - 2400000) alike, whose fractions start at noon; an MJD
    (JD - 2400000.5) is passed as mjd + 0.5. The boundaries are those of
    Instrument.get_epoch_groups (math.bin_by_time with day_frac 0, which
    rounds to the nearest local midnight): the two group exposures alike.

    :param jd: float or array, the Julian dates [days]
    :param longitude: float, the observatory's east longitude [deg], from
                      observatory_longitude; there is no default

    :return: int or array of int, the nights
    """
    import numpy as np
    shift = wrap_longitude(longitude) / 360.0
    return np.floor(np.asarray(jd, dtype=float) + shift).astype(int)


_SITE_LONGITUDES = dict()


def site_longitude(site: str) -> float:
    """
    The east longitude of an observatory named as astropy names it
    (EarthLocation.of_site, an instrument's EARTH_LOCATION), in [-180, 180)
    degrees

    astropy reads its site registry (sites.json) from data.astropy.org the
    first time and keeps it in its cache. Without it, and without a network,
    this stops and says so: a night is never cut at noon UT by default.

    :param site: str, the site's name (e.g. 'La Silla Observatory', 'cfht')

    :return: float, the east longitude [deg]
    """
    key = str(site)
    if key in _SITE_LONGITUDES:
        return _SITE_LONGITUDES[key]
    from astropy.coordinates import EarthLocation
    try:
        location = EarthLocation.of_site(key)
    except Exception as exc:                                # noqa: BLE001
        try:
            from astropy.config import paths
            cache = paths.get_cache_dir()
        except Exception:                                   # noqa: BLE001
            cache = "astropy's cache"
        emsg = ('the longitude of the observatory {0!r} (EARTH_LOCATION) '
                'could not be had from astropy\'s site registry ({1}: {2}). '
                'Nights are cut at local noon there, so it is needed. astropy '
                'downloads its registry (sites.json from data.astropy.org) '
                'once and keeps it in {3}: run once with network access, or '
                'check the name against '
                'EarthLocation.get_site_names()').format(
            key, type(exc).__name__, exc, cache)
        raise ValueError(emsg) from exc
    _SITE_LONGITUDES[key] = wrap_longitude(location.lon.to_value('deg'))
    return _SITE_LONGITUDES[key]


#: the headers' names of an instrument that LBL spells otherwise
#:    (lbl.instruments.select.InstDict): the TNG writes HARPN for HARPS-N
INSTRUMENT_ALIASES = {'HARPN': 'HARPSN'}

_INSTRUMENT_SITES = dict()


def _family(name) -> str:
    """An instrument's family: NIRPS for NIRPS, NIRPS_HE and NIRPS_HA, SPIROU
    for SPIRou, HARPSN for HARPS-N and HARPN, MAROONX for MAROON-X"""
    import re
    family = re.sub(r'[^A-Z0-9]', '', str(name).upper().split('_')[0])
    return INSTRUMENT_ALIASES.get(family, family)


def instrument_site(instrument: str, data_source=None) -> str:
    """
    The site an instrument names in EARTH_LOCATION, from LBL's own class
    (lbl.instruments.select.InstDict) made with LBL's default parameters

    :param instrument: str, LBL's name of the instrument ('SPIROU',
                       'NIRPS_HE'...), or as a header writes it ('SPIRou',
                       'NIRPS', 'HARPN', 'MAROON-X'), matched by family
    :param data_source: str or None, the data source; None takes them all,
                        which must agree

    :return: str, the site, as EarthLocation.of_site names it
    """
    key = (str(instrument).upper(), None if data_source is None
           else str(data_source))
    if key in _INSTRUMENT_SITES:
        return _INSTRUMENT_SITES[key]
    from lbl.core.base_classes import ParamDict
    from lbl.core.parameters import params
    from lbl.instruments import select
    names = [n for n in select.InstDict if n.upper() == key[0]]
    if not names:
        names = [n for n in select.InstDict
                 if _family(n) == _family(instrument)]
    if not names:
        raise ValueError('LBL has no class for the instrument {0!r}: its '
                         'observatory, where nights are cut at local noon, is '
                         'unknown ({1})'.format(
                             instrument, ', '.join(select.InstDict)))
    sites = set()
    for name in names:
        sources = select.InstDict[name]
        wanted = ([key[1]] if key[1] is not None and key[1] in sources
                  else list(sources))
        for source in wanted:
            if name.upper() == 'GENERIC':
                raise ValueError('the Generic instrument has no site of its '
                                 'own: give its parameters (EARTH_LOCATION) '
                                 'rather than its name')
            inst = sources[source](params.copy(), ParamDict())
            site = inst.params['EARTH_LOCATION']
            if site in (None, ''):
                raise ValueError('{0} ({1}) sets no EARTH_LOCATION'
                                 ''.format(name, source))
            sites.add(str(site))
    if len(sites) > 1:
        raise ValueError('the instrument {0!r} names several sites ({1}): '
                         'give LBL\'s name and data source'.format(
                             instrument, ', '.join(sorted(sites))))
    _INSTRUMENT_SITES[key] = sites.pop()
    return _INSTRUMENT_SITES[key]


def observatory_longitude(instrument, data_source=None) -> float:
    """
    THE ONE PLACE LBL takes the longitude its nights are cut at from: the
    east longitude [deg, -180 to 180] of an instrument's observatory, the
    site of its EARTH_LOCATION (astropy's EarthLocation.of_site)

    :param instrument: LBL's name of the instrument ('SPIROU', 'NIRPS_HE',
                       or as a header writes it: 'NIRPS', 'HARPN'...), an
                       instrument object (lbl.instruments) or its
                       parameters (anything with an EARTH_LOCATION entry,
                       which is how a Generic instrument gives its site)
    :param data_source: str or None, the data source, with a name

    :return: float, the east longitude [deg]
    """
    from collections.abc import Mapping
    params = getattr(instrument, 'params', instrument)
    if isinstance(params, Mapping):
        site = params.get('EARTH_LOCATION', None)
        if site in (None, ''):
            name = params.get('INSTRUMENT', None)
            if name in (None, ''):
                raise ValueError('these parameters set no EARTH_LOCATION and '
                                 'no INSTRUMENT: the observatory, where '
                                 'nights are cut at local noon, is unknown')
            source = params.get('DATA_SOURCE', data_source)
            site = instrument_site(name, source)
        return site_longitude(site)
    if instrument in (None, '', '?'):
        raise ValueError('no instrument named: the observatory, where nights '
                         'are cut at local noon, is unknown')
    return site_longitude(instrument_site(instrument, data_source))


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
