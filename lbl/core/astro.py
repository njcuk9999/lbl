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
jband = Band('j', minimum=1148.178, maximum=13494.41, mean=1248.414,
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
