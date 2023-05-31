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
             ref='SDSS')
gband = Band('g', minimum=379.764, maximum=555.304, mean=4750.82,
             ref='SDSS')
rband = Band('r', minimum=541.823, maximum=699.442, mean=6204.29,
             ref='SDSS')
iband = Band('i', minimum=669.241, maximum=840.032, mean=7519.27,
             ref='SDSS')
zband = Band('z', minimum=796.470, maximum=1087.333, mean=8992.26,
             ref='SDSS')
yband = Band('y', minimum=965.000, maximum=1195.000, mean=1080.00,
             ref='MKO/NSFCam')
jband = Band('J', minimum=1175.00, maximum=1335.00, mean=1255.00,
             ref='MKO/NSFCam')
hband = Band('H', minimum=1525.00, maximum=1665.00, mean=1595.00,
             ref='MKO/NSFCam')
kband = Band('K', minimum=2025.00, maximum=2265.00, mean=2145.00,
             ref='MKO/NSFCam')

# add all bands
bands = [uband, gband, rband, iband, zband, yband, jband, hband, kband]

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
