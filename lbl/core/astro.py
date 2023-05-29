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
    min: float = None
    max: float = None
    mean: float = None

    def __init__(self, label, min=None, max=None, mean=None, ref=None):
        self.name = label
        self.min = min
        self.max = max
        self.mean = mean
        self.ref = ref

# =============================================================================
# Define functions
# =============================================================================
def choose_bands(bands: List[Band], wavemin: float, wavemax: float
                 ) -> Tuple[List[str], List[float], List[float], List[bool]]:
    """
    Choose bands to use for a given wavelength range

    :param bands: list of bands
    :param wavemin: float, the minimum wavelength in nm
    :param wavemax: float, the maximum wavelength in nm
    :return:
    """
    bandnames = []
    blue_end = []
    red_end = []
    use_regions = []
    # loop around bands
    for band in bands:
        cond = band.min > wavemin
        cond &= band.max < wavemax
        # only add if within limits
        if cond:
            bandnames.append(band.name)
            blue_end.append(band.min)
            red_end.append(band.max)
            use_regions.append(True)
    # return lists
    return bandnames, blue_end, red_end, use_regions


# =============================================================================
# Define bands using
#    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse
# =============================================================================
uband = Band('u', min=305.511, max=403.064, mean=3572.18,
             ref='SDSS')
gband = Band('g', min=379.764, max=555.304, mean=4750.82,
             ref='SDSS')
rband = Band('r', min=541.823, max=699.442, mean=6204.29,
                ref='SDSS')
iband = Band('i', min=669.241, max=840.032, mean=7519.27,
                ref='SDSS')
zband = Band('z', min=796.470, max=1087.333, mean=8992.26,
                ref='SDSS')
yband = Band('y', min=965.000, max=1195.000, mean=1080.00,
                ref='MKO/NSFCam')
jband = Band('J', min=1175.00, max=1335.00, mean=1255.00,
                ref='MKO/NSFCam')
hband = Band('H', min=1525.00, max=1665.00, mean=1595.00,
                ref='MKO/NSFCam')
kband = Band('K', min=2025.00, max=2265.00, mean=2145.00,
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
