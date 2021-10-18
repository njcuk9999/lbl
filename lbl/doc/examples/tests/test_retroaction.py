#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-10-18

@author: artigau
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import etienne_tools as et
from tqdm import tqdm
from scipy.special import erf
from scipy.optimize import curve_fit


# =============================================================================
# Define variables
# =============================================================================
# TEMPLATE_FILE = 'templates/Template_s1d_GL699_sc1d_v_file_AB.fits'
# TEMPLATE_FILE = 'templates/Template_s1d_GL873_sc1d_v_file_AB.fits'
TEMPLATE_FILE = 'templates/Template_s1d_FP_sc1d_v_file_AB.fits'

# =============================================================================
# Define functions
# =============================================================================
def erf2(x, amp, scale):
    return erf(x/scale)*amp

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # for a given delta v, what is the appropriate feedback?
    template = Table.read(TEMPLATE_FILE)
    # get the wave and flux
    wave = template['wavelength']
    flux = template['flux']
    # get the gradient
    grad = flux - et.doppler_shift(wave, flux, 1)
    grad[np.abs(grad) > 10 * et.sigma(grad)] = np.nan
    # set up a table
    tbl = Table()
    tbl['DV'] = np.arange(-3000.0, 3000.0, 50.5)
    tbl['DV_RECOVERED'] = 0.0
    # loop around rows
    for i in tqdm(range(len(tbl['DV']))):
        residual = flux - et.doppler_shift(wave, flux, tbl['DV'][i])
        keep = np.isfinite(grad) * np.isfinite(residual)
        tbl['DV_RECOVERED'][i] = np.nanmean(grad[keep] * residual[keep]) / np.nanmean(grad[keep] ** 2)
    # fit the dv recovered
    p0 = [3000, 3000]
    fit, pcov = curve_fit(erf2, tbl['DV'], tbl['DV_RECOVERED'], p0=p0)
    print(fit)
    # plot
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(tbl['DV'], tbl['DV_RECOVERED'])
    ax[0].plot(tbl['DV'], tbl['DV'])
    ax[0].plot(tbl['DV'], erf2(tbl['DV'], *fit), 'g.')
    ax[0].set(xlabel='input velocity [m/s]', ylabel='output velocity [m/s]')
    ax[1].plot(tbl['DV'], np.gradient(tbl['DV_RECOVERED']) / np.gradient(tbl['DV']))
    ax[1].set(xlabel='input velocity [m/s]', ylabel='ratio of derivatives')
    plt.show()

# =============================================================================
# End of code
# =============================================================================
