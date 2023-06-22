#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-17

@author: cook
"""
import platform
from typing import Any, Dict, List

import matplotlib
import numpy as np
from astropy.table import Table

from lbl.core import base
from lbl.core import base_classes
from lbl.core import math as mp
from lbl.instruments import select

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.spirou.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
InstrumentsType = select.InstrumentsType
LblException = base_classes.LblException
# plt module
PLT_MOD = None


# =============================================================================
# Define standard functions
# =============================================================================
def import_matplotlib():
    """
    Load matplotlib in a way where everyone (in theory) can use it
    :return:
    """
    global PLT_MOD
    # deal with plt already set
    if PLT_MOD is not None:
        return PLT_MOD
    # fix for MacOSX plots freezing
    if platform.system() == 'darwin':
        gui_env = ['MacOSX', 'Qt5Agg', 'GTKAgg', 'TKAgg', 'WXAgg', 'Agg']
    else:
        gui_env = ['Qt5Agg', 'GTKAgg', 'TKAgg', 'WXAgg', 'Agg']
    for gui in gui_env:
        # noinspection PyBroadException
        try:
            matplotlib.use(gui, force=True)
            import matplotlib.pyplot as plt
            plt.show()
            plt.close()
            PLT_MOD = plt
            return plt
        except Exception as _:
            continue
    return None


# =============================================================================
# Define plot functions
# =============================================================================
# blank plot (for copying)
def plot_blank(inst: InstrumentsType, **kwargs):
    """
    This is a blank plot
    :param inst:  Instrument, instrument this plot is used for
    :param kwargs: replace with explicit keyword arguments

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # plot functions here
    _ = frame  # remove this
    _ = kwargs  # remove this
    # show and close plot
    plt.show()
    plt.close()


def compute_plot_ccf(inst: InstrumentsType, dvgrid: np.ndarray,
                     ccf_vector: np.ndarray, ccf_fit: np.ndarray,
                     gcoeffs: np.ndarray):
    """
    CCF Debug plot

    :param inst: Instrument, instrument this plot is used for
    :param dvgrid: np.ndarray, the rv velocity grid
    :param ccf_vector: np.ndarray, the ccf vector
    :param ccf_fit: np.ndarray, the ccf fit vector
    :param gcoeffs: np.ndarray the ccf fit coefficients

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # -------------------------------------------------------------------------
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # plot specific switch
    if not inst.params['PLOT_COMPUTE_CCF']:
        return
    # -------------------------------------------------------------------------
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # -------------------------------------------------------------------------
    # plot functions here
    frame.plot(-(dvgrid - inst.params['BERV']) / 1000, ccf_vector)
    frame.plot(-(dvgrid - inst.params['BERV']) / 1000, ccf_fit)
    # construct title
    targs = [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE'],
             -gcoeffs[0] / 1000, gcoeffs[1] / 1000,
             -(gcoeffs[0] - inst.params['BERV']) / 1000]
    title = ('CCF Plot\nOBJ_SCI={0} OBJ_TEMP={1}\nCCF: cent={2:.4f} km/s '
             'ewid={3:.4f} km/s\nCCF systemic velo={4:.4f} km/s')
    # set labels and title
    frame.set(xlabel='BERV-corrected RV [km/s]', ylabel='Normalized CCF',
              title=title.format(*targs))
    # -------------------------------------------------------------------------
    # show and close plot
    plt.grid(color='lightgrey', linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.close()



def compute_plot_sysvel(inst: InstrumentsType, dvgrid, ccf_vector, gcoeffs,
                        gfit, props):
    """
    This is a blank plot
    :param inst:  Instrument, instrument this plot is used for
    :param kwargs: replace with explicit keyword arguments

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # plot specific switch
    if not inst.params['PLOT_COMPUTE_SYSVEL']:
        return
    # set up plot
    fig, frames = plt.subplots(ncols=1, nrows=2, sharex='all')
    # plot functions here
    frames[0].plot(dvgrid / 1000, ccf_vector, label='CCF')
    frames[0].plot(dvgrid / 1000, gfit, color='r', linestyle='--',  label='fit')
    frames[0].legend(loc=0)
    frames[0].set(ylabel='CCF')
    frames[0].grid(color='grey', linestyle='--', linewidth=0.5)
    frames[1].plot(dvgrid / 1000, ccf_vector - gfit)
    frames[1].set(xlabel='dv [km/s]', ylabel='residuals')
    frames[1].grid(color='grey', linestyle='--', linewidth=0.5)
    # construct title
    targs = [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE'],
             props['VSYS'] / 1000, props['FWHM'] / 1000, props['SNR']]
    title = ('CCF Plot\nOBJ_SCI={0} OBJ_TEMP={1}\nCCF: Vsys={2:.4f} km/s '
             'fwhm={3:.4f} km/s\nCCF SNR={4:.4f}')
    plt.suptitle(title.format(*targs))
    # show and close plot
    plt.show()
    plt.close()


def compute_line_plot(inst: InstrumentsType, plot_dict: Dict[str, Any]):
    """
    Compute RV line plot

    :param inst: Instrument, instrument this plot is used for
    :param plot_dict: a dictionary of

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # -------------------------------------------------------------------------
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # plot specific switch
    if not inst.params['PLOT_COMPUTE_LINES']:
        return
    # -------------------------------------------------------------------------
    # extract values from plot dict
    wavegrid = plot_dict['WAVEGRID']
    model = plot_dict['MODEL']
    plot_orders = plot_dict['PLOT_ORDERS']
    line_orders = plot_dict['LINE_ORDERS']
    ww_ord_line = plot_dict['WW_ORD_LINE']
    spec_ord_line = plot_dict['SPEC_ORD_LINE']
    model_ord_line = plot_dict['MODEL_ORD_LINE']
    # deal with plot orders being an integer (we want a list)
    if isinstance(plot_orders, int):
        plot_orders = [plot_orders]
    # -------------------------------------------------------------------------
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=2, sharex='all')
    # -------------------------------------------------------------------------
    # storage for used labels
    used_labels = []
    # loop around orders
    for ord_num in plot_orders:
        # add template label (only if not present)
        label1 = 'template'
        if label1 in used_labels:
            label1 = None
        else:
            used_labels.append(label1)
        # plot the template
        frame[0].plot(wavegrid[ord_num], model[ord_num], color='grey', lw=3,
                      alpha=0.3, label=label1)
        # plot the lines
        for line_it in range(len(line_orders)):
            # get colour of line
            colour = ['red', 'green', 'blue'][line_it % 3]
            # only plot the orders we want
            if line_orders[line_it] == ord_num:
                # add template label (only if not present)
                label2 = 'line'
                if label2 in used_labels:
                    label2 = None
                else:
                    used_labels.append(label2)
                # plot the line
                frame[0].plot(ww_ord_line[line_it], spec_ord_line[line_it],
                              color=colour, label=label2)
                frame[1].plot(ww_ord_line[line_it],
                              spec_ord_line[line_it] - model_ord_line[line_it],
                              color=colour, label=label2)
    # construct title
    title = 'Line spectrum {0}\nOBJ_SCI={1} OBJ_TEMP={2}'
    if len(plot_orders) == 1:
        targs = ['order {0}'.format(plot_orders[0])]
    else:
        str_orders = np.array(plot_orders).astype(str)
        targs = ['orders {0}'.format(','.join(str_orders))]
    targs += [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE']]
    # set labels and title
    frame[0].set(ylabel='Arbitrary flux', title=title.format(*targs))
    frame[1].set(xlabel='Wavelength [nm]', ylabel='Residuals')

    frame[0].legend(loc=0)
    # -------------------------------------------------------------------------
    # show and close plot
    plt.show()
    plt.close()


def compil_cumulative_plot(inst: InstrumentsType, vrange: List[np.ndarray],
                           pdf: List[np.ndarray], pdf_fit: List[np.ndarray],
                           plot_path: str):
    """
    This is the probability density fucntion plot for all files

    :param inst:  Instrument, instrument this plot is used for
    :param vrange: list of np.ndarray, the velocities array [m/s] for each
                   file (for this object)
    :param pdf: list of np.ndarray, the probability density function array
                for each file (for this object)
    :param pdf_fit: list of np.ndarray, the fit to the probability density
                    function for each file (for this object)
    :param plot_path: str, the absolute path and filename to save the plot to

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # plot specific switch
    if not inst.params['PLOT_COMPIL_CUMUL']:
        return
    # -------------------------------------------------------------------------
    # set up plot
    fig, frames = plt.subplots(ncols=1, nrows=2)
    # loop around list elements
    for it in range(len(vrange)):
        # plot functions here
        frames[0].plot(vrange[it] / 1000.0, pdf[it],
                       alpha=0.2, color='grey')
        frames[1].plot(vrange[it] / 1000.0, pdf[it] - pdf_fit[it],
                       alpha=0.2, color='grey')
    # construct title
    title = 'OBJECT_SCIENCE = {0}    OBJECT_TEMPLATE_{1}'
    targs = [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE']]
    # set labels and titles
    frames[0].set(xlabel='Velocity [km/s]',
                  ylabel='Distribution function of RVs')
    frames[1].set(xlabel='Velocity [km/s]',
                  ylabel='Distribution function of RVs - gaussfit')
    # set title
    plt.suptitle(title.format(*targs))
    # show and close plot
    plt.savefig(plot_path)
    plt.close()


def compil_binned_band_plot(inst: InstrumentsType, rdb_table: Table):
    """
    This is the binned in bands plot (doesn't look at individual regions)

    :param inst:  Instrument, instrument this plot is used for
    :param rdb_table: astropy.table.Table - the rdb table

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # plot specific switch
    if not inst.params['PLOT_COMPIL_BINNED']:
        return
    # -------------------------------------------------------------------------
    # get the band names from params
    band1_name = inst.params['COMPILE_BINNED_BAND1']
    band2_name = inst.params['COMPILE_BINNED_BAND2']
    band3_name = inst.params['COMPILE_BINNED_BAND3']
    # get arrays from rdb table
    rjd = rdb_table['rjd']
    vrad = rdb_table['vrad']
    svrad = rdb_table['svrad']
    vrad_band1 = rdb_table['vrad_{0}'.format(band1_name)]
    svrad_band1 = rdb_table['svrad_{0}'.format(band1_name)]
    vrad_band2 = rdb_table['vrad_{0}'.format(band2_name)]
    svrad_band2 = rdb_table['svrad_{0}'.format(band2_name)]
    vrad_band3 = rdb_table['vrad_{0}'.format(band3_name)]
    svrad_band3 = rdb_table['svrad_{0}'.format(band3_name)]
    # get colour (band2 - band3)
    vrad_colour = vrad_band2 - vrad_band3
    svrad_colour = np.sqrt(svrad_band2 ** 2 + svrad_band3 ** 2)
    # normalize by median value
    nvrad_band1 = vrad_band1 - mp.nanmedian(vrad_band1)
    nvrad = vrad - mp.nanmedian(vrad)
    nvrad_colour = vrad_colour - mp.nanmedian(vrad_colour)
    # colour name
    colour_name = '{0} - {1}'.format(band2_name, band3_name)
    # -------------------------------------------------------------------------
    # set up plot
    fig, frames = plt.subplots(ncols=1, nrows=2, sharex='all')
    # -------------------------------------------------------------------------
    # plot functions here

    # plot band 1
    frames[0].errorbar(rjd, nvrad_band1, fmt='.r', yerr=svrad_band1, alpha=0.2,
                       label=band1_name)
    # plot all vrad
    frames[0].errorbar(rjd, nvrad, fmt='.k', yerr=svrad, alpha=0.8,
                       label='all')
    # plot colour
    frames[1].errorbar(rjd, nvrad_colour, fmt='.g', yerr=svrad_colour,
                       alpha=0.5, label=colour_name)
    # -------------------------------------------------------------------------
    # add legend
    frames[0].legend(loc=0)
    # construct axis titles
    title0 = '{0} velocity'.format(band1_name)
    title1 = '{0} velocity difference'.format(colour_name)
    # set labels
    frames[0].set(xlabel='rjd', ylabel='RV [m/s]', title=title0)
    frames[1].set(xlabel='rjd', ylabel='RV [m/s]', title=title1)
    # construct main title
    title = 'OBJECT_SCIENCE = {0}    OBJECT_TEMPLATE_{1}'
    targs = [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE']]
    # set super title
    plt.suptitle(title.format(*targs))
    # -------------------------------------------------------------------------
    # show and close plot
    plt.show()
    plt.close()


def mask_plot_ccf(inst: InstrumentsType, dvgrid: np.ndarray,
                  ccf_vector: np.ndarray, sys_vel: float):
    """
    CCF Debug plot

    :param inst: Instrument, instrument this plot is used for
    :param dvgrid: np.ndarray, the rv velocity grid
    :param ccf_vector: np.ndarray, the ccf vector
    :param sys_vel: float, the systemic velocity

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # -------------------------------------------------------------------------
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # plot specific switch
    if not inst.params['PLOT_MASK_CCF']:
        return
    # -------------------------------------------------------------------------
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # -------------------------------------------------------------------------
    # plot functions here
    frame.plot(dvgrid, ccf_vector)
    # construct title
    targs = [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE'],
             sys_vel]
    title = 'OBJ_SCI={0} OBJ_TEMP={1}\nSystem Velocity: {2:.4f} km/s '
    # set labels and title
    frame.set(xlabel='Velocity [km/s]', ylabel='CCF contrast',
              title=title.format(*targs))
    # -------------------------------------------------------------------------
    # show and close plot
    plt.show()
    plt.close()


def ccf_vector_plot(inst: InstrumentsType, ddvecs: Dict[int, np.ndarray],
                    ccf_waters: Dict[int, np.ndarray],
                    ccf_others: Dict[int, np.ndarray], objname: str):
    # import matplotlib
    plt = import_matplotlib()
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    if not inst.params['PLOT_CCF_VECTOR_PLOT']:
        return
    # set up plot
    fig, frame = plt.subplots(ncols=2, nrows=1, sharex='all')
    # plot functions here
    for iteration in ddvecs:
        # get this iterations values
        ddvec = ddvecs[iteration]
        ccf_water = ccf_waters[iteration]
        ccf_other = ccf_others[iteration]
        # plot ccf water and ccf others
        frame[0].plot(ddvec, ccf_water, alpha=0.5,
                      label='Iteration {0}'.format(iteration + 1))
        frame[1].plot(ddvec, ccf_other, alpha=0.5,
                      label='Iteration {0}'.format(iteration + 1))
    # set labels
    frame[0].set(xlabel='dv [km/s]', ylabel='ccf power', title='Water ccf')
    frame[1].set(xlabel='dv [km/s]', ylabel='ccf power', title='Dry ccf')
    # add legend
    frame[0].legend(loc=0)
    frame[1].legend(loc=0)
    # title
    plt.suptitle('Object = {0}'.format(objname))
    # show and close plot
    plt.show()
    plt.close()


def tellu_corr_plot(inst: InstrumentsType, wave_vector: np.ndarray,
                    sp_tmp: np.ndarray, trans: np.ndarray,
                    template_vector: np.ndarray,
                    template_flag: bool, objname: str):
    # import matplotlib
    plt = import_matplotlib()
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    if not inst.params['PLOT_TELLU_CORR_PLOT']:
        return
    # get parameters from inst
    trans_threshold = inst.params['TELLUCLEAN_TRANSMISSION_THRESHOLD']
    mask_domain_lower = inst.params['TELLUCLEAN_MASK_DOMAIN_LOWER']
    mask_domain_upper = inst.params['TELLUCLEAN_MASK_DOMAIN_UPPER']
    # normalize spectra
    med = np.nanmedian(sp_tmp)
    sp_tmp = sp_tmp / med
    template = template_vector / np.nanmedian(template_vector)
    # nan array with transmission lower than our limit

    mask = np.ones_like(trans)
    mask[trans < np.exp(trans_threshold)] = np.nan
    mask[~np.isfinite(sp_tmp)] = np.nan

    # mask wave based on domain limits
    keep = wave_vector > mask_domain_lower
    keep &= wave_vector < mask_domain_upper
    # calculate a scaling between sp and trans
    scale = np.nanpercentile((sp_tmp / trans * mask)[keep], 99.5)
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1, sharex='all')
    # plot functions here
    frame.plot(wave_vector, sp_tmp / scale, color='red', label='input')
    frame.plot(wave_vector, sp_tmp / (trans * mask * scale), color='green',
               label='input/abso', alpha=0.5)
    frame.plot(wave_vector, trans, color='orange', alpha=0.5, label='abso')
    # plot template if present
    if template_flag:
        frame.plot(wave_vector, sp_tmp / template, color='cyan', alpha=0.5,
                   label='sp/template')
    # add legend
    frame.legend(loc=0)
    frame.set(xlabel='Wavelength [nm]', ylabel='Normalized flux\n transmission',
              title='Object = {0}'.format(objname), ylim=[0, 1.1])
    # show and close plot
    plt.show()
    plt.close()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
