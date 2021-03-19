#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-17

@author: cook
"""
import matplotlib
import numpy as np
from typing import Any, Dict

from lbl.core import base
from lbl.core import base_classes
from lbl.instruments import default

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'instruments.spirou.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
Instrument = default.Instrument
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
    gui_env = ['Qt5Agg', 'GTKAgg', 'TKAgg', 'WXAgg', 'Agg']
    for gui in gui_env:
        # noinspection PyBroadException
        try:
            matplotlib.use(gui, force=True)
            import matplotlib.pyplot as plt
            PLT_MOD = plt
            return plt
        except Exception as _:
            continue
    return None


# =============================================================================
# Define plot functions
# =============================================================================
# blank plot (for copying)
def plot_blank(inst: Instrument, **kwargs):
    """
    This is a blank plot
    :param inst:  Instrument, instrument this plot is used for
    :param kwargs: replace with explicit keyword arguments

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # this is a debug plot skip is this is True
    # TODO: This line is only required if debug plot
    if not inst.params['DEBUG_PLOT']:
        return
    # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # plot functions here
    _ = frame   # remove this
    _ = kwargs  # remove this
    # show and close plot
    plt.show()
    plt.close()


def plot_ccf(inst: Instrument, dvgrid: np.ndarray,
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
    # this is a debug plot skip is this is True
    if not inst.params['DEBUG_PLOT']:
        return
        # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # -------------------------------------------------------------------------
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
    # -------------------------------------------------------------------------
    # plot functions here
    frame.plot(-dvgrid / 1000, ccf_vector)
    frame.plot(-dvgrid / 1000, ccf_fit)
    # construct title
    targs = [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE'],
             gcoeffs[0] / 1000, gcoeffs[1] / 1000]
    title = ('CCF Plot\nOBJ_SCI={0} OBJ_TEMP={1}\nCCF: cent={2:.4f} km/s '
             'ewid={3:.4f} km/s')
    # set labels and title
    frame.set(xlabel='RV [km/s]', ylabel='Normalized CCF',
              title=title.format(*targs))
    # -------------------------------------------------------------------------
    # show and close plot
    plt.show()
    plt.close()


def line_plot(inst: Instrument, plot_dict: Dict[str, Any]):
    """
    Compute RV line plot

    :param inst: Instrument, instrument this plot is used for
    :param plot_dict: a dictionary of

    :return: None - plots
    """
    # import matplotlib
    plt = import_matplotlib()
    # -------------------------------------------------------------------------
    # this is a debug plot skip is this is True
    if not inst.params['DEBUG_PLOT']:
        return
        # this is a plot skip if this is True
    if not inst.params['PLOT']:
        return
    # -------------------------------------------------------------------------
    # extract values from plot dict
    wavegrid = plot_dict['WAVEGRID']
    model = plot_dict['MODEL']
    plot_orders = plot_dict['PLOT_ORDERS']
    line_orders = plot_dict['LINE_ORDERS']
    ww_ord_line = plot_dict['WW_ORD_LINE']
    spec_ord_line = plot_dict['SPEC_ORD_LINE']
    # deal with plot orders being an integer (we want a list)
    if isinstance(plot_orders, int):
        plot_orders = [plot_orders]
    # -------------------------------------------------------------------------
    # set up plot
    fig, frame = plt.subplots(ncols=1, nrows=1)
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
        frame.plot(wavegrid[ord_num], model[ord_num], color='grey', lw=3,
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
                frame.plot(ww_ord_line[line_it], spec_ord_line[line_it],
                           color=colour, label=label2)
    # construct title
    title = ('Line spectrum {0}\nOBJ_SCI={1} OBJ_TEMP={2}')
    if len(plot_orders) == 1:
        targs = ['order {0}'.format(plot_orders[0])]
    else:
        str_orders = np.array(plot_orders).astype(str)
        targs = ['orders {0}'.format(','.join(str_orders))]
    targs += [inst.params['OBJECT_SCIENCE'], inst.params['OBJECT_TEMPLATE']]
    # set labels and title
    frame.set(xlabel='Wavelength [nm]', ylabel='Arbitrary flux',
              title=title.format(*targs))
    frame.legend(loc=0)
    # -------------------------------------------------------------------------
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
