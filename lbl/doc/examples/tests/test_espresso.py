#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-10-18

@author: artigau
"""
from lbl import compil


# =============================================================================
# Define variables
# =============================================================================
# create keyword argument dictionary
keyword_args = dict()
# add keyword arguments
keyword_args['INSTRUMENT'] = 'ESPRESSO'
keyword_args['DATA_DIR'] = '/Volumes/courlan/ESPRESSO'
keyword_args['TEMPLATE_SUBDIR'] = 'templates'
keyword_args['BLAZE_FILE'] = 'dummy_blaze.fits'
keyword_args['TEMPLATE_FILE'] = 'Template_LHS-1140.fits'
keyword_args['PLOT'] = True
keyword_args['PLOT_COMPUTE_CCF'] = keyword_args['PLOT']
keyword_args['PLOT_COMPUTE_LINES'] = keyword_args['PLOT']
keyword_args['PLOT_COMPIL_CUMUL'] = keyword_args['PLOT']
keyword_args['PLOT_COMPIL_BINNED'] = keyword_args['PLOT']
keyword_args['SKIP_DONE'] = False
keyword_args['MASK_SUBDIR'] = '/Volumes/courlan/ESPRESSO/masks'
keyword_args['INPUT_FILE'] = '/Volumes/courlan/ESPRESSO/science/LHS-1140-tc/ES*.fits'
# add objects
objs = ['LHS-1140']
templates = ['LHS-1140']
# set which object to run
num = 0


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run compile
    dict_tbl = compil(object_science=objs[num], object_template=templates[num],
                      **keyword_args)

# =============================================================================
# End of code
# =============================================================================
