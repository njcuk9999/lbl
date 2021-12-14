#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for CARMENES (TOI-1452)

Created on 2021-10-18

@author: artigau, cook
"""
from lbl import lbl_reset
from lbl import lbl_compil
from lbl import lbl_compute
from lbl import lbl_mask
from lbl import lbl_template
from lbl import lbl_telluclean

# =============================================================================
# Define variables
# =============================================================================
# define working directory
working = '/scratch3/lbl/data/carmenes/'
# create keyword argument dictionary
keyword_args = dict()
# add keyword arguments
keyword_args['INSTRUMENT'] = 'CARMENES'
keyword_args['DATA_DIR'] = working
keyword_args['DATA_TYPE'] = 'SCIENCE'
keyword_args['TEMPLATE_SUBDIR'] = 'templates'
keyword_args['BLAZE_FILE'] = None
keyword_args['TEMPLATE_FILE'] = 'Template_TOI-1452.fits'
keyword_args['PLOT'] = False
keyword_args['PLOT_COMPUTE_CCF'] = True
keyword_args['PLOT_COMPUTE_LINES'] = True
keyword_args['PLOT_COMPIL_CUMUL'] = True
keyword_args['PLOT_COMPIL_BINNED'] = True
keyword_args['SKIP_DONE'] = False
keyword_args['MASK_SUBDIR'] = 'masks'
keyword_args['INPUT_FILE'] = 'car-*.fits'
keyword_args['OVERWRITE'] = True
# add objects
objs = ['TOI-1452']
templates = ['TOI-1452']
teffs = [3248]
# set which object to run
num = 0

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run clean (reset everything)
    _ = lbl_reset(object_science=objs[num], object_template=templates[num],
                  **keyword_args)
    # run tellu-clean
    _ = lbl_telluclean(object_science=objs[num], object_template=templates[num],
                       telluclean_use_template=False, **keyword_args)
    # run template
    _ = lbl_template(object_science=objs[num], object_template=templates[num],
                     **keyword_args)
    # run tellu-clean
    _ = lbl_telluclean(object_science=objs[num], object_template=templates[num],
                       **keyword_args)
    # run template
    _ = lbl_template(object_science=objs[num], object_template=templates[num],
                     **keyword_args)
    # run template
    _ = lbl_template(object_science=objs[num], object_template=templates[num],
                     **keyword_args)
    # run mask code
    _ = lbl_mask(object_science=objs[num], object_template=templates[num],
                 object_teff=teffs[num], **keyword_args)
    # run compute
    _ = lbl_compute(object_science=objs[num], object_template=templates[num],
                    **keyword_args)
    # run compile
    _ = lbl_compil(object_science=objs[num], object_template=templates[num],
                   **keyword_args)

# =============================================================================
# End of code
# =============================================================================
