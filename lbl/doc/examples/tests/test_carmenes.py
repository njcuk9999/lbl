#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for CARMENES (TOI-1452)

Created on 2021-10-18

@author: artigau, cook
"""
from lbl import clean
from lbl import compil
from lbl import compute
from lbl import mask
from lbl import template

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
keyword_args['TEMPLATE_SUBDIR'] = 'templates'
keyword_args['BLAZE_FILE'] = None
keyword_args['TEMPLATE_FILE'] = 'Template_TOI-1452.fits'
keyword_args['PLOT'] = False
keyword_args['PLOT_COMPUTE_CCF'] = True
keyword_args['PLOT_COMPUTE_LINES'] = True
keyword_args['PLOT_COMPIL_CUMUL'] = True
keyword_args['PLOT_COMPIL_BINNED'] = True
keyword_args['SKIP_DONE'] = False
keyword_args['MASK_SUBDIR'] = working + 'masks'
keyword_args['INPUT_FILE'] = working + 'science/TOI-1452-tc/car-*.fits'
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
    _ = clean(object_science=objs[num], object_template=templates[num],
                    **keyword_args)
    # run template
    tbl0 = template(object_science=objs[num], object_template=templates[num],
                    **keyword_args)
    # run mask code
    tbl1 = mask(object_science=objs[num], object_template=templates[num],
                object_teff=teffs[num], **keyword_args)
    # run compute
    tbl2 = compute(object_science=objs[num], object_template=templates[num],
                   **keyword_args)
    # run compile
    tbl3 = compil(object_science=objs[num], object_template=templates[num],
                  **keyword_args)


# =============================================================================
# End of code
# =============================================================================
