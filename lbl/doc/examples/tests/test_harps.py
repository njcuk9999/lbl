#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for HARPS (Proxima-tc)

Created on 2021-10-18

@author: artigau, cook
"""
from lbl import compil
from lbl import compute
from lbl import mask
from lbl import template

# =============================================================================
# Define variables
# =============================================================================
# define working directory
working = '/data/lbl/data/harps/'
# create keyword argument dictionary
keyword_args = dict()
# add keyword arguments
keyword_args['INSTRUMENT'] = 'HARPS'
keyword_args['DATA_DIR'] = working
keyword_args['TEMPLATE_SUBDIR'] = 'templates'
keyword_args['BLAZE_FILE'] = 'HARPS.2014-09-02T21_06_48.529_blaze_A.fits'
keyword_args['TEMPLATE_FILE'] = 'Template_Proxima-tc_HARPS.fits'
keyword_args['PLOT'] = False
keyword_args['PLOT_COMPUTE_CCF'] = keyword_args['PLOT']
keyword_args['PLOT_COMPUTE_LINES'] = keyword_args['PLOT']
keyword_args['PLOT_COMPIL_CUMUL'] = keyword_args['PLOT']
keyword_args['PLOT_COMPIL_BINNED'] = keyword_args['PLOT']
keyword_args['SKIP_DONE'] = False
keyword_args['MASK_SUBDIR'] = working + 'masks'
keyword_args['INPUT_FILE'] = working + 'science/Proxima-tc/HARPS*_e2ds_A.fits'
keyword_args['OVERWRITE'] = True
# add objects
objs = ['Proxima-tc']
templates = ['Proxima-tc']
teffs = [3042]
# set which object to run
num = 0


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

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
