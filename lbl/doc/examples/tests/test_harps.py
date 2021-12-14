#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for HARPS (Proxima-tc)

Created on 2021-10-18

@author: artigau, cook
"""
from lbl import clean
from lbl import compil
from lbl import compute
from lbl import mask
from lbl import template
from lbl import preclean

# =============================================================================
# Define variables
# =============================================================================
# define working directory
working = '/scratch3/lbl/data/harps/'
# create keyword argument dictionary
keyword_args = dict()
# add keyword arguments
keyword_args['INSTRUMENT'] = 'HARPS'
keyword_args['DATA_DIR'] = working
keyword_args['DATA_TYPE'] = 'SCIENCE'
keyword_args['TEMPLATE_SUBDIR'] = 'templates'
keyword_args['BLAZE_FILE'] = 'HARPS.2014-09-02T21_06_48.529_blaze_A.fits'
keyword_args['TEMPLATE_FILE'] = 'Template_Proxima-tc_HARPS.fits'
keyword_args['PLOT'] = False
keyword_args['PLOT_COMPUTE_CCF'] = True
keyword_args['PLOT_COMPUTE_LINES'] = True
keyword_args['PLOT_COMPIL_CUMUL'] = True
keyword_args['PLOT_COMPIL_BINNED'] = True
keyword_args['SKIP_DONE'] = False
keyword_args['MASK_SUBDIR'] = 'masks'
keyword_args['INPUT_FILE'] = 'HARPS*_e2ds_A.fits'
keyword_args['OVERWRITE'] = True
# add objects
objs = ['Proxima']
templates = ['Proxima']
teffs = [3042]
# set which object to run
num = 0


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run clean (reset everything)
    _ = clean(object_science=objs[num], object_template=templates[num],
                    **keyword_args)
    # run pre-clean
    _ = preclean(object_science=objs[num], object_template=templates[num],
                 preclean_use_template=False, **keyword_args)
    # run template
    _ = template(object_science=objs[num], object_template=templates[num],
                 **keyword_args)
    # run pre-clean
    _ = preclean(object_science=objs[num], object_template=templates[num],
                 **keyword_args)
    # run template
    _ = template(object_science=objs[num], object_template=templates[num],
                 **keyword_args)
    # run mask code
    _ = mask(object_science=objs[num], object_template=templates[num],
             object_teff=teffs[num], **keyword_args)
    # run compute
    _ = compute(object_science=objs[num], object_template=templates[num],
                **keyword_args)
    # run compile
    _ = compil(object_science=objs[num], object_template=templates[num],
               **keyword_args)

# =============================================================================
# End of code
# =============================================================================
