#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for ESPERESSO (LHS-1140)

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
working = '/scratch3/lbl/data/espresso/'
# create keyword argument dictionary
keyword_args = dict()
# add keyword arguments
keyword_args['INSTRUMENT'] = 'ESPRESSO'
keyword_args['DATA_DIR'] = working
keyword_args['DATA_TYPE'] = 'SCIENCE'
keyword_args['INPUT_FILE'] = 'ES*.fits'
keyword_args['PLOT'] = False
# add objects
objs = ['LHS-1140']
templates = ['LHS-1140']
teffs = [3216]
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
                       telluclean_use_template=False,
                       **keyword_args)
    # run template
    _ = lbl_template(object_science=objs[num]+'_tc',
                     object_template=templates[num]+'_tc',
                     **keyword_args)
    # run tellu-clean
    _ = lbl_telluclean(object_science=objs[num],
                       object_template=templates[num]+'_tc',
                       skip_done=False, **keyword_args)
    # run template
    _ = lbl_template(object_science=objs[num]+'_tc',
                     object_template=templates[num]+'_tc',
                     **keyword_args)
    # run mask code
    _ = lbl_mask(object_science=objs[num]+'_tc',
                 object_template=templates[num]+'_tc',
                 object_teff=teffs[num], **keyword_args)
    # run compute
    _ = lbl_compute(object_science=objs[num]+'_tc',
                    object_template=templates[num]+'_tc',
                    **keyword_args)
    # run compile
    _ = lbl_compil(object_science=objs[num]+'_tc',
                   object_template=templates[num]+'_tc',
                   **keyword_args)

# =============================================================================
# End of code
# =============================================================================
