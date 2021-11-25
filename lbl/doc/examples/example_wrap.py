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

# =============================================================================
# Define variables
# =============================================================================
# create keyword argument dictionary
keyword_args = dict()
# add keyword arguments
keyword_args['INSTRUMENT'] = 'SPIROU'
keyword_args['DATA_DIR'] = '/scratch3/lbl/data/spirou/'
keyword_args['BLAZE_FILE'] = None
keyword_args['PLOT'] = False
keyword_args['SKIP_DONE'] = True
keyword_args['OVERWRITE'] = True
# whether to clean directories (and start from an empty directory)
CLEAN = True
# add iteration variables
objs = ['FP', 'GJ1002','GJ1286','GJ1289','GL15A','GL411','GL412A','GL687',
        'GL699', 'GL905']
templates = ['FP', 'GJ1002','GJ1286','GJ1289','GL15A','GL411','GL412A','GL687',
             'GL699','GL905']
teffs = [300, 2900,2900,3250,3603,3550,3549,3420,3224,2930]
data_types = ['FP', 'SCIENCE', 'SCIENCE', 'SCIENCE', 'SCIENCE', 'SCIENCE',
              'SCIENCE', 'SCIENCE', 'SCIENCE', 'SCIENCE']



# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # clean (if used) should be done before running anything else
    if CLEAN:
        for num in range(len(objs)):
            _ = clean(object_science=objs[num],
                      object_template=templates[num],
                      **keyword_args)
    # loop around objects
    for num in range(len(objs)):
        # run template
        tbl0 = template(object_science=objs[num],
                        object_template=templates[num],
                        data_type=data_types[num],
                        **keyword_args)
        # run mask code
        tbl1 = mask(object_science=objs[num],
                    object_template=templates[num],
                    object_teff=teffs[num],
                    data_type=data_types[num],
                    **keyword_args)
        # run compute
        tbl2 = compute(object_science=objs[num],
                       object_template=templates[num],
                       data_type=data_types[num],
                       **keyword_args)
        # run compile
        tbl3 = compil(object_science=objs[num],
                      object_template=templates[num],
                      **keyword_args)

# =============================================================================
# End of code
# =============================================================================
