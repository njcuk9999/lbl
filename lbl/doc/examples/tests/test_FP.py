#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-10-18

@author: artigau
"""
from lbl import compil
from astropy.table import Table


# =============================================================================
# Define variables
# =============================================================================
# create keyword argument dictionary
keyword_args = dict()
# add keyword arguments
keyword_args['INSTRUMENT'] = 'SPIRou'
keyword_args['DATA_DIR'] = '/Users/eartigau/lbl/'
keyword_args['INPUT_FILE'] = '2*o_pp_e2dsff_tcorr_AB.fits'
keyword_args['BLAZE_FILE'] = '2498F798T802f_pp_blaze_AB.fits'
keyword_args['SKIP_DONE'] = False
# add objects
objs = ['GL699']
templates = ['GL699']
# set which object to run
num = 0

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run compile
    dict_tbl = compil(object_science=objs[num], object_template=templates[num],
                      **keyword_args)
    # write dictionary to csv file
    tbl =  Table(dict_tbl['rdb_table'])
    tbl.write('tmp.csv',overwrite = True)

# =============================================================================
# End of code
# =============================================================================
