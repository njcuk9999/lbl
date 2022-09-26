#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for Spirou (FP + Gl699) + drift correction

Created on 2021-10-18

@author: artigau, cook
"""
from lbl import lbl_reset
from lbl import lbl_compil
from lbl import lbl_compute
from lbl import lbl_mask
from lbl import lbl_template
from lbl.recipes import lbl_find


# =============================================================================
# Define variables
# =============================================================================
# define working directory
working = '/scratch3/lbl/bin/lbl/lbl/doc/examples/'
# create keyword argument dictionary
fp_config_file = working + 'spirou_fp_config_06132.yaml'
gl699_config_file = working + 'spirou_gl699_config_06132.yaml'
# whether to copy files
FIND = False

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # # run clean (reset everything)
    _ = lbl_reset(config_file=fp_config_file)
    _ = lbl_reset(config_file=gl699_config_file)
    # # find files
    if FIND:
        _ = lbl_find.main(instrument='SPIROU', config_file=gl699_config_file)
        _ = lbl_find.main(instrument='SPIROU', config_file=fp_config_file)
    # # run template
    tbl0a = lbl_template(config_file=fp_config_file)
    tbl0b = lbl_template(config_file=gl699_config_file)
    # # run mask code
    tbl1a = lbl_mask(config_file=fp_config_file)
    tbl1b = lbl_mask(config_file=gl699_config_file)
    # run compute
    tbl2a = lbl_compute(config_file=fp_config_file)
    tbl2b = lbl_compute(config_file=gl699_config_file)
    # run compile
    tbl3a = lbl_compil(config_file=fp_config_file)
    tbl3b = lbl_compil(config_file=gl699_config_file)

# =============================================================================
# End of code
# =============================================================================
