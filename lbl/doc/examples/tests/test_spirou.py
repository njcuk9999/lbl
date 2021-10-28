#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for Spirou (FP + Gl699) + drift correction

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
working = '/data/lbl/bin/lbl/lbl/doc/examples/'
# create keyword argument dictionary
fp_config_file = working + 'spirou_fp_config.yaml'
gl699_config_file = working + 'spirou_gl699_config.yaml'


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # run template
    tbl0a = template(config_file=fp_config_file)
    tbl0b = template(config_file=gl699_config_file)
    # run mask code
    tbl1a = mask(config_file=fp_config_file)
    tbl1b = mask(config_file=gl699_config_file)
    # run compute
    tbl2a = compute(config_file=fp_config_file)
    tbl2b = compute(config_file=gl699_config_file)
    # run compile
    tbl3a = compil(config_file=fp_config_file)
    tbl3b = compil(config_file=gl699_config_file)

# =============================================================================
# End of code
# =============================================================================
