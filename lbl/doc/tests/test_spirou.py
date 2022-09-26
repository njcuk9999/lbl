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

# =============================================================================
# Define variables
# =============================================================================
# define working directory
working = '/scratch3/lbl/bin/lbl/lbl/doc/examples/tests/'
# create keyword argument dictionary
fp_config_file = working + 'spirou_fp_config.yaml'
gl699_config_file = working + 'spirou_gl699_config.yaml'

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run clean (reset everything)
    #_ = lbl_reset(config_file=fp_config_file)
    _ = lbl_reset(config_file=gl699_config_file)
    # run template
    #_ = lbl_template(config_file=fp_config_file)
    _ = lbl_template(config_file=gl699_config_file)
    # run mask code
    _ = lbl_mask(config_file=fp_config_file)
    _ = lbl_mask(config_file=gl699_config_file)
    # run compute
    _ = lbl_compute(config_file=fp_config_file)
    _ = lbl_compute(config_file=gl699_config_file)
    # run compile
    _ = lbl_compil(config_file=fp_config_file)
    _ = lbl_compil(config_file=gl699_config_file)

# =============================================================================
# End of code
# =============================================================================
