#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test of the LBL for Spirou (FP + Gl699) + drift correction

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
working = '/scratch3/lbl/bin/lbl/lbl/doc/examples/'
# create keyword argument dictionary
fp_config_file = working + 'spirou_fp_config.yaml'
gl699_config_file = working + 'spirou_gl699_config.yaml'


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # run clean (reset everything)
    _ = clean(config_file=fp_config_file)
    _ = clean(config_file=gl699_config_file)
    # run template
    _ = template(config_file=fp_config_file)
    _ = template(config_file=gl699_config_file)
    # run mask code
    _ = mask(config_file=fp_config_file)
    _ = mask(config_file=gl699_config_file)
    # run compute
    _ = compute(config_file=fp_config_file)
    _ = compute(config_file=gl699_config_file)
    # run compile
    _ = compil(config_file=fp_config_file)
    _ = compil(config_file=gl699_config_file)

# =============================================================================
# End of code
# =============================================================================
