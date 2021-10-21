#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-10-18

@author: artigau
"""
from lbl import compil
from lbl import compute

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
    # run compute
    #tbl1a = compute(config_file=fp_config_file)
    tbl1b = compute(config_file=gl699_config_file)
    # run compile
    #tbl2a = compil(config_file=fp_config_file)
    tbl2b = compil(config_file=gl699_config_file)

# =============================================================================
# End of code
# =============================================================================
