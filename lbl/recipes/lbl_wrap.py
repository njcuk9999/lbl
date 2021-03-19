#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-19

@author: cook
"""
from lbl import compute
from lbl import compil

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # set the config file
    config_file = '/data/spirou/bin/lbl/config.yaml'
    # run the compute code
    compute(config_file=config_file)
    # run the compil code
    compil(config_file=config_file)


# =============================================================================
# End of code
# =============================================================================
