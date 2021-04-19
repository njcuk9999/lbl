#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-04-19

@author: cook
"""
import lbl

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":

    # set parameters
    config_file = '/data/spirou/bin/lbl/config.yaml'
    w = 100

    dict_tbl = lbl.compil(config_file=config_file,
                          object_science='FP',
                          object_template='FP')

# =============================================================================
# End of code
# =============================================================================
