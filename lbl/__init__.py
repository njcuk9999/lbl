#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
from lbl.core import base
from lbl import recipes

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl'
__STRNAME__ = 'LBL'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# clean recipe
lbl_reset = recipes.lbl_reset.main
# tellu-clean recipe
lbl_telluclean = recipes.lbl_telluclean.main
# alias to lbl template
lbl_template = recipes.lbl_template.main
# alias to lbl mask
lbl_mask = recipes.lbl_mask.main
# alias to lbl noise
lbl_noise = recipes.lbl_noise.main
# compute recipe
lbl_compute = recipes.lbl_compute.main
# compile recipe
lbl_compil = recipes.lbl_compile.main
# wrapper code
lbl_wrap = recipes.lbl_wrap.main


# =============================================================================
# Define functions
# =============================================================================


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
