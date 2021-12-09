#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-15

@author: cook
"""
from lbl.recipes import lbl_clean
from lbl.recipes import lbl_compute
from lbl.recipes import lbl_compile
from lbl.recipes import lbl_template
from lbl.recipes import lbl_mask
from lbl.recipes import lbl_noise
from lbl.recipes import lbl_preclean

# =============================================================================
# Define variables
# =============================================================================
# alias to lbl clean
clean = lbl_clean
# alias to lbl template
template = lbl_template
# alias to lbl mask
mask = lbl_mask
# alias to lbl noise
noise = lbl_noise
# alias to lbl compute
compute = lbl_compute
# alias to lbl compile
compil = lbl_compile
# alias to lbl compile
preclean = lbl_preclean

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # print hello world
    print('Hello World')

# =============================================================================
# End of code
# =============================================================================
