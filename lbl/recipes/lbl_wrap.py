#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-19

@author: cook
"""
from astropy.table import Table

from lbl import lbl_compute
from lbl import lbl_compil
from lbl import lbl_template
from lbl import lbl_mask
from lbl import lbl_noise
from lbl import lbl_telluclean

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    # set the data directory
    data_dir = '/data/spirou/data/lbl/'
    # set the instrument
    instrument = 'SPIROU'
    # load a csv file to control objects in loop
    todo_table = Table.read('todo.csv', format='csv')
    # loop around objects
    for row in range(len(todo_table)):
        # get the object science name and object template name
        object_science = todo_table['OBJ_SCI'][row]
        object_template = todo_table['OBJ_TEMPLATE'][row]

        if instrument != 'SPIROU':
            # run telluric cleaning (without template)
            lbl_telluclean(instrument=instrument, data_dir=data_dir,
                           object_science=object_science,
                           object_template=object_template,
                           telluclean_use_template=False)
            # make the template (if not present)
            lbl_template(instrument=instrument, data_dir=data_dir,
                         object_science=object_science,
                         object_template=object_template,
                         overwrite=True)
            lbl_telluclean(instrument=instrument, data_dir=data_dir,
                           object_science=object_science,
                           object_template=object_template,
                           telluclean_use_template=True)
        # make the template (if not present)
        lbl_template(instrument=instrument, data_dir=data_dir,
                     object_science=object_science,
                     object_template=object_template, overwrite=True)
        # make the mask (if not present)
        lbl_mask(instrument=instrument, data_dir=data_dir,
                 object_science=object_science,
                 object_template=object_template)
        # make the noise model (if not present)
        lbl_noise(instrument=instrument, data_dir=data_dir,
                  object_science=object_science,
                  object_template=object_template)
        # run the compute code
        lbl_compute(instrument=instrument, data_dir=data_dir,
                    object_science=object_science,
                    object_template=object_template)
        # run the compile code
        lbl_compil(instrument=instrument, data_dir=data_dir,
                   object_science=object_science,
                   object_template=object_template)

# =============================================================================
# End of code
# =============================================================================
