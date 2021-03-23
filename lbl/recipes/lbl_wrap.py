#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2021-03-19

@author: cook
"""
from astropy.table import Table

from lbl import compute
from lbl import compil

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
        # run the compute code
        compute(instrument=instrument, data_dir=data_dir,
                object_science=object_science,
                object_template=object_template)
        # run the compile code
        compil(instrument=instrument, data_dir=data_dir,
               object_science=object_science,
               object_template=object_template)


# =============================================================================
# End of code
# =============================================================================
