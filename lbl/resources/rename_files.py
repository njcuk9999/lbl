"""
Remove non ascii characters from filenames
"""
import sys
import os
import shutil

from lbl.core import base
from lbl.core import base_classes


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'resources.misc.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get classes
log = base_classes.log

# =============================================================================
# Define functions
# =============================================================================
def main():
    # get path to use from sys.argv
    if len(sys.argv) == 1:
        path = os.getcwd()
    else:
        path = sys.argv[1]
    # ----------------------------------------------------------------------
    # walk around the directory and find all files
    for root, dirs, files in os.walk(path):
        # loop around files
        for filename in files:
            # get the full path
            fullpath = os.path.join(root, filename)
            # get the new filename
            newfilename = fullpath.encode('ascii', 'ignore').decode('ascii')
            # if the new filename is different
            if newfilename != fullpath:
                # print the change
                print('Changing: {0} to {1}'.format(fullpath, newfilename))
                # move the file
                shutil.move(fullpath, newfilename)

# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    main()