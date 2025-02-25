import os
import shutil
import sys

from setuptools import setup

def get_version() -> str:
    """
    Get the version from the version file
    :return:
    """
    # try to open version file
    try:
        with open('lbl/version.txt', 'r') as vfile:
            vtext = vfile.readlines()
    except Exception as e:
        print('Error: Could not read version file')
        print('Error: {0}'.format(e))
        sys.exit(1)
    # return version
    return vtext[0]

setup(version=get_version())
