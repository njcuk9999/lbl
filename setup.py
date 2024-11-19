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

def load_requirements() -> list:
    """
    Load requirements from file
    :return:
    """
    requirements = 'requirements.txt'
    # storage for list of modules
    modules = []
    # open requirements file
    with open(requirements, 'r') as rfile:
        lines = rfile.readlines()
    # get modules from lines in requirements file
    for line in lines:
        if len(line) == '':
            continue
        if line.startswith('#'):
            continue
        else:
            modules.append(line)
    # return modules
    return modules


setup(version=get_version(),
      include_package_data=True,
      install_requires=load_requirements())
