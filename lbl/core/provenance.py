#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What an LBL product was made from, as a token: a short hash of its input
files, kept beside the product in <product>.inputs (a small JSON file).

A step that finds its product on disk keeps it only when the token of its
inputs is the one recorded; otherwise it makes the product again. Adding
spectra to an object thus makes again its template, its mask, the velocities
of its spectra and its tables, down to the report, instead of keeping a
product made from fewer spectra.

Input files are described by name, size and date (fast, for many files), or
by their content (a template, a mask: rewriting one with the same content
then changes nothing downstream).

@author: artigau
"""
import datetime
import hashlib
import json
import os
from typing import Iterable, List, Optional

# =============================================================================
# Define variables
# =============================================================================
#: what is appended to a product's name for the file of its token
SUFFIX = '.inputs'


# =============================================================================
# Define functions
# =============================================================================
def describe(paths: Iterable[str], content: bool = False) -> List[list]:
    """
    The input files as the token sees them, in the order of their names

    :param paths: the input files
    :param content: bool, True: name and checksum of the content; False:
                    name, size and date

    :return: list of [name, checksum] or [name, size, date]
    """
    out = []
    for path in sorted(map(str, paths), key=os.path.basename):
        name = os.path.basename(path)
        # an input that is not a file on disk (an instrument may hand over
        #   spectra it reads otherwise) is described by its name
        if not os.path.isfile(path):
            out.append([name])
        elif content:
            from lbl.core import io
            out.append([name, io.generate_checksum(path)])
        else:
            stat = os.stat(path)
            out.append([name, stat.st_size, int(stat.st_mtime)])
    return out


def token(paths: Iterable[str], content: bool = False,
          extra=None) -> str:
    """
    The token of a product's inputs

    :param paths: the input files
    :param content: bool, describe the files by their content (describe)
    :param extra: anything else the product depends on (JSON-able)

    :return: str, 16 hexadecimal characters
    """
    blob = dict(files=describe(paths, content), extra=extra)
    text = json.dumps(blob, sort_keys=True, default=str).encode()
    return hashlib.sha1(text).hexdigest()[:16]


def sidecar(product: str) -> str:
    """The file that holds a product's token"""
    return str(product) + SUFFIX


def recorded(product: str) -> Optional[str]:
    """The token recorded beside a product, or None"""
    try:
        with open(sidecar(product)) as handle:
            return json.load(handle).get('token')
    except (OSError, ValueError, AttributeError):
        return None


def record(product: str, value: str, n_inputs: Optional[int] = None):
    """Record a product's token beside it, once the product is written"""
    with open(sidecar(product), 'w') as handle:
        json.dump(dict(token=value, inputs=n_inputs,
                       written=datetime.datetime.now().isoformat(
                           timespec='seconds')), handle, indent=1)


def is_current(product: str, value: str) -> bool:
    """Whether a product is on disk with this token beside it"""
    return os.path.exists(product) and recorded(product) == value


def older_than(product: str, paths: Iterable[str]) -> bool:
    """Whether a product is older than one of its inputs: the test for a
    product made before tokens existed, which has none beside it"""
    stamp = os.path.getmtime(product)
    return any(os.path.getmtime(path) > stamp for path in map(str, paths)
               if os.path.isfile(path))


# =============================================================================
# End of code
# =============================================================================
