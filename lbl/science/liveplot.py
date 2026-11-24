#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The plots a run shows as it goes, without stopping it.

A run that a front end watches (a window that starts runs, a package apart
from LBL) is given the folder of its plots in LBL_LIVE_PLOTS (ENV): every
plot of LBL's steps is then saved there as a PDF, listed in index.json
(INDEX: file, title, step, object, time), and closed, instead of being
shown in a window that would wait to be closed; the front end shows them
between the steps and the log as they come. A plot
that LBL draws for every file (a CCF per spectrum, its lines) is saved for
the FIRST file of each step and object only (claim). Without the variable
nothing changes: the plots are shown as before, when PLOT asks for them.

Created on 2026-11-18

@author: artigau
"""
import datetime
import json
import os
import re
from typing import Optional

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'lbl.science.liveplot.py'
#: the variable that names the folder of a run's plots
ENV = 'LBL_LIVE_PLOTS'
#: the list of the plots saved, in that folder
INDEX = 'index.json'


# =============================================================================
# Define functions
# =============================================================================
def folder() -> Optional[str]:
    """The folder of the run's plots, None when nobody watches"""
    return os.environ.get(ENV) or None


def active() -> bool:
    """Whether the plots go to a watching window rather than to the screen"""
    return folder() is not None


def _slug(text: str) -> str:
    return re.sub(r'[^0-9A-Za-z._+-]+', '_', str(text)).strip('_')[:80]


def claim(key: str) -> bool:
    """
    Whether this plot is the first of its key in the run: True once, False
    afterwards, across the run's processes (a lock file made exclusively),
    so that a plot drawn per file is made for the first file only, and not
    drawn at all for the others

    :param key: str, the plot (its step and object, e.g. compute_ccf_GJ1)

    :return: bool
    """
    where = folder()
    if where is None:
        return True
    os.makedirs(where, exist_ok=True)
    try:
        handle = os.open(os.path.join(where, '.%s.claim' % _slug(key)),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(handle)
    return True


def save(fig, key: str, title: str, step: str = '', obj: str = '') -> str:
    """
    The figure saved in the run's plot folder (PDF, vector) and listed in
    its index, then closed

    :param fig: matplotlib Figure
    :param key: str, what the plot is, unique in the run (names the file)
    :param title: str, what the window calls it
    :param step: str, the step that drew it (template, compute...)
    :param obj: str, the object

    :return: str, the PDF's path
    """
    where = folder()
    os.makedirs(where, exist_ok=True)
    listed = read_index(where)
    name = '%02d_%s.pdf' % (len(listed) + 1, _slug(key))
    path = os.path.join(where, name)
    fig.savefig(path)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:                                         # noqa: BLE001
        pass
    listed.append(dict(file=name, title=str(title), step=str(step),
                       object=str(obj), key=str(key),
                       time=datetime.datetime.now().isoformat(
                           timespec='seconds')))
    partial = os.path.join(where, INDEX + '.part')
    with open(partial, 'w', encoding='utf-8') as handle:
        json.dump(listed, handle, indent=1)
    os.replace(partial, os.path.join(where, INDEX))
    return path


def read_index(where: Optional[str] = None) -> list:
    """The plots of a run's plot folder, in the order they came; [] when
    there are none"""
    where = where or folder()
    if not where:
        return []
    try:
        with open(os.path.join(where, INDEX), encoding='utf-8') as handle:
            return list(json.load(handle))
    except (OSError, ValueError):
        return []


def finish(plt, key: str, title: str, step: str = '', obj: str = ''):
    """
    The end of one of LBL's plots: saved for the window when a run is
    watched, shown on the screen otherwise (as LBL always did)

    :param plt: matplotlib.pyplot
    :param key, title, step, obj: as save()
    """
    if active():
        save(plt.gcf(), key, title, step, obj)
    else:
        plt.show()
