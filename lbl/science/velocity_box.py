#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The velocity of each spectrum of a star, kept in a small box: one JSON file
per object, DATA_DIR/velocities/<object>.json, made the first time a step
needs a starting point and brought up to date as the measurements improve.

A spectrum's entry, under its file name:

    v      the star's velocity in that spectrum [m/s], the BERV taken out
           (relativistic composition): what the template registers the
           spectrum at (template_scatter.star_shift)
    err    its uncertainty [m/s] (NaN for a CCF)
    by     what measured it: 'ccf' (against the PHOENIX model at the star's
           Teff, a few hundred m/s), 'template' (the template's Bouchy
           passes), 'lbl' (LBL's own velocity)
    mjd    the spectrum's time, berv its BERV [m/s], when the date of the
           measurement

and the box's 'systemic': the median of the CCF velocities, a clean
systemic velocity of the star against the model.

An entry is replaced by a measurement of a higher rank (ccf < template <
lbl) or by a newer one of the same rank, never by one of a lower rank. The
corrected spectra of PCA2D (<object>_pca2d) are the same exposures and share
their object's box; an injection test (<object>_inj) has its own, its
velocities carrying the injected signals.

The file is written whole (a temporary file renamed over it), under a lock
where the system offers one, so that two runs writing at once both keep
their entries.

@author: artigau
"""
import datetime
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np

from lbl.core import math as mp

#: the folder of the boxes, under DATA_DIR
FOLDER = 'velocities'
#: the rank of each kind of measurement: a higher one replaces a lower one
RANK = dict(ccf=0, template=1, lbl=2)
#: the suffix of PCA2D's corrected spectra (lbl.recipes.lbl_pca2d.SUFFIX):
#:   their object shares the box of the object without it
PCA2D_SUFFIX = '_pca2d'


def box_object(objname: str) -> str:
    """The object whose box a spectrum's object reads: itself, or the object
    PCA2D corrected (<object>_pca2d is the same exposures)"""
    name = str(objname)
    if name.endswith(PCA2D_SUFFIX):
        name = name[:-len(PCA2D_SUFFIX)]
    return name


def path(data_dir: str, objname: str) -> str:
    """The box of an object: DATA_DIR/velocities/<object>.json"""
    return os.path.join(str(data_dir), FOLDER, box_object(objname) + '.json')


def read(data_dir: str, objname: str) -> Dict[str, Any]:
    """The box of an object, empty when there is none (or it cannot be
    read): dict(object, systemic, spectra)"""
    filename = path(data_dir, objname)
    try:
        with open(filename, encoding='utf-8') as handle:
            box = json.load(handle)
    except (OSError, ValueError):
        box = dict()
    box.setdefault('object', box_object(objname))
    box.setdefault('systemic', None)
    box.setdefault('spectra', dict())
    return box


@contextmanager
def _locked(filename: str):
    """A lock on the box while it is read, merged and written (fcntl where
    there is one; none elsewhere)"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    handle = open(filename + '.lock', 'a')
    try:
        try:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        handle.close()


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


def _number(value) -> Optional[float]:
    """A float for JSON, None for NaN or None"""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def update(data_dir: str, objname: str, entries: Dict[str, Dict[str, Any]],
           by: str, systemic: Optional[Dict[str, Any]] = None
           ) -> Dict[str, Any]:
    """
    Merge measurements into an object's box and write it

    :param data_dir: str, DATA_DIR
    :param objname: str, the object (its box: box_object)
    :param entries: dict, file name (any path: its base name is the key) ->
                    dict(v, err, mjd, berv)
    :param by: str, what measured them (RANK)
    :param systemic: dict or None, the box's systemic velocity, replaced
                     when given

    :return: dict, the box as written
    """
    if by not in RANK:
        raise ValueError('unknown kind of measurement: {0}'.format(by))
    filename = path(data_dir, objname)
    with _locked(filename):
        box = read(data_dir, objname)
        spectra = box['spectra']
        when = _now()
        for name, entry in entries.items():
            v = _number(entry.get('v'))
            if v is None:
                continue
            key = os.path.basename(str(name))
            old = spectra.get(key)
            if old is not None and RANK.get(old.get('by'), -1) > RANK[by]:
                continue
            spectra[key] = dict(v=v, err=_number(entry.get('err')), by=by,
                                mjd=_number(entry.get('mjd')),
                                berv=_number(entry.get('berv')), when=when)
        if systemic is not None:
            box['systemic'] = dict(systemic, when=when)
        tmp = filename + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(box, handle, indent=1, sort_keys=True)
        os.replace(tmp, filename)
    return box


def velocities(box: Dict[str, Any], files: List[str]) -> Dict[str, Any]:
    """
    The box's velocities of the given spectra, in their order

    :return: dict(v, err [m/s], by (str, '' when missing), found (bool)),
             each an array of len(files)
    """
    n = len(files)
    out = dict(v=np.full(n, np.nan), err=np.full(n, np.nan),
               by=np.array([''] * n, dtype=object), found=np.zeros(n, bool))
    spectra = box.get('spectra', dict())
    for it, name in enumerate(files):
        entry = spectra.get(os.path.basename(str(name)))
        if entry is None or entry.get('v') is None:
            continue
        out['v'][it] = float(entry['v'])
        err = entry.get('err')
        out['err'][it] = np.nan if err is None else float(err)
        out['by'][it] = str(entry.get('by', ''))
        out['found'][it] = True
    return out


def star_velocity(observed: float, berv: float) -> float:
    """The star's velocity [m/s] from the velocity of its lines as observed
    (topocentric, as a CCF against a model at rest measures it) and the
    BERV: observed = star (+) -BERV, relativistically, the inverse of
    template_scatter.star_shift"""
    return float(mp.add_velocities(observed, berv))
