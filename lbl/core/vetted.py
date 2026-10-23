#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LBL's vetted templates: high-SNR templates of slowly rotating M dwarfs, one
per 100 K of Teff and per instrument, for a star whose own template would be
poor (few spectra, a narrow BERV range) to be measured against.

A wrap names one as OBJECT_COMPARISON = 'VETTED_<object>' (the objects are
in lbl/resources/vetted_templates.yaml, with why each was chosen; the
choice itself is in lbl/doc/vetted_templates). Nothing else changes in the
wrap: when the template LBL looks for is not on disk, the depot file is
downloaded from MODEL_REPO_URL into DATA_DIR/models and written as an LBL
template, with LBL's own Savitzky-Golay columns, under the name the
instrument gives the template of VETTED_<object>.

@author: artigau
"""
import os
from typing import Dict, List, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table

from lbl.core import base
from lbl.core import base_classes
from lbl.core import io

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'core.vetted.py'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get log
log = io.log
#: the prefix of a vetted template's object name in a wrap
PREFIX = 'VETTED_'
#: the list of the vetted templates
LIST_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'resources', 'vetted_templates.yaml')
#: the Teff range a vetted template is offered as a best match for [K]
TEFF_RANGE = (2900.0, 4300.0)
#: speed of light [m/s]
SPEED_OF_LIGHT = 299792458.0

_LIST = None


# =============================================================================
# Define functions
# =============================================================================
def _load() -> Dict[str, List[dict]]:
    global _LIST
    if _LIST is None:
        import yaml
        with open(LIST_FILE) as handle:
            _LIST = yaml.safe_load(handle) or dict()
    return _LIST


def entries(instrument: str) -> List[dict]:
    """
    The vetted templates of an instrument (NIRPS_HE, SPIROU), coolest first,
    each with its object name in a wrap (name = VETTED_<object>)

    :param instrument: str, the LBL instrument

    :return: list of dict (object, name, teff, nfiles, prot, file...)
    """
    found = _load().get(str(instrument).upper(), []) or []
    out = [dict(entry, name=PREFIX + entry['object']) for entry in found]
    return sorted(out, key=lambda entry: entry['teff'])


def is_vetted(objname: Optional[str]) -> bool:
    """Whether an object name is a vetted template's (VETTED_<object>)"""
    return str(objname or '').upper().startswith(PREFIX)


def entry_of(instrument: str, objname: str) -> Optional[dict]:
    """The vetted template an object name points to, or None"""
    for entry in entries(instrument):
        if entry['name'].upper() == str(objname).upper():
            return entry
    return None


def best_match(instrument: str, teff: float) -> Optional[dict]:
    """
    The vetted template of the nearest Teff, for a star of TEFF_RANGE; None
    outside it or for an instrument without vetted templates

    :param instrument: str, the LBL instrument
    :param teff: float, the star's Teff [K]

    :return: dict or None
    """
    try:
        teff = float(teff)
    except (TypeError, ValueError):
        return None
    if not TEFF_RANGE[0] <= teff <= TEFF_RANGE[1]:
        return None
    found = entries(instrument)
    if not found:
        return None
    return min(found, key=lambda entry: abs(entry['teff'] - teff))


def fetch(inst, objname: str, template_path: str):
    """
    Put the vetted template OBJNAME at TEMPLATE_PATH, as an LBL template:
    its depot file is downloaded once into DATA_DIR/models, then written
    with the instrument's own template writer. Nothing happens if the
    template is there.

    :param inst: Instrument, the instrument of the run
    :param objname: str, VETTED_<object>
    :param template_path: str, where the instrument looks for its template

    :return: None
    """
    if os.path.exists(template_path):
        return
    instrument = inst.params['INSTRUMENT']
    entry = entry_of(instrument, objname)
    if entry is None:
        names = ', '.join(e['name'] for e in entries(instrument)) or 'none'
        emsg = ('{0} is not a vetted template of {1} (there are: {2}; '
                'lbl/resources/vetted_templates.yaml)')
        raise base_classes.LblException(emsg.format(objname, instrument,
                                                    names))
    model_dir = os.path.join(inst.params['DATA_DIR'], 'models')
    os.makedirs(model_dir, exist_ok=True)
    depot_file = os.path.join(model_dir, entry['file'])
    url = '{0}/{1}'.format(inst.params['MODEL_REPO_URL'], entry['file'])
    io.get_urlfile(url, 'vetted template ' + entry['name'], depot_file)
    msg = ('Vetted template {0}: {1} K, {2} spectra, written as {3}')
    log.general(msg.format(entry['name'], entry['teff'], entry['nfiles'],
                           template_path))
    write_template(inst, depot_file, template_path, entry['name'])


def write_template(inst, depot_file: str, template_path: str,
                   objname: Optional[str] = None):
    """
    A depot file (lbl/doc/vetted_templates/make_vetted.py) written as an LBL
    template: the flux where at least half of the spectra contributed (as
    LBL reads APERO's templates), the same flux for both order parities (the
    depot has none apart), and LBL's Savitzky-Golay columns

    :param inst: Instrument, the instrument of the run
    :param depot_file: str, the depot file
    :param template_path: str, the LBL template to write
    :param objname: str, VETTED_<object>, the object the template is of (in
                    its header, as LBL's template recipe writes it); the
                    depot's OBJECT if None

    :return: None
    """
    with fits.open(depot_file) as hdul:
        header = hdul[0].header.copy()
        table = Table(hdul['TEMPLATE'].data)
        sci = Table(hdul['SCI_TABLE'].data) if 'SCI_TABLE' in hdul else None
    wave = np.asarray(table['wavelength'], dtype=float)
    flux = np.asarray(table['flux'], dtype=float)
    eflux = np.asarray(table['eflux'], dtype=float)
    rms = np.asarray(table['rms'], dtype=float)
    n_valid = np.asarray(table['n_valid'], dtype=float)
    bad = n_valid < np.nanmax(n_valid) / 2
    flux[bad] = np.nan
    # the grid's step [m/s], uniform in velocity
    dv_grid = float(np.nanmedian(np.diff(np.log(wave)))) * SPEED_OF_LIGHT
    savgol = inst.calculate_savgol_template(
        dv_grid=dv_grid, flux_dict=dict(flux=flux, flux_odd=flux,
                                        flux_even=flux))
    if sci is not None and 'BERV' in sci.colnames:
        berv = np.asarray(sci['BERV'], dtype=float)
        coverage = len(np.unique(np.floor(berv[np.isfinite(berv)])))
    else:
        coverage = 0
    sci_table = dict()
    if sci is not None:
        for col in sci.colnames:
            sci_table[col] = np.asarray(sci[col])
    props = dict(wavelength=wave, flux=flux, eflux=eflux, rms=rms,
                 flux_odd=flux, eflux_odd=eflux, rms_odd=rms,
                 flux_even=flux, eflux_even=eflux, rms_even=rms,
                 template_coverage=coverage, total_nobs_berv=coverage,
                 template_nobs=int(header.get('VETNFIL', 0)),
                 savgol_fluxes=savgol,
                 template_type='VETTED_SAVGOL' if savgol else 'VETTED',
                 planets=None)
    # the template is the vetted star's, not the run's: its header says so
    #   (lbl_hkeys writes OBJECT_SCIENCE and OBJECT_COMPARISON)
    if objname is None:
        objname = PREFIX + str(header.get('OBJECT', 'UNKNOWN'))
    kept = {key: inst.params.get(key) for key in ('OBJECT_SCIENCE',
                                                  'OBJECT_COMPARISON')}
    try:
        for key in kept:
            inst.params[key] = objname
        inst.write_template(template_path, props,
                            io.LBLHeader.from_fits(header), sci_table)
    finally:
        for key, value in kept.items():
            inst.params[key] = value


# =============================================================================
# End of code
# =============================================================================
