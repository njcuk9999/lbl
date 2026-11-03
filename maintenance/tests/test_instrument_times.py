#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The time keys of the instrument classes: nights are cut at local noon and the
plot date is right, whichever of an MJD or a BJD the class has in KW_MJDATE.

Created on 2026-11-02

@author: artigau
"""
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from lbl.core import base_classes, io, parameters
from lbl.instruments import select

#: the classes that have a BJD key
BJD_CLASSES = [('HARPS', 'ORIG'), ('HARPS', 'ESO'), ('HARPSN', 'ORIG'),
               ('HARPSN', 'ESO'), ('ESPRESSO', 'None'), ('SOPHIE', 'None'),
               ('CORALIE', 'None')]
#: from real headers: (MJD-OBS, BJD) of a HARPS, a HARPS-N and an ESPRESSO file
REAL_TIMES = {('HARPS', 'ORIG'): (53817.03237667, 2453817.53488381),
              ('HARPSN', 'ESO'): (58212.9454282406, 2458213.45542024),
              ('ESPRESSO', 'None'): (58720.1032638, 2458720.61403175)}


def _inst(instrument, source='None', **extra):
    args = base_classes.ParamDict()
    args.set('INSTRUMENT', instrument, source='test')
    args.set('DATA_SOURCE', source, source='test')
    for key, value in extra.items():
        args.set(key, value, source='test')
    klass = select.InstDict[instrument][source]
    return klass(parameters.params.copy(), args)


@pytest.mark.parametrize('instrument,source', BJD_CLASSES)
def test_epochs_cut_at_local_noon(instrument, source):
    """Two exposures an hour on either side of local midnight are one night,
    and two an hour on either side of local noon are two"""
    inst = _inst(instrument, source)
    kw_mjd = inst.params['KW_MJDATE']
    zero = 2400000.5 if 'BJD' in kw_mjd else 0.0
    # local midnight at the class's site, in UT days
    site = EarthLocation.of_site(inst.params['EARTH_LOCATION'])
    midnight = 58000 + ((-site.lon.value + 360) / 360) % 1
    hour = 1 / 24
    times = np.array([midnight - hour, midnight + hour,
                      midnight + 0.5 - hour, midnight + 0.5 + hour]) + zero
    groups, values = inst.get_epoch_groups(Table({kw_mjd: times}))
    assert values[0] == values[1] == values[2]
    assert values[3] != values[2]
    assert len(groups) == 2


@pytest.mark.parametrize('instrument,source', BJD_CLASSES)
def test_plot_date_is_not_a_julian_date(instrument, source):
    inst = _inst(instrument, source)
    kw_mjd = inst.params['KW_MJDATE']
    mjd = 58212.9454282406
    header = io.LBLHeader()
    header[kw_mjd] = mjd + 2400000.5 if 'BJD' in kw_mjd else mjd
    plot_date = inst.get_plot_date(header)
    assert plot_date == pytest.approx(Time(mjd, format='mjd').plot_date)
    # days since 1970, not a year near 8600
    assert 1e4 < plot_date < 1e5


@pytest.mark.parametrize('key,times', REAL_TIMES.items())
def test_true_mjd_key_where_a_real_file_shows_one(key, times):
    """A real file of these classes has MJD-OBS, and the exposure's start is
    within the exposure time and the barycentric offset of the BJD"""
    inst = _inst(*key)
    assert inst.params['KW_MJDATE'] == 'MJD-OBS'
    mjd, bjd = times
    assert abs(inst.to_mjd(bjd) - mjd) < 0.02


def test_to_mjd_leaves_an_mjd_alone():
    inst = _inst('HARPS', 'ORIG')
    assert inst.to_mjd(58000.25) == 58000.25
    assert inst.to_mjd(2458000.75) == pytest.approx(58000.25)
    values = inst.to_mjd(np.array([58000.25, 2458000.75]))
    assert values == pytest.approx([58000.25, 58000.25])


def test_template_time_limits_are_mjds_on_a_bjd_class(tmp_path):
    """TEMPLATE_MJDSTART and TEMPLATE_MJDEND are MJDs, HARPS has a BJD in
    KW_MID_EXP_TIME: a file inside the limits is kept, one outside is not"""
    inst = _inst('HARPS', 'ORIG', TEMPLATE_MJDSTART=58000.0,
                 TEMPLATE_MJDEND=58010.0, DATA_TYPE='SCIENCE')
    paths = []
    for name, mjd in (('in', 58005.0), ('out', 58020.0)):
        hdr = fits.Header()
        hdr[inst.params['KW_MID_EXP_TIME']] = mjd + 2400000.5
        path = tmp_path / (name + '.fits')
        fits.PrimaryHDU(np.zeros((2, 2)), header=hdr).writeto(path)
        paths.append(str(path))
    kept = inst.filter_files(paths)
    assert [p.split('/')[-1] for p in kept] == ['in.fits']
