#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KW_EXPTIME is an exposure time on every class, not another time key: LBL divides
it by 86400 to match an exposure to the velocities of a template's table.

Created on 2026-11-02

@author: artigau
"""
import pytest

from lbl.core import base_classes, parameters
from lbl.instruments import select


def _instances():
    for instrument, sources in select.InstDict.items():
        if instrument == 'Generic':
            continue
        for source, klass in sources.items():
            yield instrument, source, klass


@pytest.mark.parametrize('instrument,source,klass', list(_instances()))
def test_exposure_time_is_not_a_time_key(instrument, source, klass):
    args = base_classes.ParamDict()
    inst = klass(parameters.params.copy(), args)
    key = inst.params['KW_EXPTIME']
    assert key not in (inst.params['KW_BJD'], inst.params['KW_MID_EXP_TIME'],
                       inst.params['KW_MJDATE'])


@pytest.mark.parametrize('instrument,source', [('HARPS', 'ESO'),
                                               ('HARPSN', 'ESO'),
                                               ('ESPRESSO', 'None')])
def test_exposure_time_of_the_eso_format(instrument, source):
    """Real HARPS-N and ESPRESSO S2D files have EXPTIME in seconds (900.0)"""
    klass = select.InstDict[instrument][source]
    inst = klass(parameters.params.copy(), base_classes.ParamDict())
    assert inst.params['KW_EXPTIME'] == 'EXPTIME'
