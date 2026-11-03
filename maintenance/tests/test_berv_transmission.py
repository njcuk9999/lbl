#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exactly one of get_berv and get_berv_transmission carries the header's BERV: the
first for a grid in the observer's frame, the second for a barycentric one.

Created on 2026-11-02

@author: artigau
"""
import pytest

from lbl.core import base_classes, io, parameters
from lbl.instruments import select

#: the classes whose wave grid is barycentric (get_berv returns zero)
BARYCENTRIC = [('HARPS', 'ESO'), ('HARPSN', 'ESO'), ('ESPRESSO', 'None'),
               ('NIRPS_HA', 'ESO'), ('NIRPS_HE', 'ESO')]


@pytest.mark.parametrize('instrument,source', BARYCENTRIC)
def test_one_of_the_two_carries_the_berv(instrument, source):
    klass = select.InstDict[instrument][source]
    args = base_classes.ParamDict()
    args.set('OBJECT_SCIENCE', 'STAR', source='test')
    inst = klass(parameters.params.copy(), args)
    header = io.LBLHeader()
    header[inst.params['KW_BERV']] = -12.5
    assert inst.get_berv(header) == 0.0
    assert inst.get_berv_transmission(header) == pytest.approx(-12500.0)
