#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The wavelength domain of HARPS-N is the one of its real files: 65 S2D of one star,
200 of another and one more, read through HarpsN_ESO, run from 387.48 to 387.55 nm
and end at 691.06 to 691.20 nm. HARPS's (378.06 to 691.48) is not it.

Created on 2026-11-02

@author: artigau
"""
import pytest

from lbl.core import base_classes, parameters
from lbl.instruments import select


@pytest.mark.parametrize('source', ['ORIG', 'ESO'])
def test_harpsn_domain_from_real_files(source):
    klass = select.InstDict['HARPSN'][source]
    inst = klass(parameters.params.copy(), base_classes.ParamDict())
    assert 387.47 < inst.wavemin < 387.56
    assert 691.05 < inst.wavemax < 691.21
    # the g band starts at 379.8 nm, 8 nm before the first order: it is
    #   kept, and measured on the part the spectrum covers
    assert inst.get_binned_parameters()['bands'] == ['g']
