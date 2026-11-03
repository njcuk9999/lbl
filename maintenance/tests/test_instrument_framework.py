#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What the instrument base class promises: a method a class does not implement
says so with its name, an error is raised where one is built, and a value the
user gave is never overwritten by the class.

Created on 2026-11-02

@author: artigau
"""
import numpy as np
import pytest
from astropy.io import fits

from lbl.core import base_classes, parameters
from lbl.instruments import default, select

KWARGS = 'lbl.instruments.select.parse_args() [KWARGS]'


def test_not_implemented_says_what():
    inst = default.Instrument('TEST')
    with pytest.raises(NotImplementedError, match='populate_sci_table'):
        inst.populate_sci_table('a', dict(), None)
    with pytest.raises(NotImplementedError, match='load_blaze_from_science'):
        inst.load_blaze_from_science('a', np.ones(2), None, '.')


def test_neid_says_it_is_not_implemented():
    from lbl.instruments import neid
    with pytest.raises(NotImplementedError, match='NEID'):
        neid.Neid(parameters.params.copy(), base_classes.ParamDict())


@pytest.mark.parametrize('instrument,source', [('SPIROU', 'CADC'),
                                               ('NIRPS_HA', 'CADC'),
                                               ('NIRPS_HE', 'CADC')])
def test_missing_flux_extension_stops(tmp_path, instrument, source):
    klass = select.InstDict[instrument][source]
    inst = klass(parameters.params.copy(), base_classes.ParamDict())
    inst.params.set('FLUX_EXTENSION_NAME', None, source='test')
    path = tmp_path / 'x.fits'
    fits.PrimaryHDU(np.ones((2, 2))).writeto(path)
    with pytest.raises(base_classes.LblException,
                       match='FLUX_EXTENSION_NAME'):
        inst.load_science_file(str(path))


def _user_args(**values):
    args = base_classes.ParamDict()
    for key, value in values.items():
        args.set(key, value, source=KWARGS)
    return args


def _registered():
    for instrument, sources in select.InstDict.items():
        if instrument == 'Generic':
            continue
        for source, klass in sources.items():
            yield instrument, source, klass


@pytest.mark.parametrize('instrument,source,klass', list(_registered()))
def test_param_set_keeps_a_user_value(instrument, source, klass):
    """After construction, param_set does not overwrite what the user gave
    (the ESSP constructors run their param_override a second time)"""
    args = _user_args(SNR_THRESHOLD=123.5, HP_WIDTH=321.5)
    inst = klass(parameters.params.copy(), args)
    assert inst.params['SNR_THRESHOLD'] == 123.5
    inst.param_override()
    assert inst.params['SNR_THRESHOLD'] == 123.5
    assert inst.params['HP_WIDTH'] == 321.5
    # a value the class set itself can still be changed by the class
    inst.param_set('READ_OUT_NOISE', 99.0, source='test')
    assert inst.params['READ_OUT_NOISE'] == 99.0


def _generic_args(**extra):
    """the keys the generic class asks for, as a wrap gives them"""
    params = parameters.params
    values = dict(GENERIC_INSTRUMENT='MYSPEC', GENERIC_DATA_SOURCE='DRS',
                  GENERIC_WAVEMIN=380.0, GENERIC_WAVEMAX=690.0,
                  EARTH_LOCATION='La Silla Observatory')
    import inspect
    from lbl.instruments import generic
    text = inspect.getsource(generic.Generic.param_override)
    for line in text.splitlines():
        if "generic_validate('" not in line:
            continue
        key = line.split("generic_validate('")[1].split("'")[0]
        if key in values:
            continue
        current = params[key]
        values[key] = 'KEY_' + key if current is None else current
    values.update(extra)
    return _user_args(**values)


def test_generic_keeps_a_user_value():
    """The generic class takes the user's values first (it validates them),
    then sets its own: those must not overwrite the user's"""
    from lbl.instruments import generic
    args = _generic_args(FLUX_EXTENSION_NAME='SCI', OBJECT_LOGG=5.25)
    params = parameters.params.copy()
    inst = generic.Generic(params, args)
    assert inst.params['FLUX_EXTENSION_NAME'] == 'SCI'
    assert inst.params['OBJECT_LOGG'] == 5.25
    # the class's own value where the user gave none
    assert inst.params['KW_REF_KEY'] == 'DPRTYPE'
    # INSTRUMENT is replaced by the instrument Generic stands for
    assert inst.params['INSTRUMENT'] == 'MYSPEC'
    assert inst.params['DATA_SOURCE'] == 'DRS'
