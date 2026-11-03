#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Each class names its own instrument: its name, the files it writes, its site, its
wavelength domain and its keys are not those of the class it was copied from.

Created on 2026-11-02

@author: artigau
"""
import pytest
from astropy.coordinates import EarthLocation

from lbl.core import base_classes, parameters
from lbl.instruments import select


def _inst(instrument, source='None'):
    klass = select.InstDict[instrument][source]
    # no argument: what the class itself sets, before update_from_args
    return klass(parameters.params.copy(), base_classes.ParamDict())


def _all():
    for instrument, sources in select.InstDict.items():
        if instrument == 'Generic':
            continue
        for source in sources:
            yield instrument, source


@pytest.mark.parametrize('instrument,source', list(_all()))
def test_site_is_known_to_astropy(instrument, source):
    inst = _inst(instrument, source)
    EarthLocation.of_site(inst.params['EARTH_LOCATION'])


@pytest.mark.parametrize('instrument,source', list(_all()))
def test_class_sets_its_own_instrument(instrument, source):
    assert _inst(instrument, source).params['INSTRUMENT'] == instrument


def test_essp_classes_do_not_share_files():
    names = dict()
    for instrument in ('HARPS', 'HARPSN', 'EXPRES', 'NEID'):
        inst = _inst(instrument, 'ESSP')
        assert inst.name == instrument + '_ESSP'
        for attr in ('default_template_name', 'default_mask_name',
                     'default_sample_wave_name'):
            value = getattr(inst, attr)
            assert value not in names, (instrument, value, names.get(value))
            names[value] = instrument


@pytest.mark.parametrize('instrument,essp_source,other_source',
                         [('EXPRES', 'ESSP', 'ORIG')])
def test_essp_domain_is_the_instrument_domain(instrument, essp_source,
                                              other_source):
    essp = _inst(instrument, essp_source)
    other = _inst(instrument, other_source)
    assert (essp.wavemin, essp.wavemax) == (other.wavemin, other.wavemax)
    assert essp.params['COMPIL_WAVE_MIN'] == essp.wavemin


def test_neid_essp_domain_is_neid():
    """the domain of the (unregistered) Neid class"""
    essp = _inst('NEID', 'ESSP')
    assert essp.params['EARTH_LOCATION'] == 'WIYN'
    assert essp.wavemin == pytest.approx(379.66)
    assert essp.wavemax == pytest.approx(822.36)


def test_coralie_at_la_silla_with_its_own_keys():
    inst = _inst('CORALIE')
    assert inst.params['EARTH_LOCATION'] == 'La Silla Observatory'
    # every header key of the class has the ESO prefix of its DRS
    for key in ('KW_SNR', 'KW_EXT_SNR', 'KW_BERV', 'KW_BJD'):
        assert inst.params[key].startswith('HIERARCH ESO '), key
