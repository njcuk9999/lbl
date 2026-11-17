#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The injection test (lbl.science.injection, lbl_wrap FAKE_KEPLERIAN): the
signals as given, their velocity, the BERV every instrument class gives with
them, the injected object's folder, the wrap's names, and what is recovered.

Created on 2026-11-17

@author: artigau
"""
import os

import numpy as np
import pytest
import yaml

from lbl.core import base_classes
from lbl.core import math as mp
from lbl.instruments.default import Instrument
from lbl.recipes import lbl_wrap
from lbl.science import injection


class Fake(Instrument):
    """An instrument whose BERV is 12 km/s and whose time is the header's"""

    def __init__(self, data_dir, objname, data_type='SCIENCE'):
        super().__init__('FAKE')
        self.params = base_classes.ParamDict()
        self.params['DATA_DIR'] = str(data_dir)
        self.params['SCIENCE_SUBDIR'] = 'science'
        self.params['OBJECT_SCIENCE'] = objname
        self.params['DATA_TYPE'] = data_type

    def get_berv(self, sci_hdr):
        return 12000.0

    def get_rjd_value(self, header):
        return header['RJD']


class Child(Fake):
    """A class whose get_berv calls its parent's"""

    def get_berv(self, sci_hdr):
        return super().get_berv(sci_hdr) + 0.0


def _record(folder, fake):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, injection.RECORD), 'w') as handle:
        yaml.safe_dump(dict(star='X', signals=injection.signals(fake)),
                       handle)


def test_the_signals_as_given():
    one = injection.signals(dict(k=5, period=10))
    assert one == [dict(k=5.0, period=10.0, e=0.0, omega=90.0, phase=0.0)]
    two = injection.signals(dict(k=[5, 3], period=[10, 40],
                                 phase=[0.25, 0.5], e=[0, 0.2],
                                 omega=[90, 30]))
    assert [s['k'] for s in two] == [5.0, 3.0]
    assert two[1]['e'] == 0.2 and two[1]['phase'] == 0.5
    listed = injection.signals([dict(K=5, period=10, t0=2460001.0)])
    assert listed[0]['t0'] == 2460001.0 and 'phase' not in listed[0]
    assert injection.signals(None) == [] and injection.signals({}) == []
    for bad in (dict(k=[5, 3], period=[10]), dict(k=5), dict(k=5, period=0),
                dict(k=5, period=10, e=1.0), dict(k=5, period=10, mass=1),
                dict(k=5, period=10, phase=0.1, t0=2460000.0),
                dict(k='five', period=10)):
        with pytest.raises(base_classes.LblException):
            injection.signals(bad)


def test_the_velocity_is_the_star_s():
    """phase 0: a conjunction at EPOCH, the star then approaching; a phase
    moves the conjunction back by phase * period"""
    found = injection.signals(dict(k=5.0, period=8.0))
    epoch = injection.EPOCH
    assert abs(injection.velocity(epoch, found)[0]) < 1e-9
    assert np.isclose(injection.velocity(epoch + 2.0, found)[0], -5.0)
    assert np.isclose(injection.velocity(epoch - 2.0, found)[0], 5.0)
    later = injection.signals(dict(k=5.0, period=8.0, phase=0.25))
    assert np.isclose(injection.velocity(epoch, later)[0], -5.0)
    # two signals add
    both = injection.signals(dict(k=[5.0, 2.0], period=[8.0, 8.0]))
    assert np.isclose(injection.velocity(epoch + 2.0, both)[0], -7.0)
    # the conjunction of an orbit given by its periastron
    ecc = injection.signals(dict(k=5.0, period=8.0, e=0.3, omega=40.0,
                                 tp=epoch))
    t0 = injection.conjunction(ecc[0])
    # at the conjunction, nu + omega = 90 deg: v = K e cos(omega)
    assert np.isclose(injection.velocity(t0, ecc)[0],
                      5.0 * 0.3 * np.cos(np.radians(40.0)))


def test_get_berv_adds_the_recorded_signals(tmp_path):
    fake = dict(k=5.0, period=8.0)
    _record(str(tmp_path / 'science' / 'STAR_inj'), fake)
    header = dict(RJD=injection.EPOCH + 2.0 - 2400000.0)
    expected = mp.add_velocities(12000.0, -5.0)
    assert Fake(tmp_path, 'STAR').get_berv(header) == 12000.0
    assert np.isclose(Fake(tmp_path, 'STAR_inj').get_berv(header), expected)
    # what is derived from it: its telluric cleaning, its PCA2D correction
    for name in ('STAR_inj_tc', 'STAR_inj_pca2d', 'STAR_inj_pca2d_v2'):
        assert np.isclose(Fake(tmp_path, name).get_berv(header), expected)
    # not a calibration, and once for a class calling its parent's
    assert Fake(tmp_path, 'STAR_inj', 'FP').get_berv(header) == 12000.0
    assert np.isclose(Child(tmp_path, 'STAR_inj').get_berv(header), expected)
    # PCA2D's readers say the folder of each file
    inst = Fake(tmp_path, 'STAR')
    inst.spectrum_folder = str(tmp_path / 'science' / 'STAR_inj')
    assert np.isclose(inst.get_berv(header), expected)
    # no time: said, never taken for no injection
    with pytest.raises(injection.InjectionError):
        Fake(tmp_path, 'STAR_inj').get_berv(dict())


def test_pca2d_reads_the_injection(tmp_path):
    """berv_of and observer_berv fall back on the header when get_berv
    fails, but not when the injection cannot be done"""
    from lbl.pca2d import readers
    _record(str(tmp_path / 'science' / 'STAR_inj'), dict(k=5.0, period=8.0))
    inst = Fake(tmp_path, 'STAR')
    path = str(tmp_path / 'science' / 'STAR_inj' / 'a.fits')
    readers.as_science(inst, path)
    header = dict(RJD=injection.EPOCH + 2.0 - 2400000.0)
    assert np.isclose(readers.berv_of(inst, header) * 1000.0,
                      mp.add_velocities(12000.0, -5.0))
    with pytest.raises(injection.InjectionError):
        readers.berv_of(inst, dict())


def test_the_injected_folder(tmp_path):
    science = tmp_path / 'science'
    (science / 'STAR').mkdir(parents=True)
    for name in ('a.fits', 'b.fits'):
        (science / 'STAR' / name).write_text(name)
    found = injection.signals(dict(k=5.0, period=8.0))
    said = []
    name, previous = injection.prepare(str(science), 'STAR', found,
                                       log=said.append)
    assert name == 'STAR_inj' and previous is None
    target = science / 'STAR_inj'
    assert sorted(os.listdir(target)) == ['a.fits', 'b.fits',
                                          injection.RECORD]
    assert (target / 'a.fits').read_text() == 'a.fits'
    assert injection.read(str(target)) == found
    assert 'K = 5 m/s, P = 8 d, phase 0' in said[0]
    # again: the same signals; a spectrum gone from the star goes from it
    os.remove(science / 'STAR' / 'b.fits')
    name, previous = injection.prepare(str(science), 'STAR', found)
    assert previous == found
    assert sorted(os.listdir(target)) == ['a.fits', injection.RECORD]
    # an injected object is not injected again
    with pytest.raises(base_classes.LblException):
        injection.prepare(str(science), 'STAR_inj', found)


def test_the_wrap_measures_the_injected_objects(tmp_path, monkeypatch):
    calls = []
    for module in ('lbl_template', 'lbl_mask', 'lbl_compute', 'lbl_compile'):
        monkeypatch.setattr(getattr(lbl_wrap, module), 'main',
                            lambda _m=module, **kw: calls.append((_m, kw)))
    for star in ('STAR1', 'STAR2'):
        (tmp_path / 'science' / star).mkdir(parents=True)
        (tmp_path / 'science' / star / 'a.fits').write_text('a')
    # what LBL made from STAR1_inj with other signals is put aside
    (tmp_path / 'lblrv' / 'STAR1_inj_STAR1_inj').mkdir(parents=True)
    _record(str(tmp_path / 'science' / 'STAR1_inj'), dict(k=1.0, period=3.0))
    fake = dict(k=[5.0], period=[8.0])
    lbl_wrap.main(dict(INSTRUMENT='HARPS', DATA_SOURCE='ORIG',
                       DATA_DIR=str(tmp_path), DATA_TYPES=['SCIENCE'] * 2,
                       OBJECT_SCIENCE=['STAR1', 'STAR2'],
                       OBJECT_COMPARISON=['STAR1', 'VETTED_GL699'],
                       OBJECT_TEFF=[3500, 3600], RUN_LBL_COMPUTE=True,
                       REPORT=False, FAKE_KEPLERIAN=fake))
    computed = [(kw['object_science'], kw['object_comparison'])
                for name, kw in calls if name == 'lbl_compute']
    assert computed == [('STAR1_inj', 'STAR1_inj'),
                        ('STAR2_inj', 'VETTED_GL699')]
    # never handed to LBL's recipes
    assert all('FAKE_KEPLERIAN' not in kw for _n, kw in calls)
    assert not (tmp_path / 'lblrv' / 'STAR1_inj_STAR1_inj').exists()
    aside = list((tmp_path / 'pca2d' / 'STAR1_inj' / 'superseded').glob(
        '*/lblrv/STAR1_inj_STAR1_inj'))
    assert len(aside) == 1
    assert injection.read(str(tmp_path / 'science' / 'STAR2_inj')) == \
        injection.signals(fake)


def test_what_is_recovered():
    """A series holding exactly the injected signal: K and the phase
    recovered, within their posteriors"""
    from lbl.pca2d import keplerfit
    rng = np.random.default_rng(3)
    found = injection.signals(dict(k=[6.0], period=[7.3], phase=[0.3]))
    t = np.sort(rng.uniform(0.0, 200.0, 80)) + 60100.0
    err = np.full(t.size, 1.0)
    v = injection.velocity(t + 2400000.0, found) + rng.normal(0, 1.0, t.size)
    post = keplerfit.posteriors(t, v, err, [7.3], steps=600, burn=200)
    row = injection.recovery(post, found)[0]
    assert row['scenario'] == 'circular'
    assert abs(row['dk'][0]) < 3 * (row['k'][2] - row['k'][1])
    assert abs(row['dlam'][0]) < 15.0
    assert abs(row['sigma']) < 3.0


def test_an_eccentric_signal_is_recovered_with_its_phase():
    """The Keplerian fit's mean longitude M + omega is the injected one"""
    from lbl.pca2d import keplerfit
    rng = np.random.default_rng(5)
    found = injection.signals(dict(k=[10.0], period=[11.0], phase=[0.6],
                                   e=[0.3], omega=[50.0]))
    t = np.sort(rng.uniform(0.0, 300.0, 120)) + 60100.0
    err = np.full(t.size, 0.5)
    v = injection.velocity(t + 2400000.0, found) + rng.normal(0, 0.5, t.size)
    post = keplerfit.posteriors(t, v, err, [11.0], steps=1500, burn=500)
    row = injection.recovery(post, found)[0]
    assert row['scenario'] == 'keplerian'
    assert abs(row['dk'][0]) < 1.0
    assert abs(row['dlam'][0]) < 10.0
    assert abs(row['e'][0] - 0.3) < 0.1


def test_the_window_writes_and_reads_the_signals():
    from lbl.gui import wrapgen, wrapload
    state = dict(on=True, signals=[dict(k='5', period=12.3, phase='',
                                        e=0, omega=90),
                                   dict(k=3, period=40, phase=0.25, e=0.2,
                                        omega=30)])
    fake = wrapgen.injection_params(state)
    assert fake == dict(k=[5.0, 3.0], period=[12.3, 40.0], phase=[0.0, 0.25],
                        e=[0.0, 0.2], omega=[90.0, 30.0])
    assert wrapgen.injection_params(dict(on=False, signals=state['signals'])) \
        is None
    assert 'K and P' in wrapgen.injection_params(dict(on=True, signals=[]))
    assert 'period' in wrapgen.injection_params(
        dict(on=True, signals=[dict(k=5, period=-1)]))
    # in the wrap, and back
    text = wrapgen.render(dict(INSTRUMENT='SPIROU', DATA_SOURCE='APERO',
                               OBJECT_SCIENCE=['STAR'], FAKE_KEPLERIAN=fake))
    assert "rparams['FAKE_KEPLERIAN'] = {'k': [5.0, 3.0]" in text
    assert 'an injection test' in text
    back = wrapload.to_state(wrapload.read_text(text)['rparams'])['state']
    assert back['injection']['on'] is True
    assert [s['k'] for s in back['injection']['signals']] == [5.0, 3.0]
    assert 'FAKE_KEPLERIAN' not in back['extra']
    # a t0 is read as the phase it means
    rows, said = wrapload.injection_state(dict(k=5, period=10,
                                               t0=injection.EPOCH - 2.5))
    assert np.isclose(rows[0]['phase'], 0.25) and 't0' in said


def test_the_run_s_steps_name_the_injected_objects():
    from lbl.gui import wrapgen
    plan = wrapgen.steps(dict(OBJECT_SCIENCE=['STAR'],
                              OBJECT_COMPARISON=['STAR'],
                              RUN_LBL_COMPUTE=True, RUN_LBL_COMPILE=True,
                              PCA2D=dict(n_star=1),
                              FAKE_KEPLERIAN=dict(k=5, period=10)))
    objects = [step['obj'] for step in plan]
    assert 'STAR_inj' in objects and 'STAR_inj_pca2d' in objects
    assert 'STAR' not in objects and 'STAR_pca2d' not in objects


def test_the_report_says_what_was_recovered():
    from lbl.pca2d import keplerfit, texreport
    rng = np.random.default_rng(7)
    signal = injection.signals(dict(k=4.0, period=9.0, phase=0.1))[0]
    t = np.sort(rng.uniform(0.0, 250.0, 90)) + 60100.0
    err = np.full(t.size, 1.0)
    clean = injection.velocity(t + 2400000.0, [signal])
    planets = texreport.injected_planets(None, 'X') + [
        {'name': 'injected 1', 'short': 'inj1', 'period': 9.0,
         'k': 4.0, 't0': injection.conjunction(signal), 'source': 'injected',
         'injected': signal}]
    posts = {}
    for label, scale in (('delivered', 1.0), ('corrected', 0.5)):
        v = scale * clean + rng.normal(0.0, 1.0, t.size)
        posts[label] = keplerfit.posteriors(t, v, err, [9.0], steps=600,
                                            burn=200)
    text = texreport.injection_section(posts, planets,
                                       dict(label='delivered'),
                                       dict(label='corrected'), 'X')
    assert 'the injection test' in text
    assert 'K = 4 m/s, P = 9 d, phase 0.1' in text
    assert 'corrected / delivered' in text
    # the corrected K about half the delivered one
    ratio = injection.ratio(posts['corrected'], posts['delivered'], 0,
                            'circular')
    assert 0.3 < ratio[0] < 0.75
    assert texreport.injection_section(posts, planets[:0], {}, {}, 'X') == ''


def test_other_signals_make_another_pca2d_cube():
    import copy
    from lbl.pca2d import config as pconfig
    base = copy.deepcopy(pconfig.DEFAULTS)
    key = pconfig.cache_key(base)
    signals = injection.signals(dict(k=5.0, period=8.0))
    other = copy.deepcopy(base)
    other['registration']['injection'] = signals
    again = copy.deepcopy(other)
    again['registration']['injection'] = injection.signals(
        dict(k=6.0, period=8.0))
    assert pconfig.cache_key(other) != key
    assert pconfig.cache_key(again) != pconfig.cache_key(other)
    # none: the key of every cube already built
    del other['registration']['injection']
    assert pconfig.cache_key(other) == key


def test_the_signals_among_the_known_planets(tmp_path):
    """known=True: PLANETS reads the injected signals with the catalog's
    orbits of the star, so that the template follows both"""
    from lbl.science import planets
    both = injection.signals(dict(k=[5.0, 3.0], period=[8.0, 20.0],
                                  known=True))
    assert all(signal['known'] for signal in both)
    one = injection.signals(dict(k=[5.0, 3.0], period=[8.0, 20.0],
                                 known=[True, False]))
    assert one[0]['known'] and 'known' not in one[1]
    with pytest.raises(base_classes.LblException):
        injection.signals(dict(k=5.0, period=8.0, known='maybe'))
    assert 'a known planet' in injection.describe(both[:1])
    # the star in a personal catalog of DATA_DIR, with one planet
    with open(tmp_path / planets.PERSONAL, 'w') as handle:
        yaml.safe_dump(dict(stars=dict(STAR=dict(planets=[dict(
            name='STAR b', period=3.0, k=2.0, t0=2460000.5)]))), handle)
    folder = str(tmp_path / 'science' / 'STAR_inj')
    _record(folder, dict(k=[5.0, 3.0], period=[8.0, 20.0],
                         known=[True, False]))
    names = planets.star_names('STAR_inj')
    found = planets.orbits_for(names, str(tmp_path), science_dir=folder)
    assert [o['name'] for o in found['orbits']] == ['STAR b', 'injected 1']
    assert found['orbits'][1]['k'] == 5.0
    # the star's own spectra, or no folder: the catalog alone
    plain = planets.orbits_for(planets.star_names('STAR'), str(tmp_path),
                               science_dir=str(tmp_path / 'science' / 'STAR'))
    assert [o['name'] for o in plain['orbits']] == ['STAR b']
    # a star no catalog knows: the injected signals alone
    _record(str(tmp_path / 'science' / 'NOBODY_XYZ_inj'),
            dict(k=4.0, period=6.0, known=True))
    alone = planets.orbits_for(['NOBODY_XYZ_inj', 'NOBODY_XYZ'],
                               str(tmp_path), science_dir=str(
                                   tmp_path / 'science' / 'NOBODY_XYZ_inj'))
    assert [o['name'] for o in alone['orbits']] == ['injected 1']
    assert alone['catalog'].endswith(injection.RECORD)
    # the velocity PLANETS registers the spectra with: the injected one too
    bjd = injection.EPOCH + 1.5
    assert np.isclose(planets.star_velocity(bjd, alone['orbits'])[0],
                      injection.velocity(bjd, injection.signals(
                          dict(k=4.0, period=6.0)))[0])


def test_the_window_holds_known():
    from lbl.gui import wrapgen, wrapload
    state = dict(on=True, known=True, signals=[dict(k=5, period=12)])
    fake = wrapgen.injection_params(state)
    assert fake['known'] is True
    text = wrapgen.render(dict(OBJECT_SCIENCE=['STAR'],
                               FAKE_KEPLERIAN=fake))
    back = wrapload.to_state(wrapload.read_text(text)['rparams'])['state']
    assert back['injection']['known'] is True
    assert 'known' not in back['injection']['signals'][0]


def test_an_injected_period_is_fitted_once():
    from lbl.pca2d import texreport
    known = [dict(name='b', period=3.0, source='archive'),
             dict(name='8 d', period=8.0, source='configuration')]
    injected = [dict(name='injected 1', period=8.0, source='injected')]
    merged = texreport.with_injected(known, injected)
    assert [p['name'] for p in merged] == ['b', 'injected 1']


def test_the_corner_marks_the_injected_values(tmp_path):
    """The triangle of the posteriors puts a star at the values an injected
    signal was given, its angles on the branch of the samples"""
    from lbl.pca2d import keplerfit, texreport
    rng = np.random.default_rng(13)
    signal = injection.signals(dict(k=6.0, period=9.0, phase=0.3, e=0.3,
                                    omega=200.0))[0]
    t = np.sort(rng.uniform(0.0, 250.0, 90)) + 60100.0
    err = np.ones(t.size)
    v = injection.velocity(t + 2400000.0, [signal])
    posts = {label: keplerfit.posteriors(
        t, v + rng.normal(0.0, 1.0, t.size), err, [9.0], steps=600,
        burn=200) for label in ('delivered', 'corrected')}
    planets = [dict(name='injected 1', short='inj1', period=9.0, k=6.0,
                    source='injected', injected=signal)]
    truths = texreport.corner_truths(posts, planets, 'keplerian',
                                     'delivered')
    assert truths['K_0'] == 6.0 and truths['e_0'] == 0.3
    centre = np.median(posts['delivered']['keplerian']['samples']['omega_0'])
    # 200 deg, on the samples' branch (-160 when they sit about -160)
    assert abs(truths['omega_0'] - centre) < 30.0
    assert np.isclose(truths['omega_0'] % 360.0, 200.0)
    assert 'e_0' not in texreport.corner_truths(posts, planets, 'circular',
                                                'delivered')
    assert texreport.corner_truths(posts, [dict(name='b', period=9.0)],
                                   'keplerian', 'delivered') == {}
    path = texreport.figure_posterior_corner(
        posts, planets, dict(label='delivered', star='X'),
        dict(label='corrected', star='X'), 'keplerian',
        str(tmp_path / 'corner.pdf'))
    assert path and os.path.exists(path)
