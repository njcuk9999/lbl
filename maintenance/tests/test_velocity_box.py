#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The velocity box (lbl.science.velocity_box): each spectrum's velocity kept
in DATA_DIR/velocities/<object>.json, filled the first time by a CCF against
the PHOENIX model and brought up to date by better measurements; the
template's passes start from it.

Created on 2026-11-24

@author: artigau
"""
import json
import os

import numpy as np

from lbl.core.math import speed_of_light_ms
from lbl.science import general, template_scatter as ts, velocity_box as vb


def test_a_better_measurement_replaces_a_worse_never_the_reverse(tmp_path):
    vb.update(str(tmp_path), 'STAR', {'/x/a.fits': dict(v=-21000.0),
                                      'b.fits': dict(v=-21100.0)}, 'ccf',
              systemic=dict(v=-21050.0, by='ccf', n=2))
    vb.update(str(tmp_path), 'STAR', {'a.fits': dict(v=-21010.0, err=1.5)},
              'lbl')
    vb.update(str(tmp_path), 'STAR', {'a.fits': dict(v=-20000.0, err=9.0),
                                      'b.fits': dict(v=-21090.0, err=3.0)},
              'template')
    box = vb.read(str(tmp_path), 'STAR')
    assert box['spectra']['a.fits']['v'] == -21010.0
    assert box['spectra']['a.fits']['by'] == 'lbl'
    assert box['spectra']['b.fits']['by'] == 'template'
    assert box['systemic']['v'] == -21050.0
    with open(tmp_path / 'velocities' / 'STAR.json') as handle:
        assert json.load(handle)['spectra']['b.fits']['err'] == 3.0


def test_the_corrected_spectra_start_from_their_object_s_box(tmp_path):
    """<object>_pca2d keeps its own box (its velocities are cleaner), and
    starts from its object's for the spectra it lacks; an injection test
    has its own"""
    assert vb.path('/d', 'GJ1_pca2d') != vb.path('/d', 'GJ1')
    assert vb.parent_object('GJ1_pca2d') == 'GJ1'
    assert vb.parent_object('GJ1_inj') is None
    vb.update(str(tmp_path), 'GJ1', {'a.fits': dict(v=1.0),
                                     'b.fits': dict(v=2.0)}, 'template',
              systemic=dict(v=1.5, by='ccf', n=2))
    vb.update(str(tmp_path), 'GJ1_pca2d', {'a.fits': dict(v=1.2)},
              'template')
    got = vb.velocities(vb.read_start(str(tmp_path), 'GJ1_pca2d'),
                        ['/p/a.fits', '/p/b.fits', '/p/c.fits'])
    assert list(got['found']) == [True, True, False]
    assert list(got['v'][:2]) == [1.2, 2.0]
    assert vb.read_start(str(tmp_path), 'GJ1_pca2d')['systemic']['v'] == 1.5
    assert list(vb.read(str(tmp_path), 'GJ1_pca2d')['spectra']) == ['a.fits']


def test_a_nan_velocity_is_not_kept(tmp_path):
    vb.update(str(tmp_path), 'S', {'a.fits': dict(v=np.nan)}, 'ccf')
    assert vb.read(str(tmp_path), 'S')['spectra'] == {}


def test_the_star_s_velocity_registers_the_spectrum_at_rest():
    """star_velocity is the inverse of template_scatter.star_shift: the
    star's velocity from what the lines show and the BERV takes the
    observed lines back to rest"""
    observed, berv = -15000.0, 25000.0
    star = vb.star_velocity(observed, berv)
    shift = ts.star_shift(np.array([berv]), np.array([star]))[0]
    assert np.isclose(shift, -np.arctanh(observed / speed_of_light_ms),
                      rtol=0, atol=1e-12)


def _model(wave):
    """lines in the H band, the photometric band the CCF reads"""
    rng = np.random.default_rng(5)
    centres = rng.uniform(1510.0, 1790.0, 400)
    depth = rng.uniform(0.1, 0.6, 400)
    flux = np.ones_like(wave)
    for c, d in zip(centres, depth):
        flux -= d * np.exp(-0.5 * ((wave - c) / 0.01) ** 2)
    return flux


def _shifted(velocity, wave):
    """the model's lines redshifted by velocity [m/s]"""
    ratio = np.sqrt((1 + velocity / speed_of_light_ms)
                    / (1 - velocity / speed_of_light_ms))
    return _model(wave / ratio)


def test_the_ccf_against_the_model_finds_a_spectrum_s_velocity():
    m_wave = np.linspace(1500.0, 1800.0, 600000)
    reference = general.sysvel_reference(1505.0, 1795.0, m_wave,
                                         _model(m_wave))
    wave = np.linspace(1505.0, 1795.0, 200000)
    for velocity in (-35000.0, 12345.0):
        found, _, _, signif = general.sysvel_ccf(
            wave, _shifted(velocity, wave), reference)
        assert abs(1000 * found - velocity) < 100 and signif > 5


def test_the_template_starts_from_the_box_the_ccf_only_once(tmp_path,
                                                             monkeypatch):
    """the first time, each spectrum's velocity by a CCF against the model,
    into the box with the systemic velocity; the start is the box less
    it; the second time the box is read, no CCF"""
    m_wave = np.linspace(1500.0, 1800.0, 600000)
    monkeypatch.setattr(general, 'get_stellar_models',
                        lambda inst, model_dir: (m_wave, _model(m_wave)))
    observed = {'a.fits': -20000.0, 'b.fits': -26000.0, 'c.fits': -14000.0}
    berv = np.array([10000.0, 4000.0, 16000.0])
    wave = np.linspace(1505.0, 1795.0, 200000)
    read = []

    def fake_spectrum(inst, filename, load_science, blaze, calib_dir):
        read.append(filename)
        return wave, _shifted(observed[os.path.basename(filename)], wave), \
            None

    monkeypatch.setattr(ts, 'spectrum_1d', fake_spectrum)

    class Inst:
        params = dict(OBJECT_TEFF=3300)

    files = ['/s/a.fits', '/s/b.fits', '/s/c.fits']
    start, sysvel = ts.start_velocities(
        Inst(), files, None, None, '', berv, np.zeros(3), 'STAR',
        str(tmp_path), '', lambda x, **k: x)
    star = np.array([vb.star_velocity(observed[os.path.basename(f)], b)
                     for f, b in zip(files, berv)])
    assert len(read) == 3
    assert abs(sysvel - np.median(star)) < 100
    assert np.all(np.abs(start - (star - sysvel)) < 100)
    box = vb.read(str(tmp_path), 'STAR')
    assert {e['by'] for e in box['spectra'].values()} == {'ccf'}
    again, sysvel2 = ts.start_velocities(
        Inst(), files, None, None, '', berv, np.zeros(3), 'STAR',
        str(tmp_path), '', lambda x, **k: x)
    assert len(read) == 3 and sysvel2 == sysvel
    assert np.allclose(again, start)


def test_known_orbits_win_over_a_ccf_not_over_the_template(tmp_path):
    vb.update(str(tmp_path), 'S', {'a.fits': dict(v=-100.0),
                                   'b.fits': dict(v=-300.0)}, 'ccf',
              systemic=dict(v=-200.0, by='ccf', n=2))
    vb.update(str(tmp_path), 'S', {'b.fits': dict(v=-250.0, err=2.0)},
              'template')

    class Inst:
        params = dict()

    orbits = np.array([5.0, -5.0])
    start, _ = ts.start_velocities(Inst(), ['a.fits', 'b.fits'], None, None,
                                   '', np.zeros(2), orbits, 'S',
                                   str(tmp_path), '', lambda x, **k: x)
    assert list(start) == [5.0, -50.0]
