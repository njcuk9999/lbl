#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
What lbl_pca2d does for lbl_wrap's passes of PCA2D: an earlier pass's
correction deleted rather than moved aside, the passes recorded in the
correction's done.json, and LBL's velocities settled or not from one pass
to the next.

Created on 2026-11-25

@author: artigau
"""
import json
import os

import numpy as np

from lbl.recipes import lbl_pca2d, lbl_wrap


def _corrected(data_dir, name='STAR_pca2d'):
    """a corrected object and an LBL product made from it"""
    folder = os.path.join(data_dir, 'science', name)
    os.makedirs(folder)
    with open(os.path.join(folder, 'a.fits'), 'wb') as handle:
        handle.write(b'x' * 1000)
    rdb = os.path.join(data_dir, 'lblrdb')
    os.makedirs(rdb)
    with open(os.path.join(rdb, 'lbl_%s_%s.rdb' % (name, name)), 'w') as h:
        h.write('rdb')
    return folder


def test_an_earlier_pass_is_deleted_a_previous_run_moved_aside(tmp_path):
    said = []
    work = str(tmp_path / 'pca2d' / 'STAR')
    folder = _corrected(str(tmp_path))
    lbl_pca2d.supersede(str(tmp_path), folder, 'STAR_pca2d', work,
                        lambda *a: said.append(a), discard=True)
    assert not os.path.exists(folder)
    assert not os.listdir(str(tmp_path / 'lblrdb'))
    assert not os.path.exists(os.path.join(work, 'superseded'))
    assert 'deleted' in said[-1][0]
    os.rmdir(str(tmp_path / 'lblrdb'))
    folder = _corrected(str(tmp_path))
    lbl_pca2d.supersede(str(tmp_path), folder, 'STAR_pca2d', work,
                        lambda *a: said.append(a))
    assert not os.path.exists(folder)
    aside = os.listdir(os.path.join(work, 'superseded'))
    assert len(aside) == 1 and 'moved' in said[-1][0]


def test_the_passes_are_recorded_with_the_correction(tmp_path):
    state = tmp_path / 'done.json'
    state.write_text(json.dumps(dict(stamp='abc', first='abc', passes=None,
                                     settled=False)))
    lbl_pca2d.STATES['STAR_pca2d'] = str(state)
    lbl_pca2d.mark_passes(['STAR_pca2d', 'OTHER_pca2d'], 3, True)
    got = json.loads(state.read_text())
    assert got['passes'] == 3 and got['settled'] is True
    assert got['stamp'] == 'abc'


def test_lbl_s_velocities_settle_when_they_no_longer_move():
    """PASS_FRAC of the spectra within PASS_NSIG sigma of the previous
    pass's, the change of all of them aside"""
    rng = np.random.default_rng(1)
    names = ['f%d.fits' % i for i in range(40)]
    before = {n: (v, 2.0) for n, v in zip(names, rng.normal(0, 100, 40))}
    # all moved by 30 m/s (the template's frame), each by 0.3 sigma
    after = {n: (v + 30.0 + rng.normal(0, 0.6), 2.0)
             for n, (v, _) in before.items()}
    assert lbl_wrap._settled('STAR', before, after, 2)
    # a signal grown by 5 % (100 m/s, errors of 2 m/s): not settled
    grown = {n: (1.05 * v, 2.0) for n, (v, _) in before.items()}
    assert not lbl_wrap._settled('STAR', before, grown, 2)


def test_the_star_frame_is_lbl_s_velocities_less_their_median():
    frame = lbl_wrap._star_frame({'a': (1000.0, 1.0), 'b': (1100.0, 1.0),
                                  'c': (900.0, 1.0)})
    assert frame == {'a': 0.0, 'b': 0.1, 'c': -0.1}
    assert lbl_wrap._star_frame({'a': (1.0, 1.0)}) == {}


def test_the_star_frame_leaves_out_what_follows_the_berv():
    """a pattern proportional to the BERV is taken out (the confusion of
    the two frames), an orbit stays; plain floats (PCA2D's request is
    YAML)"""
    rng = np.random.default_rng(3)
    berv = np.linspace(-15000.0, 15000.0, 30)
    orbit = 100.0 * np.sin(np.arange(30) * 0.7)
    found = {'f%d' % i: (2000.0 + o + 0.01 * b, 2.0, b)
             for i, (o, b) in enumerate(zip(orbit, berv))}
    frame = lbl_wrap._star_frame(found)
    values = 1000.0 * np.array([frame['f%d' % i] for i in range(30)])
    assert abs(np.corrcoef(values, berv)[0, 1]) < 0.05
    assert np.corrcoef(values, orbit)[0, 1] > 0.99
    assert all(type(v) is float for v in frame.values())
    _ = rng
