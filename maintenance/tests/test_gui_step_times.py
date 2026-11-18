#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
How long each step of a run took, and what is left of a run going
(lbl.gui.runner.Job: started and ended on every step, timing): the steps
to come timed on the same kind already done in this run, else on the
earlier runs of the data folder; the step going on its counter.

Created on 2026-11-18

@author: artigau
"""
import json

from lbl.gui import runner


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _job(tmp_path, steps):
    folder = tmp_path / 'gui' / 'runs' / 'r2'
    folder.mkdir(parents=True)
    job = runner.Job('r2', 'r2', str(folder), str(folder / 'wrap.py'),
                     'python', str(tmp_path), dict(steps=steps))
    job.state = 'running'
    return job


def test_steps_are_timed_and_the_rest_estimated(tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(runner.time, 'time', clock)
    plan = [dict(kind='template', obj='STAR'), dict(kind='compute',
                                                    obj='STAR'),
            dict(kind='template', obj='STAR_pca2d'),
            dict(kind='compute', obj='STAR_pca2d')]
    job = _job(tmp_path, plan)
    job._advance('template', 'STAR')
    clock.now = 1092.0
    job._advance('compute', 'STAR')
    template = job.steps[0]
    assert template['state'] == 'done'
    assert template['ended'] - template['started'] == 92.0
    # compute going, a third of its files done in 60 s: 120 s left; then
    #   the corrected spectra's template like the first (92 s); their
    #   compute has no time yet
    clock.now = 1152.0
    job.steps[1]['progress'] = 1.0 / 3.0
    timing = job.timing()
    assert abs(timing['remaining'] - (120.0 + 92.0)) < 1e-6
    assert timing['unknown'] == 1
    assert abs(timing['end'] - (1152.0 + 212.0)) < 1e-6
    # compute done: it times the corrected spectra's compute too
    clock.now = 1300.0
    job._advance('template', 'STAR_pca2d')
    assert job.steps[1]['ended'] - job.steps[1]['started'] == 208.0
    timing = job.timing()
    assert timing['unknown'] == 0
    assert abs(timing['remaining'] - (92.0 + 208.0)) < 1e-6
    # kept for the runs to come
    job.record()
    with open(tmp_path / 'gui' / 'runs' / 'r2' / 'run.json') as handle:
        saved = json.load(handle)['steps']
    assert saved[0]['ended'] - saved[0]['started'] == 92.0


def test_the_earlier_runs_time_what_this_one_has_not_done(tmp_path,
                                                         monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(runner.time, 'time', clock)
    earlier = tmp_path / 'gui' / 'runs' / 'r1'
    earlier.mkdir(parents=True)
    with open(earlier / 'run.json', 'w') as handle:
        json.dump(dict(steps=[
            dict(kind='pca2d_fit', state='done', started=0.0, ended=300.0),
            dict(kind='pca2d_fit', state='done', started=0.0, ended=500.0),
            dict(kind='pca2d_fit', state='done', started=0.0, ended=400.0),
            dict(kind='compute', state='failed', started=0.0, ended=9.0)]),
            handle)
    job = _job(tmp_path, [dict(kind='pca2d_cube', obj='STAR'),
                          dict(kind='pca2d_fit', obj='STAR')])
    job._advance('pca2d_cube')
    timing = job.timing()
    # the fit: the median of the earlier runs' (400 s); the cube: none
    assert timing['remaining'] == 400.0 and timing['unknown'] == 1
    assert runner.step_durations(str(tmp_path / 'gui' / 'runs')) == {
        'pca2d_fit': [300.0, 500.0, 400.0]}
    job.state = 'done'
    assert job.timing() == {}
