#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The plots a run shows as it goes (lbl.science.liveplot): saved for the
window that watches the run instead of shown in windows that wait to be
closed, the first file's only for a plot drawn per file, listed in order;
and the window's side (lbl.gui.runner: the folder given to the run, the
plots in its summary).

Created on 2026-11-18

@author: artigau
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

from lbl.science import liveplot                                # noqa: E402


class _Inst:
    """the parameters a plot reads"""

    def __init__(self, plot=False):
        self.name = 'FAKE'
        self.params = dict(PLOT=plot, PLOT_COMPUTE_CCF=True,
                           PLOT_MASK_CCF=True, OBJECT_SCIENCE='STAR',
                           OBJECT_COMPARISON='STAR', BERV=0.0,
                           COMPIL_SLOPE_REF_WAVE=1600.0)


def test_without_a_window_nothing_changes(monkeypatch):
    monkeypatch.delenv(liveplot.ENV, raising=False)
    assert not liveplot.active()
    assert liveplot.claim('anything') and liveplot.claim('anything')
    shown = []
    monkeypatch.setattr(plt, 'show', lambda *a, **k: shown.append(1))
    plt.figure()
    liveplot.finish(plt, 'k', 'title')
    assert shown == [1]
    plt.close('all')


def test_a_watched_run_saves_and_lists_its_plots(tmp_path, monkeypatch):
    folder = tmp_path / 'plots'
    monkeypatch.setenv(liveplot.ENV, str(folder))
    monkeypatch.setattr(plt, 'show', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('a watched run never opens a window')))
    assert liveplot.claim('compute_ccf_STAR')
    assert not liveplot.claim('compute_ccf_STAR')
    plt.figure(); plt.plot([0, 1], [0, 1])
    liveplot.finish(plt, 'template_STAR', 'template', 'template', 'STAR')
    fig = plt.figure(); plt.plot([0, 1], [1, 0])
    liveplot.save(fig, 'mask_ccf_STAR', 'mask CCF', 'mask', 'STAR')
    listed = liveplot.read_index(str(folder))
    assert [row['title'] for row in listed] == ['template', 'mask CCF']
    assert [row['file'] for row in listed] == ['01_template_STAR.pdf',
                                               '02_mask_ccf_STAR.pdf']
    assert all((folder / row['file']).exists() for row in listed)


def test_lbl_s_plots_go_to_the_window_first_file_only(tmp_path, monkeypatch):
    """PLOT off: no plot without a window; with one, the CCF of the first
    spectrum only, and no window opened"""
    from lbl.science import plot
    monkeypatch.setattr(plt, 'show', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('no window')))
    dv = np.linspace(-20000, 20000, 81)
    ccf = 1 - np.exp(-0.5 * (dv / 3000) ** 2)
    monkeypatch.delenv(liveplot.ENV, raising=False)
    plot.compute_plot_ccf(_Inst(plot=False), dv, ccf, ccf, [0.0, 3000.0])
    folder = tmp_path / 'plots'
    monkeypatch.setenv(liveplot.ENV, str(folder))
    for _ in range(3):
        plot.compute_plot_ccf(_Inst(plot=False), dv, ccf, ccf, [0.0, 3000.0])
    plot.mask_plot_ccf(_Inst(plot=False), dv / 1000, ccf, 0.0)
    listed = liveplot.read_index(str(folder))
    assert [row['step'] for row in listed] == ['compute', 'mask']
    # the template's own plots
    grid = np.linspace(1590.0, 1610.0, 4001)
    flux = 1 - 0.3 * np.exp(-0.5 * ((grid - 1600.0) / 0.01) ** 2)
    plot.template_plot(_Inst(), grid, flux, flux * 0.01, 'STAR')
    x = np.log(np.linspace(1599.0, 1601.0, 3000))
    plot.template_velocity_plot(_Inst(), np.array([1.0, -1.0, 2.0]),
                                np.arange(3), 'spectrum', [12.0, 3.0],
                                x, np.ones(x.size), x, np.ones(x.size),
                                'STAR')
    listed = liveplot.read_index(str(folder))
    assert [row['title'] for row in listed][-2:] == [
        'template', 'template velocities, the spectra registered']


def test_the_window_gives_the_folder_and_reads_the_plots(tmp_path):
    from lbl.gui import runner
    run = tmp_path / 'gui' / 'runs' / 'r1'
    run.mkdir(parents=True)
    assert runner.plots_of(str(run)) == []
    folder = runner.plots_folder(str(run))
    os.makedirs(folder)
    (run / 'plots' / '01_template_STAR.pdf').write_bytes(b'%PDF-1.4')
    with open(os.path.join(folder, 'index.json'), 'w') as handle:
        json.dump([dict(file='01_template_STAR.pdf', title='template',
                        step='template', object='STAR'),
                   dict(file='02_gone.pdf', title='gone')], handle)
    found = runner.plots_of(str(run))
    assert len(found) == 1 and found[0]['title'] == 'template'
    assert found[0]['path'].endswith('01_template_STAR.pdf')
    job = runner.Job('r1', 'r1', str(run), str(run / 'wrap.py'), 'python',
                     str(tmp_path), dict())
    assert job.summary()['plots'] == found
