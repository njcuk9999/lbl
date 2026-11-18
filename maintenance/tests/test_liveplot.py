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
                                np.array([2.0, 2.5, 1.5]),
                                np.array([60100.1, 60101.2, 60150.3]),
                                'MJD (KW_MID_EXP_TIME)', [12.0, 3.0],
                                x, np.ones(x.size), x, np.ones(x.size),
                                'STAR')
    # the registered pixels coloured by their spectrum's epoch
    plot.template_velocity_plot(_Inst(), np.array([1.0, -1.0]),
                                np.array([2.0, 2.5]),
                                np.array([60100.1, 60101.2]), 'MJD', [3.0],
                                np.r_[x, x], np.ones(2 * x.size), x,
                                np.ones(x.size), 'OTHER',
                                cloud_c=np.r_[np.full(x.size, 60100.1),
                                              np.full(x.size, 60101.2)])
    listed = liveplot.read_index(str(folder))
    assert [row['title'] for row in listed][-3:] == [
        'template', 'template velocities, the spectra registered',
        'template velocities, the spectra registered']


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


def test_the_template_s_velocities_have_times_and_errors():
    """the table of spectra's MJD (LBL's keyword names), a BJD brought back
    to an MJD, and each spectrum's velocity in the log"""
    from lbl.science import template_scatter as ts
    times, label = ts._times(dict(KW_MID_EXP_TIME=[60100.5, 60101.5]), 2)
    assert list(times) == [60100.5, 60101.5] and 'KW_MID_EXP_TIME' in label
    times, _ = ts._times(dict(BJD=[2460100.5, 2460101.0]), 2)
    assert np.allclose(times, [60100.0, 60100.5])
    times, label = ts._times(dict(FILENAME=['a', 'b']), 2)
    assert not np.isfinite(times).any() and label == ''
    said = []
    import lbl.science.template_scatter as module
    original = module.log.general
    module.log.general = lambda message: said.append(message)
    try:
        ts._log_velocities(dict(FILENAME=['/x/a.fits', '/x/b.fits']),
                           np.array([10.0, 14.0]), np.array([1.5, 2.0]),
                           np.array([60100.5, np.nan]), 'MJD')
    finally:
        module.log.general = original
    first = said[1].split()
    assert first[:4] == ['1/2', 'a.fits', 'MJD', '60100.50000']
    assert first[4:] == ['-2.00', '+-', '1.50', 'm/s']
    assert 'b.fits' in said[2] and '+2.00' in said[2]
    assert 'rms 2.00 m/s' in said[-1]


def test_the_lines_plot_is_divided_by_the_order_s_median(tmp_path,
                                                         monkeypatch):
    """compute's plot of the lines: the template, the lines and their
    residuals divided by the order's median flux"""
    from lbl.science import plot
    monkeypatch.setenv(liveplot.ENV, str(tmp_path / 'plots'))
    drawn = []
    monkeypatch.setattr(liveplot, 'finish',
                        lambda plt_, *a, **k: drawn.append(plt_.gcf()))
    wave = np.linspace(1600.0, 1610.0, 500)[None, :]
    model = 1000.0 * (1 - 0.3 * np.exp(-0.5 * ((wave - 1605.0) / 0.02) ** 2))
    plot_dict = dict(WAVEGRID=wave, MODEL=model, PLOT_ORDERS=0,
                     LINE_ORDERS=[0], WW_ORD_LINE=[wave[0, 240:260]],
                     SPEC_ORD_LINE=[model[0, 240:260] * 1.01],
                     MODEL_ORD_LINE=[model[0, 240:260]])
    inst = _Inst()
    inst.params['PLOT_COMPUTE_LINES'] = True
    plot.compute_line_plot(inst, plot_dict)
    top, bottom = drawn[0].axes
    assert np.isclose(np.median(top.lines[0].get_ydata()), 1.0, atol=0.01)
    assert np.all(np.abs(top.lines[1].get_ydata()) < 1.1)
    assert np.all(np.abs(bottom.lines[0].get_ydata()) < 0.02)
    assert 'median of the order' in top.get_ylabel()


def test_the_pca2d_fit_plot_once_per_fit(tmp_path, monkeypatch):
    """PCA2D's fit: its eigenvalues and the correlations of its components
    with the observing conditions, one plot per fit (a second pass gets its
    own), nothing without a window"""
    from lbl.pca2d import twoframe
    rng = np.random.default_rng(0)
    comps = [('b1', rng.normal(size=40)), ('b2', rng.normal(size=40))]
    rho = np.array([[0.7, -0.1], [0.05, -0.5]])
    outdir = tmp_path / 'STAR' / 's1-o7'
    outdir.mkdir(parents=True)
    monkeypatch.delenv(liveplot.ENV, raising=False)
    twoframe.live_fit_plot(str(outdir), [], [5.0, 1.0], 100.0, rho, comps,
                           ['BERV', 'airmass'])
    folder = tmp_path / 'plots'
    monkeypatch.setenv(liveplot.ENV, str(folder))
    for _ in range(2):
        twoframe.live_fit_plot(str(outdir), [], [5.0, 1.0], 100.0, rho,
                               comps, ['BERV', 'airmass'])
    titles = [row['title'] for row in liveplot.read_index(str(folder))]
    assert titles == ['PCA2D fit: eigenvalues, correlations with the'
                      ' conditions', 'PCA2D fit: eigenvalues, correlations'
                      ' with the conditions, pass 2']


def test_an_injected_signal_is_drawn_over_the_velocities(tmp_path,
                                                         monkeypatch):
    """the velocities against MJD with the injection test's signals drawn
    over them"""
    from lbl.science import plot
    monkeypatch.setenv(liveplot.ENV, str(tmp_path / 'plots'))
    drawn = []
    monkeypatch.setattr(liveplot, 'finish',
                        lambda plt_, *a, **k: drawn.append(plt_.gcf()))
    curve = (np.linspace(60100.0, 60300.0, 50), np.sin(np.arange(50.0)),
             'injected: K = 10 m/s, P = 200 d, phase 0')
    plot.template_velocity_plot(_Inst(), np.array([1.0, -1.0]),
                                np.array([2.0, 2.5]),
                                np.array([60100.1, 60201.2]), 'MJD', [3.0],
                                np.array([]), np.array([]), np.array([]),
                                np.array([]), 'STAR', injected=curve)
    labels = [line.get_label() for line in drawn[0].axes[1].lines]
    assert any(label.startswith('injected: K = 10') for label in labels)
