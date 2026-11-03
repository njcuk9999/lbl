#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lbl_reset removes LBL's products and nothing else (run on a temporary tree only);
lbl_demo stops at once while it has no address; lbl_setup's wrap file argument
is optional; the parameters say what they are.

Created on 2026-11-02

@author: artigau
"""
import sys

import pytest

from lbl.core import base_classes, io, parameters
from lbl.recipes import lbl_reset


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')
    return path


def test_reset_removes_derived_spectra_and_keeps_the_data(tmp_path,
                                                          monkeypatch):
    # the command line is pytest's: the recipe must not read it
    monkeypatch.setattr(sys, 'argv', ['lbl_reset'])
    # and the model files are not downloaded into the temporary tree
    from lbl.instruments import default
    monkeypatch.setattr(default.Instrument, 'get_model_files',
                        lambda *args, **kwargs: None)
    science = tmp_path / 'science'
    kept = [_touch(science / 'STAR' / 'a.fits'),
            _touch(science / 'TOI_tcx' / 'a.fits'),
            _touch(science / 'STAR' / 'b_tc.fits'),
            _touch(tmp_path / 'calib' / 'sample_wave_grid_harps.fits.bak'),
            _touch(tmp_path / 'calib' / 'harps.fits'),
            _touch(tmp_path / 'calib' / 'HARPS.blaze_A.fits')]
    gone = [_touch(science / 'STAR_tc' / 'a.fits'),
            _touch(science / 'STAR_pca2d' / 'a.fits'),
            _touch(science / 'STAR_tc_pca2d' / 'a.fits'),
            _touch(science / 'STAR_tc_pca2d_demo' / 'a.fits'),
            _touch(science / 'STAR_pca2d' / 'figures' / 'f.pdf'),
            _touch(tmp_path / 'templates' / 'Template_STAR_HARPS_ORIG.fits'),
            _touch(tmp_path / 'calib' / 'sample_wave_grid_harps_orig.fits')]
    args = dict(instrument='HARPS', data_source='ORIG',
                data_dir=str(tmp_path))
    lbl_reset.main(**args)
    for path in kept:
        assert path.exists(), path
    for path in gone:
        assert not path.exists(), path


def test_clean_directory_takes_one_name_as_a_name(tmp_path):
    """a string is not a list of substrings: 'grid.fits' is in
    'sample_wave_grid.fits' as a substring, and must stay"""
    keep = _touch(tmp_path / 'grid.fits')
    drop = _touch(tmp_path / 'sample_wave_grid.fits')
    io.clean_directory(str(tmp_path), include_files='sample_wave_grid.fits')
    assert keep.exists() and not drop.exists()


def test_demo_stops_without_an_address(monkeypatch):
    from lbl.recipes import lbl_demo

    def no_input(*_):
        raise AssertionError('asked for a directory')
    monkeypatch.setattr('builtins.input', no_input)
    with pytest.raises(base_classes.LblException, match='URL_TO_DATA'):
        lbl_demo.main()


def test_setup_wrap_file_is_optional(monkeypatch):
    from lbl.core import wrap_setup
    monkeypatch.setattr(sys, 'argv', ['lbl_setup'])
    args = wrap_setup.get_args(description='test')
    assert args['wrap_file'] == 'None'


def test_parameters_say_what_they_are():
    params = parameters.params
    assert params.instances['USE_TQDM'].dtype is bool
    assert params.instances['GENERIC_DATA_SOURCE'].options is None
    for key in ('SCIENCE_MASK_TYPE', 'FP_MASK_TYPE', 'LFC_MASK_TYPE'):
        assert params.instances[key].source is not None
    assert 'instrument' in params.instances['KW_INSTRUMENT'].description
    assert 'COMPUTE_MODEL_PLOT_ORDERS' in params.not_none
