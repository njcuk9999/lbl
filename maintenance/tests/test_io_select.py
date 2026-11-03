#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small things of the input and output layer: yaml keys in any case, a list of
header keys skips an empty one, and messages that do not raise on their own.

Created on 2026-11-02

@author: artigau
"""
import pytest

from lbl.core import base_classes, io
from lbl.instruments import select


def test_lower_case_yaml_keys(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('instrument: HARPS\ndata_source: ORIG\nsnr_threshold: 12\n')
    inputs = select.parse_args(['INSTRUMENT', 'CONFIG_FILE'],
                               dict(config_file=str(path)), 'test',
                               parse=False)
    assert inputs['INSTRUMENT'] == 'HARPS'
    assert inputs['SNR_THRESHOLD'] == 12


def test_header_key_list_skips_an_empty_value():
    header = io.LBLHeader()
    header['KEY1'] = ''
    header['KEY2'] = 'None'
    header['KEY3'] = 5.0
    assert header.get_hkey(['KEY1', 'KEY2', 'KEY3'], dtype=float) == 5.0


def test_missing_file_without_a_kind(tmp_path):
    with pytest.raises(base_classes.LblException, match='cannot be found'):
        io.check_file_exists(str(tmp_path / 'nothing.fits'))


def test_missing_table_message(tmp_path):
    with pytest.raises(base_classes.LblException, match='csv table'):
        io.load_table(str(tmp_path / 'nothing.csv'), fmt='csv')


def test_download_message_names_the_file(tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(io.wget, 'download',
                        lambda url, path: open(path, 'w').close())
    monkeypatch.setattr(io.log, 'general', messages.append)
    io.get_urlfile('http://example/x.fits', 'PHOENIX model',
                   str(tmp_path / 'x.fits'))
    assert any('Downloaded PHOENIX model file' in m for m in messages)
    assert not any('tapas' in m for m in messages)
