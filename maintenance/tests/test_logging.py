#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The log file of a run: plain text, every level, and still the one written to
after an LblException (which must not swap the run's log for ~/lbl).

Created on 2026-11-02

@author: artigau
"""
import pytest

from lbl.core import base_classes, logger


@pytest.fixture
def run_log(tmp_path, monkeypatch):
    """a Log as lbl_misc.move_log makes it, in a temporary DATA_DIR/log"""
    path = tmp_path / 'LOG-test.log'
    log = logger.Log(filename=str(path))
    monkeypatch.setattr(base_classes, 'log', log)
    yield log, path
    for handler in list(log.logger.handlers):
        if handler.get_name() == 'file':
            handler.close()
            log.logger.removeHandler(handler)


def _flush(log):
    for handler in log.logger.handlers:
        handler.flush()


def test_log_file_is_plain_and_keeps_every_level(run_log):
    log, path = run_log
    # VERBOSE=0: the console shows warnings only, the file still all
    log.update_console(0)
    log.general('a general line')
    log.warning('a warning line')
    _flush(log)
    text = path.read_text()
    assert 'a general line' in text and 'a warning line' in text
    assert '\x1b[' not in text
    console = [h for h in log.logger.handlers if h.get_name() == 'console']
    assert console[0].level == log.WARNING
    log.update_console(2)


def test_exception_message_goes_to_the_run_log(run_log):
    log, path = run_log
    with pytest.raises(base_classes.LblException):
        raise base_classes.LblException('the exception message')
    # a caught one too, and what follows it
    try:
        raise base_classes.LblLowCCFSNR('a low snr')
    except base_classes.LblLowCCFSNR:
        pass
    log.general('after the exceptions')
    _flush(log)
    text = path.read_text()
    for line in ('the exception message', 'a low snr', 'after the exceptions'):
        assert line in text
    assert [h for h in log.logger.handlers if h.get_name() == 'file']
