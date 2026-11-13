"""
lbl.core.io.make_dir with several runs starting at once in one DATA_DIR: a
directory another run made between the check and the creation is not an
error (it stopped every wrap of a queue but the first with FileExistsError,
on rali, 2026-11-13).

    PYTHONPATH=<this checkout> python -m pytest maintenance/tests/test_io_make_dir.py
"""
from pathlib import Path

import pytest

from lbl.core import io
from lbl.core.base_classes import LblException


def test_a_directory_made_meanwhile_by_another_run_is_used(tmp_path,
                                                          monkeypatch):
    # another run made log/ just after this one looked
    (tmp_path / "log").mkdir()
    monkeypatch.setattr(Path, "exists", lambda self: False)
    assert io.make_dir(str(tmp_path), "log", "log") == str(tmp_path / "log")


def test_a_subdirectory_and_its_parent_made_meanwhile(tmp_path, monkeypatch):
    (tmp_path / "log" / "lbl_template").mkdir(parents=True)
    monkeypatch.setattr(Path, "exists", lambda self: False)
    out = io.make_dir(str(tmp_path), "log", "log", subdir="lbl_template")
    assert out == str(tmp_path / "log" / "lbl_template")


def test_it_makes_the_directory_and_its_parent(tmp_path):
    out = io.make_dir(str(tmp_path), "log", "log", subdir="lbl_template")
    assert Path(out).is_dir()


def test_a_data_dir_that_does_not_exist_is_still_an_error(tmp_path):
    with pytest.raises(LblException):
        io.make_dir(str(tmp_path / "nowhere"), "log", "log",
                    subdir="lbl_template")
