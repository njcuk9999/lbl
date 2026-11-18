"""The correction of many exposures in threads (reconstruct._correct_threaded):
every file counted once, as it is done, whichever thread finishes first, so
that the bar moves with the first file finished rather than waiting on the
first of the list."""
import os
import time
import types

from lbl.pca2d import reconstruct


def test_files_counted_as_they_are_done(tmp_path, monkeypatch):
    paths = []
    for i in range(8):
        path = tmp_path / ("f%d.fits" % i)
        path.write_text("x")
        paths.append(str(path))
    done = []

    def correct_file(model, row, path, outdir, *args, **kwargs):
        # the first of the list is the slowest: the others finish first
        time.sleep(0.3 if path.endswith("f0.fits") else 0.01)
        done.append(os.path.basename(path))
        return path + ".out", 90, 100

    monkeypatch.setattr(reconstruct, "correct_file", correct_file)
    args = types.SimpleNamespace(refit=False, corrected_dir=str(tmp_path),
                                 n_star=0, n_earth=1, kernel_halfwidth=8,
                                 overwrite=True, max_sky_ratio=None)
    pairs = [(path, dict(bjd=0.0, berv=0.0)) for path in paths]
    pairs.append((str(tmp_path / "gone.fits"), dict(bjd=0.0, berv=0.0)))
    covered = []
    written, skipped, refitted = reconstruct._correct_threaded(
        dict(coeffs=None), args, None, pairs, shifter=None, mask_mode="none",
        alive_by_file=None, names_by_night=None, members=None,
        clipped_by_file=None, floored_by_file=None, column_cards=None,
        velocity_fwhm=None, covered=covered, workers=4)
    assert (written, skipped, refitted) == (8, 1, 0)
    assert len(covered) == 8 and covered == [90.0] * 8
    # f0 started first and finished last
    assert done[-1] == "f0.fits"


class _Args(types.SimpleNamespace):
    """the stage's arguments: what is not set is None"""

    def __getattr__(self, name):
        return None


def test_the_residuals_of_a_threaded_correction_are_written(tmp_path,
                                                             monkeypatch):
    """correct_many in threads hands every file to write_residuals (it once
    emptied its list of files after the threads, so that save_residuals
    wrote none)"""
    names = ['f%d.fits' % i for i in range(5)]
    for name in names:
        (tmp_path / name).write_text('x')
    given = []
    monkeypatch.setattr(reconstruct, '_correct_threaded',
                        lambda model, args, config, pairs, *a, **k:
                        (len(pairs), 0, 0))
    monkeypatch.setattr(reconstruct, 'write_residuals',
                        lambda model, args, config, paths:
                        given.extend(paths) or len(paths))
    args = _Args(all=True, by_night=False, source_dir=str(tmp_path),
                 refit=False, weight='flux', mask='none', workers=4,
                 fits=str(tmp_path / 'components.fits'),
                 corrected_dir=str(tmp_path / 'out'),
                 residuals_dir=str(tmp_path / 'residuals'))
    model = dict(coeffs=dict(filename=names), n_star=1, n_earth=2)
    reconstruct.correct_many(model, args)
    assert [os.path.basename(p) for p in given] == names
