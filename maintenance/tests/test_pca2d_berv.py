"""PCA2D's BERV, from LBL's instrument profile only (2026-10-27): the one
it applies (get_berv, zero on a barycentric grid) and the observer's, which
the telluric lines follow (get_berv + get_berv_transmission), for the
river plots' green line, the observer's frame and the bundles of a night."""
import numpy as np
import pytest
from astropy.table import Table

from lbl.pca2d import nights, readers
from lbl.pca2d.figures import sequence


class _Profile:
    """An instrument profile: what is left to apply, and the tellurics'"""
    def __init__(self, applied, transmission):
        self.applied, self.transmission = applied, transmission

    def get_berv(self, header):
        return self.applied

    def get_berv_transmission(self, header):
        return self.transmission


def test_the_observer_berv_whatever_frame_the_grid_is_in():
    # APERO's t.fits: the grid in the observer's frame
    assert readers.observer_berv(_Profile(13604.75, 0.0), {}) == 13.60475
    # the S2D of the TNG: barycentric, get_berv zero (HARPS-N ESO)
    assert readers.observer_berv(_Profile(0.0, -20573.36), {}) == -20.57336

    class Broken(_Profile):
        def get_berv(self, header):
            raise KeyError('BERV')
    assert np.isnan(readers.observer_berv(Broken(0, 0), {}))


def test_the_green_line_follows_the_observer_not_the_applied_berv():
    meta = Table(dict(berv=np.zeros(3), berv_obs=[-20.0, 0.5, 21.0]))
    assert list(sequence.observer_bervs(meta)) == [-20.0, 0.5, 21.0]
    # a cube made before berv_obs was kept: the applied BERV
    old = Table(dict(berv=[1.0, 2.0]))
    assert list(sequence.observer_bervs(old)) == [1.0, 2.0]


def test_the_bundles_of_a_night_take_the_profile_berv(tmp_path):
    files = [str(tmp_path / ('NIRPS.2024-08-24T06:%02d:00.000t.fits' % m))
             for m in range(4)]
    bervs = dict(zip(files, [10.0, 10.05, 10.5, 10.55]))
    _, labels = nights.group_files(files, True, bundle=dict(berv=0.1),
                                   berv_of=bervs.get)
    assert len(set(labels)) == 2
    # no profile to ask: stops, rather than read the headers itself
    with pytest.raises(RuntimeError, match='instrument'):
        nights.group_files(files, True, bundle=dict(berv=0.1))
