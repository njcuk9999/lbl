"""The river plots' rows (lbl.pca2d.figures.sequence.river_rows) and the
BERV of the bundles figure (lbl.pca2d.cube.member_berv): both by the
observer's BERV, so that the green line, which follows it, always runs one
way down the page, a grid already barycentric (BERV applied 0) included."""
import numpy as np
from astropy import constants

from lbl.pca2d.cube import member_berv
from lbl.pca2d.figures.sequence import berv_track, river_rows


def test_the_rows_by_the_observer_s_berv_on_a_barycentric_grid():
    rng = np.random.default_rng(4)
    # ESO's S2D: the BERV the fit applies is 0 for every row, the
    #   observer's (LBL's profile) is not; the cube's rows in time order
    berv_obs = rng.uniform(-4.2, 2.4, 46)
    keep = np.ones(46, dtype=bool)
    keep[[3, 17]] = False
    rows = river_rows(keep, berv_obs)
    assert set(rows) == set(np.where(keep)[0])
    track = berv_track(1187.8, berv_obs[rows])
    # the green line, one way all down the page
    assert np.all(np.diff(track) >= 0) or np.all(np.diff(track) <= 0)
    assert np.all(np.diff(berv_obs[rows]) >= 0)


def test_the_bundles_berv_is_the_observer_s():
    # a barycentric grid: applied 0, the observer's from the profile
    assert member_berv(dict(berv=0.0, berv_obs=2.41)) == 2.41
    # without the observer's: the applied one
    assert member_berv(dict(berv=12.3)) == 12.3
    # known orbits: the star's frame, composed relativistically
    c = constants.c.to("km/s").value
    v = member_berv(dict(berv=-0.0, berv_obs=20.0, vstar=0.01))
    assert v == (20.0 - 0.01) / (1.0 - 20.0 * 0.01 / c ** 2)
