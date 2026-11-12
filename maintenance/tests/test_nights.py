"""One definition of an observing night for all of LBL
(lbl.core.astro.night_number): a night runs from local mean noon to the next
local mean noon at the observatory, the site the instrument's class names in
EARTH_LOCATION (lbl.core.astro.observatory_longitude). The compile's epochs,
PCA2D's coadds, its correct stage and holdout, the reports and the window
agree on which exposures are one night.

    PYTHONPATH=<this checkout> python -m pytest maintenance/tests/test_nights.py
"""
import datetime as dt
import os

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.table import Table

from lbl.core import astro, base_classes, parameters
from lbl.core.astro import night_number, observatory_longitude
from lbl.gui import targets
from lbl.instruments import select
from lbl.pca2d import config as spconfig
from lbl.pca2d import diagnostics, lblscan, nights, reconstruct
from lbl.report import series, stats

#: NIRPS, HARPS, CORALIE: La Silla; SPIRou, ESPaDOnS: Mauna Kea (CFHT)
LA_SILLA = observatory_longitude("NIRPS_HE")
MAUNA_KEA = observatory_longitude("SPIROU")
SECOND = 1.0 / 86400.0

#: a Chilean night: 23:30 UT on 2024-10-11 and 01:30 and 08:00 UT the next day,
#:    then the next night's first exposure at 23:40 UT
STAMPS = [dt.datetime(2024, 10, 11, 23, 30), dt.datetime(2024, 10, 12, 1, 30),
          dt.datetime(2024, 10, 12, 8, 0), dt.datetime(2024, 10, 12, 23, 40)]
#: a night on Mauna Kea: 05:30 UT (19:30 HST) to 15:00 UT (05:00 HST) the same
#:    UT day, across noon UT (02:00 HST), then the next night at 05:40 UT
HAWAII = [dt.datetime(2024, 10, 12, 5, 30), dt.datetime(2024, 10, 12, 11, 50),
          dt.datetime(2024, 10, 12, 12, 10), dt.datetime(2024, 10, 12, 15, 0),
          dt.datetime(2024, 10, 13, 5, 40)]


def _jd(stamp):
    return 2451545.0 + (stamp - dt.datetime(2000, 1, 1, 12, 0)).total_seconds() / 86400


JD = np.array([_jd(s) for s in STAMPS])
JD_HAWAII = np.array([_jd(s) for s in HAWAII])


def test_the_sites_are_those_of_the_instruments():
    assert LA_SILLA == pytest.approx(-70.73, abs=0.01)
    assert MAUNA_KEA == pytest.approx(-155.47, abs=0.01)
    # as a header names it, matched by family, and from the parameters
    assert observatory_longitude("NIRPS") == LA_SILLA
    assert observatory_longitude("SPIRou") == MAUNA_KEA
    assert observatory_longitude("HARPN") == observatory_longitude("HARPSN")
    assert observatory_longitude(dict(EARTH_LOCATION="cfht")) == MAUNA_KEA
    # a Generic instrument has no site but the one its parameters give
    with pytest.raises(ValueError, match="EARTH_LOCATION"):
        observatory_longitude("Generic")
    with pytest.raises(ValueError, match="no class"):
        observatory_longitude("NOT_AN_INSTRUMENT")
    with pytest.raises(ValueError):
        observatory_longitude(None)


@pytest.mark.parametrize("longitude", [-70.73, -155.47, 5.71, 0.0, -17.88,
                                       179.9, -179.9])
def test_the_night_changes_at_local_mean_noon(longitude):
    noon = 2460600.0 - longitude / 360.0        # JD at local mean noon
    assert night_number(noon - SECOND, longitude) + 1 \
        == night_number(noon + SECOND, longitude)
    # local midnight, half a day later, is inside the night
    assert night_number(noon + 0.4, longitude) \
        == night_number(noon + 0.6, longitude) \
        == night_number(noon + SECOND, longitude) == 2460600
    # a longitude written east of 180 cuts the same nights
    assert night_number(noon + 0.3, longitude + 360.0) == 2460600


def test_the_number_is_the_julian_day_of_the_local_date():
    # an rjd and an MJD + 0.5 give the boundary of the JD
    assert list(night_number(JD - 2400000.0, LA_SILLA)) == \
        list(night_number(JD, LA_SILLA) - 2400000)
    assert list(night_number(JD - 2400000.5 + 0.5, LA_SILLA)) == \
        list(night_number(JD, LA_SILLA) - 2400000)
    # the Chilean night that began on 2024-10-11: JD 2460595 at its noon UT
    assert list(night_number(JD, LA_SILLA)) == [2460595] * 3 + [2460596]
    with pytest.raises(ValueError):
        night_number(JD, None)


def test_mauna_kea_nights_are_no_longer_cut_in_two():
    numbers = night_number(JD_HAWAII, MAUNA_KEA)
    assert len(set(numbers[:4])) == 1 and numbers[4] == numbers[0] + 1
    # noon UT, the old boundary, cut it between 11:50 and 12:10 UT
    assert np.floor(JD_HAWAII[1]) != np.floor(JD_HAWAII[2])
    labels = [nights.night_label(s, MAUNA_KEA) for s in HAWAII]
    assert labels == ["2024-10-11"] * 4 + ["2024-10-12"]


def test_the_cube_s_labels_and_the_julian_nights_agree():
    labels = [nights.night_label(s, LA_SILLA) for s in STAMPS]
    assert labels == ["2024-10-11"] * 3 + ["2024-10-12"]
    numbers = night_number(JD, LA_SILLA)
    assert len(set(numbers[:3])) == 1 and numbers[3] == numbers[0] + 1
    # the label is the date of the Julian day number, to the second on either
    #   side of local noon, at any longitude
    rng = np.random.default_rng(3)
    for longitude in (LA_SILLA, MAUNA_KEA, 5.71, 120.0):
        noon = dt.datetime(2024, 3, 1, 12) - dt.timedelta(hours=longitude / 15)
        stamps = [noon + dt.timedelta(seconds=s) for s in (-1, 1)]
        stamps += [noon + dt.timedelta(hours=float(h))
                   for h in rng.uniform(-60, 60, 50)]
        for stamp in stamps:
            day = int(night_number(_jd(stamp), longitude))
            date = dt.date(2000, 1, 1) + dt.timedelta(days=day - 2451545)
            assert nights.night_label(stamp, longitude) == date.isoformat()


def _instrument(name, source):
    args = base_classes.ParamDict()
    args.set("INSTRUMENT", name, source="test")
    args.set("DATA_SOURCE", source, source="test")
    return select.InstDict[name][source](parameters.params.copy(), args)


CLASSES = [(name, source) for name, sources in select.InstDict.items()
           for source in sources if name != "Generic"]


@pytest.mark.parametrize("name,source", CLASSES)
def test_the_compile_s_epochs_are_these_nights(name, source):
    """Instrument.get_epoch_groups (math.bin_by_time) and night_number cut
    the same nights, for every class: a second on either side of local noon
    is two nights, and random times fall in the same groups."""
    inst = _instrument(name, source)
    longitude = observatory_longitude(inst)
    assert longitude == observatory_longitude(name, source)
    noon = 60000.5 - longitude / 360.0 + np.arange(-3, 4)  # MJD, local noon
    rng = np.random.default_rng(7)
    mjd = np.concatenate([noon - SECOND, noon + SECOND,
                          rng.uniform(59990.0, 60010.0, 400)])
    _, epochs = inst.get_epoch_groups(Table({inst.params["KW_MJDATE"]: mjd}))
    numbers = night_number(mjd + 0.5, longitude)
    pairs = set(zip(np.round(epochs, 6), numbers))
    assert len(pairs) == len(set(numbers)) == len(set(np.round(epochs, 6)))
    # the second before local noon and the second after: two nights
    assert np.all(night_number(noon - SECOND + 0.5, longitude) + 1
                  == night_number(noon + SECOND + 0.5, longitude))


def test_a_night_across_midnight_ut_is_one_night_everywhere():
    coeffs = Table(dict(bjd=JD, b1=[0.1, 0.2, 0.3, 0.4]))
    index = reconstruct.night_index(coeffs, LA_SILLA)
    assert len(index) == 2
    assert len(set(diagnostics.night_of(JD[:3], LA_SILLA))) == 1
    held = diagnostics.holdout_rows(JD, "even", LA_SILLA)
    assert held[0] == held[1] == held[2] != held[3]
    rjd = JD - 2400000.0
    binned = lblscan.nightly(rjd, np.zeros(4), np.ones(4), LA_SILLA)
    assert binned.shape[1] == 2
    table = {series.TIME: rjd, series.VELOCITY: np.zeros(4),
             series.ERROR: np.ones(4)}
    rows = {label: text for label, text, _ in stats.summarise(table, LA_SILLA)}
    assert rows["Nights"] == "2"
    # without the observatory the summary counts no night, at no boundary
    rows = {label: text for label, text, _ in stats.summarise(table, None)}
    assert "Nights" not in rows and "Nightly rms" not in rows
    records = {str(i): dict(mjd=float(r - 0.5), instrument="NIRPS", berv=0.0)
               for i, r in enumerate(rjd)}
    assert targets.summary(records)["nights"] == 2
    assert len(set(targets.points(records)["night"])) == 2


def test_a_mauna_kea_night_is_one_night_everywhere():
    rjd = JD_HAWAII - 2400000.0
    assert len(reconstruct.night_index(Table(dict(bjd=JD_HAWAII)),
                                       MAUNA_KEA)) == 2
    held = diagnostics.holdout_rows(JD_HAWAII, "odd", MAUNA_KEA)
    assert len(set(held[:4])) == 1 and held[4] != held[0]
    assert lblscan.nightly(rjd, np.zeros(5), np.ones(5),
                           MAUNA_KEA).shape[1] == 2
    table = {series.TIME: rjd, series.VELOCITY: np.zeros(5),
             series.ERROR: np.ones(5)}
    rows = {label: text for label, text, _ in stats.summarise(table, MAUNA_KEA)}
    assert rows["Nights"] == "2"
    records = {str(i): dict(mjd=float(r - 0.5), instrument="SPIRou", berv=0.0)
               for i, r in enumerate(rjd)}
    assert targets.summary(records)["nights"] == 2
    # an instrument the window does not know: no night count rather than one
    #   at another boundary
    unknown = {k: dict(v, instrument="?") for k, v in records.items()}
    assert targets.summary(unknown)["nights"] is None
    assert set(targets.points(unknown)["night"]) == {None}


def test_the_configuration_names_the_observatory():
    assert nights.config_longitude({"lbl": {"instrument": "SPIROU",
                                            "data_source": "APERO"}}) \
        == MAUNA_KEA
    assert nights.config_longitude({"input": {"instrument": "HARPS"}}) \
        == LA_SILLA
    assert nights.config_longitude({"input": {"format": "tfits"}}) is None
    with pytest.raises(SystemExit, match="--config"):
        nights.run_longitude({"input": {}}, None, need="the test")
    assert nights.run_longitude({"input": {}}, None) is None


def test_stacking_needs_the_observatory(tmp_path):
    files = [str(tmp_path / ("SPIROU.2024-10-12T%02d:%02d:00.000t.fits" % hm))
             for hm in ((5, 30), (11, 50), (12, 10), (15, 0))]
    _, labels = nights.group_files(files, True, longitude=MAUNA_KEA)
    assert labels == ["2024-10-11"] * 4
    with pytest.raises(RuntimeError, match="local noon"):
        nights.group_files(files, True)
    # one row per file: no night, no observatory needed
    _, labels = nights.group_files(files, False)
    assert len(set(labels)) == 4


def test_the_cube_records_its_nights_and_a_stage_checks_them(tmp_path):
    cube = tmp_path / "cube"
    cube.mkdir()
    meta = Table(dict(filename=["a.fits"], bjd=[2460600.6], n_exposures=[2]))
    nights.record_longitude(meta, {"lbl": {"instrument": "NIRPS_HE"}})
    meta.write(str(cube / "meta.fits"))
    assert nights.cube_longitude(str(cube)) == pytest.approx(LA_SILLA)
    # the cube alone says it
    assert nights.run_longitude(None, str(cube), need="x") \
        == pytest.approx(LA_SILLA)
    # another observatory than the cube's stops
    with pytest.raises(SystemExit, match="not the same observatory"):
        nights.run_longitude({"input": {"instrument": "SPIROU"}}, str(cube))
    # a cube without the record, coadded at noon UT by an older LBL
    old = tmp_path / "old"
    old.mkdir()
    Table(dict(filename=["a.fits"], bjd=[2460600.6], n_exposures=[2])).write(
        str(old / "meta.fits"))
    assert nights.cube_longitude(str(old)) is None
    with pytest.raises(SystemExit):
        nights.run_longitude(None, str(old), need="x")


def _nominal(instrument, data_source):
    path = os.path.join(os.path.dirname(spconfig.__file__), "nominal.yaml")
    return spconfig.load_config(path, instrument=instrument,
                                data_source=data_source)


def test_the_night_boundary_decides_which_cube_is_reused(monkeypatch):
    """A cube coadded at noon UT is not taken for one coadded at local noon,
    nor one of La Silla for one of Mauna Kea; a cube of single exposures
    that picks no night is the same either way, and keeps its key."""
    cfg = _nominal("NIRPS_HE", "CADC")
    cfg["input"]["nightly_stack"] = "auto"
    key = spconfig.cache_key(cfg)
    monkeypatch.setattr(nights, "config_longitude", lambda config: 0.0)
    assert spconfig.cache_key(cfg) != key
    monkeypatch.undo()
    assert spconfig.cache_key(cfg) == key
    cfg["input"]["nightly_stack"] = False
    single = spconfig.cache_key(cfg)
    monkeypatch.setattr(nights, "config_longitude", lambda config: 0.0)
    assert spconfig.cache_key(cfg) == single
    # a list of nights is a list of local nights
    cfg["quality"]["nights"] = [60600]
    shifted = spconfig.cache_key(cfg)
    monkeypatch.undo()
    assert spconfig.cache_key(cfg) != shifted


def test_without_astropy_s_site_registry_it_says_what_is_missing(
        tmp_path, monkeypatch):
    """Offline and with an empty cache, the site cannot be had: it stops and
    says so, it never falls back to noon UT."""
    from astropy.config import paths
    from astropy.utils.data import conf
    monkeypatch.setattr(astro, "_SITE_LONGITUDES", {})
    monkeypatch.setattr(EarthLocation, "_site_registry", None)
    with paths.set_temp_cache(str(tmp_path)), \
            conf.set_temp("allow_internet", False):
        with pytest.raises(ValueError, match="sites.json"):
            astro.site_longitude("La Silla Observatory")


def test_an_archive_product_s_date_is_not_the_exposure_s():
    # the date in an ESO archive product's name is when the archive
    #   processed it: taken, every exposure would be one night
    for name in ("r.NIRPS.ADP.2025-04-11T12-47-12.315_S2D_BLAZE_TCORR_A.fits",
                 "ADP.2025-04-11T12:47:12.315.fits",
                 "ADP.2025-04-11T12_47_12.315.fits"):
        assert nights.timestamp_from_name(name) is None
    for name in ("NIRPS.2023-01-20T08:42:08.941t.fits",
                 "r.NIRPS.2023-01-20T08_42_08.941_S2D_A.fits"):
        assert nights.timestamp_from_name(name) == dt.datetime(
            2023, 1, 20, 8, 42, 8)


def test_the_time_of_an_s2d_comes_from_its_primary_header(tmp_path):
    """ESO's S2D: DATE-OBS in the primary header alone, the flux in
    extension 1, the name of the archive search (no time in it)"""
    from astropy.io import fits

    def s2d(name, primary):
        head = fits.Header()
        head.update(primary)
        path = str(tmp_path / name)
        fits.HDUList([fits.PrimaryHDU(header=head), fits.ImageHDU(
            np.ones((2, 4), "f4"), name="SCIDATA")]).writeto(path)
        return path
    files = [s2d("r.NIRPS.ADP.2025-04-11T12-47-12.3%02d_S2D_A.fits" % k,
                 {"DATE-OBS": stamp.isoformat()})
             for k, stamp in enumerate(STAMPS)]
    _, labels = nights.group_files(files, True, extension=1,
                                   longitude=LA_SILLA)
    assert len(set(labels)) == 2
    # MJD-OBS when there is no DATE-OBS
    mjd = s2d("r.NIRPS.ADP.x_S2D_A.fits", {"MJD-OBS": 60041.25})
    assert nights._stamp_from_header(mjd, 1) == dt.datetime(2023, 4, 7, 6)
    bare = s2d("r.NIRPS.ADP.y_S2D_A.fits", {})
    with pytest.raises(RuntimeError, match="DATE-OBS or MJD-OBS"):
        nights.group_files([bare], True, extension=1, longitude=LA_SILLA)
