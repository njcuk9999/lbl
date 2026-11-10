"""The Data Explorer tab (lbl.gui.explore): the tables, their error columns,
the template each was measured against, the TAPAS opacities, the FIP, and
the server refusing a table outside the data folders."""
import json
import os
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from lbl.gui import app, explore


def _rdb(path, n=40, period=7.0, amplitude=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rjd = np.sort(60000 + rng.uniform(0, 200, n))
    vrad = amplitude * np.sin(2 * np.pi * rjd / period) + rng.normal(0, 1, n)
    table = Table(dict(rjd=rjd, vrad=vrad, svrad=np.ones(n),
                       fwhm=rng.normal(5, 0.1, n),
                       sig_fwhm=np.full(n, 0.1),
                       BERV=np.linspace(-20, 25, n),
                       MJDMID=rjd - 0.5,
                       local_file_name=["f%02d.fits" % i for i in range(n)]))
    table["vrad"][3] = np.nan
    table.write(str(path), format="ascii.rdb", overwrite=True)
    return table


@pytest.fixture
def data(tmp_path):
    (tmp_path / "lblrdb").mkdir()
    (tmp_path / "templates").mkdir()
    (tmp_path / "models" / "template_SPIROU_CADC").mkdir(parents=True)
    (tmp_path / "models" / "tellurics").mkdir()
    rdb = tmp_path / "lblrdb" / "lbl_GJ_1_GJ_1.rdb"
    _rdb(rdb, amplitude=8.0)
    # the companion names the template; a _pca2d template of the same
    #   object must not be taken for it
    header = fits.Header(dict(LBL_OBJC="GJ_1", LBLINSTR="SPIROU",
                              LBLIMODE="CADC"))
    fits.PrimaryHDU(header=header).writeto(
        tmp_path / "lblrdb" / "lbl_GJ_1_GJ_1.fits")
    wave = np.exp(np.linspace(np.log(1000), np.log(1100), 20001))
    flux = 1 - 0.5 * np.exp(-0.5 * ((wave - 1050) / 0.01) ** 2)
    for name in ("LBL_Template_GJ_1_spirou.fits",
                 "LBL_Template_GJ_1_pca2d_spirou.fits"):
        Table(dict(wavelength=wave, flux=flux)).write(
            tmp_path / "templates" / name)
    Table(dict(wavelength=wave, flux=flux)).write(
        tmp_path / "models" / "template_SPIROU_CADC"
        / "LBL_Template_VETTED_GL1_spirou.fits")
    _rdb(tmp_path / "lblrdb" / "lbl_TOI_1_VETTED_GL1.rdb")
    return tmp_path


def test_the_tables_newest_first_and_the_window_opens_on_every_spectrum(data):
    folder = data / "lblrdb"
    _rdb(folder / "lbl2_GJ_1_GJ_1.rdb")
    now = os.path.getmtime(folder / "lbl2_GJ_1_GJ_1.rdb")
    os.utime(folder / "lbl_GJ_1_GJ_1.rdb", (now - 30, now - 30))
    os.utime(folder / "lbl_TOI_1_VETTED_GL1.rdb", (now - 3600, now - 3600))
    files = explore.rdb_files(str(data))
    assert [f["name"] for f in files["files"]][0] == "lbl2_GJ_1_GJ_1.rdb"
    # the nights' table is written just after that of every spectrum: the
    #   latest run opens on the latter
    assert files["latest"].endswith("lbl_GJ_1_GJ_1.rdb")


def test_a_table_its_errors_its_defaults_and_its_template(data):
    table = explore.read_rdb(str(data / "lblrdb" / "lbl_GJ_1_GJ_1.rdb"),
                             str(data))
    assert table["errors"]["vrad"] == "svrad"
    assert table["errors"]["fwhm"] == "sig_fwhm"
    assert "svrad" not in table["errors"]
    assert (table["x"], table["y"], table["c"]) == ("MJDMID", "vrad", "BERV")
    # JSON has no NaN: a missing value is null
    assert table["values"]["vrad"][3] is None
    json.dumps(table, allow_nan=False)
    assert table["labels"][0] == "f00.fits"
    assert table["template"]["name"] == "LBL_Template_GJ_1_spirou.fits"
    # no companion: the template from the table's name, a vetted one in
    #   models/template_*
    vetted = explore.template_of(
        str(data / "lblrdb" / "lbl_TOI_1_VETTED_GL1.rdb"), str(data))
    assert vetted["name"] == "LBL_Template_VETTED_GL1_spirou.fits"


def test_the_template_is_cut_to_what_is_shown(data):
    rdb = str(data / "lblrdb" / "lbl_GJ_1_GJ_1.rdb")
    wide = explore.template_view(rdb, str(data), nbin=500)
    # the min and max of each bin: the line's depth survives
    assert len(wide["x"]) == 1000 and min(wide["y"]) < 0.55
    narrow = explore.template_view(rdb, str(data), 1049.9, 1050.1, nbin=500)
    assert len(narrow["x"]) < 1000 and narrow["range"] == [1049.9, 1050.1]
    assert "tapas" not in wide


def test_tapas_per_molecule_the_instrument_s_site_first(data):
    folder = data / "models" / "tellurics"
    wave = np.linspace(1000, 1100, 100001)
    for site, depth in (("LaSilla", 0.5), ("MaunaKea", 0.9)):
        trans = 1 - depth * np.exp(-0.5 * ((wave - 1050) / 0.005) ** 2)
        Table(dict(wavelength=wave, H2O=trans, O2=np.ones_like(wave),
                   CH4=trans)).write(folder / ("%s_tapas.fits" % site))
    picked = explore.tapas_file(str(data), "SPIROU")
    assert os.path.basename(picked["path"]) == "MaunaKea_tapas.fits"
    view = explore.template_view(str(data / "lblrdb" / "lbl_GJ_1_GJ_1.rdb"),
                                 str(data), nbin=300)
    molecules = view["tapas"]["molecules"]
    # O2 has nothing there: left out; the opacity is -ln(transmission)
    assert set(molecules) == {"H2O", "CH4"}
    assert max(molecules["H2O"]["y"]) == pytest.approx(-np.log(0.1),
                                                       rel=1e-3)
    narrow = explore.template_view(str(data / "lblrdb"
                                        / "lbl_GJ_1_GJ_1.rdb"),
                                   str(data), 1049.99, 1050.01, nbin=300)
    assert max(narrow["tapas"]["molecules"]["H2O"]["y"]) == pytest.approx(
        -np.log(0.1), rel=1e-3)


def test_the_fip_finds_a_sinusoid_and_its_period(data):
    rdb = data / "lblrdb" / "lbl_S_S.rdb"
    _rdb(rdb, n=60, period=7.0, amplitude=6.0)
    fip = explore.fip_of(str(rdb), "vrad")
    assert fip["err"] == "svrad"
    assert fip["best"]["oa"]["period"] == pytest.approx(7.0, rel=0.02)
    assert fip["best"]["oa"]["fip"] < 1e-3
    assert len(fip["period"]) == explore.FIP_BINS
    assert len(fip["prob"]) == 60
    # a column without its error: at its robust scatter
    assert explore.fip_of(str(rdb), "BERV")["err"] is None


@pytest.fixture
def server(data, monkeypatch):
    monkeypatch.setattr(app, "STATE_FILE", str(data / "state.json"))
    app.Handler.ctx = app.Context(str(data))
    srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_the_server_reads_tables_of_the_data_folder_only(server, data,
                                                         tmp_path_factory):
    query = "data_dir=%s&path=%s" % (data, data / "lblrdb"
                                     / "lbl_GJ_1_GJ_1.rdb")
    with urllib.request.urlopen(server + "/api/explore/rdb?" + query) as r:
        assert json.loads(r.read())["rows"] == 40
    with urllib.request.urlopen(server + "/api/explore/template?" + query
                                + "&n=200") as r:
        assert json.loads(r.read())["template"]["object"] == "GJ_1"
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "lbl_X_X.rdb"
    _rdb(elsewhere)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(server + "/api/explore/rdb?data_dir=%s"
                               "&path=%s" % (data, elsewhere))
    assert error.value.code == 403
