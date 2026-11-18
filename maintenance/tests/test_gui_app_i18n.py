"""The two languages, and the server end to end on a free port."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from lbl.gui import app, i18n


def test_both_languages_have_every_word():
    assert set(i18n.EN) == set(i18n.FR)
    for key in i18n.EN:
        assert i18n.EN[key].count("%s") == i18n.FR[key].count("%s"), key
    for key, (labels, helps) in i18n.OPTIONS.items():
        assert len(labels) == 2 and all(labels), key
        assert helps is None or (len(helps) == 2 and all(helps)), key


def test_every_option_and_step_is_explained_in_both_languages():
    from lbl.gui import lblinfo
    for row in lblinfo.pca2d_options("NIRPS_HE"):
        for lang in ("en", "fr"):
            text = i18n.option_text(row["key"], lang, row.get("description"))
            assert len(text["help"]) > 80 and not text["lbl"], row["key"]
    for row in lblinfo.lbl_switches():
        assert "help_sw_%s" % row["key"] in i18n.EN, row["key"]
    # an option the window does not know yet shows LBL's own words
    text = i18n.option_text("new_key", "fr", "what LBL says")
    assert text == dict(label="new_key", help="", lbl="what LBL says")


def test_no_long_dashes():
    words = " ".join(list(i18n.EN.values()) + list(i18n.FR.values()))
    words += " ".join(h for _, helps in i18n.OPTIONS.values() for h in helps)
    assert "\u2014" not in words and "\u2013" not in words


def test_french_says_vous_never_tu():
    words = " ".join(i18n.FR.values()).lower().replace("'", " ").split()
    assert not {"tu", "toi", "ton", "ta", "tes"} & set(words)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "STATE_FILE", str(tmp_path / "state.json"))
    (tmp_path / "data" / "science" / "STAR").mkdir(parents=True)
    (tmp_path / "data" / "reports").mkdir()
    (tmp_path / "data" / "reports" / "r.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "secret.pdf").write_bytes(b"%PDF-1.4")
    app.Handler.ctx = app.Context(str(tmp_path / "data"))
    srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1], tmp_path
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, response.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def test_pages_and_json(server):
    base, tmp = server
    status, page = _get(base + "/")
    assert status == 200 and b"<title>LBL</title>" in page
    strings = json.loads(_get(base + "/api/strings?lang=fr")[1])
    assert strings["tab_targets"] == "Cibles"
    state = json.loads(_get(base + "/api/state")[1])
    assert state["state"]["data_dir"].endswith("data")
    built = _post(base + "/api/build", dict(
        data_dir=str(tmp / "data"), instrument="NIRPS_HE", source="CADC",
        targets=["STAR"], teff={"STAR": 3300}))
    assert built["errors"] == [] and "lbl_wrap.main" in built["wrap"]


def test_files_are_served_from_the_data_folder_only(server):
    base, tmp = server
    status, body = _get(base + "/file?path=%s" % (tmp / "data" / "reports"
                                                    / "r.pdf"))
    assert status == 200 and body.startswith(b"%PDF")
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(base + "/file?path=%s" % (tmp / "secret.pdf"))
    assert error.value.code == 403


def test_the_targets_say_when_there_is_no_folder(server, tmp_path):
    base, tmp = server
    import shutil
    from astropy.io import fits
    # a target is a folder with spectra; an empty one is said apart
    fits.PrimaryHDU().writeto(tmp / "data" / "science" / "STAR" / "a.fits")
    (tmp / "data" / "science" / "EMPTY").mkdir()
    got = json.loads(_get(base + "/api/targets?data_dir=%s" % (tmp / "data"))[1])
    assert got["science"] == "ok" and [r["name"] for r in got["targets"]] == ["STAR"]
    assert got["empty"] == ["EMPTY"]
    shutil.rmtree(tmp / "data" / "science" / "STAR")
    got = json.loads(_get(base + "/api/targets?data_dir=%s" % (tmp / "data"))[1])
    assert got["science"] == "empty" and got["targets"] == []
    assert got["empty"] == ["EMPTY"]
    shutil.rmtree(tmp / "data" / "science")
    got = json.loads(_get(base + "/api/targets?data_dir=%s" % (tmp / "data"))[1])
    assert got["science"] == "missing"


#: an injection test ticked in the window, after its wrap was shown
INJECTED = dict(on=True, known=False, signals=[dict(k=100, period=20,
                                                    phase=0, e=0, omega=90)])


def test_a_wrap_shown_before_a_change_runs_the_settings_as_they_are(server):
    """the page's text built before the injection test was ticked: not
    edited by hand, the run is the settings' (FAKE_KEPLERIAN, <target>_inj);
    edited by hand, it runs as it is only once confirmed, and the steps
    planned are those of the text"""
    from lbl.gui import wrapgen
    base, tmp = server
    state = dict(data_dir=str(tmp / "data"), instrument="NIRPS_HE",
                 source="CADC", targets=["STAR"], teff={"STAR": 3300})
    stale = _post(base + "/api/build", state)["wrap"]
    assert "FAKE_KEPLERIAN" not in stale
    rparams = wrapgen.build(dict(state, injection=INJECTED))["rparams"]
    fresh = wrapgen.to_run(stale, rparams, "x", edited=False)
    assert "FAKE_KEPLERIAN" in fresh["text"] and fresh["differs"] == []
    assert any(step["obj"] == "STAR_inj"
               for step in wrapgen.steps(fresh["rparams"]))
    kept = wrapgen.to_run(stale, rparams, "x", edited=True)
    assert kept["text"] == stale and kept["differs"] == ["FAKE_KEPLERIAN"]
    assert not any("_inj" in step["obj"]
                   for step in wrapgen.steps(kept["rparams"]))
    # the window asks before running it (nothing is queued)
    req = urllib.request.Request(
        base + "/api/queue", headers={"Content-Type": "application/json"},
        data=json.dumps(dict(state=dict(state, injection=INJECTED), name="x",
                             wrap=stale, edited=True)).encode())
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req)
    assert error.value.code == 409
    assert json.loads(error.value.read())["differs"] == ["FAKE_KEPLERIAN"]
    assert not (tmp / "data" / "gui" / "runs").exists()
