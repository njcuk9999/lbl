"""The archive search's fetch (archives.py): where a group's spectra go,
that nothing there is moved or overwritten, and that the wrap reads what a
folder came from. The fetch's getters are stood in for: no archive is asked."""
import json
import os
import time

from lbl.gui import archives, wrapgen


def _fake_fetch(monkeypatch, spectra, tmp_path):
    """lbl.gui.fetch's spectra and stars stood in for, its cache in tmp_path:
    nothing is asked of an archive, nor of SIMBAD or the TIC"""
    import sys
    import types
    from lbl.gui import fetch
    stars = types.SimpleNamespace(stellar_teff=lambda ident: dict(teff=None))
    for name, module in (("spectra", spectra), ("stars", stars)):
        monkeypatch.setitem(sys.modules, "lbl.gui.fetch." + name, module)
        monkeypatch.setattr(fetch, name, module, raising=False)
    monkeypatch.setenv("LBL_GUI_CACHE", str(tmp_path / "cache"))


def _group(names, lbl="NIRPS_HE", source="ESO"):
    def get(one, science, calib):
        with open(os.path.join(science, one["name"]), "w") as handle:
            handle.write("spectrum")
        return ""
    return dict(key="", obj="GJ1", lbl=lbl, source=source,
                input="*TCORR_A.fits", params=dict(BLAZE_CORRECTED=False),
                spectra=[dict(id=n, name=n) for n in names], get=get)


def test_a_folder_never_mixes_two_readers(tmp_path):
    group = _group(["a_TCORR_A.fits", "b_TCORR_A.fits"])
    assert archives.folder_for(str(tmp_path), group) == "GJ1"
    mine = tmp_path / "science" / "GJ1"
    mine.mkdir(parents=True)
    (mine / "a_TCORR_A.fits").write_text("x")
    # the same spectra, some there already: the same folder
    assert archives.folder_for(str(tmp_path), group) == "GJ1"
    # another reader's spectra there: a folder of its own
    (mine / "2950000t.fits").write_text("x")
    assert archives.folder_for(str(tmp_path), group) == "GJ1_NIRPS_HE_ESO"


def test_a_fetch_adds_and_says_where_from(tmp_path):
    keep = tmp_path / "science" / "GJ1"
    keep.mkdir(parents=True)
    (keep / "a_TCORR_A.fits").write_text("mine")
    box = archives.Archives()
    group = _group(["a_TCORR_A.fits", "b_TCORR_A.fits"])
    box.search = dict(state="done", target="GJ 1", teff=3478.0, groups=[
        dict(id="NIRPS:", inst="NIRPS", label="NIRPS", archive="ESO",
             lbl="NIRPS_HE", source="ESO", input=group["input"],
             params=group["params"], n=2, have=1, folder="GJ1", size_mb=24)])
    box._groups = {"NIRPS:": group}
    box.start_fetch(str(tmp_path), ["NIRPS:"])
    for _ in range(100):
        if box.status()["fetch"]["state"] != "running":
            break
        time.sleep(0.05)
    assert box.status()["fetch"]["state"] == "done"
    # what was there is as it was; the missing one is fetched
    assert (keep / "a_TCORR_A.fits").read_text() == "mine"
    assert (keep / "b_TCORR_A.fits").read_text() == "spectrum"
    origin = json.loads((keep / archives.ARCHIVE_FILE).read_text())
    assert origin["lbl"] == "NIRPS_HE" and origin["source"] == "ESO"
    assert origin["teff"] == 3478.0
    # the bar's times, that its time left and its duration are read from
    status = box.status()
    bar = status["fetch"]["bars"][0]
    assert bar["started"] <= bar["ended"] <= status["now"]


def test_the_wrap_takes_the_reader_of_a_fetched_folder(tmp_path):
    folder = tmp_path / "science" / "GJ1"
    folder.mkdir(parents=True)
    (folder / archives.ARCHIVE_FILE).write_text(json.dumps(dict(
        lbl="NIRPS_HE", source="ESO", input="*TCORR_A.fits",
        params=dict(BLAZE_CORRECTED=False))))
    state = dict(data_dir=str(tmp_path), instrument="NIRPS_HE", source="ESO",
                 targets=["GJ1"], teff={"GJ1": 3478}, pca2d_on=False)
    built = wrapgen.build(state)
    assert built["errors"] == []
    assert built["rparams"]["INPUT_FILE"] == "*TCORR_A.fits"
    assert built["rparams"]["BLAZE_CORRECTED"] is False
    # read with another source: refused, saying which
    built = wrapgen.build(dict(state, source="CADC"))
    assert any("ESO" in e for e in built["errors"])


def test_two_readers_found_together_get_two_folders(tmp_path, monkeypatch):
    import types
    found = {"NIRPS": [_group(["n.fits"])],
             "ESPRESSO": [_group(["e.fits"], lbl="ESPRESSO", source="None")]}
    fake = types.SimpleNamespace(
        identify=lambda target: dict(main="HD 225213"),
        groups=lambda target, inst, root, ident: found.get(inst, []),
        INSTRUMENTS={k: dict(label=k, archive=k, size=1.0) for k in
                     archives.INSTRUMENTS})
    _fake_fetch(monkeypatch, fake, tmp_path)
    box = archives.Archives()
    box.start_search("GJ 1", ["NIRPS", "ESPRESSO"], str(tmp_path))
    for _ in range(100):
        if box.status()["search"]["state"] != "running":
            break
        time.sleep(0.05)
    folders = {r["lbl"]: r["folder"] for r in box.status()["search"]["groups"]}
    # whichever answered first keeps the star's name, the other its reader's
    first, second = sorted(folders.values())
    assert first == "GJ1" and second in ("GJ1_ESPRESSO_None", "GJ1_NIRPS_HE_ESO")


def test_a_search_stops_and_drops_a_late_answer(tmp_path, monkeypatch):
    # an archive that does not answer and one that does: stopped, the
    #   search keeps what came and drops what comes late
    import threading
    import types
    release = threading.Event()

    def groups(target, inst, root, ident):
        if inst == "SOPHIE":
            release.wait(5)
        return [_group(["%s.fits" % inst], lbl=inst, source="None")]
    fake = types.SimpleNamespace(
        identify=lambda target: dict(main="V* AD Leo"), groups=groups,
        INSTRUMENTS={k: dict(label=k, archive=k, size=1.0) for k in
                     archives.INSTRUMENTS})
    _fake_fetch(monkeypatch, fake, tmp_path)
    box = archives.Archives()
    box.start_search("AD Leo", ["HARPS", "SOPHIE"], str(tmp_path))
    for _ in range(100):
        if box.status()["search"]["instruments"]["HARPS"]["state"] == "done":
            break
        time.sleep(0.05)
    status = box.stop_search()
    assert status["search"]["state"] == "done"
    assert status["search"]["instruments"]["SOPHIE"]["state"] == "stopped"
    release.set()
    time.sleep(0.5)
    status = box.status()
    assert [g["inst"] for g in status["search"]["groups"]] == ["HARPS"]
    texts = " ".join(line["text"] for line in status["log"])
    assert "search stopped" in texts and "SOPHIE" in texts
    # every line in the window's logs too
    kept = (tmp_path / "gui" / "logs" / "archives.log").read_text()
    assert "search stopped" in kept


def test_several_stars_at_once_and_none_out_of_reach(tmp_path, monkeypatch):
    # two stars, two instruments, every request at once; a star SOPHIE's
    #   telescope never sees is not asked for there
    import types
    asked = []
    places = {"GJ 1": dict(main="HD 225213", ra=1.38, dec=-37.7),
              "AD Leo": dict(main="V* AD Leo", ra=154.9, dec=19.87)}

    def groups(target, inst, root, ident):
        asked.append((target, inst))
        return [dict(_group(["%s.fits" % inst], lbl=inst, source="None"),
                     obj=target.replace(" ", ""))]
    fake = types.SimpleNamespace(
        identify=lambda target: places[target], groups=groups,
        INSTRUMENTS={k: dict(label=k, archive=k, size=1.0) for k in
                     archives.INSTRUMENTS})
    _fake_fetch(monkeypatch, fake, tmp_path)
    box = archives.Archives()
    box.start_search("GJ 1, AD Leo", ["HARPS", "SOPHIE"], str(tmp_path))
    for _ in range(100):
        if box.status()["search"]["state"] != "running":
            break
        time.sleep(0.05)
    search = box.status()["search"]
    assert search["targets"] == ["GJ 1", "AD Leo"]
    assert search["cells"]["GJ 1|SOPHIE"]["state"] == "skipped"
    assert ("GJ 1", "SOPHIE") not in asked and ("AD Leo", "SOPHIE") in asked
    assert {(r["target"], r["inst"]) for r in search["groups"]} == {
        ("GJ 1", "HARPS"), ("AD Leo", "HARPS"), ("AD Leo", "SOPHIE")}
    assert len({r["folder"] for r in search["groups"]}) == 3
    assert search["stars"]["AD Leo"]["place"].startswith("10:19:")
