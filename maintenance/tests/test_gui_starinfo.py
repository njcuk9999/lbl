"""The star of the Data Explorer (lbl.gui.starinfo): the relations it derives
with (as their papers print them, within their ranges), the velocity seen
from the Earth and the telluric window, the planets of the NASA Exoplanet
Archive, the summary and the record kept of it. No service is asked."""
import datetime
import json
import math
import os

import pytest

from lbl.gui import explore, starinfo


# =============================================================================
# The relations
# =============================================================================
def _gj3470(**changes):
    """GJ 3470 as SIMBAD, Gaia and AllWISE give it (2026-11-11)."""
    data = dict(sptype="M2.0V", otype="BY*", G=11.360, BP=12.564, RP=10.269,
                J=8.794, H=8.206, K=7.989, W1=7.829, W2=7.781,
                M_G=9.0225, M_Ks=5.6516)
    data.update(changes)
    return data


def _value(rows, label, how):
    row = next(r for r in rows if r["label"] == label and r["how"] == how)
    return float(row["value"].split()[0])


def test_mann_radius_and_mass_as_printed():
    mode, rows, _ = starinfo.derive(_gj3470())
    assert mode == "mdwarf"
    m_ks = 5.6516
    # Mann et al. (2016, erratum) Table 1: R = 1.9515 - 0.3520 M + 0.01680 M^2
    assert _value(rows, "R★", "M_Ks") == pytest.approx(
        1.9515 - 0.3520 * m_ks + 0.01680 * m_ks ** 2, abs=6e-4)
    assert _value(rows, "M★", "M_Ks") == pytest.approx(
        0.5858 + 0.3872 * m_ks - 0.1217 * m_ks ** 2 + 0.0106 * m_ks ** 3
        - 2.7262e-4 * m_ks ** 4, abs=6e-4)


def test_mann_radius_with_feh_multiplies():
    """'for the equations including [Fe/H] the right-hand side is
    multiplied by (1 + f[Fe/H])' (Mann et al. 2016, Table 1 notes)."""
    _, rows, _ = starinfo.derive(_gj3470())
    feh = _value(rows, "[Fe/H]", "W1−W2, BP−RP, M_G")
    m_ks = 5.6516
    poly = 1.9305 - 0.3466 * m_ks + 0.01647 * m_ks ** 2
    assert _value(rows, "R★", "M_Ks, [Fe/H]") == pytest.approx(
        poly * (1 + 0.04458 * feh), abs=6e-4)


def test_duque_feh_is_their_eq4():
    _, rows, _ = starinfo.derive(_gj3470())
    x, y, z = 7.829 - 7.781, 12.564 - 10.269, 9.0225
    expected = 5.87 - 1.40 * x + 5.08 * y - 2.45 * z - 0.788 * y ** 2 \
        + 0.1094 * z ** 2
    assert _value(rows, "[Fe/H]", "W1−W2, BP−RP, M_G") \
        == pytest.approx(expected, abs=6e-3)


def test_duque_out_of_range_gives_none_and_says_why():
    """Barnard's star: a saturated W1 makes W1-W2 0.79."""
    _, rows, notes = starinfo.derive(_gj3470(W1=4.386, W2=3.600))
    assert not any(r["label"] == "[Fe/H]" for r in rows)
    assert any(n["key"] == "star_note_out" and n["args"][0] == "W1−W2"
               for n in notes)
    # nor a radius with [Fe/H]
    assert not any(r["how"] == "M_Ks, [Fe/H]" for r in rows)


def test_mann_teff_relations():
    _, rows, _ = starinfo.derive(_gj3470())
    x, jh = 12.564 - 10.269, 8.794 - 8.206
    alone = 3500 * (3.245 - 2.4309 * x + 1.043 * x ** 2 - 0.2127 * x ** 3
                    + 0.01649 * x ** 4)
    with_jh = 3500 * (3.172 - 2.475 * x + 1.082 * x ** 2 - 0.2231 * x ** 3
                      + 0.01738 * x ** 4 + 0.08776 * jh + 0.04355 * jh ** 2)
    assert _value(rows, "Teff", "BP−RP") == pytest.approx(alone, abs=1)
    assert _value(rows, "Teff", "BP−RP, J−H") == pytest.approx(
        with_jh, abs=1)
    assert any(r["how"] == "BP−RP, [Fe/H]" for r in rows)


def test_mann_range():
    _, rows, notes = starinfo.derive(_gj3470(M_Ks=10.5, K=7.989))
    assert not any(r["label"] == "R★" for r in rows)
    assert any(n["args"][0] == "M_Ks" for n in notes)


def test_delfosse_range():
    _, rows, _ = starinfo.derive(_gj3470(M_Ks=9.7))
    assert not any(r["how"] == "M_K" for r in rows)
    _, rows, _ = starinfo.derive(_gj3470(M_Ks=6.0))
    expected = 10 ** (1e-3 * (1.8 + 6.12 * 6 + 13.205 * 36 - 6.2315 * 216
                              + 0.37529 * 1296))
    assert _value(rows, "M★", "M_K") == pytest.approx(expected,
                                                            abs=6e-4)


def test_kinds_of_star():
    assert starinfo.derive(dict(sptype="DA2", M_G=11.0))[0] == "wd"
    assert starinfo.derive(dict(sptype="K1III", M_G=0.5))[0] == "other"
    assert starinfo.derive(dict(sptype="B2V", M_G=-1.5, BP=0.0,
                                RP=0.2))[0] == "early"
    assert starinfo.derive(dict(sptype=None, M_G=12.0))[0] == "mdwarf"


def test_afgk_gaia_then_sequence():
    data = dict(sptype="G8V", BP=4.0, RP=3.1, M_G=5.2)
    mode, rows, _ = starinfo.derive(data, dict(
        teff_gspphot=5500.0, logg_gspphot=4.5, mh_gspphot=-0.1,
        radius_flame=0.9, lum_flame=0.7, mass_flame=0.92))
    assert mode == "afgk_gaia"
    assert _value(rows, "Teff", "GSP-Phot") == 5500
    mode, rows, notes = starinfo.derive(data, None)
    assert mode == "afgk_seq"
    # BP-RP 0.9 and M_G 5.2 sit on G8V (0.900, 5.10)
    assert _value(rows, "Teff", "BP−RP, M_G") == pytest.approx(5480)
    assert notes[0]["key"] == "star_note_sequence"
    # far above the sequence (a subgiant), placed by its spectral type
    mode, rows, _ = starinfo.derive(dict(sptype="G2V", BP=4.0, RP=3.1,
                                         M_G=3.0), None)
    assert mode == "afgk_seq" and _value(rows, "Teff", "G2V") == 5770


# =============================================================================
# The velocity seen from the Earth, the tellurics
# =============================================================================
def test_compose_is_relativistic():
    c = starinfo._speed_of_light_kms()
    assert starinfo.compose(26.3, 0.0) == pytest.approx(26.3, abs=1e-9)
    assert starinfo.compose(12.0, 12.0) == pytest.approx(0.0, abs=1e-9)
    # 30 - (-30): the composition of velocities, not 60
    assert starinfo.compose(30.0, -30.0) == pytest.approx(
        60.0 / (1 + 900.0 / c ** 2), rel=1e-12)


def test_berv_as_astropy_without_rotation():
    """The Earth's barycentric velocity on the star's direction is
    astropy's barycentric correction, but for the Earth's rotation."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    dates, berv = starinfo.bervs(119.7743, 15.3913, 2026)
    assert len(dates) == 365 and dates[0] == datetime.date(2026, 1, 1)
    star = SkyCoord(ra=119.7743 * u.deg, dec=15.3913 * u.deg)
    site = EarthLocation.from_geodetic(lon=-70.73 * u.deg, lat=-29.26 * u.deg,
                                       height=2400 * u.m)
    for day in (0, 80, 170, 260, 350):
        expected = star.radial_velocity_correction(
            kind="barycentric", obstime=Time(dates[day].isoformat()),
            location=site).to(u.km / u.s).value
        assert berv[day] == pytest.approx(expected, abs=0.5)


def test_telluric_window():
    """GJ 3470 (RV 26.3 km/s): its lines on the tellurics from early
    September to early December (astropy at La Silla: 8 Sep to 6 Dec)."""
    out = starinfo.tellurics(119.7743, 15.3913, 26.341, 2026)
    assert len(out["windows"]) == 1
    first, last = out["windows"][0]
    assert "2026-09-04" <= first <= "2026-09-12"
    assert "2026-12-01" <= last <= "2026-12-09"
    # faster than the Earth ever moves toward it: never
    assert starinfo.tellurics(119.7743, 15.3913, -110.0, 2026)["windows"] \
        == []
    assert starinfo.tellurics(119.7743, 15.3913, None) is None


def test_telluric_window_across_the_new_year():
    """A star whose window spans 31 December and 1 January gets one."""
    _, berv = starinfo.bervs(10.0, 0.0, 2026)
    out = starinfo.tellurics(10.0, 0.0, berv[0], 2026)
    assert any(a > b for a, b in out["windows"])


# =============================================================================
# The planets
# =============================================================================
def test_planet_keys():
    keys = starinfo.planet_keys(
        ["NAME Barnard's star", "GJ 699", "HIP 87937", "TIC 325554331",
         "Gaia DR3 4472832130942575872", "Gaia DR2 4472832130942575872"],
        ["GJ699"])
    assert keys["gaia_dr3_id"] == ["Gaia DR3 4472832130942575872"]
    assert keys["tic_id"] == ["TIC 325554331"]
    assert keys["hip_name"] == ["HIP 87937"]
    assert "Barnard's star" in keys["hostname"] and "GJ699" in \
        keys["hostname"]


def test_reference_link():
    link = ("<a refstr=BONFILS_ET_AL__2012 href=https://ui.adsabs.harvard.edu"
            "/abs/2012A%26A...546A..27B/abstract target=ref> Bonfils et al."
            " 2012 </a>")
    assert starinfo.reference(link) == dict(
        text="Bonfils et al. 2012",
        url="https://ui.adsabs.harvard.edu/abs/2012A%26A...546A..27B/abstract")
    assert starinfo.reference("") is None


def test_ask_planets(monkeypatch):
    asked = []

    def tap(base, adql, timeout=0):
        asked.append((base, adql))
        row = {key.strip(): "" for key in starinfo.PLANET_COLUMNS.split(",")}
        two = dict(row, pl_name="X c", pl_orbper="20.5", pl_bmasse="5.1",
                   pl_bmassprov="Msini", disc_year="2021", st_teff="3300",
                   st_teff_reflink="<a href=https://x/y target=ref>Y 2020</a>")
        one = dict(row, pl_name="X b", pl_orbper="3.25", pl_rvamp="4.2",
                   pl_controv_flag="1", st_teff="3300")
        return [two, one]
    monkeypatch.setattr(starinfo, "tap", tap)
    monkeypatch.setattr(starinfo, "_DOWN", {})
    out = starinfo.ask_planets(["Gaia DR3 12", "TIC 34"], ["X"])
    assert asked[0][0] == starinfo.NASA_TAP
    assert "gaia_dr3_id IN ('Gaia DR3 12')" in asked[0][1]
    assert [p["name"] for p in out["planets"]] == ["X b", "X c"]
    assert out["planets"][0]["controversial"]
    assert out["planets"][1]["mass_kind"] == "Msini"
    assert out["host"]["teff"]["value"] == 3300
    assert out["host"]["teff"]["ref"]["text"] == "Y 2020"


def test_ask_planets_unreachable(monkeypatch):
    def tap(base, adql, timeout=0):
        raise OSError("down")
    monkeypatch.setattr(starinfo, "tap", tap)
    monkeypatch.setattr(starinfo, "_DOWN", {})
    assert starinfo.ask_planets(["TIC 34"], ["X"]) is None
    # left alone afterwards
    assert not starinfo._usable(starinfo.NASA_TAP)


# =============================================================================
# The summary and its record
# =============================================================================
def _star():
    return dict(main="LP 424-4", asked="GJ 3470", server="simbad.harvard.edu",
                ra=119.7743, dec=15.3913, plx=34.0, plx_err=0.03,
                pmra=-185.7, pmdec=-57.0, sptype="M2.0V", otype="BY*",
                rv=26.341, rv_err=0.004,
                mags=dict(G=11.36, J=8.794, H=8.206, K=7.989),
                ids=["GJ 3470", "TIC 19028197", "Gaia DR3 654687847820642560"],
                fe_h=[dict(teff="3616", log_g="4.79", fe_h="0.07")],
                vsini=[1.5])


@pytest.fixture
def offline(monkeypatch):
    calls = dict(simbad=0)

    def simbad(names):
        calls["simbad"] += 1
        return _star() if "GJ 3470" in names else None
    monkeypatch.setattr(starinfo, "ask_simbad", simbad)
    monkeypatch.setattr(starinfo, "ask_gaia", lambda sid: dict(
        source_id=sid, phot_g_mean_mag=11.36, phot_bp_mean_mag=12.564,
        phot_rp_mean_mag=10.269, parallax=34.017, parallax_error=0.026,
        pmra=-185.7, pmdec=-57.0, ruwe=1.14, **{"from": "gea.esac.esa.int"}))
    monkeypatch.setattr(starinfo, "ask_wise", lambda *a: dict(
        W1=7.829, W2=7.781, sep=0.2, source="dc.g-vo.org"))
    monkeypatch.setattr(starinfo, "ask_planets", lambda ids, names: dict(
        planets=[dict(name="GJ 3470 b", period=3.3366, k=13.4)],
        host=dict(teff=dict(value=3600.0, ref=None))))
    return calls


def test_summarise(offline):
    out = starinfo.summarise(["GJ_3470", "GJ 3470"])
    assert out["state"] == "done" and out["title"] == "LP 424-4"
    keys = [g["key"] for g in out["groups"]]
    assert keys == ["position", "photometry", "derived", "published"]
    derived = out["groups"][2]
    assert derived["mode"] == "mdwarf"
    assert any(r["label"] == "[Fe/H]" for r in derived["rows"])
    assert out["planets"]["source"] == "nasa"
    assert out["tellurics"]["windows"]
    json.dumps(out)
    assert starinfo.summarise(["nobody"])["state"] == "unknown"


def test_summarise_falls_back_on_lbl_planets(offline, monkeypatch):
    monkeypatch.setattr(starinfo, "ask_planets", lambda ids, names: None)
    out = starinfo.summarise(["GJ 3470"], fallback_planets=lambda: dict(
        planets=[dict(name="GJ 3470 b", period=3.3366, k=8.2,
                      reference="a paper")]))
    assert out["planets"]["source"] == "lbl"
    assert out["planets"]["rows"][0]["k"] == 8.2


def test_summarise_simbad_unreachable(monkeypatch):
    def simbad(names):
        raise OSError("SIMBAD: no server answered")
    monkeypatch.setattr(starinfo, "ask_simbad", simbad)
    out = starinfo.summarise(["GJ 3470"])
    assert out["state"] == "failed" and "no server" in out["error"]


def test_summary_kept(offline, tmp_path, monkeypatch):
    first = starinfo.summary(str(tmp_path), "GJ_3470", ["GJ 3470"])
    assert first["state"] == "done" and offline["simbad"] == 1
    assert os.path.exists(tmp_path / starinfo.CACHE)
    again = starinfo.summary(str(tmp_path), "GJ_3470", ["GJ 3470"])
    assert again["when"] == first["when"] and offline["simbad"] == 1
    starinfo.summary(str(tmp_path), "GJ_3470", ["GJ 3470"], refresh=True)
    assert offline["simbad"] == 2
    # older than KEEP_DAYS: asked again
    path = tmp_path / starinfo.CACHE
    kept = json.loads(path.read_text())
    kept["GJ_3470"]["when"] = (datetime.datetime.now() - datetime.timedelta(
        days=starinfo.KEEP_DAYS + 1)).isoformat(timespec="seconds")
    path.write_text(json.dumps(kept))
    starinfo.summary(str(tmp_path), "GJ_3470", ["GJ 3470"])
    assert offline["simbad"] == 3


def test_summary_failure_not_kept(tmp_path, monkeypatch):
    def simbad(names):
        raise OSError("down")
    monkeypatch.setattr(starinfo, "ask_simbad", simbad)
    assert starinfo.summary(str(tmp_path), "X", ["X"])["state"] == "failed"
    assert not os.path.exists(tmp_path / starinfo.CACHE)


def test_star_of_names_the_folder(tmp_path, monkeypatch):
    """A corrected table's star is its folder's less LBL's suffixes, asked
    as SIMBAD writes it."""
    (tmp_path / "lblrdb").mkdir()
    (tmp_path / "science" / "GJ_3470_pca2d").mkdir(parents=True)
    rdb = tmp_path / "lblrdb" / "lbl_GJ_3470_pca2d_GJ_3470.rdb"
    rdb.write_text("rjd\tvrad\n---\t----\n1\t2\n")
    seen = {}

    def summary(data_dir, key, names, refresh=False, fallback_planets=None):
        seen.update(data_dir=data_dir, key=key, names=names, refresh=refresh)
        return dict(state="done")
    monkeypatch.setattr(starinfo, "summary", summary)
    explore.star_of(str(rdb), refresh=True)
    assert seen["key"] == "GJ_3470" and seen["refresh"]
    assert "GJ 3470" in seen["names"]
    assert os.path.samefile(seen["data_dir"], tmp_path)


def test_speed_of_light_not_hard_coded():
    source = open(starinfo.__file__, encoding="utf-8").read()
    for literal in ("299792", "2.99792", "3e5", "3e8"):
        assert literal not in source
    assert not math.isnan(starinfo._speed_of_light_kms())
