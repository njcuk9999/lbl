"""SIMBAD's Teff (simbad.py): the names asked and the value kept; SIMBAD
itself is not asked."""
from lbl.gui import simbad


def test_the_names_asked_follow_apero_and_the_headers():
    assert simbad.candidates("TOIM4508") == ["TOIM4508", "TOI-4508"]
    assert simbad.candidates("TYC_3980M1081M1") == ["TYC_3980M1081M1",
                                                    "TYC 3980-1081-1"]
    # the folder's name unknown to SIMBAD: its headers' OBJECT after it
    assert simbad.candidates("Gliese1214", "GJ1214") == [
        "Gliese1214", "GJ 1214", "GJ1214"]


def test_a_measurement_far_from_the_spectral_type_is_left_out():
    # NLTT 5556, an M4.5V (3110 K): 3000 K from its paper, a catalogue's
    #   4500 K twice (SIMBAD's mesFe_H, 2026-10-25)
    answer = simbad.summarise("LP 991-84", "M4.5V", [3000.0, 4500.0, 4500.0])
    assert (answer["teff"], answer["n"], answer["n_all"]) == (3000, 1, 3)
    # no measurement: the spectral type's (Pecaut & Mamajek 2013)
    answer = simbad.summarise("X", "M4.5V", [])
    assert (answer["teff"], answer["source"]) == (3110, "sptype")
    # neither
    assert simbad.summarise("X", None, [])["teff"] is None


def test_the_spectral_types_read():
    assert simbad.teff_of_type("dM5.0") == 3060
    assert simbad.teff_of_type("M3.5Ve") == 3270
    assert simbad.teff_of_type("G8III") is None
