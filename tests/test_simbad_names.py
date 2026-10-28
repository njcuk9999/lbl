"""lbl.core.simbad: a TOI in any spelling (TOI_1234, TOI 1234, TOI1234,
TOI-1234), and the spellings SIMBAD is asked in. No network: the answers
of SIMBAD itself are not tested here."""
from lbl.core import simbad
from lbl.pca2d import archive


def test_a_toi_in_any_spelling():
    for name in ('TOI-1452', 'TOI 1452', 'TOI_1452', 'TOI1452', 'toi_1452',
                 'TOI-1452.01', 'TOI1452_tc', 'TOI_1452_pca2d_demo',
                 'TOI-01452'):
        assert simbad.toi_number(name) == '1452', name
    for name in ('TIC 1452', 'GJ 1452', 'TOIX', '', None, 'K2-18'):
        assert simbad.toi_number(name) is None, name


def test_the_spellings_simbad_is_asked_in():
    assert simbad.name_variants('TOI_1452')[-1] == 'TOI-1452'
    assert 'TOI-1452' in simbad.name_variants('TOI1452_pca2d_demo')
    assert 'TOI-1452' in simbad.name_variants('TOI-1452.01')
    assert simbad.name_variants('Gl699')[-1] == 'GJ 699'
    assert simbad.name_variants('GJ_15A')[-1] == 'GJ 15 A'
    # the name as written comes first, and nothing is asked twice
    names = simbad.name_variants('TOI-1452')
    assert names[0] == 'TOI-1452' and len(names) == len(set(names))


def test_the_toi_archive_finds_an_underscored_toi():
    # 2026-10-27: TOI_1234 did not fetch the TOI table of the archive
    for folder in ('TOI_4860', 'TOI4860_tc', 'TOI 4860'):
        assert archive.identifiers({}, folder=folder)['toi'] == '4860'
