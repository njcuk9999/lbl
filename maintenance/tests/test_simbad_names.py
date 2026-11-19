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
    # a folder named after its star and its instrument or data source
    assert 'HD 10700' in simbad.name_variants('HD10700_EXPRES')
    assert 'GJ 229' in simbad.name_variants('Gliese229_HARPS_ORIG')
    assert 'AD Leo' in simbad.name_variants('ADLeo_HARPSN_ESO')
    assert 'Epsilon Eridani' in simbad.name_variants('EpsilonEridani_HARPSN_ESO')
    assert 'TYC 3980-1081-1' in simbad.name_variants('TYC_3980M1081M1')
    assert 'TOI-4508' in simbad.name_variants('TOIM4508')
    # the name as written comes first, and nothing is asked twice
    names = simbad.name_variants('TOI-1452')
    assert names[0] == 'TOI-1452' and len(names) == len(set(names))


def test_a_host_named_after_its_toi_is_asked_for_as_the_archive_names_it():
    # an underscored or unseparated TOI number in the folder name is asked
    #   of the archive's confirmed planets as the host name it writes
    for folder in ('TOI_4860', 'TOI4860_tc', 'TOI 4860', 'TOI-4860.01'):
        assert 'TOI-4860' in archive.identifiers({}, folder=folder)['names']
    # and a name that is not a TOI adds none
    names = archive.identifiers({'main_id': 'GJ 699'}, folder='GL699')['names']
    assert not any(name.startswith('TOI') for name in names)
    assert 'Gl 699' in names


def test_the_report_fits_tess_candidates_and_merges_confirmed_twins(
        monkeypatch, tmp_path):
    # the archive stood in for: one confirmed planet, its TOI, a candidate
    #   and a false positive
    def answer(query):
        if 'from pscomppars' in query:
            return [dict(pl_name='TOI-9 b', pl_letter='b', pl_orbper='3.0',
                         pl_tranmid='', pl_rvamp='5.0')]
        assert 'tid = 12' in query
        return [dict(toi='9.01', pl_orbper='3.0001', pl_tranmid='2459000.5',
                     tfopwg_disp='CP'),
                dict(toi='9.02', pl_orbper='10.78', pl_tranmid='2459351.6',
                     tfopwg_disp='PC'),
                dict(toi='9.03', pl_orbper='1.5', tfopwg_disp='FP')]
    monkeypatch.setattr(archive, '_get', answer)
    got = archive.planets({'ids': ['TIC 12']}, folder='TOI9',
                          cache=str(tmp_path), refresh=True)
    assert [(p['name'], p['source']) for p in got['planets']] == [
        ('TOI-9 b', 'confirmed'), ('TOI-9.02', 'TOI')]
    # the confirmed planet takes its TOI's transit time
    assert got['planets'][0]['t0'] == 2459000.5
    assert got['planets'][1]['disposition'] == 'PC'
