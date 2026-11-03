"""
lbl.core.simbad.tap goes to SIMBAD's mirror when a server does not answer,
and asks the server that answered first afterwards.
"""
import pytest
from astropy.table import Table

from lbl.core import simbad


class FakeClient:
    """A client whose servers named in `down` raise, as an unreachable one
    does; the others answer one row"""

    def __init__(self, down):
        self.down = set(down)
        self.server = None
        self.asked = []

    def query_tap(self, query):
        self.asked.append(self.server)
        if self.server in self.down:
            raise ConnectionError('no connection to %s' % self.server)
        return Table({'main_id': ['* eps Eri']})


@pytest.fixture
def fake(monkeypatch):
    def make(down):
        client = FakeClient(down)
        monkeypatch.setattr(simbad, '_SIMBAD', client)
        monkeypatch.setattr(simbad, 'SERVERS', ['simbad.cds.unistra.fr',
                                                'simbad.harvard.edu'])
        return client
    return make


def test_the_mirror_answers_when_the_cds_does_not(fake):
    client = fake(down={'simbad.cds.unistra.fr'})
    rows = simbad.tap('SELECT main_id FROM basic')
    assert rows == [{'main_id': '* eps Eri'}]
    assert client.asked == ['simbad.cds.unistra.fr', 'simbad.harvard.edu']
    # the next query goes to the mirror first
    simbad.tap('SELECT main_id FROM basic')
    assert client.asked[-1] == 'simbad.harvard.edu'
    assert len(client.asked) == 3
    assert simbad.SERVERS[0] == 'simbad.harvard.edu'


def test_the_cds_alone_when_it_answers(fake):
    client = fake(down=set())
    simbad.tap('SELECT main_id FROM basic')
    assert client.asked == ['simbad.cds.unistra.fr']


def test_no_server_answering_names_them_all(fake):
    fake(down={'simbad.cds.unistra.fr', 'simbad.harvard.edu'})
    with pytest.raises(OSError) as info:
        simbad.tap('SELECT main_id FROM basic')
    assert 'simbad.cds.unistra.fr' in str(info.value)
    assert 'simbad.harvard.edu' in str(info.value)
