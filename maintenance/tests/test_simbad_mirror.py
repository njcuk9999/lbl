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
        monkeypatch.setattr(simbad, '_DOWN', {})
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


def test_a_server_that_holds_a_query_is_left_after_its_deadline(
        fake, monkeypatch):
    # 2026-11-19: the CfA mirror sent its answer a trickle at a time, the
    #   read timeout started again at each one, a query held 478 s
    import time
    client = fake(down={'simbad.cds.unistra.fr'})
    held = client.query_tap

    def slow(query):
        if client.server == 'simbad.harvard.edu':
            time.sleep(5)
        return held(query)
    client.query_tap = slow
    monkeypatch.setattr(simbad, 'DEADLINE', 0.2)
    start = time.time()
    with pytest.raises(OSError) as info:
        simbad.tap('SELECT main_id FROM basic')
    assert time.time() - start < 2 and 'TimeoutError' in str(info.value)
    # both left aside: the next query does not wait for them again
    asked = len(client.asked)
    start = time.time()
    with pytest.raises(OSError) as info:
        simbad.tap('SELECT main_id FROM basic')
    assert time.time() - start < 0.1 and len(client.asked) == asked
    assert 'did not answer' in str(info.value)
    # asked again once DOWN_FOR is over
    monkeypatch.setattr(simbad, 'DOWN_FOR', 0.0)
    client.down = set()
    client.query_tap = held
    assert simbad.tap('SELECT main_id FROM basic') == [
        {'main_id': '* eps Eri'}]


def test_a_query_refused_leaves_the_server_asked(fake):
    from pyvo.dal.exceptions import DALQueryError
    client = fake(down=set())

    def refused(query):
        client.asked.append(client.server)
        raise DALQueryError('bad ADQL')
    client.query_tap = refused
    with pytest.raises(OSError):
        simbad.tap('SELECT nothing FROM nowhere')
    assert simbad._DOWN == {}
