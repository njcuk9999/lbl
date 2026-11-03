"""
general.get_stellar_models takes the PHOENIX models from LBL's mirror first
(trimmed, gzipped, their own names), from the original site when the mirror
does not have them, and never pairs a wave file of one with a model of the
other.
"""
import os

import numpy as np
import pytest
from astropy.io import fits

from lbl.science import general

SITE = 'https://phoenix.astro.physik.uni-goettingen.de/data/'
MIRROR = 'https://raw.githubusercontent.com/eartigau/lbl-phoenix-mirror/main/'


class FakeInstrument:
    def __init__(self, mirror=MIRROR):
        self.params = dict(
            STELLAR_WAVE_URL=SITE + 'HiResFITS/',
            STELLAR_WAVE_FILE='WAVE_PHOENIX-ACES-AGSS-COND-2011.fits',
            STELLAR_MODEL_URL=SITE + 'HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
                                     '{ZSTR}{ASTR}/',
            STELLAR_MODEL_FILE='lte{TEFF}-{LOGG}-{ZVALUE}{ASTR}'
                               '.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
            STELLAR_MIRROR_URL=mirror,
            STELLAR_MIRROR_SUFFIX='_350-2550nm.fits.gz')

    def get_stellar_model_format_dict(self, params):
        return dict(TEFF='04500', LOGG='4.50', ZVALUE='0.0', ZSTR='Z-0.0',
                    ASTR='')


@pytest.fixture
def downloads(monkeypatch):
    """wget.download replaced: the mirror's files have 3 samples, the
    site's 5; `down` names the hosts that fail"""
    state = dict(asked=[], down=set())

    def download(url, out):
        state['asked'].append(url)
        host = 'mirror' if url.startswith(MIRROR) else 'site'
        if host in state['down']:
            raise OSError('HTTP Error 404: Not Found')
        size = 3 if host == 'mirror' else 5
        fits.PrimaryHDU(np.arange(size, dtype=float) + 1).writeto(out)
    monkeypatch.setattr(general.wget, 'download', download)
    return state


def test_the_mirror_first(tmp_path, downloads):
    wave, flux = general.get_stellar_models(FakeInstrument(), str(tmp_path))
    assert wave.size == flux.size == 3
    assert all(url.startswith(MIRROR) for url in downloads['asked'])
    assert downloads['asked'][1].endswith(
        'HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte04500-4.50-0.0.'
        'PHOENIX-ACES-AGSS-COND-2011-HiRes_350-2550nm.fits.gz')
    # read again from the disk, nothing downloaded
    general.get_stellar_models(FakeInstrument(), str(tmp_path))
    assert len(downloads['asked']) == 2


def test_the_site_when_the_mirror_fails(tmp_path, downloads):
    downloads['down'].add('mirror')
    wave, flux = general.get_stellar_models(FakeInstrument(), str(tmp_path))
    assert wave.size == flux.size == 5


def test_a_model_missing_on_the_mirror_takes_the_site_wave_file(tmp_path,
                                                                downloads,
                                                                monkeypatch):
    # the mirror has its wave file but not this model: both come from the
    #   site, never the mirror's wave file with the site's model
    original = general.wget.download

    def no_model(url, out):
        if url.startswith(MIRROR) and 'lte' in url:
            raise OSError('HTTP Error 404: Not Found')
        original(url, out)
    monkeypatch.setattr(general.wget, 'download', no_model)
    wave, flux = general.get_stellar_models(FakeInstrument(), str(tmp_path))
    assert wave.size == flux.size == 5


def test_an_original_pair_on_disk_is_used_as_it_is(tmp_path, downloads):
    folder = tmp_path / 'phoenix'
    folder.mkdir()
    for name in ('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits',
                 'lte04500-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'):
        fits.PrimaryHDU(np.ones(5)).writeto(os.path.join(folder, name))
    wave, flux = general.get_stellar_models(FakeInstrument(), str(tmp_path))
    assert wave.size == 5 and downloads['asked'] == []


def test_no_mirror(tmp_path, downloads):
    general.get_stellar_models(FakeInstrument(mirror=None), str(tmp_path))
    assert all(url.startswith(SITE) for url in downloads['asked'])


def test_both_failing_names_both(tmp_path, downloads):
    downloads['down'].update({'mirror', 'site'})
    with pytest.raises(general.LblException) as info:
        general.get_stellar_models(FakeInstrument(), str(tmp_path))
    assert 'the mirror' in str(info.value)
    assert 'the original site' in str(info.value)
