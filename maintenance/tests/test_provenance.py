"""
lbl.core.provenance: the token of what a product was made from, beside it,
so that a step makes its product again when its inputs changed (spectra
added to an object), and keeps it otherwise.
"""
import os
import time

from astropy.table import Table

from lbl.core import provenance
from lbl.pca2d import cube


def _files(folder, names, text='x'):
    out = []
    for name in names:
        path = folder / name
        path.write_text(text)
        out.append(str(path))
    return out


def test_the_token_follows_the_files(tmp_path):
    files = _files(tmp_path, ['a.fits', 'b.fits'])
    first = provenance.token(files)
    assert provenance.token(list(reversed(files))) == first
    assert provenance.token(files + _files(tmp_path, ['c.fits'])) != first
    # by content: rewritten with the same bytes, the same token
    by_content = provenance.token(files, content=True)
    _files(tmp_path, ['a.fits'])
    assert provenance.token(files, content=True) == by_content
    _files(tmp_path, ['a.fits'], text='y')
    assert provenance.token(files, content=True) != by_content


def test_a_product_is_current_only_with_its_token(tmp_path):
    files = _files(tmp_path, ['a.fits'])
    product = str(tmp_path / 'table.rdb')
    value = provenance.token(files)
    assert not provenance.is_current(product, value)
    open(product, 'w').close()
    assert not provenance.is_current(product, value)
    provenance.record(product, value, len(files))
    assert provenance.is_current(product, value)
    assert not provenance.is_current(product, provenance.token(
        files + _files(tmp_path, ['b.fits'])))


def test_an_input_newer_than_the_product(tmp_path):
    product = tmp_path / 'template.fits'
    product.write_text('t')
    old = time.time() - 100
    os.utime(product, (old, old))
    assert provenance.older_than(str(product), _files(tmp_path, ['a.fits']))
    assert not provenance.older_than(str(product), [str(tmp_path / 'none')])


def test_a_spectrum_not_on_disk_is_described_by_its_name(tmp_path):
    assert provenance.describe(['synth_000']) == [['synth_000']]


def test_the_exposures_of_a_cached_cube(tmp_path):
    Table({'filename': ['a.fits', 'a.fits', 'b.fits']}).write(
        str(tmp_path / 'meta.fits'))
    assert cube._cached_exposures(str(tmp_path)) == 2
    Table({'filename': ['a.fits', 'b.fits', 'c.fits'],
           'row': [0, 0, 1], 'group': ['n', 'n', 'm'],
           'bjd': [1.0, 1.0, 2.0], 'berv': [0.0, 0.0, 0.0]}).write(
        str(tmp_path / cube.MEMBERS_FILE), format='ascii.csv')
    assert cube._cached_exposures(str(tmp_path)) == 3
