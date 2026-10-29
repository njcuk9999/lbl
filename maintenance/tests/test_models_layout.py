"""DATA_DIR/models in subfolders: LBL's downloads by kind (phoenix,
tellurics, resproj, masks), the vetted templates in
template_<INSTRUMENT>_<SOURCE>, the PCA2D observer bases one keeps in
pca2d_<INSTRUMENT>_<SOURCE>. Older versions of LBL kept every file at the
top of models/; such a folder is sorted the first time a file is looked
for.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_models_layout.py
"""
import os

import pytest
from astropy.io import fits

from lbl.core import io, vetted
from lbl.pca2d import basis
from lbl.recipes.lbl_wrap import _template_of

from test_pca2d_basis import _fake_run


def test_a_file_at_the_top_of_models_is_moved_into_its_subfolder(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    (models / 'tapas_lbl.fits').write_text('tapas')
    path = io.model_path(str(models), 'tellurics', 'tapas_lbl.fits')
    assert path == str(models / 'tellurics' / 'tapas_lbl.fits')
    assert open(path).read() == 'tapas'
    assert not (models / 'tapas_lbl.fits').exists()
    # a file not there yet: its path, the folder made for the download
    path = io.model_path(str(models), 'phoenix', 'lte03000.fits')
    assert path == str(models / 'phoenix' / 'lte03000.fits')
    assert (models / 'phoenix').is_dir() and not os.path.exists(path)
    # an absolute name is someone's own file, left where it is
    assert io.model_path(str(models), 'resproj', '/a/b.fits') == '/a/b.fits'


class _Inst:
    def __init__(self, data_dir):
        self.params = dict(DATA_DIR=str(data_dir), INSTRUMENT='SPIROU',
                           DATA_SOURCE='APERO')


def test_a_vetted_template_is_kept_in_models_not_with_the_templates(
        tmp_path):
    inst = _Inst(tmp_path)
    templates = tmp_path / 'templates'
    templates.mkdir()
    name = 'Template_s1d_VETTED_GJ411_sc1d_v_file_AB.fits'
    # one written in templates/ before is moved
    (templates / name).write_text('vetted')
    path = vetted.template_path(inst, name, str(templates))
    assert path == str(tmp_path / 'models' / 'template_SPIROU_APERO' / name)
    assert open(path).read() == 'vetted'
    assert not (templates / name).exists()
    # and the wrap's report finds it there
    lbl_name = 'LBL_Template_VETTED_GJ411_spirou.fits'
    (tmp_path / 'models' / 'template_SPIROU_APERO' / lbl_name).write_text('')
    assert _template_of(str(tmp_path), 'VETTED_GJ411', 'SPIROU') == str(
        tmp_path / 'models' / 'template_SPIROU_APERO' / lbl_name)


def _basis_file(folder, label, source='CADC'):
    head = fits.Header()
    head['BASISNAM'] = label
    head['INSTRUME'] = 'SPIROU'
    head['DATASRC'] = source
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'PCA2D_Observer_%s_spirou.fits' % label)
    fits.PrimaryHDU(header=head).writeto(path)
    return path


def test_a_kept_basis_is_read_first_and_keep_moves_one(tmp_path):
    data_dir = str(tmp_path)
    components = os.path.join(data_dir, 'pca2d', 'components')
    kept = os.path.join(data_dir, 'models', 'pca2d_SPIROU_CADC')
    first = _basis_file(components, 'spirou_SNR15-25')
    assert basis.basis_path('spirou_SNR15-25', data_dir, 'SPIROU',
                            'CADC') == first
    out = basis.keep('spirou_SNR15-25', data_dir, 'SPIROU')
    assert out == os.path.join(kept, os.path.basename(first))
    assert not os.path.exists(first)
    # read there, with or without the DATA_SOURCE
    assert basis.basis_path('spirou_SNR15-25', data_dir, 'SPIROU',
                            'CADC') == out
    assert basis.basis_path('spirou_SNR15-25', data_dir, 'SPIROU') == out
    # a basis of the same name left in components is not the one read
    _basis_file(components, 'spirou_SNR15-25')
    assert basis.basis_path('spirou_SNR15-25', data_dir, 'SPIROU',
                            'CADC') == out
    assert '(kept)' in basis.compatible_bases(data_dir, 'SPIROU', 'CADC')
    # keeping the kept one again changes nothing
    assert basis.keep(os.path.basename(out), data_dir, 'SPIROU') == out


def test_a_basis_that_does_not_say_its_source_cannot_be_kept(tmp_path):
    components = tmp_path / 'pca2d' / 'components'
    components.mkdir(parents=True)
    fits.PrimaryHDU().writeto(components / 'PCA2D_Observer_x_spirou.fits')
    with pytest.raises(ValueError, match='DATA_SOURCE'):
        basis.keep('x', str(tmp_path), 'SPIROU')


def test_a_fit_of_the_name_of_a_kept_basis_updates_it_there(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    _fake_run(str(run))
    kept = tmp_path / 'models' / 'pca2d_NIRPS_HE_CADC'
    kept.mkdir(parents=True)
    (kept / 'PCA2D_Observer_2026a_nirps_he.fits').write_text('old')
    path = basis.save(str(run), '2026a', str(tmp_path), ['A', 'B'],
                      instrument='NIRPS_HE', data_source='CADC')
    assert path == str(kept / 'PCA2D_Observer_2026a_nirps_he.fits')
    assert basis.load('2026a', str(tmp_path), 'NIRPS_HE',
                      'CADC').header['NEARTH'] == 3
    # another name: in components, as before
    other = basis.save(str(run), '2026b', str(tmp_path), ['A', 'B'],
                       instrument='NIRPS_HE', data_source='CADC')
    assert other == str(tmp_path / 'pca2d' / 'components' /
                        'PCA2D_Observer_2026b_nirps_he.fits')


def test_the_window_lists_the_kept_bases_and_keeps_one(tmp_path):
    from lbl.gui import lblinfo
    data_dir = str(tmp_path)
    _basis_file(os.path.join(data_dir, 'pca2d', 'components'), 'a')
    _basis_file(os.path.join(data_dir, 'models', 'pca2d_SPIROU_CADC'), 'b')
    rows = {row['name']: row for row in lblinfo.bases(data_dir, 'SPIROU',
                                                      'CADC')}
    assert rows['b']['kept'] and not rows['a']['kept']
    assert rows['b']['folder'] == os.path.join('models', 'pca2d_SPIROU_CADC')
    assert 'error' in lblinfo.keep_basis(data_dir, '../x.fits', 'SPIROU')
    out = lblinfo.keep_basis(data_dir, 'PCA2D_Observer_a_spirou.fits',
                             'SPIROU')
    assert out['path'].startswith(os.path.join(data_dir, 'models',
                                               'pca2d_SPIROU_CADC'))
    assert all(row['kept'] for row in lblinfo.bases(data_dir, 'SPIROU',
                                                    'CADC'))


def test_a_models_folder_from_before_is_sorted_at_once(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    names = ['lte03100-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',
             'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', 'tapas_lbl.fits',
             'temperature_gradient_3500.fits', 'vetted_GJ411_spirou.fits',
             'vetted_GL699_nirps_he.fits', 'my_mask.fits',
             'lte03000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fitsx.tmp']
    for name in names:
        (models / name).write_text(name)
    io.sort_models(str(models), 'SPIROU', 'CADC')
    left = sorted(p.name for p in models.iterdir() if p.is_file())
    # another instrument's vetted file, someone's own file and a broken
    #   download stay where they are
    assert left == sorted(names[5:])
    assert (models / 'phoenix' / names[0]).exists()
    assert (models / 'phoenix' / names[1]).exists()
    assert (models / 'tellurics' / 'tapas_lbl.fits').exists()
    assert (models / 'resproj' / 'temperature_gradient_3500.fits').exists()
    assert (models / 'template_SPIROU_CADC' /
            'vetted_GJ411_spirou.fits').exists()


def test_the_science_template_is_not_looked_for_beside_a_vetted_one(
        tmp_path):
    # the science template is looked for in templates/, not built from the
    #   comparison's path with the object's name, which would put it in
    #   models/template_SPIROU_CADC beside the vetted template
    from lbl.science import general

    class Asked(Exception):
        pass

    class Inst:
        params = dict(DATA_TYPE='SCIENCE', OBJECT_SCIENCE='KIC7847566',
                      OBJECT_COMPARISON='VETTED_GJ4063',
                      ROUGH_CCF_MIN_RV=-3e5, ROUGH_CCF_MAX_RV=3e5,
                      ROUGH_CCF_STEP_RV=500, ROUGH_CCF_FILTER_SIZE=100,
                      CCF_CLEAN_BANDS=[])

        def get_mask_systemic_vel(self, mask_file):
            return 0.0

        def load_mask(self, mask_file):
            return None

        def load_template(self, filename):
            raise Asked(filename)

    vetted_template = str(tmp_path / 'models' / 'template_SPIROU_CADC' /
                          'LBL_Template_VETTED_GJ4063_spirou.fits')
    science = str(tmp_path / 'templates' /
                  'LBL_Template_KIC7847566_spirou.fits')
    with pytest.raises(Asked) as asked:
        general.get_systemic_vel_props(Inst(), vetted_template, 'mask.fits',
                                       science)
    assert str(asked.value) == science


def test_a_basis_in_any_subfolder_of_models_is_found(tmp_path):
    # a basis put in another subfolder of models/ (here
    #   template_NIRPS_HE_CADC): LBL and the window look in every subfolder
    from lbl.gui import lblinfo
    data_dir = str(tmp_path)
    folder = os.path.join(data_dir, 'models', 'template_NIRPS_HE_CADC')
    head = fits.Header()
    head['BASISNAM'] = 'NIRPS_SNR45-90'
    head['INSTRUME'] = 'NIRPS_HE'
    head['DATASRC'] = 'CADC'
    os.makedirs(folder)
    path = os.path.join(folder, 'PCA2D_Observer_NIRPS_SNR45-90_nirps_he.fits')
    fits.PrimaryHDU(header=head).writeto(path)
    assert basis.basis_path('NIRPS_SNR45-90', data_dir, 'NIRPS_HE',
                            'CADC') == path
    rows = lblinfo.bases(data_dir, 'NIRPS_HE', 'CADC')
    assert [(r['name'], r['kept'], r['compatible']) for r in rows] == [
        ('NIRPS_SNR45-90', True, True)]
    assert rows[0]['folder'] == os.path.join('models', 'template_NIRPS_HE_CADC')
    # the instrument's own kept folder comes first for the same file
    _basis_file(os.path.join(data_dir, 'models', 'pca2d_NIRPS_HE_CADC'),
                'other')
    first = os.path.join(data_dir, 'models', 'pca2d_NIRPS_HE_CADC')
    assert basis.kept_folders(data_dir, 'NIRPS_HE', 'CADC')[0] == first


def test_a_missing_blaze_falls_back_to_the_default_of_the_setup(tmp_path):
    # models/default-blaze/blaze_<INSTRUMENT>_<SOURCE>.fits, used when the
    #   blaze a file names is not in calib/
    import types
    from lbl.core.base_classes import LblException
    from lbl.instruments.default import Instrument

    inst = types.SimpleNamespace(params=dict(
        DATA_DIR=str(tmp_path), INSTRUMENT='NIRPS_HE', DATA_SOURCE='ESO'))
    inst.default_blaze_file = lambda: Instrument.default_blaze_file(inst)
    calib = tmp_path / 'calib'
    calib.mkdir()
    # the night's blaze, when it is there
    (calib / 'night_BLAZE_A.fits').write_text('')
    assert Instrument.blaze_path(inst, str(calib), 'night_BLAZE_A.fits') \
        == str(calib / 'night_BLAZE_A.fits')
    # no default: LBL stops, saying where one goes
    with pytest.raises(LblException, match='default-blaze/blaze_NIRPS_HE_ESO'):
        Instrument.blaze_path(inst, str(calib), 'other_BLAZE_A.fits')
    default = tmp_path / 'models' / 'default-blaze' / 'blaze_NIRPS_HE_ESO.fits'
    default.parent.mkdir(parents=True)
    default.write_text('')
    for named in ('other_BLAZE_A.fits', None):
        assert Instrument.blaze_path(inst, str(calib), named) == str(default)
