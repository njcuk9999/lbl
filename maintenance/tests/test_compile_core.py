"""The arithmetic of lbl_compile (lbl.science.general.make_rdb_table and
make_rdb_table2) on line tables whose answer is known: a small instrument
object hands synthetic lblrv tables to the real functions."""
import warnings

import numpy as np
from astropy.table import Table

from lbl.core import io
from lbl.core import math as mp
from lbl.instruments import default
from lbl.science import general

io.log.console_verbosity = 0

#: the CCF e-width written in the synthetic headers, in m/s
CCF_EW = 2000.0


class SynthInst:
    """The methods make_rdb_table calls, on tables held in memory"""
    name = 'SYNTH'

    def __init__(self, tables, mjd, berv=None, ccf_ew=CCF_EW, **params):
        self.tables = tables
        self.mjd = np.array(mjd, dtype=float)
        self.berv = np.zeros(len(mjd)) if berv is None else np.array(berv)
        self.ccf_ew = ccf_ew
        self.params = dict(COMPIL_WAVE_MIN=400.0, COMPIL_WAVE_MAX=1000.0,
                           COMPIL_MAX_PIXEL_WIDTH=500,
                           COMPIL_FRAC_TIME_MEAS=0.1, COMPIL_FP_EWID=5.0,
                           COMPIL_CUT_PEARSONR=-1, COMPIL_SLOPE_REF_WAVE=550.0,
                           KW_CCF_EW='CCF_EW', RESPROJ_TABLES=None,
                           COMPIL_FORCE_SIGMA_PER_LINE=False, USE_TQDM=False,
                           COMPIL_ADD_UNIFORM_WAVEBIN=False,
                           COMPIL_NUM_UNIFORM_WAVEBIN=6, DATA_TYPE='SCIENCE',
                           PLOT=False, PLOT_COMPIL_CUMUL=False,
                           PLOT_COMPIL_BINNED=False, KW_MJDATE='MJD',
                           OBJECT_SCIENCE='SYN', OBJECT_COMPARISON='SYN')
        self.params.update(params)

    def files(self):
        return np.array(['synth_%04d' % it for it in range(len(self.tables))])

    def rdb_columns(self):
        return np.array(['MJD', 'CCF_EW']), [False, False]

    def load_lblrv_file(self, name):
        row = int(str(name).split('_')[1])
        hdr = io.LBLHeader()
        hdr['MJD'] = float(self.mjd[row])
        hdr['CCF_EW'] = float(self.ccf_ew)
        hdr['ROW'] = row
        return self.tables[row].copy(), hdr

    def get_berv(self, hdr):
        return float(self.berv[hdr['ROW']])

    def fix_lblrv_header(self, hdr):
        return hdr

    def get_rjd_value(self, hdr):
        return float(hdr['MJD']) + 0.5

    def get_plot_date(self, hdr):
        return float(hdr['MJD'])

    def get_binned_parameters(self):
        return dict(bands=['g', 'r'], blue_end=[400.0, 600.0],
                    red_end=[600.0, 1000.0],
                    region_names=['', '_0-2000', '_2000-4000'],
                    region_low=[0, 0, 2000], region_high=[4000, 2000, 4000],
                    use_regions=[True, True])

    def get_uniform_binned_parameters(self, binned):
        return default.Instrument.get_uniform_binned_parameters(self, binned)

    def get_epoch_groups(self, rdb_table):
        # one epoch per integer MJD
        values = np.floor(np.array(rdb_table['MJD'], dtype=float))
        return np.unique(values), values


def make_tables(dv, sdv, wave, xpix=None, d2v=None):
    """One lblrv-like table per epoch, from arrays (epochs, lines)"""
    tables = []
    nlines = dv.shape[1]
    xpix = np.linspace(1, 3999, nlines) if xpix is None else xpix
    for row in range(dv.shape[0]):
        table = Table()
        table['ORDER'] = np.zeros(nlines, dtype=int)
        table['WAVE_START'] = wave
        table['WAVE_END'] = wave * (1 + 1e-4)
        table['XPIX'] = xpix
        table['NPIXLINE'] = np.full(nlines, 20)
        table['dv'] = dv[row]
        table['sdv'] = sdv[row]
        table['contrast'] = np.ones(nlines)
        table['sig_contrast'] = np.full(nlines, 0.01)
        table['d2v'] = np.zeros(nlines) if d2v is None else d2v[row]
        table['sd2v'] = np.full(nlines, 1000.0)
        table['d3v'] = np.zeros(nlines)
        table['sd3v'] = np.full(nlines, 1000.0)
        tables.append(table)
    return tables


def run_rdb(inst):
    """The real make_rdb_table on a SynthInst: (rdb fits data, rdb table)"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out, _ = general.make_rdb_table(inst, 'x.rdb', inst.files(), '.')
    return out, out['RDB']


def synthetic_star(nepoch=12, nlines=400, seed=1, sigma=5.0, offsets=0.0,
                   velocity=None):
    """Lines of known per-line offsets around a known velocity per epoch"""
    rng = np.random.default_rng(seed)
    wave = np.sort(rng.uniform(410.0, 990.0, nlines))
    if velocity is None:
        velocity = rng.normal(0, 3.0, nepoch)
    offset = offsets * rng.normal(0, 1, nlines)
    sdv = np.full((nepoch, nlines), sigma)
    dv = (velocity[:, None] + offset[None, :] +
          rng.normal(0, 1, (nepoch, nlines)) * sdv)
    return wave, velocity, offset, dv, sdv


def test_the_fwhm_column_is_the_ccf_fwhm_plus_dw_over_it():
    """fwhm = FWHM + dW / FWHM with FWHM = 2.3548 CCF_EW (dW = FWHM dFWHM,
    Artigau et al. 2022) and dW = 8 ln 2 d2v"""
    wave, _, _, dv, sdv = synthetic_star()
    d2v = np.full_like(dv, 30000.0)
    inst = SynthInst(make_tables(dv, sdv, wave, d2v=d2v), np.arange(12.0))
    _, rdb = run_rdb(inst)
    fwhm = mp.fwhm_value() * CCF_EW
    dw = 8 * np.log(2) * 30000.0
    assert np.allclose(rdb['dW'], dw)
    assert np.allclose(rdb['fwhm'], fwhm + dw / fwhm)
    assert np.allclose(rdb['sig_fwhm'], np.array(rdb['sdW']) / fwhm)


def test_an_lblrv_file_without_finite_velocity_gives_a_nan_row():
    """The compile does not stop on it (np.min of an empty array): its row
    is NaN, the others are measured"""
    wave, velocity, _, dv, sdv = synthetic_star()
    dv[4] = np.nan
    inst = SynthInst(make_tables(dv, sdv, wave), np.arange(12.0))
    _, rdb = run_rdb(inst)
    vrad = np.array(rdb['vrad'], dtype=float)
    assert np.isnan(vrad[4]) and np.isnan(rdb['vrad_achromatic'][4])
    others = np.delete(np.arange(12), 4)
    assert np.all(np.isfinite(vrad[others]))
    assert np.all(np.abs(vrad[others] - velocity[others]) < 1.5)


def capture(func, *args, names=None, **kwargs):
    """Run func; return its result and its local variables on return"""
    import sys
    saved = dict()

    def tracer(frame, event, arg):
        if event == 'return' and frame.f_code is func.__code__:
            for key, value in frame.f_locals.items():
                if names is None or key in names:
                    saved[key] = value
    previous = sys.getprofile()
    sys.setprofile(tracer)
    try:
        result = func(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    return result, saved


def test_per_line_zero_points_are_the_offsets_of_the_lines():
    """Lines with offsets of 20 m/s rms, 30% of them missing at random in
    each epoch: per_line_mean measures each line's offset (Artigau et al.
    2022, section 4.7), and once removed the missing lines no longer add
    their offsets to the velocities"""
    wave, velocity, offset, dv, sdv = synthetic_star(offsets=20.0, seed=3)
    rng = np.random.default_rng(4)
    dv[rng.random(dv.shape) < 0.3] = np.nan
    inst = SynthInst(make_tables(dv, sdv, wave), np.arange(12.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out, saved = capture(general.make_rdb_table, inst, 'x.rdb',
                             inst.files(), '.', names=['per_line_mean'])
    plm = np.array(saved['per_line_mean'])
    good = np.isfinite(plm)
    assert np.sum(good) > 0.95 * len(plm)
    assert np.corrcoef(plm[good], offset[good])[0, 1] > 0.95
    # weighted mean of the offsets at zero: plm = offset - their mean
    assert abs(np.median(plm[good] - (offset[good] - np.mean(offset)))) < 1.0
    vrad = np.array(out[0]['RDB']['vrad'], dtype=float)
    resid = vrad - velocity
    # noise alone: 5 m/s / sqrt(280 lines) = 0.30 m/s per epoch
    assert np.std(resid - np.mean(resid)) < 0.45


def test_a_band_with_too_few_lines_is_nan_not_zero():
    """Three lines in the g band (5 needed): its columns are NaN"""
    wave, velocity, _, dv, sdv = synthetic_star()
    wave = np.sort(np.concatenate([[450.0, 500.0, 550.0],
                                   np.linspace(610.0, 990.0,
                                               len(wave) - 3)]))
    inst = SynthInst(make_tables(dv, sdv, wave), np.arange(12.0))
    _, rdb = run_rdb(inst)
    for col in ('vrad_g', 'svrad_g', 'vrad_g_0-2000', 'svrad_g_2000-4000'):
        assert np.all(np.isnan(np.array(rdb[col], dtype=float)))
    assert np.all(np.isfinite(np.array(rdb['vrad_r'], dtype=float)))


def test_the_nightly_table_ignores_unmeasured_values():
    """Three exposures a night: a band measured in two of them gives their
    weighted mean, not NaN nor 0 +- 0 (a value of 0 +- 0, as in an rdb
    written before, counts as unmeasured); other columns are averaged over
    their finite values"""
    table = Table()
    table['MJD'] = [100.1, 100.2, 100.3, 101.1, 101.2, 101.3]
    table['vrad_x'] = [1.0, 3.0, np.nan, 0.0, 5.0, 7.0]
    table['svrad_x'] = [1.0, 1.0, np.nan, 0.0, 2.0, 2.0]
    table['BERV'] = [10.0, np.nan, 12.0, 1.0, 2.0, 3.0]
    inst = SynthInst([], table['MJD'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rdb2 = general.make_rdb_table2(inst, table)
    assert np.allclose(rdb2['vrad_x'], [2.0, 6.0])
    assert np.allclose(rdb2['svrad_x'], [np.sqrt(0.5), np.sqrt(2.0)])
    assert np.allclose(rdb2['BERV'], [11.0, 2.0])


def test_a_calibration_run_has_its_line_velocities_and_fwhm_in_m_s():
    """FP files: the DV and SDV extensions hold the line velocities (they
    were left at zero), and with CCF_EW = 0 in the headers the fwhm uses
    COMPIL_FP_EWID in km/s: 5 km/s gives a FWHM of 11.77 km/s"""
    wave, _, _, dv, sdv = synthetic_star()
    d2v = np.full_like(dv, -40000.0)
    inst = SynthInst(make_tables(dv, sdv, wave, d2v=d2v), np.arange(12.0),
                     ccf_ew=0.0, DATA_TYPE='FP')
    out, rdb = run_rdb(inst)
    assert np.allclose(out['DV'], dv, equal_nan=True)
    assert np.allclose(out['SDV'], sdv, equal_nan=True)
    fwhm = mp.fwhm_value() * 5000.0
    dw = 8 * np.log(2) * -40000.0
    assert np.allclose(rdb['fwhm'], fwhm + dw / fwhm)
    assert np.all(np.array(rdb['fwhm']) > 0.9 * fwhm)


def test_a_one_file_run_gives_its_velocity():
    """A line seen once has no offset to remove: a one-file compile gives
    the weighted mean of the lines, where it used to fail"""
    wave, velocity, _, dv, sdv = synthetic_star(nepoch=1)
    inst = SynthInst(make_tables(dv, sdv, wave), np.arange(1.0))
    _, rdb = run_rdb(inst)
    assert abs(rdb['vrad'][0] - mp.odd_ratio_mean(dv[0], sdv[0])[0]) < 1e-9


def test_a_short_run_keeps_lines_seen_twice_with_their_offset():
    """Three files, lines with offsets, some lines missing in one file: the
    zero point of those lines is the mean of their two differences, and they
    are not lost by the 100 sigma cut"""
    wave, velocity, offset, dv, sdv = synthetic_star(nepoch=3, offsets=20.0,
                                                     seed=8)
    dv[0, :100] = np.nan
    inst = SynthInst(make_tables(dv, sdv, wave), np.arange(3.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out, saved = capture(general.make_rdb_table, inst, 'x.rdb',
                             inst.files(), '.', names=['per_line_mean'])
    plm = np.array(saved['per_line_mean'])
    assert np.all(np.isfinite(plm[:100]))
    assert np.corrcoef(plm[:100], offset[:100])[0, 1] > 0.9


def test_numbers_written_as_text_are_averaged_per_night():
    """An airmass that the header gives as text is averaged over the night,
    as a number; real text, or the same number in every row (an object
    called 10700), stays as its first value"""
    table = Table()
    table['MJD'] = [100.1, 100.2, 101.1]
    table['AIRMASS'] = ['1.2', '1.4', '1.1']
    table['OBJECT'] = ['10700', '10700', '10700']
    table['DATE'] = ['2024-01-01', '2024-01-01', '2024-01-02']
    table['TAU'] = ['NaN', 'NaN', 'NaN']
    inst = SynthInst([], table['MJD'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rdb2 = general.make_rdb_table2(inst, table)
    assert rdb2['AIRMASS'].dtype.kind == 'f'
    assert np.allclose(rdb2['AIRMASS'], [1.3, 1.1])
    assert list(rdb2['OBJECT']) == ['10700', '10700']
    assert list(rdb2['DATE']) == ['2024-01-01', '2024-01-02']
    assert list(rdb2['TAU']) == ['NaN', 'NaN']
