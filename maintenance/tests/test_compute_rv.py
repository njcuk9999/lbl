"""lbl.science.general.compute_rv on a synthetic star with a closed form
(Gaussian lines of known depth and width on a log-wavelength axis): the
spectrum is synthetic, the measurement is the real code."""
import copy

import numpy as np
import pytest
from astropy.table import Table

from lbl.core import base_classes, io
from lbl.core import math as mp
from lbl.science import activity_lines, general

C_MS = mp.speed_of_light_ms
#: width of the lines (sigma), in m/s: FWHM 8 km/s
SIGMA = 8000.0 / mp.fwhm_value()
#: size of a pixel, in m/s
PIX_MS = 1000.0
#: step of the template grid, in m/s
TEMPLATE_STEP = 250.0


class Quiet:
    """Takes the place of the LBL logger"""
    def __getattr__(self, name):
        def nothing(*args, **kwargs):
            return None
        return nothing


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    silent = Quiet()
    monkeypatch.setattr(general, 'log', silent)
    monkeypatch.setattr(activity_lines, 'log', silent)
    monkeypatch.setattr(io, 'log', silent)


class Header(dict):
    """What compute_rv reads from a header"""
    def get_hkey(self, key, dtype=None, required=True):
        value = self[key]
        return dtype(value) if dtype is not None else value


class Star:
    """Lines at fixed places: F(u) = 1 - sum d_i G(u - u_i), u = c ln(wave)"""

    def __init__(self, wave0=500.0, norders=8, npix=2048, seed=1):
        rng = np.random.default_rng(seed)
        self.norders, self.npix = norders, npix
        step = PIX_MS / C_MS
        self.wave = np.zeros((norders, npix))
        uu0 = np.log(wave0)
        for order in range(norders):
            self.wave[order] = np.exp(uu0 + np.arange(npix) * step)
            uu0 = np.log(self.wave[order, -1]) - 100 * step
        umin = C_MS * np.log(self.wave.min()) - 4e5
        umax = C_MS * np.log(self.wave.max()) + 4e5
        centres = [umin]
        while centres[-1] < umax:
            centres.append(centres[-1] + rng.uniform(25e3, 50e3))
        self.centres = np.array(centres[1:-1])
        self.depths = rng.uniform(0.05, 0.7, size=len(self.centres))
        xx = np.arange(npix) / (npix - 1) - 0.5
        self.blaze = np.tile(0.25 + 0.75 * np.cos(np.pi * xx) ** 2,
                             (norders, 1))

    def flux(self, uu, shift=0.0):
        out = np.ones_like(uu)
        for centre, depth in zip(self.centres + shift, self.depths):
            near = np.abs(uu - centre) < 8 * SIGMA
            out[near] -= depth * np.exp(-0.5 * ((uu[near] - centre) / SIGMA)
                                        ** 2)
        return out

    def template_table(self):
        umin = C_MS * np.log(self.wave.min()) - 3.5e5
        umax = C_MS * np.log(self.wave.max()) + 3.5e5
        uu = np.arange(umin, umax, TEMPLATE_STEP)
        table = Table()
        table['wavelength'] = np.exp(uu / C_MS)
        table['flux'] = self.flux(uu)
        table['rms'] = np.full(len(uu), 1e-3)
        return table

    def mask_table(self, offset=0.0, centres=None, depths=None):
        """Minima at the line centres (w_mask > 0), maxima half way between
        lines (w_mask < 0), displaced by offset (m/s)"""
        centres = self.centres if centres is None else centres
        depths = self.depths if depths is None else depths
        mids = 0.5 * (centres[1:] + centres[:-1])
        uu = np.concatenate([centres[1:-1], mids[1:-1]]) + offset
        weight = np.concatenate([np.ones(len(centres) - 2),
                                 -np.ones(len(mids) - 2)])
        depth = np.concatenate([depths[1:-1], -np.full(len(mids) - 2, 0.01)])
        order = np.argsort(uu)
        table = Table()
        table['ll_mask_s'] = np.exp(uu[order] / C_MS)
        table['ll_mask_e'] = table['ll_mask_s']
        table['w_mask'] = weight[order]
        table['value'] = np.ones(len(uu))
        table['depth'] = depth[order]
        table['line_snr'] = np.full(len(uu), 100.0)
        return table

    def observe(self, velocity=0.0, flux=1e4, seed=None):
        """The spectrum at velocity (m/s, receding, exact relativistic
        shift), times blaze, with shot and readout noise"""
        delta = -C_MS * np.arctanh(velocity / C_MS)
        data = np.zeros_like(self.wave)
        for order in range(self.norders):
            uu = C_MS * np.log(self.wave[order]) + delta
            data[order] = flux * self.blaze[order] * self.flux(uu)
        if seed is None:
            return data
        rng = np.random.default_rng(seed)
        return data + np.sqrt(np.abs(data) + 25.0) * rng.standard_normal(
            data.shape)


class Inst:
    """The instrument object: what compute_rv and its helpers call"""

    def __init__(self, star, mask=None, **params):
        self.star = star
        self._mask = star.mask_table() if mask is None else mask
        self._template = star.template_table()
        self.params = dict(
            USE_NOISE_MODEL=False, HP_WIDTH=500, KW_MID_EXP_TIME='MJDMID',
            COMPUTE_RV_N_ITERATIONS=20, COMPUTE_MODEL_PLOT_ORDERS=[0],
            COMPUTE_LINE_MIN_PIX_WIDTH=5, COMPUTE_LINE_NSIG_THRES=8,
            COMPUTE_RV_BULK_ERROR_CONVERGENCE=1.0,
            COMPUTE_RV_MAX_N_GOOD_ITERS=15, COMPUTE_RMS_SIGCLIP_THRES=5,
            READ_OUT_NOISE=5.0, RESPROJ_TABLES=None,
            NOISE_SAMPLING_WIDTH=2e5, NOISE_SAMPLING_NSTEP=4,
            DATA_TYPE='SCIENCE', PLOT=False, PLOT_COMPUTE_LINES=False,
            PLOT_COMPUTE_CCF=False, PLOT_COMPUTE_SYSVEL=False,
            USE_SAVGOL_TEMPLATE=False, COMPUTE_FLAG_SPURIOUS_SIG=10,
            ROTBROAD=None, BERV=0.0, ROUGH_CCF_MIN_RV=-3e5,
            ROUGH_CCF_MAX_RV=3e5, ROUGH_CCF_EWIDTH_GUESS=2000,
            COMPUTE_CCF_HP_SCALE=30, CCF_SNR_MIN=7, ITERATION=-1, TOTAL=-1,
            REF_TABLE_FMT='csv')
        self.params.update(params)

    def get_wave_solution(self, science_filename=None, data=None,
                          header=None):
        return self.star.wave

    def get_berv(self, header):
        return header['BERV']

    def load_mask(self, filename):
        return self._mask

    def load_template(self, filename, get_hdr=False):
        return self._template

    def get_sample_wave_grid(self, calib_dir, science_file):
        return self.star.wave


def setup(tmp_path, star=None, mask=None, **params):
    """A star, its instrument, reference table and template splines"""
    star = Star() if star is None else star
    inst = Inst(star, mask=mask, **params)
    ref = general.make_ref_dict(inst, str(tmp_path / 'ref_table.csv'), False,
                                ['synthetic'], 'mask', str(tmp_path))
    splines = general.spline_template(inst, 'template', 0.0, str(tmp_path))
    return star, inst, ref, splines


def run(inst, data, ref, splines, vsys=0.0, model_velocity=0.0):
    """The real compute_rv: (line table, outputs); ref is left as it was"""
    table = copy.deepcopy(ref)
    header = Header(MJDMID=60000.0, BERV=0.0)
    props = dict(SCI_TABLE=None, VSYS=vsys, FWHM=8000.0, MASK_SYS_VEL=0.0)
    return general.compute_rv(inst, 0, np.array(data), header,
                              splines=splines, ref_table=table,
                              blaze=inst.star.blaze, systemic_props=props,
                              systemic_all=np.full(1, np.nan),
                              mjdate_all=np.zeros(1),
                              model_velocity=model_velocity,
                              science_file='synthetic.fits',
                              mask_file='mask')


def bulk(table):
    """Bulk velocity and error of a line table, and its number of lines"""
    dv = np.array(table['dv'], dtype=float)
    sdv = np.array(table['sdv'], dtype=float)
    mean, err = mp.odd_ratio_mean(dv, sdv)
    return mean, err, int(np.sum(np.isfinite(dv)))


def test_a_start_off_in_velocity_ends_at_the_same_velocity(tmp_path):
    """The sigma clip of a pass is made against the model of that pass: the
    flanks of deep lines, clipped while the model is 1 km/s away, come back
    when it has reached the spectrum, so a start off by 1 km/s gives the
    same velocity and the same lines as a start at the right velocity"""
    star, inst, ref, splines = setup(tmp_path)
    data = star.observe(velocity=0.0, seed=10)
    v0, e0, n0 = bulk(run(inst, data, ref, splines, vsys=0.0)[0])
    for offset in (-1000.0, 1000.0):
        v1, e1, n1 = bulk(run(inst, data, ref, splines, vsys=offset)[0])
        assert abs(n1 - n0) <= 2
        assert abs(v1 - v0) < 0.05 * e0
        assert abs(e1 / e0 - 1) < 0.01
