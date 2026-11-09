"""The real template recipe on synthetic files: a known star, on the EXPRES class
with only the place the files come from replaced."""
import os
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from lbl.core import io, math as mp, parameters
from lbl.instruments import expres, select
from lbl.recipes import lbl_template as recipe
from lbl.science import template_scatter

C_MS = mp.speed_of_light_ms
NPIX = 2048
NORD = 6
STEP_MS = 800.0


def _ratio(velocity):
    """The relativistic Doppler ratio of a velocity in m/s"""
    beta = np.asarray(velocity) / C_MS
    return np.sqrt((1 + beta) / (1 - beta))


class Files:
    """Spectra of one star on one wave map, one array per file: `noise_even`
    multiplies the noise of the even orders, `levels` the flux of each file,
    and `holes` lists the (file, order, first pixel, last pixel) missing"""

    def __init__(self, nfile=12, counts=4e4, noise_even=1.0, levels=None,
                 holes=(), bervamp=20000.0, seed=3):
        self.names = ['synth_{0:03d}'.format(it) for it in range(nfile)]
        rng = np.random.default_rng(seed)
        self.wave = np.zeros((NORD, NPIX))
        start = 520.0
        for order in range(NORD):
            self.wave[order] = start * np.exp(np.arange(NPIX) * STEP_MS / C_MS)
            start = self.wave[order, int(NPIX * 0.75)]
        self.berv = bervamp * np.cos(np.linspace(0, 3, nfile))
        self.blaze = np.tile(np.sinc(0.7 * (np.arange(NPIX) / NPIX - 0.5) * 2) ** 2,
                             (NORD, 1))
        self.levels = np.ones(nfile) if levels is None else np.asarray(levels)
        self.lines = np.sort(rng.uniform(self.wave.min(), self.wave.max(), 400))
        self.depths = rng.uniform(0.05, 0.6, self.lines.size)
        self.counts, self.noise_even = counts, noise_even
        self.holes, self.seed = holes, seed
        self.cache = {}

    def star(self, rest):
        """The spectrum on a wavelength vector in the star's frame"""
        tau = np.zeros_like(rest)
        for line, depth in zip(self.lines, self.depths):
            near = np.abs(rest / line - 1) < 12 * 2200.0 / C_MS
            tau[near] += -np.log(1 - depth) * np.exp(
                -0.5 * (np.log(rest[near] / line) * C_MS / 2200.0) ** 2)
        return np.exp(-tau)

    def flux(self, name):
        if name not in self.cache:
            it = self.names.index(name)
            rng = np.random.default_rng(self.seed * 100 + it)
            image = self.star(self.wave / _ratio(-self.berv[it]))
            image = image * self.blaze * self.counts * self.levels[it]
            sigma = np.sqrt(image)
            sigma[::2] = sigma[::2] * self.noise_even
            image = image + rng.normal(0.0, 1.0, image.shape) * sigma
            for fileno, order, first, last in self.holes:
                if fileno == it:
                    image[order, first:last] = np.nan
            self.cache[name] = image
        return np.array(self.cache[name])


class Instrument(expres.Expres):
    """The EXPRES class; only the files are replaced"""

    def __init__(self, params, args, files):
        super().__init__(params, args)
        self.files = files

    def science_files(self, directory):
        return list(self.files.names)

    def blaze_file(self, directory):
        return None

    def filter_files(self, science_files):
        return science_files

    def template_file(self, directory, tkind, required=True):
        return os.path.join(directory, 'LBL_Template_synth.fits')

    def load_header(self, filename, kind='fits file', extnum=None, extname=None):
        header = fits.Header()
        header['BERVS'] = self.files.berv[self.files.names.index(str(filename))]
        return io.LBLHeader.from_fits(header, str(filename))

    def load_science_file(self, science_file):
        return (self.files.flux(str(science_file)),
                self.load_header(science_file))

    def get_wave_solution(self, science_filename=None, data=None, header=None):
        return np.array(self.files.wave)

    def load_blaze_from_science(self, science_file, sci_image, sci_hdr,
                                calib_directory, normalize=True):
        return np.array(self.files.blaze), False

    def get_berv(self, sci_hdr):
        return float(sci_hdr['BERVS'])

    def populate_sci_table(self, filename, tdict, sci_hdr, berv=0.0):
        tdict = self.add_dict_list_value(tdict, 'FILENAME', str(filename))
        return self.add_dict_list_value(tdict, 'BERV', berv)


def _run(files, data_dir, **over):
    """run_template on these files; its locals, and the template it wrote"""
    arguments = sys.argv
    sys.argv = sys.argv[:1]
    try:
        args = select.parse_args(
            recipe.ARGS_TEMPLATE,
            dict(INSTRUMENT='EXPRES', DATA_SOURCE='ORIG', DATA_DIR=str(data_dir),
                 DATA_TYPE='SCIENCE', OBJECT_SCIENCE='SYNTH',
                 OBJECT_COMPARISON='SYNTH', VERBOSE=1),
            'test')
    finally:
        sys.argv = arguments
    inst = Instrument(parameters.params.copy(), args, files)
    inst.params['OVERWRITE'] = True
    inst.params['MODEL_REPO_URL'] = None
    inst.params['COMPIL_WAVE_MIN'] = float(np.floor(files.wave.min() + 1))
    inst.params['COMPIL_WAVE_MAX'] = float(np.ceil(files.wave.max() - 1))
    inst.params['COMPIL_SLOPE_REF_WAVE'] = float(np.median(files.wave))
    inst.params['USE_TQDM'] = False
    inst.params['PLOT'] = False
    inst.params['SMALL_RV_SHIFT'] = True
    for key in over:
        inst.params[key] = over[key]
    out = recipe.run_template(inst, 'SYNTH', 'science')
    with fits.open(out['template_file']) as hdul:
        table = Table(hdul['TEMPLATE'].data)
    return out, table


def test_a_low_snr_order_cuts_its_own_rms_and_not_the_other_parity(tmp_path):
    # only the even orders are too noisy: their flux and their rms go, the
    #   odd orders' stay, each rms with the flux of its own parity
    files = Files(noise_even=200.0)
    _, table = _run(files, tmp_path)
    even, odd = np.array(table['flux_even']), np.array(table['flux_odd'])
    assert np.isfinite(odd).sum() > 1000
    assert np.array_equal(np.isfinite(table['rms_even']), np.isfinite(even))
    assert np.array_equal(np.isfinite(table['rms_odd']), np.isfinite(odd))


def test_a_point_one_file_lacks_is_the_mean_of_the_files_that_have_it():
    # two files in a bin, the second three times brighter: where it lacks a
    #   point, the bin is the first file's value, with no step
    wave = np.linspace(600.0, 601.0, 50)
    shape = 1 - 0.5 * np.exp(-0.5 * ((wave - 600.5) / 0.05) ** 2)
    weight = np.full(wave.size, 2.0)
    faint, bright = 100.0 * weight * shape, 300.0 * weight * shape
    bright[20:25] = np.nan
    flux_bin, weight_bin, norm_bin = (np.zeros(wave.size) for _ in range(3))
    for flux in (faint, bright):
        recipe.add_to_bin(flux_bin, weight_bin, norm_bin, flux, weight,
                          recipe.file_level(flux, weight))
    assert np.allclose(flux_bin / norm_bin, shape / np.median(shape))
    # the weight is the faint file's alone there, both files' elsewhere
    assert np.allclose(weight_bin[20:25], 2.0)
    assert np.allclose(np.delete(weight_bin, range(20, 25)), 4.0)


def test_a_hole_in_one_file_does_not_empty_its_bin(tmp_path):
    # (BERVBIN) a hole in one order of one file, three times brighter than
    #   the others: every bin keeps the points there, and the hole's bin has
    #   no step
    levels = np.ones(24)
    levels[0] = 3.0
    files = Files(nfile=24, levels=levels, holes=[(0, 2, 900, 960)])
    out, _ = _run(files, tmp_path, TEMPLATE_MEDBINMAX=12,
                  TEMPLATE_METHOD='BERVBIN')
    wavegrid, cube = out['wavegrid'], out['built']['flux_cube']
    hole = mp.doppler_shift(files.wave[2, [905, 955]], -files.berv[0])
    inside = (wavegrid > hole[0]) & (wavegrid < hole[1])
    assert inside.sum() > 20
    assert np.all(np.isfinite(cube[inside]))
    # the hole's bin against the median of the bins, inside the hole and
    #   just outside: the same to the noise
    near = (wavegrid > hole[0] - 0.3) & (wavegrid < hole[1] + 0.3) & ~inside
    ratio = cube[:, out['built']['ibin'][0]] / np.nanmedian(cube, axis=1)
    assert abs(np.nanmedian(ratio[inside]) - np.nanmedian(ratio[near])) < 0.01


def test_the_low_pass_is_hp_width_on_the_grid_it_is_applied_to(tmp_path):
    # the template's grid is 3.2 times finer than the pixels (0.25 km/s
    #   against 0.8): the width in points is that of the grid
    assert recipe.lowpass_points(500, 250.0) == 2000
    out, _ = _run(Files(nfile=6), tmp_path)
    width_kms = out['hp_width'] * out['grid_step_magic'] / 1000
    assert out['grid_step_magic'] == 250.0
    assert width_kms == pytest.approx(out['inst'].params['HP_WIDTH'])


class Moving(Files):
    """The same star, moving: `vel` is its velocity in each file [m/s]"""

    def __init__(self, vel, **kwargs):
        super().__init__(**kwargs)
        self.vel = np.asarray(vel)

    def flux(self, name):
        if name not in self.cache:
            it = self.names.index(name)
            rng = np.random.default_rng(self.seed * 100 + it)
            velocity = mp.add_velocities(-self.berv[it], self.vel[it])
            image = self.star(self.wave / _ratio(velocity))
            image = image * self.blaze * self.counts
            image = image + rng.normal(0.0, 1.0, image.shape) * np.sqrt(image)
            self.cache[name] = image
        return np.array(self.cache[name])


def test_the_scatter_template_is_the_star_within_its_uncertainty(tmp_path):
    # SCATTER (the default): the template and its derivative are the star's,
    #   their scatter about it is the uncertainty they carry, and the orders
    #   join without a step
    files = Files(nfile=24)
    _, table = _run(files, tmp_path)
    wave, flux = np.array(table['wavelength']), np.array(table['flux'])
    good = np.isfinite(flux) & (wave > files.wave.min() + 1) & \
        (wave < files.wave.max() - 1)
    wave, flux = wave[good], flux[good]
    truth = files.star(wave)
    level = mp.lowpassfilter(flux / truth, 501)
    assert np.ptp(level) < 3e-3
    ratio = np.std(flux - truth * level) / np.median(table['eflux'][good])
    assert 0.8 < ratio < 1.3
    # the derivative per step of the grid
    dln, h = np.log(wave[1] / wave[0]), 0.01
    slope = (files.star(wave * np.exp(h * dln)) -
             files.star(wave * np.exp(-h * dln))) / (2 * h)
    error = np.array(table['flux_savgol_d1'])[good] - slope * level
    ratio = np.std(error) / np.median(table['eflux_savgol_d1'][good])
    assert 0.8 < ratio < 1.35


def test_the_scatter_template_measures_the_velocities_of_the_star(tmp_path):
    # SMALL_RV_SHIFT False: a star moving by +-1.5 km/s, its velocities found
    #   (up to their mean, which the template takes) to a few m/s
    vel = 1500.0 * np.sin(np.linspace(0, 9, 24))
    out, _ = _run(Moving(vel, nfile=24), tmp_path, SMALL_RV_SHIFT=False)
    rv0 = np.array(out['rv0'])
    assert np.std((rv0 - rv0.mean()) - (vel - vel.mean())) < 8.0


def _own_template_velocities(files, resel):
    """The velocity of each file against the SCATTER template made of all
    the files, the window resel resolution elements wide (converged)"""
    arguments = sys.argv
    sys.argv = sys.argv[:1]
    try:
        args = select.parse_args(
            recipe.ARGS_TEMPLATE,
            dict(INSTRUMENT='EXPRES', DATA_SOURCE='ORIG', DATA_DIR='.',
                 DATA_TYPE='SCIENCE', OBJECT_SCIENCE='SYNTH',
                 OBJECT_COMPARISON='SYNTH', VERBOSE=1), 'test')
    finally:
        sys.argv = arguments
    inst = Instrument(parameters.params.copy(), args, files)
    nfile = len(files.names)
    step = np.log(1 + 250.0 / C_MS)
    lngrid = np.log(files.wave.min()) + np.arange(
        int(np.log(files.wave.max() / files.wave.min()) / step)) * step
    fit_params = dict(fwhm=resel / inst.params['APPROX_RESOLUTION'],
                      clip=5.0, obs_clip=2.0)
    orders = list(range(NORD))
    hp_pix = np.full(NORD, 500)
    data = template_scatter.read_orders(
        inst, files.names, lambda name: (files.flux(name), None),
        np.array(files.blaze), '', orders, hp_pix, lambda x, **kw: x)
    shifts = template_scatter.star_shift(files.berv, np.zeros(nfile))
    fits = {io_: template_scatter.fit_order(data, io_, shifts, lngrid,
                                            fit_params, 500, 4)
            for io_ in orders}
    rv = np.zeros(nfile)
    for _ in range(5):
        dv = template_scatter.measure_velocities(
            data, fits, template_scatter.star_shift(files.berv, rv), lngrid,
            scan=False, scan_step=step)
        rv = mp.add_velocities(rv, dv)
    return rv


def test_the_scatter_template_does_not_pull_the_velocities_to_itself():
    # 15 m/s added to spectra of SNR ~60, each measured against the template
    #   made of them all, less the same without it: the default window gives
    #   the velocity back (bar the ~1/N of the template a spectrum is), a
    #   window of 0.4 resolution element (3 steps of the grid, templatepy's)
    #   pulls it towards the template, by 16 % as on 40 NIRPS spectra of
    #   GJ 1028: there the template copies each spectrum's own noise sharply
    nfile = 24
    vel = 15.0 * np.sin(2 * np.pi * np.arange(nfile) / 7.3)
    slopes = []
    for resel in (parameters.params['TEMPLATE_SCATTER_FWHM'], 0.4):
        still = _own_template_velocities(
            Moving(np.zeros(nfile), nfile=nfile, counts=4e3,
                   bervamp=26000.0), resel)
        moving = _own_template_velocities(
            Moving(vel, nfile=nfile, counts=4e3, bervamp=26000.0), resel)
        slopes.append(np.polyfit(vel, moving - still, 1)[0])
    assert slopes[0] > 0.97
    assert slopes[1] < 0.9
