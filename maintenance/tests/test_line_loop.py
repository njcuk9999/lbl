"""The line-by-line kernel of compute_rv (lbl.core.fastmath.line_loop) on a
model with known additions: each residual projection table reads its own
signal, the line statistics do not depend on the tables, and a line that fails
in a pass has no velocity rather than the one of an earlier pass."""
import numpy as np

from lbl.core import fastmath

NPIX = 400
NLINE = 9
XPIX = np.arange(NPIX, dtype=float)
WAVE = (600.0 + 0.002 * XPIX)[None, :].copy()
LEVEL = 2000.0
BLAZE = np.full((1, NPIX), LEVEL)
B_RATIO = np.full((1, NPIX), 1.3)
NORM = np.full((1, NPIX), 0.9)
RMS = np.full((1, NPIX), 5.0)
STARTS = 2 + 42 * np.arange(NLINE)


def _model():
    flux = np.ones(NPIX)
    for center in np.arange(20.0, NPIX, 40.0):
        flux -= 0.4 * np.exp(-0.5 * ((XPIX - center) / 4.0) ** 2)
    model = (flux * LEVEL)[None, :]
    dmodel = np.gradient(model, axis=1) / 500.0
    d2model = np.gradient(dmodel, axis=1) / 500.0
    d3model = np.gradient(d2model, axis=1) / 500.0
    return model, dmodel, d2model, d3model


MODEL, DMODEL, D2MODEL, D3MODEL = _model()
GRAD = (1e-3 * np.sin(XPIX / 9.0))[None, :]


def run(sci, proj_models, iteration=3, last=True, model=None, out=None,
        mask_keep=None):
    """One pass of line_loop; out (the arrays of a previous pass) is updated"""
    model = MODEL if model is None else model
    nkey = proj_models.shape[0]
    if out is None:
        out = dict(arrs=[np.full(NLINE, np.nan) for _ in range(9)],
                   meanxpix=np.zeros(NLINE), meanblaze=np.zeros(NLINE),
                   rmsratio=np.zeros(NLINE),
                   npixline=np.zeros(NLINE, dtype=np.int64),
                   chi2=np.zeros(NLINE),
                   proj=np.full((nkey, NLINE), np.nan),
                   sproj=np.full((nkey, NLINE), np.nan))
        out['arrs'][8] = np.zeros(NLINE)
    if mask_keep is None:
        mask_keep = np.ones(NLINE, dtype=bool)
    fastmath.line_loop(iteration, last, np.zeros(NLINE, dtype=np.int64),
                       600.0 + 0.002 * STARTS, 600.0 + 0.002 * (STARTS + 40),
                       STARTS.astype(np.int64), (STARTS + 40).astype(np.int64),
                       mask_keep, WAVE, sci[None, :], RMS, model, DMODEL,
                       D2MODEL, D3MODEL, BLAZE, B_RATIO, NORM, proj_models, 3,
                       *out['arrs'], out['meanxpix'], out['meanblaze'],
                       out['rmsratio'], out['npixline'], out['chi2'],
                       out['proj'], out['sproj'],
                       np.zeros(NLINE, dtype=bool), np.zeros(NLINE, dtype=bool))
    names = ['dv', 'sdv', 'd0v', 'sd0v', 'd2v', 'sd2v', 'd3v', 'sd3v', 'frac']
    out.update(zip(names, out['arrs']))
    return out


def _sci_with_gradient(delta):
    """S = M + delta * (b n) * g: a projection of delta on the table g"""
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 5.0, NPIX)
    return MODEL[0] + delta * B_RATIO[0] * NORM[0] * GRAD[0] + noise


def test_every_table_reads_the_same_signal():
    sci = _sci_with_gradient(7.0)
    one = run(sci, GRAD[None, :, :])
    two = run(sci, np.array([GRAD, GRAD]))
    # the first table as before, the second identical to it
    assert np.allclose(two['proj'][0], one['proj'][0], rtol=0, atol=0)
    assert np.allclose(two['proj'][1], two['proj'][0], rtol=0, atol=0)
    assert np.allclose(two['sproj'][1], two['sproj'][0], rtol=0, atol=0)
    assert abs(np.median(two['proj'][1]) - 7.0) < 3 * np.median(
        two['sproj'][1])


def test_line_statistics_do_not_depend_on_the_tables():
    sci = _sci_with_gradient(7.0)
    none = run(sci, np.zeros((0, 1, 1)))
    two = run(sci, np.array([GRAD, GRAD]))
    assert np.array_equal(none['rmsratio'], two['rmsratio'])
    assert np.array_equal(none['chi2'], two['chi2'])
    # pure noise of the declared rms: a ratio near 1
    assert abs(np.median(none['rmsratio']) - 1) < 0.2
    # the velocity is computed before the tables, as before
    assert np.array_equal(none['dv'], two['dv'])


def test_a_line_that_fails_in_a_later_pass_has_no_velocity():
    sci = MODEL[0] + 12.0 * DMODEL[0]
    out = run(sci, np.zeros((0, 1, 1)), iteration=1, last=False)
    assert np.all(np.isfinite(out['dv']))
    # the model of line 4 turns NaN: its window is rejected in the next pass
    model = MODEL.copy()
    model[0, STARTS[4] + 10] = np.nan
    out = run(sci, np.zeros((0, 1, 1)), iteration=2, last=True, model=model,
              out=out)
    assert np.isnan(out['dv'][4]) and np.isnan(out['sdv'][4])
    assert np.isnan(out['d2v'][4])
    assert np.all(np.isfinite(np.delete(out['dv'], 4)))


def test_a_line_flagged_bad_has_no_velocity_in_the_later_passes():
    sci = MODEL[0] + 12.0 * DMODEL[0]
    out = run(sci, np.zeros((0, 1, 1)), iteration=1, last=False)
    keep = np.ones(NLINE, dtype=bool)
    keep[2] = False
    out = run(sci, np.zeros((0, 1, 1)), iteration=2, last=False, out=out,
              mask_keep=keep)
    assert np.isnan(out['dv'][2])
    assert np.isfinite(out['dv'][3])
