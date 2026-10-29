"""YAML configuration handling with defaults and a reproducibility hash."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re

import yaml

from lbl.core import astro


class _ConfigLoader(yaml.SafeLoader):
    """yaml.SafeLoader without YAML 1.1's base-60 numbers.

    PyYAML follows YAML 1.1, where digits separated by a colon are a number in
    base 60: an unquoted `1267:2` is read as 1267 * 60 + 2 = 76022, and even
    `2450:0.5` as 147000.5. The windows in output.windows are written exactly
    that way, centre:width in nm, so three of the eight became numbers in the
    tens of thousands, landed outside the grid, and were skipped by the figure
    scripts without a word. Only the ones whose centre had a decimal point
    survived, because `1200.3:2` is not a base-60 pattern.

    Everything else is SafeLoader: the same ints, floats, booleans and nulls,
    so no other value in a config changes type.
    """


_ConfigLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in entries
            if tag not in ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float")]
    for first, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()}
# PyYAML 6's own two patterns, each minus its base-60 alternative
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"""^(?:[-+]?0b[0-1_]+
                    |[-+]?0[0-7_]+
                    |[-+]?(?:0|[1-9][0-9_]*)
                    |[-+]?0x[0-9a-fA-F_]+)$""", re.X),
    list("-+0123456789"))
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
                    |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def read_yaml(stream):
    """A YAML document, read the way this package's configs mean it."""
    return yaml.load(stream, Loader=_ConfigLoader)


def check_windows(windows, domain):
    """Say out loud about any window no figure can be drawn in.

    The figure scripts skip such a window in a subprocess whose output is only
    shown when it fails, so a window that could not be drawn used to vanish from
    the PDF with nothing said. A number where a centre:width was meant is the
    signature of the base-60 reading above, from a config read some other way.
    """
    from .logger import log

    if domain.get("wave_min") is None or domain.get("wave_max") is None:
        return
    lo, hi = float(domain["wave_min"]), float(domain["wave_max"])
    for spec in windows or []:
        centre, _, width = str(spec).partition(":")
        try:
            c, w = float(centre), float(width or 2.0)
        except ValueError:
            log("output.windows: %r is not centre:width in nm, so it is not"
                " drawn" % (spec,), "warn")
            continue
        if not lo <= c <= hi:
            hint = ("; a number here is what YAML 1.1 makes of an unquoted"
                    " centre:width, 1267:2 being read as 76022"
                    if isinstance(spec, (int, float)) else "")
            log("output.windows: %s is centred at %.1f nm, outside the domain"
                " %.1f-%.1f nm, so it is not drawn%s" % (spec, c, lo, hi, hint),
                "warn")

# Every tunable lives here; config.yaml only needs to override what differs.
DEFAULTS = {
    "input": {
        # exposures read, resampled and high-passed at once, in processes,
        # when the cube is built (cube._parallel_map): the same cube to the
        # bit, and left out of the cache key, since it says nothing about
        # what the cube holds
        "workers": 4,
        # How the spectra are read, from the instrument's LBL class
        # (PCA2D_INPUT_FORMAT): 'tfits' = APERO order-by-order spectra, read
        # by tfits.py with the extensions below; 's2d' or 'e2ds' = order by
        # order through the LBL class (readers.py); 's1d' = the resampled
        # s1d_v products
        "format": "tfits",
        # the t.fits extensions (PCA2D_TFITS_EXTENSIONS): flux, wave, blaze,
        # recon and sky
        "extensions": None,
        # the header keywords read into the metadata when PCA2D reads the
        # files itself, {name: [keyword, ...]} (the KW_* of the LBL class,
        # Instrument.pca2d_header_keys)
        "header_keys": None,
        # bring a barycentric wavelength grid back to the observer's frame
        # (PCA2D_TO_OBSERVER)
        "to_observer": False,
        # The input ROOT, and not the folder of spectra: one run reads
        # <directory>/<object>/, so the same config serves every target on the
        # disk and a run needs nothing but a name. See spectra_dir().
        "directory": "data",
        "pattern": "*.fits",         # PCA2D_INPUT_PATTERN of the LBL class
        "max_files": None,           # None = all; small int for quick tests
        "object": None,              # keep only this OBJECT (None = no filter).
                                     # It is also the FOLDER: one run reads
                                     # <directory>/<object>/.
        # What to match in the headers, when the folder is not named as the
        # observer typed the target: two campaigns of one star on two
        # instruments live in two folders (a run is one instrument, and two
        # instruments must not share a folder or an LBL object) while their
        # headers name the same star. None means the folder's own name.
        "object_header": None,
        # "auto" (the default), True or False. Coadding a night is not a
        # modelling choice, it exists so the fit holds in memory, and fitting
        # the individual spectra is strictly more information. "auto" weighs
        # the fit's footprint against output.max_memory_gb, a config value and
        # not the machine's RAM, so that the decision stays a pure function of
        # the configuration the cube cache key hashes. See cube.resolve_stacking.
        "nightly_stack": "auto",     # coadd the exposures of each night before
                                     # the PCA. Every exposure is still
                                     # registered with its own BERV first, so
                                     # nothing is smeared; with several
                                     # exposures per visit this makes the cube
                                     # (and the decomposition) that many times
                                     # smaller (PCA2D_NIGHTLY_STACK)
        # --- 'tfits' only -------------------------------------------------
        # with stacking, each night cut into bundles of at most this BERV
        # range (km/s), this many minutes or this many exposures: a long
        # sequence (a transit) is not one sky, and in the observer's frame a
        # coadd smears the star by its BERV range (nights.split_nights).
        # 0.1 km/s keeps an ordinary visit of a few exposures (a few tens of
        # m/s of BERV) one row and cuts a sequence of several hours (hundreds
        # of m/s). None: whole nights
        "bundle_minutes": None,
        "bundle_berv": 0.1,
        "bundle_max": None,
        "orders": None,              # None = every order; or an explicit list
        "min_order_finite_fraction": 0.05,   # skip an order with fewer usable
                                     # pixels than this. An order inside a
                                     # saturated water band can be lost
                                     # entirely; that is expected, not an
                                     # error.
        "blaze_min_frac": 0.05,      # reject pixels where the blaze has fallen
                                     # below this fraction of its order peak
    },
    "domain": {
        # nm, the instrument's domain (wavemin and wavemax of its LBL class)
        "wave_min": None,
        "wave_max": None,
        "grid_source": "magic",      # 'magic'  = build the log-uniform grid from
                                     #            wave0 + dv below (the only
                                     #            option that works for t.fits,
                                     #            which carry no common grid)
                                     # 'native' = recycle the grid stored in the
                                     #            first s1d_v file
        # nm, anchor of the magic grid: a whole number of nm at or below the
        # instrument's wave_min (Instrument.pca2d_settings)
        "wave0": None,
        # Where dv comes from. False: the number below. True: the finest pixel
        # step in the first spectrum, times SMART_DV_FRACTION, and THE NUMBER
        # BELOW IS NOT READ AT ALL. A float here is that fraction. The measured
        # step is kept as domain.pixel_dv for the record.
        "smart_dv": False,
        # km/s per pixel (500 m/s). Read only when smart_dv is false, and may
        # be null when it is true, since nothing would look at it.
        "dv": 0.5,
        # The photometric bands, nm, LBL's own (lbl.core.astro.bands, u to
        # K); the instrument's layer keeps those that overlap its domain. The
        # PCA is fitted on the part of them the grid covers, and only there
        # (grids.band_mask).
        "bands": {band.name: [band.minimum, band.maximum]
                  for band in astro.bands},
    },
    # What is PUBLISHED about the target, which the pipeline never measures and
    # never needs, except to mark the periods a component must not vary at. The
    # coefficient periodogram takes them: a basis component varying at a planet's
    # period subtracts that planet out of the spectra, and the velocities then
    # come back cleaner precisely because the signal is gone. Per object, under
    # objects.<NAME>.target.
    "target": {
        "planets": [],               # known orbital periods, days
        "prot": None,                # published rotation period, days
    },
    "registration": {
        "frame": "barycentric",      # 'barycentric' (stellar rest frame) or
                                     # 'observer' (telluric rest frame, no shift)
        "target_berv": 0.0,          # km/s; frame the spectra are registered to
        "spline_order": 3,           # cubic spline; the BERV shift is fractional
        "mask_threshold": 0.999,     # good-pixel fraction required after resampling
        # valid native pixels dropped on either side of every gap before the
        # spline: it bends toward the straight line a gap is filled with, and
        # beside an OH core that leaves sky in samples that read as valid
        "edge_pixels": 1,
        # the Keplerian orbits of the star's companions (period, k, e, omega,
        # tp or t0, as lbl/resources/planets/*.yaml): the star's frame is then
        # the BERV composed with their velocity at each exposure, not the BERV
        # alone (readers.star_frame). Set per object by LBL's PLANETS option.
        # Empty: the BERV alone
        "orbits": [],
        # THE VELOCITY OF EACH EXPOSURE, {file name: km/s}, from a free fit of
        # them (LBL's template, when it iterates on the velocities: its
        # SCI_TABLE's VSYS): the star's frame is then the BERV composed with
        # that, a planet and the star's activity followed whatever they are.
        # Taken over the orbits for the files it names. Set per object by
        # lbl_pca2d. Empty: the orbits, or the BERV alone
        "velocities": {},
    },
    "highpass": {
        "method": "savgol",          # 'savgol' or 'none'
        # The Savitzky-Golay width IN KM/S, so the filter does the same thing
        # to a line whatever the grid step; `window`, in samples, is derived
        # from it once dv is known (resolve_highpass). A config that sets
        # `window` and not this keeps its window, which is how a run saved
        # that way reads back, cube key included.
        "width_kms": 100.0,
        "window": 201,               # samples (odd): what 100 km/s is at dv = 0.5
        "polyorder": 2,
        "mode": "divide",            # 'divide'  -> y = ln(f / lowpass(f))
                                     # 'log_sub' -> y = ln(f) - lowpass(ln f)
        "frame": "target",           # s1d only: apply before ('observer') or
                                     # after ('target') the BERV registration.
                                     # 'target', because the high-pass against
                                     # a template needs the common grid.
                                     # t.fits always filters on the destination
                                     # grid, per order -- the native sampling is
                                     # not uniform in velocity.
        # THE HIGH-PASS ON THE RESIDUAL TO THE TEMPLATE rather than on the
        # spectrum. A savgol run over a whole spectrum is pulled down by
        # every deep line, and when a telluric sweeps across the core of a
        # line with the BERV this bias changes FROM ONE EXPOSURE TO THE NEXT:
        # there then remains, in the star frame, a residual that follows the
        # BERV without being stellar. Over a season's BERV range a telluric
        # line moves by a tenth of a nm, and the depth it adds at the core of
        # a stellar line goes from nothing to most of it, intermittently.
        #
        # With this option the high-pass is done in TWO PASSES: a median
        # template of all the registered exposures, savgol on the RESIDUAL to
        # the template, then the stellar structure restored from the template
        # alone. No line is masked, nothing enters or leaves depending on the
        # BERV. Requires frame: target, since the high-pass has to be done in
        # the star frame.
        "against_template": True,
    },
    "quality": {
        "min_snr": 1.0,              # reject if the band SNR is below this
        "require_positive_flux": True,   # reject frames with median flux <= 0
        "max_nan_fraction": 0.5,     # reject if more of the band than this is NaN
        # Epoch window, as rjd = jd - 2400000. Both default to null, i.e. keep
        # everything: excluding part of a time series is a claim about the data,
        # not a hygiene default, and it must be made in the config where it is
        # visible (an instrument's commissioning season, for example).
        # `quality` is hashed into the cube cache key, so setting either of
        # these rebuilds the cube rather than silently reusing one built over a
        # different window.
        # Discard a sample where the sky emission the pipeline subtracted was
        # more than this many times the stellar flux. null keeps everything.
        # Hashed into the cube cache key only when set.
        "max_sky_ratio": None,
        # Drop a sample that has a masked neighbour within this many samples on
        # BOTH sides: it survived the cuts but its neighbourhood did not, so its
        # high pass and any shift of it were built from samples that are not
        # there. null disables the check.
        # Hashed into the cube cache key only when set.
        "isolated_window": None,
        "min_rjd": None,
        "max_rjd": None,
        "nights": None,
        # THE SPECTROGRAPH MODE: the list of modes kept, read from the
        # headers' DRS mode (PCA2D_DRS_MODES of the instrument's LBL class).
        # null keeps everything, the ordinary case of a single mode. Another
        # mode (another fibre, another resolution) has its own wavelength
        # solution and velocity zero point, so it is another instrument from
        # the point of view of a common basis. Hashed into the cube cache key
        # when set.
        "drs_modes": None,
    },
    "weights": {
        "mode": "photon",            # 'photon' or 'uniform'
        "snr_pixel_scale": 1.0,      # EXTSN is per e2ds pixel; rescale if wanted
        # THE PER-PIXEL THRESHOLD IS AN SNR THRESHOLD, not a depth one. The
        # bottom of a deep line carries fewer photons, hence a larger
        # fractional RMS, but it is where the velocity content is densest:
        # what has to be rejected is the pixel whose noise exceeds what the
        # photons explain, not the one that is dark.
        #
        # sigma being the photon sigma of the log-flux, 1/sigma IS the local
        # SNR. 5 is low on purpose: reject the dead pixel, not sort by
        # brightness. Set to null, `ln_clip_low` takes over again.
        "min_pixel_snr": 5.0,
        "ln_clip_low": None,         # y below this -> zero weight. Off: a
                                     # cut on ln f removes the bottom of every
                                     # deep line (at -0.5, every line deeper
                                     # than 39% of the continuum), where the
                                     # velocity content is; min_pixel_snr
                                     # above is the cut to use
        "ln_clip_high": None,        # y above this -> zero weight (None = keep)
        "sigma_floor_frac": 0.0,     # add this (in ln units) in quadrature
        "telluric_mode": "recon",        # 'recon'     = the paired
                                         #               *_s1d_v_recon_A.fits
                                         # 'from_data' = airmass regression
                                         # 'file'      = external model
                                         # 'none'      = no telluric weighting
        "telluric_ramp_zero": 0.5,       # transmission giving zero weight
        "telluric_ramp_one": 1.0,        # transmission giving full weight
        "telluric_file": None,           # for mode 'file': (wavelength_nm, T)
        "telluric_power": 2.0,           # for modes 'from_data' / 'file'
        "telluric_min_transmission": None,
        # Width in pixels of the boxcar used to measure the noise from the
        # sample-to-sample scatter, or null to trust the photon model alone.
        # The larger of the two sigmas is used. Hashed into the cube cache key
        # only when set, so turning it on rebuilds the cube and leaving it off
        # changes no existing key.
        "empirical_noise_box": None,
        "min_good_fraction": 0.2,    # drop grid columns with fewer good spectra
    },
    "output": {
        # The cube's storage dtype is NOT here: it is cube.CUBE_DTYPE, hard
        # float32, for the reasons written beside it.
        # The output ROOT: every product of a run lands in
        # <directory>/<object>/<M>-<N>/, corrected spectra included, so that
        # nothing is ever written beside the input files.
        "directory": "outputs",
        "tag": "run",
        # Rebuildable intermediates, deliberately NOT under the output root:
        # a cube costs twenty minutes and does not depend on where the products
        # of a run are asked to go, so pointing --out-dir somewhere new must
        # not orphan it.
        "cache_directory": "cache",
        # Where a run's products are kept, when that is not the internal disk:
        # each run folder, and LBL's, is a link to its place under it, made
        # before anything is written (storage.py).
        # None here, so that a clone on another machine does not aim at a disk
        # it does not have; config.yaml names the one this campaign uses.
        "fits_directory": None,
        "use_cache": True,
        # Whether a cube already in the cache is READ back. False builds it
        # again and still writes it, which is what --rebuild-cube means: a
        # rebuild that kept nothing left the fit with no cube to open, since
        # the cube stage is the only thing that writes one. use_cache False is
        # the other decision, no cache at all, and it stays that.
        "reuse_cache": True,
        "max_memory_gb": 12.0,       # refuse to allocate a cube larger than this
                                     # rather than driving the machine to swap
        # the windows the figures draw, centre:width in nm (PCA2D_WINDOWS of
        # the instrument's LBL class)
        "windows": [],
    },
    # ---------------------------------------------------------------- fit ---
    # The two-frame fit. Two component counts, never one: M in the stellar rest
    # frame and N in the observer frame, and they need not be equal.
    "twoframe": {
        # ZERO: the star is the per-parity spectrum `mean: star` takes out
        # with a coefficient of exactly 1, and a component in front of it would
        # be an exponent on the flux. Raised to look at the star's
        # variability, keep it small: with many star components the star block
        # has enough freedom to help describe an observer-frame line, through
        # the pair (feature, its derivative times the shift), and it then
        # takes part of what the observer block should hold.
        # THIS KEY COUNTS THE FREE COMPONENTS ONLY. What a user sets, the
        # PCA2D dict's n_star and --n-star, counts the star's spectra with the
        # template: n_star 1 is this 0, the template alone (star_spectra and
        # free_star_components below convert)
        "n_star": 0,
        "n_earth": 7,
        "iters": 16,
        # sweeps in a row worse than the best before the fit stops, and which
        # iterate it keeps: 'best' (lowest chi2) or 'last'
        "patience": 2,
        "keep": "best",
        "tie_parities": True,
        # Rejection threshold in ROBUST SIGMAS, 1.4826 x MAD, not in MADs:
        # 10 here is 14.83 plain MADs. `max_sigma` is the name; `max_mad` is
        # accepted as an alias because configs and logs on disk use it and the
        # value must keep meaning exactly what it meant.
        # Hierarchical template: median within each BERV bin, then median
        # across bins, so a clump of exposures sharing a barycentric velocity
        # cannot vote twice. LBL's values, so both steps of the chain bin the
        # same way. null restores the plain median.
        "template_berv_bin": None,        # m/s
        "template_berv_min_entries": 3,
        "max_mad": 10.0,
        "max_mad_rounds": 2,
        "clip": 3.0,
        "min_snr_frac": 0.5,
        # Storage for the two (rows, pixels) arrays the fit lives in. float32
        # halves them and their model temporaries, which is what makes fitting
        # every exposure possible instead of nightly means; the weighted sums
        # accumulate in float64 whatever this says.
        "dtype": "float64",
        # The static part of the model (twoframe --mean). "star", the
        # default: one star spectrum per order parity, in the STAR's frame,
        # taken out once before any component with a coefficient of exactly
        # 1. The alternatives are kept for comparison: "offset", a one-shot
        # per-parity observer-frame offset, and "full", which the correction
        # divides out with the observer block, so that a mean spectrum of the
        # star in the observer's frame goes back into the corrected spectra;
        # "iterate", one mean per order parity in EACH frame re-estimated at
        # every sweep with the components centred, separates the frames on a
        # synthetic cube but does not converge on real campaigns (the
        # star-frame mean moves more at every sweep).
        "mean": "star",
        # The one-shot star-frame median of the older modes; "iterate" makes
        # its own star-frame mean, one per parity, and ignores this.
        "template": False,
        "shift": "lanczos",
        "kernel_halfwidth": 8,
        "gap_guard": True,
        "leakage": True,
        "chunk": None,
        "order": "star_first",
        # One velocity per exposure, fitted beside the components as the
        # derivative of the reconstructed star carried to that exposure's BERV.
        # It exists to keep the star's own motion OUT of the observer block:
        # without it the block describes the shift, and dividing the block out
        # of the flux writes that velocity into the corrected spectrum. It is
        # fitted and never divided out, since taking it out of the data would
        # remove the signal being looked for. Off by default, as a check of
        # what it does; `--velocity-term` or `velocity_term: true` turns it
        # on.
        "velocity_term": False,
        # The shift is measured only on columns inside a photometric band whose
        # telluric transmission stays above this in 90% of the exposures. Zero
        # turns the transmission cut off and keeps the band cut. Fitting the
        # star's velocity where the flux is mostly atmosphere is fitting it to
        # the wrong thing, and the term then comes out larger than the
        # velocity LBL measures on the same spectra.
        "velocity_min_transmission": 0.95,
        # The instrument's resolving power, lambda/dlambda, its LBL class's
        # APPROX_RESOLUTION: the unit every Savitzky-Golay here is measured
        # in. It turns no smoothing on by itself.
        "resolution": None,
        # Smooth the star's spectra per parity (--mean star) and the star
        # components P to this FWHM, in resolution elements, by LBL's own
        # filter (pca2d.resolution). Off by default: smoothing the star
        # removes line structure that LBL then misses, and the velocities
        # degrade with the width. The observer components are never smoothed
        # here.
        "star_smooth": None,
        # CROSS-VALIDATION (pca2d.diagnostics.holdout_rows): "even" or "odd",
        # the nights held out of the fit. None, the nominal: all the nights.
        "holdout": None,
        # A SAVED OBSERVER BASIS (pca2d.basis, the PCA2D dict's basis=):
        # its components are taken frozen and never updated, and only the
        # star's block and the amplitudes are fitted, for a star whose BERV
        # barely moves and cannot tell its frame from the observer's. None,
        # the nominal: the observer components are fitted. Here and not in
        # nominal.yaml, whose content stamps every correction on disk.
        "earth_basis": None,
        # with earth_basis: its amplitudes fitted and taken out FIRST, then
        # the star's spectrum estimated, so that it does not take the mean of
        # what the basis describes (a narrow BERV range). False: the star's
        # spectrum from the data as they are, as without a frozen basis
        "observer_first": True,
        # ANOTHER STAR'S TEMPLATE as the star's spectrum, one entry per member
        # of the fit, {file, velocity} or None (estimated from the member's
        # own data): set by lbl_pca2d when OBJECT_COMPARISON is another star,
        # velocity the object's minus the template's (an LBL pass upstream)
        "star_templates": None,
        # The older spelling of the two above: smooth to one resolution element
        # of this R. Still read, so the runs made with it reproduce.
        "star_resolution": None,
        # How the star-side vectors are carried and updated: 'spline', one
        # cubic B-spline evaluated at every exposure's shifted positions and
        # updated exactly (pca2d.splinestar), the default: chi2 falls at every
        # sweep and the fit is faster. 'grid': the samples moved by the
        # Lanczos kernel and updated from the normal diagonal, an
        # approximation that gives velocities of the same quality.
        "star_basis": "spline",
    },
    # ---------------------------------------------------------------- lbl ---
    # Handing both sets of spectra to LBL, the delivered ones and the corrected
    # ones, as two objects in one LBL tree. See pca2d/lbl.py.
    # ------------------------------------------------------------ correct ---
    # What comes out of a corrected file. The star components are never
    # divided out: with no template the first of them IS the star.
    "correct": {
        # exposures corrected at once, in threads (reconstruct --workers): the
        # same files as one at a time, each read, refitted and written on its
        # own. Here and not in nominal.yaml, whose content is in every
        # correction's stamp
        "workers": 4,
        "n_star": 0,
        "n_earth": None,             # None = every observer component the fit has
        # THE UNIT IS AN EXCURSION, not a sample: one sample beyond 3 sigma is
        # noise (0.27% of it is, and 1% of these residuals were), while a run
        # of samples leaning the same way over a line's width is not. Every
        # window from one sample to excursion_elements resolution elements is
        # judged by the aggregate significance of its sum, and flagged, whole,
        # beyond excursion_nsig. The variance of that sum is measured, never
        # assumed: the grid oversamples the detector pixel, so neighbouring
        # samples of the residual are strongly correlated and a window of n
        # samples holds far fewer than n independent measurements
        # (outliers.window_variance, noise_correlation). Off (null) by
        # default; 6.0 with 2.0 elements is the setting it was designed for.
        "excursion_nsig": None,
        "excursion_elements": 2.0,
        # WHAT STANDS STILL IN THE OBSERVER'S FRAME. A leftover anchored there
        # (a telluric line the correction does not reach, airglow, a detector
        # feature) need not have one sign, but it always has scatter the noise
        # does not account for: the chi2 of a column over the exposures, above
        # 1. Averaged over excess_elements resolution elements, that average is
        # certain to sqrt(2 / (N * independent columns)), so its excess becomes
        # a significance: excess_nsig sigmas of it, and a windowed chi2 above
        # excess_chi2, drop the region from every exposure. A saturated
        # telluric band reaches a column chi2 of hundreds.
        # THE LOOP'S WEIGHT IS THE MASK. The loop already down-weights each
        # sample by (clip/|z|)^2 beyond the cut (twoframe.clip_weights);
        # below this fraction of its nominal weight, the sample was not
        # described by the model and becomes NaN in the corrected file. 0.5
        # is |z| > 4.24 sigma with the clip at 3, a fraction of a percent of
        # the samples, per exposure and not in common.
        "weight_floor": None,
        # THE COHERENT BIAS OVER THE EXPOSURES, the third axis and the only
        # one that neither the cut nor the excess scatter can see:
        # clip_weights compares each exposure to the OTHER EXPOSURES, so a
        # residual present in all of them is an outlier in none. The mean
        # over the exposures, on the other hand, gains sqrt(N) in
        # significance, and reveals whole badly corrected lines (methane and
        # water, several resolution elements wide).
        # The two axes add up because they see different things: the cut
        # (nsig_cut) sees the accidents of one exposure, the bias what is
        # anchored in the observer frame and that no per-exposure criterion
        # can see. A lower nsig_cut removes more but starts to bite into the
        # star's own signal.
        "bias_nsig": 10.0,
        "bias_elements": 2.0,
        "excess_nsig": None,
        "excess_chi2": 1.25,
        "excess_elements": 2.0,
        "nsig_cut": 3.0,             # NaN beyond this many running robust sigmas
        # And then the column the exposures agree is bad. With nsig_cut set,
        # an observer column where more than column_frac of the exposures is
        # clipped AND whose survivors' reduced chi2 is still above
        # column_chi2 is dropped from every exposure (outliers.py). Pure noise
        # gives 0.973 after a 3 sigma clip and never reaches 10% clipped, so
        # these two defaults are measured thresholds and not guesses; null in
        # either leaves the columns alone.
        "column_frac": 0.10,
        "column_chi2": 1.5,
        # THE SAME TESTS IN THE STAR'S FRAME (outliers.residual_outliers),
        # each exposure carried by its shift: a star column with more than
        # star_column_frac of the exposures clipped and survivors still above
        # star_column_chi2, or whose mean residual over the exposures is
        # beyond star_bias_nsig sigmas (summed over bias_elements), is masked
        # in every exposure where it falls. Off (None) by default: what
        # another star's template, or a narrow BERV, leaves standing still on
        # the star
        "star_column_frac": None,
        "star_column_chi2": None,
        "star_bias_nsig": None,
        "column_min_rows": 10,       # fewer exposures than this: no verdict
        # The metric the amplitudes are measured in when they are refitted at
        # correction time. "flux": every sample as the fit saw it, the older
        # behaviour. "velocity": each sample weighted by (dT/dv)^2, the star's own
        # derivative there, since what a contaminant does to a radial velocity
        # is its overlap with that derivative and a contaminant flat where the
        # star has structure moves no line. It implies a refit, and it changes
        # only how b is MEASURED: what is divided out and which samples are
        # blanked are untouched, and so is the shrinkage below.
        "weight": "velocity",
        "velocity_floor": 0.05,      # what a sample with no velocity info keeps
        # DIAGNOSTICS (pca2d.diagnostics), both inactive at None. q_support:
        # divide the basis out only where the cube's transmission drops below
        # this threshold. permute_b: a seed, each night corrected with the
        # amplitudes of another (a null test).
        "q_support": None,
        "permute_b": None,
        "shrink": True,              # each component only where it is significant
        "shrink_smooth": False,      # its significance averaged over an element
        "smooth_components": [],     # observer components smoothed before division
        # A sample is blanked in every exposure as soon as one of them gave
        # it no weight (or fewer than mask_share of them did), so the whole
        # campaign carries ONE set of lines. 'exposure' blanks each
        # exposure's own, and a line set that moves from epoch to epoch is
        # scatter in the velocities. 'none' leaves the delivered flux there,
        # uncorrected.
        "mask": "common",
        # the fraction of exposures that must have weighted a sample for it
        # to be kept (PCA2D_MASK_SHARE of the instrument's LBL class)
        "mask_share": 1.0,
    },
    "lbl": {
        "prepare": True,             # write LBL's config and its run script
        "run": False,                # and run it. Hours, so it is asked for.
        # LBL's DATA_DIR, its own tree. None puts it under the output ROOT,
        # <output.directory>/lbl, before a run name or a variant adds its own
        # level: the delivered object's LBL products are the same for every
        # run of a target, and hours to make, so the runs of one output root
        # share one tree, and it does not depend on the folder the run was
        # started from.
        "directory": None,
        # The corrected object's name. {tag} becomes the run's <M>-<N>, and
        # it is in the default because without it two runs at different
        # component counts write their corrected spectra into ONE LBL science
        # folder, where LBL's glob takes the mixture and says nothing.
        "suffix": "_PCA2D_{tag}",
        "before": True,              # the delivered spectra as their own object
        "after": True,               # and the corrected ones as another
        # LBL picks the stellar model its mask comes from by this, and stops
        # without it. 'auto' reads it from the spectra, where APERO writes it
        # as OBJTEMP and PP_TEFF; a number here overrides the header.
        "teff": "auto",
        "template": None,            # OBJECT_COMPARISON; None = each its own
        # The corrected object's template is the fit's first star component
        # at its mean amplitude (lbltemplate.py), written by LBL's own writer
        # where LBL looks for it, so LBL's template step finds it and skips.
        # A template LBL made for that object is never replaced.
        # THE PDF REPORT that lbl/report writes after compile, in
        # <DATA_DIR>/reports: the table of the run's statistics, the
        # sequence, the velocity against the BERV, the periodograms, the FIP
        # and the quality of the telluric correction. True by default,
        # because it costs a few seconds at the end of a run that took
        # minutes. Set it to false when running dozens of variants in a row
        # and only the .rdb files matter.
        "report": True,
        "star_template": True,
        # With two or more star components, the ones past the first go to
        # LBL as RESPROJ tables STRPCA2..N, in the place of its DTEMP
        # gradients, and each exposure's projection on them is an rdb column.
        "strpca": True,
        # LBL's temperature-gradient projection, DTEMP<T>, measured on both
        # objects so that the report can set them side by side. 'auto' takes
        # the table of LBL's grid (3000 to 6000 K by 500) nearest the star's
        # Teff, a number the one nearest that, false none. It goes first
        # among the RESPROJ tables: the LBL installed here divides the
        # residual in place for every table, so only the first is right.
        "dtemp": "auto",
        "steps": ["template", "mask", "compute", "compile"],
        # How the spectra get into LBL's science folders: 'symlink' costs no
        # room, 'copy' holds every spectrum a second time. A disk that cannot
        # store a link (exFAT, for example) gets copies whatever this says,
        # and the run says so before it starts (lbl.link_mode).
        "link": "symlink",
        "input_file": None,          # LBL's glob inside a science folder
        # LBL's name for the instrument and its data source, as a wrap's
        # INSTRUMENT and DATA_SOURCE: they select LBL's class, which reads the
        # files and gives every instrument-dependent setting
        # (Instrument.pca2d_settings)
        "instrument": None,
        "data_source": None,
    },
}


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (new or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


# What `domain.smart_dv: true` means: sample the finest pixel on the detector
# at this fraction of its own step. Below 1 the grid is finer than the data,
# which is the only thing resampling has to guarantee.
SMART_DV_FRACTION = 0.7


def measure_pixel_dv(directory: str, pattern: str = "*.fits",
                     config: dict | None = None) -> float:
    """The smallest step between adjacent pixels, in km/s, from the first file.

    Read from the wavelength EXTENSION, order by order, and never from header
    polynomials. The minimum is taken over every order, since it is the finest
    pixel anywhere in the domain that decides how finely the common grid has to
    be sampled for no part of the spectrum to be smoothed by the resampling.

    One file, the first that opens, and not a median over many: the answer has
    to be a pure function of the configuration and the data, because it becomes
    `domain.dv` and `domain.dv` is hashed into the cube cache key.
    """
    import glob

    import numpy as np

    from .io import robust_open
    from .grids import velocity
    from .tfits import extensions_for
    from . import readers as _readers

    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise SystemExit("no file matching %s in %s" % (pattern, directory))
    # An instrument read through LBL's class has no wavelength extension to
    # open: its solution can be a table, a header polynomial or another
    # format of its own, and that is exactly what LBL's reader class knows
    # how to handle. So the grid is asked of it rather than guessing an
    # extension name that does not exist.
    bridged = config is not None and not _readers.reads_natively(config)
    extensions = ((config or {}).get("input") or {}).get("extensions")
    for path in paths[:20]:
        try:
            if bridged:
                wave = np.asarray(_readers.read_any(path, config)["wave"],
                                  dtype=np.float64)
            else:
                with robust_open(path) as hdulist:
                    _, e_wave, _, _ = extensions_for(hdulist, extensions)
                    wave = np.asarray(hdulist[e_wave].data, dtype=np.float64)
        except Exception:                                     # noqa: BLE001
            continue
        wave = np.atleast_2d(wave)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = velocity(np.diff(np.log(wave), axis=-1))
        # a NaN, a repeated wavelength or a decreasing one is not a pixel step
        step = step[np.isfinite(step) & (step > 0)]
        if step.size:
            finest, typical = float(step.min()), float(np.median(step))
            if finest < 0.5 * typical:
                from .logger import log
                log("the finest pixel step in %s is %.4f km/s against a median"
                    " of %.4f: that looks like a glitch in the wavelength"
                    " solution rather than a pixel, and it is about to set the"
                    " grid for the whole run"
                    % (os.path.basename(path), finest, typical), "warn")
            return finest
    raise SystemExit("none of the first files in %s carries a usable wavelength"
                     " grid, so domain.smart_dv has nothing to measure"
                     % directory)


def smart_dv_from_step(step: float, value=True) -> float:
    """The grid step for a measured pixel step: a fraction of it, rounded down.

    Down to the nearest 10 m/s, and down rather than to the nearest so that the
    grid stays at or below the requested fraction of a pixel. Rounding at all
    is what keeps a wavelength solution that moved by a hair between two
    reductions from re-keying the cube and orphaning twenty minutes of work.
    """
    fraction = SMART_DV_FRACTION if value is True else float(value)
    if not 0 < fraction < 1:
        raise ValueError("domain.smart_dv must be true or a fraction in (0, 1),"
                         " got %r" % (value,))
    dv = math.floor(fraction * float(step) * 100.0) / 100.0
    if dv <= 0:
        raise ValueError("smart dv resolves to %g km/s from a pixel step of %g:"
                         " below 10 m/s there is nothing left to round to"
                         % (dv, step))
    return dv


def resolve_smart_dv(cfg: dict) -> dict:
    """`domain.dv` measured from the data instead of chosen.

    A magic grid is uniform in velocity and a spectrograph is not, so one dv
    has to serve the finest pixel in the domain; anything below that is paid
    for on every sample of every exposure and buys nothing. A pixel of about
    2 km/s, sampled by a fixed dv of 0.5, is oversampled fourfold in every
    array the run allocates and every sweep the fit makes over them.

    `domain.dv` in the config is then DEAD: it is overwritten here, not
    combined with anything, and the run says so out loud rather than leaving a
    number in the file to be believed. It may also simply be null.

    Resolved here, at load, and written into `domain.dv` as a plain number, so
    that the cube cache key hashes the step that was actually used and the
    saved config says what it was. `smart_dv` may also be a number, the
    fraction to use in place of the default 0.7.
    """
    from .logger import log

    value = cfg["domain"].get("smart_dv")
    if not value:
        return cfg
    if cfg["domain"].get("grid_source", "magic") == "native":
        log("domain.smart_dv is ignored with grid_source 'native': the grid"
            " comes from the file and brings its own step", "warn")
        return cfg
    step = measure_pixel_dv(spectra_dir(cfg),
                            cfg["input"].get("pattern") or "*.fits", cfg)
    dv = smart_dv_from_step(step, value)
    fraction = SMART_DV_FRACTION if value is True else float(value)
    was = cfg["domain"].get("dv")
    log("smart dv: finest pixel is %.4f km/s, so dv = %.2f km/s, %.0f%% of it"
        % (step, dv, 100 * fraction), "value")
    if isinstance(was, (int, float)):
        log("domain.dv: the %.2f km/s in the config is NOT used, smart_dv"
            " measured the step above instead" % float(was), "warn")
    cfg["domain"]["dv"] = float(dv)
    cfg["domain"]["pixel_dv"] = float(step)

    # A window still counted in SAMPLES widens in velocity on a coarser grid
    # without a line of the config changing. Said out loud rather than
    # rescaled: what it should cover is a modelling decision. The high pass is
    # no longer one of them: it is a width in km/s (resolve_highpass), and a
    # config that still gives it in samples is told so there.
    n = cfg["weights"].get("empirical_noise_box")
    if n:
        log("  weights.empirical_noise_box is %d samples, so %.0f km/s at this"
            " step" % (n, n * dv), "warn")
    return cfg


def highpass_samples(width_kms, dv, polyorder=2):
    """The Savitzky-Golay window, in samples, for a width in km/s.

    Odd, as savgol_filter needs, and never shorter than polyorder + 2. At
    dv = 0.5 km/s, 100 km/s is 201 samples; at a step of 1.37 km/s it is 73,
    the same 100 km/s.
    """
    n = int(round(float(width_kms) / float(dv)))
    n = max(n, int(polyorder) + 2)
    return n if n % 2 else n + 1


def _file_highpass(user, layered, object_name=None, variant=None):
    """The `highpass` keys the configuration FILE sets, its layers merged in
    the order load_config applies them; the package DEFAULTS are not in it.

    A variant, the last layer, that gives the window in samples and not the
    width keeps that window: the width the layers below it set is dropped,
    the way a configuration written with a window in samples reads back.
    """
    if not layered:
        layers = [user]
    else:
        top = {k: v for k, v in user.items()
               if k not in ("general", "instruments", "objects")}
        layers = [user.get("general") or top]
        if object_name:
            layers.append((user.get("objects") or {}).get(object_name) or {})
    out = {}
    for layer in layers:
        out.update((layer or {}).get("highpass") or {})
    last = (variant or {}).get("highpass") or {}
    if last.get("window") and "width_kms" not in last:
        out.pop("width_kms", None)
    out.update(last)
    return out


def resolve_highpass(cfg, file_hp=None):
    """`highpass.window` from `highpass.width_kms` and the grid step.

    The width is a velocity so that the filter treats a line the same way on
    any grid; the window is what savgol_filter takes. It is written back as a
    plain number, so the saved config says what was used, and the cube key
    hashes it, together with the dv it came from.

    A config file that sets `window` and not `width_kms` keeps its window:
    reading a saved run written that way back has to give the cube it was
    built with, not a new one.
    """
    from .logger import log

    file_hp = file_hp or {}
    # a copy: a section the file does not touch can still be DEFAULTS' own dict
    cfg["highpass"] = hp = dict(cfg["highpass"])
    if "width_kms" in file_hp:
        width = file_hp["width_kms"]
    elif file_hp.get("window"):
        width = None
    else:
        width = hp.get("width_kms")
    hp["width_kms"] = float(width) if width else None
    dv = cfg["domain"].get("dv")
    if not dv:
        return cfg                   # smart_dv without an object: no step yet
    if width:
        window = highpass_samples(width, dv, hp.get("polyorder", 2))
        if file_hp.get("window") and int(file_hp["window"]) != window:
            log("highpass.window %d is not used: width_kms %.0f km/s is %d"
                " samples at dv = %.2f km/s"
                % (int(file_hp["window"]), float(width), window, float(dv)),
                "warn")
        hp["window"] = window
    elif cfg["domain"].get("smart_dv"):
        log("highpass.window is %d samples, a config from before width_kms,"
            " so %.0f km/s at this step"
            % (int(hp["window"]), int(hp["window"]) * float(dv)), "warn")
    return cfg


def lbl_directory(config: dict, out_root: str | None = None) -> str:
    """LBL's tree for this configuration, in full.

    `lbl.directory` when it is set, otherwise `lbl` under the output root.
    `out_root` is that root as it was BEFORE a run name or a variant added a
    level to output.directory, which is what a caller passes: the tree is
    shared by every run under one root, and the delivered object in it is
    hours of LBL that no run should have to make twice.

    Absolute, because every stage reads it back from the run's resolved
    configuration and a relative path means another folder from another
    working directory.
    """
    block = config.get("lbl") or {}
    chosen = block.get("directory")
    if not chosen:
        root = out_root or (config.get("output") or {}).get("directory") or "outputs"
        chosen = os.path.join(root, "lbl")
    return os.path.abspath(os.path.expanduser(str(chosen)))


def spectra_dir(config: dict) -> str:
    """The folder holding one object's spectra: input.directory/input.object.

    Resolved on every call rather than stored back into `input.directory`.
    Storing it would give that one key two meanings, the root in a config a
    person writes and the object's folder in a config a run saved, and loading
    a saved config would then append the object a second time.
    """
    inp = config.get("input") or {}
    root = inp.get("directory") or "data"
    obj = inp.get("object")
    if not obj:
        return root
    # a config written before the root/object split already ends in the object
    if os.path.basename(os.path.normpath(root)) == str(obj):
        return root
    return os.path.join(root, str(obj))


def has_spectra_dir(config: dict) -> bool:
    """Whether this object's folder exists, which is whether there are data.

    A configuration is read in two situations that look alike and are not: a
    run, which is about to open spectra, and a reading of the parameters with
    no data anywhere, which is what the tests and a fresh checkout do. The
    second must work: `./check.sh` on a clone with no `data/` is the first
    thing anybody does with this package.
    """
    return os.path.isdir(spectra_dir(config))


def _apply_run_overrides(cfg: dict, object_name, data_dir, out_dir) -> dict:
    """What the command line said, on top of whichever layer merged last."""
    if object_name:
        cfg["input"]["object"] = object_name
    if data_dir:
        cfg["input"]["directory"] = data_dir
    if out_dir:
        cfg["output"]["directory"] = out_dir
    return cfg


def instrument_layer(instrument: str, data_source) -> dict:
    """The instrument's own settings, as a layer of the configuration.

    They come from LBL's class for that instrument and data source
    (Instrument.pca2d_settings: its PCA2D_* parameters, its domain, its
    resolving power, the bands that overlap its domain), so that nothing in
    this package depends on which instrument the spectra come from. The
    instrument is named, as in a wrap (INSTRUMENT, DATA_SOURCE), never
    guessed from the files: the wrong class raises no error, it reads another
    fibre or applies another BERV and returns plausible numbers.
    """
    from . import readers as _readers

    try:
        inst = _readers.profile(str(instrument), str(data_source))
    except Exception as exc:                                   # noqa: BLE001
        raise SystemExit(
            "LBL has no class for INSTRUMENT %r and DATA_SOURCE %r (%s). The"
            " instrument and its data source are named as in a wrap: the"
            " lbl.instrument and lbl.data_source of the configuration, or"
            " --instrument and --data-source."
            % (instrument, data_source, exc))
    return copy.deepcopy(inst.pca2d_settings())


#: the keys of a variant file that describe it rather than configure the run
VARIANT_META = ("reuse_fit", "note")


#: Every configuration key the window can set, in the order the window shows
#: it, and what each one means in one line.
#:
#: It lives here rather than in the window's code because a report has to
#: print it. The report's summary page lists all of them with the value the
#: run actually resolved, so that two runs that differ by any of these (a
#: correction whose amplitudes are refitted in another metric, for example)
#: do not read alike on the page.
WINDOW_SETTINGS = (
    ("twoframe.n_star", "star-frame components with a free amplitude,"
                        " besides the star's spectrum (not adjusted in"
                        " depth): n_star minus 1"),
    ("twoframe.n_earth", "observer components in the basis"),
    ("twoframe.velocity_term", "one velocity per exposure fitted beside them"),
    ("twoframe.iters", "sweeps at most"),
    ("correct.shrink", "divide each observer component out only where"
                       " significant"),
    ("correct.nsig_cut", "residual clip, in running robust sigmas"),
    ("correct.column_frac", "clipped fraction above which a column goes whole"),
    ("correct.column_chi2", "and the survivors' reduced chi2 above which"),
    ("correct.weight", "metric the correction's amplitudes are measured in"),
    ("highpass.width_kms", "the Savitzky-Golay high pass, in km/s"),
    ("domain.dv", "the grid step, in km/s"),
    ("input.nightly_stack", "coadd each night: true, false, or auto"),
    ("lbl.run", "run LBL after the correction"),
    ("lbl.prepare", "write LBL's tree and its config"),
    ("lbl.before", "measure the uncorrected spectra too"),
    ("lbl.after", "measure the corrected spectra"),
    ("lbl.star_template", "give LBL the fit's own star spectrum as template"),
    ("lbl.strpca", "keep the STRPCA columns"),
    ("lbl.suffix", "what the corrected object is called in LBL"),
    ("lbl.teff", "effective temperature handed to LBL"),
    ("lbl.template", "an existing template to reuse instead"),
    ("lbl.steps", "which LBL steps run"),
    ("lbl.directory", "LBL's data tree"),
    ("lbl.link", "symlink or copy the spectra into LBL's tree"),
)


def setting_value(config, path):
    """The value `path` ("correct.weight") has in a resolved config.

    Returns the string "(not set)" for a key the config does not carry, which
    is a fact about the run worth printing rather than a hole to hide.
    """
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return "(not set)"
        node = node[part]
    return node


def instrument_stem(folder, instrument):
    """The star a folder named <STAR>_<INSTRUMENT> is a campaign of, or None.

    The suffix has to be the instrument's LBL name, or its first part when
    that name carries a mode after an underscore (<NAME>_<MODE>): a name that
    merely ends in another word is a name, and is left alone.
    """
    if not folder or not instrument:
        return None
    names = {str(instrument), str(instrument).split("_")[0]}
    pattern = "|".join(re.escape(name) for name in
                       sorted(names, key=len, reverse=True))
    match = re.match(r"^(.+?)_(?:%s)$" % pattern, str(folder), re.I)
    return match.group(1) if match else None


def load_config(path: str | None, object_name: str | None = None,
                data_dir: str | None = None, out_dir: str | None = None,
                instrument: str | None = None, variant: dict | None = None,
                data_source: str | None = None) -> dict:
    """The layers of a configuration, merged, with the instrument resolved.

    `general`, then the instrument's own settings (instrument_layer, from
    LBL's class), then `objects.<object_name>`, then `variant`, the content of
    a variants/<name>.yaml if one is named, each winning over the last, all
    on top of the package DEFAULTS. The instrument and its data source are
    LBL's names, given here or by the configuration's lbl.instrument and
    lbl.data_source (the object's block first); without them the layer is
    left out, which is enough to read the parameters but not to read
    spectra. The t.fits extensions the instrument names are handed to
    `tfits`, so every reader in the package agrees on which extension is
    which.

    A file written in the flat form, with `input:` and `domain:` at the top
    level, also loads: it is treated as one more layer on top of `general`.
    """
    user = {}
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError("config file not found: %s" % path)
        with open(path, "r", encoding="utf-8") as handle:
            user = read_yaml(handle) or {}

    layered = any(k in user for k in ("general", "instruments", "objects"))
    cfg = _deep_update(DEFAULTS, user.get("general", {}) if layered else user)
    if layered and user.get("general") is None:
        cfg = _deep_update(cfg, {k: v for k, v in user.items()
                                 if k not in ("general", "instruments", "objects")})
    if layered and user.get("instruments"):
        from .logger import log
        log("%s has an `instruments:` block, which is not read: an"
            " instrument's settings are the PCA2D_* parameters of its LBL"
            " class (lbl/instruments), and a target's go under `objects:`"
            % path, "warn")

    cfg = _apply_run_overrides(cfg, object_name, data_dir, out_dir)

    obj_block = ((user.get("objects") or {}).get(object_name) or {}
                 if object_name and layered else {})
    # the instrument as LBL names it: from the caller, else from the object's
    # block, else from the configuration's lbl block
    named = _deep_update(cfg.get("lbl") or {}, obj_block.get("lbl") or {})
    instrument = instrument or named.get("instrument")
    data_source = data_source or named.get("data_source")
    if instrument:
        cfg = _deep_update(cfg, instrument_layer(instrument, data_source))
        cfg["input"]["instrument"] = str(cfg["lbl"]["instrument"]).upper()

    if obj_block:
        cfg = _deep_update(cfg, obj_block)
    # A folder named <STAR>_<INSTRUMENT> is that star's campaign on that
    # instrument, and its headers name the star, not the folder: matched on
    # the folder's own name, none of its files would be read. Only when
    # nothing names the header already.
    stem = instrument_stem(object_name, cfg["input"].get("instrument"))
    if stem and not cfg["input"].get("object_header"):
        cfg["input"]["object_header"] = stem
    # a variant, the nominal plus what it changes, on top of everything else
    if variant:
        cfg = _deep_update(cfg, {k: v for k, v in variant.items()
                                 if k not in VARIANT_META})
    # last word to the command line: an object block may move the roots, a flag
    # passed to this run overrules it
    cfg = _apply_run_overrides(cfg, object_name, data_dir, out_dir)
    # the t.fits extensions every reader of this process uses, also when a
    # run's resolved configuration is read back; none for an instrument read
    # through its LBL class
    from . import tfits as _tfits
    _tfits.use_extensions(cfg["input"].get("extensions"))
    _tfits.use_header_keys(cfg["input"].get("header_keys"))

    # dv from the data, once the instrument's domain and the object's folder
    # are both known. Only when an object was named and its folder is there:
    # loading the file to read it, which the tests and the docs do, must not
    # need the data to be on this disk. A run saves the resolved number, so
    # every stage after this one reads a plain dv.
    if object_name and cfg["domain"].get("smart_dv") and has_spectra_dir(cfg):
        cfg = resolve_smart_dv(cfg)

    # A knob that used to exist. Removing it silently would mean a config
    # asking for float64 got float32 and no word about it.
    if cfg["output"].pop("cube_dtype", None) is not None:
        from .logger import log
        log("output.cube_dtype is no longer a setting and was dropped from this"
            " configuration: the cube is float32, in cube.CUBE_DTYPE", "warn")

    # max_sigma is the correct name for the cut; max_mad is what it was called
    # first. Both drive the same key so a config written either way behaves
    # identically, and a config carrying both is a contradiction worth refusing.
    tf = (user or {}).get("twoframe") or {}
    for new, old in (("max_sigma", "max_mad"),
                     ("max_sigma_rounds", "max_mad_rounds")):
        if new in tf:
            if old in tf and tf[old] != tf[new]:
                raise ValueError("config sets both twoframe.%s and twoframe.%s"
                                 " to different values" % (new, old))
            cfg["twoframe"][old] = tf[new]

    # the high pass's window from its width in km/s, now that dv is known
    cfg = resolve_highpass(cfg, _file_highpass(user, layered, object_name,
                                               variant))
    return cfg


def as_yaml_value(value):
    """A scalar as it should read in the file: true/false, a number, a list, or
    a quoted string when YAML would otherwise make something else of it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(as_yaml_value(v) for v in value)
    # written bare only if reading it back gives the same string: `auto` is
    # safe, `*t.fits` is an alias, `1220:4` is a base-60 number and `true` is
    # not a word (yaml-windows-as-text, which this project has been bitten by)
    text = str(value)
    try:
        plain = yaml.safe_load(text) if text.strip() else None
    except yaml.YAMLError:
        plain = None
    return text if isinstance(plain, str) and plain == text else json.dumps(text)


def update_file(path, values):
    """Write `values` into a configuration FILE, in place, keeping the comments.

    `values` maps "section.key" to a value, and only the `general:` block is
    touched, which is where the nominal lives. A key already there keeps its
    line, its indentation and whatever comment follows it, and only its value
    changes; a key that is missing is added at the end of its section; a
    section that is missing is refused, since guessing where it goes would put
    it under the wrong instrument.

    Every comment in config.yaml is an argument for a value, and yaml.safe_dump
    would delete all of them, which is why this edits lines rather than
    re-writing the document. Returns the keys it wrote.
    """
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("general:"))
    except StopIteration:
        raise SystemExit("%s has no `general:` block to write into" % path)
    end = next((i for i in range(start + 1, len(lines))
                if lines[i][:1] not in ("", " ", "#")), len(lines))

    written = []
    for name, value in (values or {}).items():
        section, key = name.split(".")
        head = next((i for i in range(start + 1, end)
                     if lines[i].strip().startswith(section + ":")), None)
        if head is None:
            raise SystemExit("%s has no `%s:` under general:" % (path, section))
        indent = len(lines[head]) - len(lines[head].lstrip()) + 2
        stop = next((i for i in range(head + 1, end)
                     if lines[i].strip() and not lines[i].startswith(" " * indent)),
                    end)
        at = next((i for i in range(head + 1, stop)
                   if lines[i].strip().startswith(key + ":")), None)
        text = "%s%s: %s" % (" " * indent, key, as_yaml_value(value))
        if at is None:
            last = max([i for i in range(head + 1, stop) if lines[i].strip()],
                       default=head)
            lines.insert(last + 1, text)
            end += 1
        else:
            comment = lines[at].split("#", 1)
            if len(comment) > 1:
                pad = max(1, 28 - len(text))
                text = "%s%s# %s" % (text, " " * pad, comment[1].strip())
            lines[at] = text
        written.append(name)
    body = "\n".join(lines)

    # Read back what was written and check that YAML agrees, BEFORE replacing
    # the file. A line can be written and still not be read: a key that appears
    # twice in its section is resolved by the last one, an indentation that does
    # not match its block belongs to another key, and either way the value would
    # be accepted here and ignored by every run afterwards. This is somebody's
    # configuration, so it is verified rather than assumed.
    check = yaml.safe_load(body) or {}
    general = check.get("general") or {}
    for name, value in (values or {}).items():
        section, key = name.split(".")
        got = (general.get(section) or {}).get(key)
        if got != value and not (isinstance(got, float) and got == value):
            raise SystemExit("%s: wrote %s: %r but it reads back as %r, so the"
                             " file was left as it was" % (path, name, value, got))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    return written


def free_star_components(n_star) -> int:
    """
    The star-frame components a fit leaves free, from n_star as a user sets
    it (the PCA2D dict, --n-star): the star's spectra, the template
    included. 1 is the template alone, not adjusted in depth; 2 adds one
    star-frame component with a free amplitude; 0 is not a fit.
    """
    count = int(n_star)
    if count < 1:
        raise ValueError(
            "n_star = %d: it counts the star's spectra, the template"
            " included. 1 is the template alone, not adjusted in depth; 2"
            " adds one star-frame component with a free amplitude" % count)
    return count - 1


def star_spectra(free: int) -> int:
    """n_star as a user sets it, from the free components of a fit"""
    return int(free) + 1


def describe_star(free: int) -> str:
    """The star side of a fit, spelled: 'n_star 1: the star's spectrum, not
    adjusted in depth' or with its free components"""
    free = int(free)
    text = ("n_star %d: the star's spectrum, not adjusted in depth"
            % star_spectra(free))
    if free:
        text += (", and %d star-frame component%s with a free amplitude"
                 " (never divided out)" % (free, "s" if free > 1 else ""))
    return text


def cache_key(config: dict) -> str:
    """Hash of the config entries that affect the registered data cube.

    Anything downstream of the cube (PCA settings, plots) is deliberately left
    out so that re-running with a different n_components reuses the cache.
    """
    relevant = {
        # input.workers is how fast, not what: out of the key
        "input": {k: v for k, v in config["input"].items() if k != "workers"},
        "quality": config["quality"],
        # domain.bands only chooses which columns the *fit* sees, so it must not
        # invalidate a cube that took twenty minutes to build
        "domain": {k: v for k, v in config["domain"].items() if k != "bands"},
        # orbits omitted when there are none, so that the key's appearance does
        # not orphan the cubes built before it (the same as against_template)
        "registration": {k: v for k, v in config["registration"].items()
                         if not (k in ("orbits", "velocities") and not v)},
        # the window the width resolved to, not the width: dv is hashed with
        # the domain, and a config in samples from before width_kms has to
        # keep the key its cube was built under
        # against_template is omitted when it is false, for the same reason as
        # the other settings added after the fact: its appearance in the
        # dictionary must not orphan the cubes already built.
        "highpass": {k: v for k, v in config["highpass"].items()
                     if k != "width_kms"
                     and not (k == "against_template" and not v)},
        # weights that are baked into the cube rather than applied later
        "weights": {
            k: config["weights"][k]
            for k in ("mode", "snr_pixel_scale", "sigma_floor_frac",
                      "telluric_mode", "empirical_noise_box")
        },
    }
    # Optional entries left unset are stripped before hashing: a knob that is
    # not set describes no behaviour, so it must not describe a different
    # cube either, and adding one must not change the key of every cube
    # already built (twenty minutes each). The same holds for a name matched
    # in the headers (hashed when set, since it decides which files are read)
    # and for the bundles, which only cut stacked nights: with stacking
    # refused they describe nothing.
    unstacked = str(relevant["input"].get("nightly_stack")).lower() == "false"
    relevant["input"] = {k: v for k, v in relevant["input"].items()
                         if not (k == "object_header" and v is None)
                         and not (k in ("bundle_minutes", "bundle_berv",
                                        "bundle_max")
                                  and (v is None or unstacked))}
    relevant["quality"] = {k: v for k, v in relevant["quality"].items()
                           if not (k in ("min_rjd", "max_rjd", "max_sky_ratio",
                                         "isolated_window", "nights",
                                         "drs_modes")
                                   and v is None)}
    relevant["weights"] = {k: v for k, v in relevant["weights"].items()
                           if not (k == "empirical_noise_box" and v is None)}
    # The Doppler shift that registers each spectrum is relativistic
    # (grids.doppler); the first-order 1 + v/c is 0.003 pixel (1.5 m/s) off
    # at a 30 km/s BERV. A cube registered with the first-order shift must not
    # be reused, so the formula is in the key. An observer-frame cube applies
    # no shift at all, and keeps its key and its telluric map.
    if relevant["registration"].get("frame") != "observer":
        relevant["doppler"] = "relativistic"
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]

