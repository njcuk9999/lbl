"""YAML configuration handling with defaults and a reproducibility hash."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re

import yaml


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
        "format": "tfits",           # 'tfits' = APERO order-by-order spectra
                                     #           (the nominal input)
                                     # 's1d'   = the resampled s1d_v products
        # The input ROOT, and not the folder of spectra: one run reads
        # <directory>/<object>/, so the same config serves every target on the
        # disk and a run needs nothing but a name. See spectra_dir().
        "directory": "data",
        "pattern": "*t.fits",
        "max_files": None,           # None = all; small int for quick tests
        "object": None,              # keep only this OBJECT (None = no filter).
                                     # It is also the FOLDER: one run reads
                                     # <directory>/<object>/.
        # What to match in the headers, when the folder is not named as the
        # observer typed the target. Barnard's star is `Gl699` in both
        # pipelines' headers, and its two campaigns live here in GL699_SPIROU
        # and GL699_NIRPS, since a run is one instrument and the two must not
        # share a folder or an LBL object. None means the folder's own name.
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
                                     # nothing is smeared; NIRPS takes 3-4
                                     # exposures a visit, so this makes the cube
                                     # (and the decomposition) 3-4 times smaller
        # --- 'tfits' only -------------------------------------------------
        # with stacking, each night cut into bundles of at most this BERV
        # range (km/s), this many minutes or this many exposures: a transit
        # sequence is not one sky, and in the observer's frame a coadd smears
        # the star by its BERV range (nights.split_nights). 0.1 km/s since
        # 2026-10-21: an ordinary NIRPS visit stays one row (median ranges 4
        # to 35 m/s over twenty targets), a five-hour sequence (600 m/s) is
        # cut. None: whole nights
        "bundle_minutes": None,
        "bundle_berv": 0.1,
        "bundle_max": None,
        "orders": None,              # None = every order; or an explicit list
        "min_order_finite_fraction": 0.05,   # skip an order with fewer usable
                                     # pixels than this. NIRPS loses orders
                                     # 44-45 and 71-74 entirely to water; that
                                     # is expected, not an error.
        "blaze_min_frac": 0.05,      # reject pixels where the blaze has fallen
                                     # below this fraction of its order peak
    },
    "domain": {
        "wave_min": 965.0,           # nm
        "wave_max": 1950.0,          # nm
        "grid_source": "magic",      # 'magic'  = build the log-uniform grid from
                                     #            wave0 + dv below (the only
                                     #            option that works for t.fits,
                                     #            which carry no common grid)
                                     # 'native' = recycle the grid stored in the
                                     #            first s1d_v file
        "wave0": 965.0,              # nm, anchor of the magic grid
        # Where dv comes from. False: the number below. True: the finest pixel
        # step in the first spectrum, times SMART_DV_FRACTION, and THE NUMBER
        # BELOW IS NOT READ AT ALL. A float here is that fraction. The measured
        # step is kept as domain.pixel_dv for the record.
        "smart_dv": False,
        # km/s per pixel (500 m/s). Read only when smart_dv is false, and may
        # be null when it is true, since nothing would look at it.
        "dv": 0.5,
        # Nominal photometric bands (MKO half-power points, nm). The PCA is
        # fitted on these and only these; see pca.fit_bands.
        "bands": {
            "Y": [970.0, 1070.0],
            "J": [1170.0, 1330.0],
            "H": [1490.0, 1780.0],
            "K": [2030.0, 2370.0],
        },
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
    },
    "highpass": {
        "method": "savgol",          # 'savgol' or 'none'
        # The Savitzky-Golay width IN KM/S, so the filter does the same thing
        # to a line whatever the grid step; `window`, in samples, is derived
        # from it once dv is known (resolve_highpass). A config that sets
        # `window` and not this keeps its window, which is how a run saved
        # before 2026-09-11 reads back, cube key included.
        "width_kms": 100.0,
        "window": 201,               # samples (odd): what 100 km/s is at dv = 0.5
        "polyorder": 2,
        "mode": "divide",            # 'divide'  -> y = ln(f / lowpass(f))
                                     # 'log_sub' -> y = ln(f) - lowpass(ln f)
        "frame": "target",           # s1d only: apply before ('observer') or
                                     # after ('target') the BERV registration.
                                     # 'target' depuis le 2026-10-16: le
                                     # passe-haut contre un template demande
                                     # la grille commune, et c'est devenu le
                                     # defaut.
                                     # t.fits always filters on the destination
                                     # grid, per order -- the native sampling is
                                     # not uniform in velocity.
        # LE PASSE-HAUT SUR LE RESIDU AU TEMPLATE plutot que sur le spectre.
        # Un savgol passe sur un spectre entier est tire vers le bas par
        # chaque raie profonde, et quand une tellurique balaie le coeur d'une
        # raie avec le BERV ce biais change D'UNE POSE A L'AUTRE: il reste
        # alors, dans le repere de l'etoile, un residu qui suit le BERV sans
        # etre stellaire. Mesure le 2026-10-16 sur HD 69830 a 688.85 nm: la
        # tellurique parcourt 0.117 nm sur les 51 km/s de BERV, et sa
        # profondeur au coeur passe de 3% a 89%, par intermittence.
        #
        # Avec cette option le passe-haut se fait en DEUX PASSES: template
        # median de toutes les poses registrees, savgol sur le RESIDU au
        # template, puis la structure stellaire rendue depuis le template
        # seul. Aucune raie n'est masquee, rien n'entre ni ne sort selon le
        # BERV. Demande frame: target, le passe-haut devant se faire dans le
        # repere de l'etoile.
        "against_template": True,
    },
    "quality": {
        "min_snr": 1.0,              # reject if the band SNR is below this
        "require_positive_flux": True,   # reject frames with median flux <= 0
        "max_nan_fraction": 0.5,     # reject if more of the band than this is NaN
        # Epoch window, as rjd = jd - 2400000. Both default to null, i.e. keep
        # everything: excluding part of a time series is a claim about the data,
        # not a hygiene default, and it must be made in the config where it is
        # visible. For NIRPS the first season sits before rjd 60200
        # (2023-09-12); on Proxima it carries most of the excess velocity
        # scatter. `quality` is hashed into the cube cache key, so setting
        # either of these rebuilds the cube rather than silently reusing one
        # built over a different window.
        # Discard a sample where the sky emission the pipeline subtracted was
        # more than this many times the stellar flux. null keeps everything.
        # Hashed into the cube cache key only when set.
        "max_sky_ratio": None,
        # Drop a sample that has a masked neighbour within this many samples on
        # BOTH sides: it survived the cuts but its neighbourhood did not, so its
        # high pass and any shift of it were built from samples that are not
        # there. 3 is the width that was asked for. null disables the check.
        # Hashed into the cube cache key only when set.
        "isolated_window": None,
        "min_rjd": None,
        "max_rjd": None,
        "nights": None,
        # LE MODE DU SPECTROGRAPHE: la liste des modes gardes, lue dans
        # HIERARCH ESO INS MODE. null garde tout, ce qui est le defaut parce
        # qu'un seul mode est le cas ordinaire. Sur les 100 poses HARPS de
        # HD 10700, 8 sont en EGGS (la fibre de 1.4") et veulent le modele
        # tellurique decale de +16 km/s la ou les 92 autres le veulent a
        # zero: autre fibre, autre solution en longueur d'onde, donc un
        # autre instrument du point de vue d'une base commune. Hashe dans la
        # cle du cache de cube quand il est mis.
        "drs_modes": None,
    },
    "weights": {
        "mode": "photon",            # 'photon' or 'uniform'
        "snr_pixel_scale": 1.0,      # EXTSN is per e2ds pixel; rescale if wanted
        # LE SEUIL PAR PIXEL EST UN SEUIL DE SNR, pas de profondeur. Un fond de
        # raie profonde porte moins de photons, donc un RMS fractionnaire plus
        # grand, mais c'est l'endroit ou le contenu en vitesse est le plus
        # dense: ce qu'il faut ecarter est le pixel dont le bruit depasse ce
        # que les photons expliquent, pas celui qui est sombre.
        #
        # sigma etant le sigma photonique du log-flux, 1/sigma EST le SNR
        # local. 5 est bas exprès: ecarter le pixel mort, pas trier par
        # brillance. Mis a null, c'est `ln_clip_low` qui reprend la main.
        "min_pixel_snr": 5.0,
        "ln_clip_low": None,         # ANCIEN: y below this -> zero weight.
                                     # Remplace le 2026-10-16 par min_pixel_snr:
                                     # a -0.5 il tuait le fond de toute raie
                                     # plus profonde que 39% du continu, 13.7%
                                     # des pixels sur HD 10700, et LBL y perdait
                                     # le tiers de ses raies
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
    },
    # ---------------------------------------------------------------- fit ---
    # The two-frame fit. Two component counts, never one: M in the stellar rest
    # frame and N in the observer frame, and they need not be equal.
    "twoframe": {
        # 2 and 7 since 2026-09-09, measured on TOI-2120 with the static sky
        # left in the data for the fit to describe. Against 5+5: the observer
        # block explains 25.8% of the weighted variance instead of 16.5%, its
        # first component 17.4% instead of 9.7%, and the leakage from it into
        # the star block falls from 0.086 to 0.0116. Five star components were
        # enough rope for the star block to help describe an observer-frame
        # line, through the (feature, its derivative x shift) pair; two are not.
        # ZERO: the star is the per-parity spectrum `mean: star` takes out
        # with a coefficient of exactly 1, and a component in front of it would
        # be an exponent on the flux. Measured better on every target it was
        # measured on (config.yaml has the numbers). The paragraph above is why
        # 2 was chosen over 5 back when there were star components at all, and
        # is kept because raising this to look at variability still has to
        # choose a number.
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
        # The static part of the model (twoframe --mean). "offset": the
        # one-shot per-parity observer-frame offset of the fit behind the
        # reports, which the correction divides out with the observer block.
        # "iterate", one mean per order parity in EACH frame re-estimated at
        # every sweep with the components centred, separates the frames on a
        # synthetic cube but does not converge yet on TOI2120 (2026-09-10: its
        # star-frame mean moved more at every sweep and chi2 turned over after
        # one), so it is not the default.
        # SETTLED 2026-09-14, and the window no longer offers a choice: one
        # star spectrum per order parity, in the STAR's frame, taken out once
        # before any component with a coefficient of exactly 1. The observer-
        # frame means, "offset" and "full", go back into the correction, which
        # cost Proxima 46 m/s; "iterate" does not converge on a whole campaign.
        # Both are still read from a config or a variant, for redoing the runs
        # that were measured on them.
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
        # of the flux writes that velocity into the corrected spectrum. Measured
        # on TOI2120 it accounts for 41% of the variance of what the correction
        # did to LBL's velocities. It is fitted and never divided out, since
        # taking it out of the data would remove the signal being looked for.
        # OFF by default since 2026-09-11, as the sanity check on what it does;
        # `--velocity-term` or `velocity_term: true` turns it on.
        "velocity_term": False,
        # The shift is measured only on columns inside a photometric band whose
        # telluric transmission stays above this in 90% of the exposures. Zero
        # turns the transmission cut off and keeps the band cut. Fitting the
        # star's velocity where the flux is mostly atmosphere is fitting it to
        # the wrong thing: unmasked, the term came out 1.8 times larger than
        # the velocity LBL measures on the same spectra.
        "velocity_min_transmission": 0.95,
        # The instrument's resolving power, lambda/dlambda, per instrument in
        # config.yaml as LBL's APPROX_RESOLUTION: the unit every Savitzky-Golay
        # here is measured in. It turns no smoothing on by itself.
        "resolution": None,
        # Smooth the star's spectra per parity (--mean star) and the star
        # components P to this FWHM, in resolution elements, by LBL's own
        # filter (pca2d.resolution). Off by default: at one element it cost
        # TOI-2120 4 m/s, at half an element 2 (2026-09-11). The observer
        # components are never smoothed here.
        "star_smooth": None,
        # VALIDATION CROISEE (pca2d.diagnostics.holdout_rows): "even" ou "odd",
        # les nuits tenues hors du fit. None, le nominal: toutes les nuits.
        "holdout": None,
        # A SAVED OBSERVER BASIS (pca2d.basis, the PCA2D dict's basis=):
        # its components are taken frozen and never updated, and only the
        # star's block and the amplitudes are fitted, for a star whose BERV
        # barely moves and cannot tell its frame from the observer's. None,
        # the nominal: the observer components are fitted. Here and not in
        # nominal.yaml, whose hash stamps every correction on disk.
        "earth_basis": None,
        # The older spelling of the two above: smooth to one resolution element
        # of this R. Still read, so the runs made with it reproduce.
        "star_resolution": None,
        # How the star-side vectors are carried and updated: 'spline', one
        # cubic B-spline evaluated at every exposure's shifted positions and
        # updated exactly (pca2d.splinestar), nominal since 2026-09-11: chi2
        # falls at every sweep, the fit is faster, and on TOI-2120 the
        # velocities tie with 'grid' (20.99 against 20.36 m/s), the samples
        # moved by the Lanczos kernel and updated from the normal diagonal.
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
        # own; 7.5 min for 420 SPIRou files one at a time. Here and not in
        # nominal.yaml, whose hash is in every correction's stamp
        "workers": 4,
        "n_star": 0,
        "n_earth": None,             # None = every observer component the fit has
        # THE UNIT IS AN EXCURSION, not a sample: one sample beyond 3 sigma is
        # noise (0.27% of it is, and 1% of these residuals were), while a run
        # of samples leaning the same way over a line's width is not. Every
        # window from one sample to excursion_elements resolution elements is
        # judged by the aggregate significance of its sum, and flagged, whole,
        # beyond excursion_nsig. The variance of that sum is measured, never
        # assumed: at 0.5 km/s the grid oversamples a 2.3 km/s SPIRou pixel and
        # the residual's rho_1 is 0.935, so 17 samples hold 3.5 independent
        # measurements and a flat 3.2 sigma bump two elements wide is the 6
        # sigma event (outliers.window_variance, noise_correlation).
        # null until it is measured on more than one target: 6.0 and 2.0 are
        # the values it was built for, and variants/0-7exc runs them.
        "excursion_nsig": None,
        "excursion_elements": 2.0,
        # WHAT STANDS STILL IN THE OBSERVER'S FRAME. A leftover anchored there
        # (a telluric line the correction does not reach, airglow, a detector
        # feature) need not have one sign, but it always has scatter the noise
        # does not account for: the chi2 of a column over the exposures, above
        # 1. Averaged over excess_elements resolution elements, that average is
        # certain to sqrt(2 / (N * independent columns)), so its excess becomes
        # a significance: excess_nsig sigmas of it, and a windowed chi2 above
        # excess_chi2, drop the region from every exposure. On TOI-2120 the
        # 1.27 um O2 band reaches a column chi2 of 222 and 1919 nm of 63.
        # LE POIDS DE LA BOUCLE EST LE MASQUE. La boucle deprecie deja chaque
        # echantillon par (clip/|z|)^2 au-dela de la coupure
        # (twoframe.clip_weights); sous cette fraction de son poids nominal,
        # l'echantillon n'a pas ete decrit par le modele et devient NaN dans le
        # fichier corrige. 0.5 est |z| > 4.24 sigma avec le clip a 3, et coute
        # 0.388% des echantillons sur TOI-2120, par pose et non en commun.
        "weight_floor": None,
        # LE BIAIS COHERENT SUR LES POSES, le troisieme axe et le seul que ni
        # la coupure ni l'exces de dispersion ne peuvent voir: clip_weights
        # compare chaque pose aux AUTRES POSES, donc un residu present dans
        # toutes n'est aberrant pour aucune. La moyenne sur les poses, elle,
        # gagne sqrt(N) = 17.8 avec 316 poses. Sur TOI-2120: 21.9% du domaine
        # au-dela de 3 sigma, 2.8% au-dela de 10, et les pires regions sont
        # 1761, 1803, 1805 et 1764-1769 nm (methane et eau), larges de 4 a 6
        # elements de resolution: des raies entieres mal corrigees.
        # NOMINAL depuis le 2026-09-26, avec nsig_cut 3.0: mesure sur
        # TOI-2120 a 0-7, rms 17.01 -> 14.63, rms par nuit 14.57 -> 11.52, et
        # sigma_K 1.23 -> 1.05 avec K a 5.54 contre 5.42, donc sans deplacer
        # l'amplitude de la planete (la coupure seule a 2.5 sigma donnait
        # 14.60 mais faisait tomber K a 3.67). Les deux axes s'additionnent
        # parce qu'ils voient des choses differentes: la coupure les accidents
        # d'une pose, le biais ce qui est ancre dans le referentiel de
        # l'observateur et qu'aucun critere par pose ne peut voir.
        "bias_nsig": 10.0,
        "bias_elements": 2.0,
        "excess_nsig": None,
        "excess_chi2": 1.25,
        "excess_elements": 2.0,
        # 3.0 est nominal depuis le 2026-09-26, avec bias_nsig 10.0. Seul, il
        # gagne 1.37 m/s; 2.5 gagnait 2.41 mais deplacait K de 1.6 sigma.
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
        "column_min_rows": 10,       # fewer exposures than this: no verdict
        # The metric the amplitudes are measured in when they are refitted at
        # correction time. "flux": every sample as the fit saw it, the older
        # behaviour. "velocity": each sample weighted by (dT/dv)^2, the star's own
        # derivative there, since what a contaminant does to a radial velocity
        # is its overlap with that derivative and a contaminant flat where the
        # star has structure moves no line. It implies a refit, and it changes
        # only how b is MEASURED: what is divided out and which samples are
        # blanked are untouched, and so is the shrinkage below. Nominal since
        # 2026-09-15 (config.yaml carries the measurements).
        "weight": "velocity",
        "velocity_floor": 0.05,
        # LES DIAGNOSTICS DE LA NUIT DU 2026-10-17 (pca2d.diagnostics), tous
        # deux inactifs a None. q_support: ne diviser la base que la ou la
        # transmission du cube descend sous ce seuil. permute_b: une graine,
        # chaque nuit corrigee avec les amplitudes d'une autre (test nul).
        "q_support": None,
        "permute_b": None,      # what a sample with no velocity info keeps
        "shrink": True,              # each component only where it is significant
        "shrink_smooth": False,      # its significance averaged over an element
        "smooth_components": [],     # observer components smoothed before division
        # SETTLED 2026-09-14, and the window no longer offers a choice: a
        # sample is blanked in every exposure as soon as one of them gave it no
        # weight, so the whole campaign carries ONE set of lines. 'exposure'
        # blanks each exposure's own, and a line set that moves from epoch to
        # epoch is scatter: 1.8 m/s on TOI-4552 with nothing divided out at
        # all. 'none' leaves the delivered flux there, uncorrected. Both are
        # still read from a config or a variant, for redoing those runs.
        "mask": "common",
    },
    "lbl": {
        "prepare": True,             # write LBL's config and its run script
        "run": False,                # and run it. Hours, so it is asked for.
        # Which LBL runs it: the conda environment its script is run in.
        # lbl-rapide is LBL's speed branch (lbl.FAST_RECIPE makes it), the
        # default since 2026-09-18; 'current' is the LBL environment.yml
        # installs beside this package, which ran everything before; a path
        # is a python.
        "environment": "lbl-rapide",
        # LBL's DATA_DIR, its own tree. None puts it under the output ROOT,
        # <output.directory>/lbl, before a run name or a variant adds its own
        # level: the delivered object's LBL products are the same for every
        # run of a target, and hours to make, so the runs of one output root
        # share one tree. It used to be `lbl` beside wherever the run was
        # started, which put SMETHELLS_20's velocities in another clone's
        # folder while its report was on the data disk (2026-09-15).
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
        # LE RAPPORT PDF que lbl/report ecrit apres compile, dans
        # <DATA_DIR>/reports: la table des statistiques du run, la sequence,
        # la vitesse contre le BERV, les periodogrammes, le FIP et la qualite
        # de la correction tellurique. Vrai par defaut, parce qu'il coute
        # quelques secondes a la fin d'un run qui en a pris des minutes. A
        # mettre a false quand on enchaine des dizaines de variantes et que
        # seuls les .rdb comptent.
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
        # store a link (exFAT, as the data disks here are) gets copies
        # whatever this says, and the run says so before it starts
        # (lbl.link_mode).
        "link": "symlink",
        "input_file": None,          # LBL's glob inside a science folder
        "instrument": None,          # LBL's name for this spectrograph, which
        "data_source": None,         # with the source selects its reader
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


def measure_pixel_dv(directory: str, pattern: str = "*t.fits",
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
    # Un instrument lu par le pont de LBL n'a pas d'extension WaveAB a ouvrir:
    # sa solution peut etre une table, un polynome d'en-tete ou un autre
    # format par instrument, et c'est justement ce que la classe de lecture de
    # LBL sait faire. On lui demande donc la grille plutot que de deviner un
    # nom d'extension qui n'existe pas (2026-10-14).
    bridged = config is not None and not _readers.reads_natively(config)
    for path in paths[:20]:
        try:
            if bridged:
                wave = np.asarray(_readers.read_any(path, config)["wave"],
                                  dtype=np.float64)
            else:
                with robust_open(path) as hdulist:
                    _, e_wave, _, _ = extensions_for(hdulist)
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
    for on every sample of every exposure and buys nothing. On SPIRou the step
    is around 2.27 km/s, which a fixed dv of 0.5 oversamples more than fourfold
    in every array the run allocates and every sweep the fit makes over them.

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
                            cfg["input"].get("pattern", "*t.fits"), cfg)
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
    dv = 0.5 km/s, 100 km/s is 201 samples; at SPIRou's smart step of
    1.37 km/s it is 73, the same 100 km/s.
    """
    n = int(round(float(width_kms) / float(dv)))
    n = max(n, int(polyorder) + 2)
    return n if n % 2 else n + 1


def _file_highpass(user, layered, instrument=None, object_name=None, variant=None):
    """The `highpass` keys the configuration FILE sets, its layers merged in
    the order load_config applies them; the package DEFAULTS are not in it.

    A variant, the last layer, that gives the window in samples and not the
    width keeps that window: the width the layers below it set is dropped,
    the way a run saved before width_kms reads back.
    """
    if not layered:
        layers = [user]
    else:
        top = {k: v for k, v in user.items()
               if k not in ("general", "instruments", "objects")}
        layers = [user.get("general") or top]
        if instrument:
            layers.append((user.get("instruments") or {}).get(instrument.upper()) or {})
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
    every run saved before 2026-09-11 is written that way, and reading one
    back has to give the cube it was built with, not a new one.
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


def detect_instrument(directory: str, pattern: str = "*t.fits") -> str:
    """Read INSTRUME from the first file that opens, and refuse to guess.

    The instrument decides which FITS extensions hold the flux, the wavelength
    solution and the blaze, and getting that wrong fails silently: reading
    FluxA from a SPIRou file raises nothing, it returns half the light with a
    different blaze and SNR. So it is read from the data and never offered as
    a command-line flag.
    """
    import glob

    from .io import robust_open

    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        # L'OEUF ET LA POULE. Le motif des fichiers est une propriete de
        # l'instrument (*t.fits sur APERO, *_S2D_A.fits sur HARPS, *e2ds_A.fits
        # sur SOPHIE), mais l'instrument se lit DANS un fichier, qu'il faut
        # donc trouver avant de le connaitre. A ce stade le motif est encore
        # celui de `general`, donc il ne vaut rien pour un instrument qui n'est
        # pas celui par defaut. INSTRUME, lui, est dans tout fichier FITS: on
        # elargit a eux tous plutot que de refuser (2026-10-14).
        paths = sorted(glob.glob(os.path.join(directory, "*.fits")))
    if not paths:
        raise SystemExit("no file matching %s in %s" % (pattern, directory))
    for path in paths[:20]:
        try:
            with robust_open(path) as hdulist:
                name = str(hdulist[0].header.get("INSTRUME", "")).strip().upper()
        except Exception:                                     # noqa: BLE001
            continue
        if name:
            return name
    raise SystemExit("none of the first files in %s declares INSTRUME"
                     % directory)


#: the keys of a variant file that describe it rather than configure the run
VARIANT_META = ("reuse_fit", "note")


#: Every configuration key the window can set, in the order the window shows
#: it, and what each one means in one line.
#:
#: It lives here rather than in gui.py because a report has to print it and a
#: report must not import tkinter. The report's summary page lists all of them
#: with the value the run actually resolved: until 2026-09-15 that page named
#: the components and the high pass and nothing else, so a run made with
#: correct.weight: velocity, which refits every amplitude in the correction,
#: read on the page exactly like a run made without it. tests/test_gui.py ties
#: this tuple to gui.ALL_OPTIONS in both directions, so a setting cannot be
#: added to the window without appearing on the page.
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
    ("lbl.environment", "which LBL: the environment its script runs in"),
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

    The suffix has to be the instrument the files themselves declare (NIRPS
    may carry its mode, _NIRPS_HE or _NIRPS_HA): a name that merely ends in
    another word is a name, and is left alone.
    """
    if not folder or not instrument:
        return None
    match = re.match(r"^(.+?)_%s(?:_H[AE])?$" % re.escape(str(instrument)),
                     str(folder), re.I)
    return match.group(1) if match else None


def load_config(path: str | None, object_name: str | None = None,
                data_dir: str | None = None, out_dir: str | None = None,
                instrument: str | None = None, variant: dict | None = None) -> dict:
    """The three layers of config.yaml, merged, with the instrument resolved.

    `general`, then `instruments.<INSTRUME>`, then `objects.<object_name>`,
    then `variant`, the content of a variants/<name>.yaml if one is named,
    each winning over the last, all on top of the package DEFAULTS. The
    instrument is read from the files themselves unless one is named, and the
    extension table it carries is registered with `tfits` so every reader in
    the package agrees on which extension is which.

    A file written in the old flat form, with `input:` and `domain:` at the top
    level, still loads: it is treated as one more layer on top of `general`.
    """
    user = {}
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError("config file not found: %s" % path)
        with open(path, "r") as handle:
            user = read_yaml(handle) or {}

    layered = any(k in user for k in ("general", "instruments", "objects"))
    cfg = _deep_update(DEFAULTS, user.get("general", {}) if layered else user)
    if layered and user.get("general") is None:
        cfg = _deep_update(cfg, {k: v for k, v in user.items()
                                 if k not in ("general", "instruments", "objects")})

    cfg = _apply_run_overrides(cfg, object_name, data_dir, out_dir)

    table = (user.get("instruments") or {}) if layered else {}
    if table:
        from . import tfits as _tfits
        _tfits.register_instruments(
            {name: dict(block.get("extensions") or {})
             for name, block in table.items() if block.get("extensions")})
    # Only look at the data when an object was named AND its folder is there:
    # loading the file to read it, which the tests and the docs do, must not
    # need a telescope. A checkout with no data under the input root resolves
    # every layer it can and leaves the two steps that read spectra undone,
    # rather than refusing to load at all; the callers that are about to run a
    # stage, cli.resolve and cli.joint_members, say plainly what a missing
    # folder means before anything else happens.
    if instrument is None and table and object_name and has_spectra_dir(cfg):
        instrument = detect_instrument(spectra_dir(cfg),
                                       cfg["input"].get("pattern", "*t.fits"))
    if instrument:
        block = dict(table.get(instrument.upper()) or {})
        block.pop("extensions", None)
        if not block and table:
            raise SystemExit(
                "config.yaml has no `instruments: %s:` block. The instrument is"
                " read from INSTRUME in the data, so add one rather than"
                " renaming anything." % instrument)
        cfg = _deep_update(cfg, block)
        cfg["input"]["instrument"] = instrument.upper()

    if object_name and layered:
        cfg = _deep_update(cfg, (user.get("objects") or {}).get(object_name, {}))
    # A folder named <STAR>_<INSTRUMENT> is that star's campaign on that
    # spectrograph, and its headers name the star: tfiles_repo holds every
    # campaign that way (TOI1078_NIRPS, OBJECT = TOI-1078, DRSOBJN = TOI1078),
    # and matched on the folder's own name every file of the first joint run
    # made from it was skipped (2026-09-18). Only when nothing names the
    # header already: GL699_NIRPS's Gl699 stays what config.yaml says.
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

    # dv from the data, once the instrument's domain and the object's folder
    # are both known. Only when an object was named, for the same reason the
    # instrument is only detected then: loading the file to read it must not
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
    cfg = resolve_highpass(cfg, _file_highpass(user, layered, instrument,
                                               object_name, variant))
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
    with open(path) as handle:
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
    with open(path, "w") as handle:
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
                         if not (k == "orbits" and not v)},
        # the window the width resolved to, not the width: dv is hashed with
        # the domain, and a config in samples from before width_kms has to
        # keep the key its cube was built under
        # against_template est omis quand il est faux, pour la meme raison que
        # les autres reglages ajoutes apres coup: son apparition dans le
        # dictionnaire ne doit pas orpheliner les cubes deja construits.
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
    # Entries left unset are stripped before hashing. Without this, adding a new
    # optional knob to `quality` changes the key of every existing config and
    # orphans cubes that took twenty minutes to build, purely because a default
    # of None appeared in the dictionary. A knob that is not set describes no
    # behaviour, so it must not describe a different cube either.
    # The epoch window is dropped from the hash when unset. Without this, adding
    # these two knobs would change the key of every existing config and orphan
    # cubes that took twenty minutes to build, purely because a default of None
    # appeared in the dictionary. Only these two are treated this way: the older
    # entries are hashed exactly as before, including their Nones, so keys
    # computed by earlier versions still match.
    # a name matched in the headers describes which files are read, so it is
    # hashed when it is set; unset it describes nothing and must not re-key
    # every cube already built
    # the bundles only cut stacked nights: with stacking refused they describe
    # nothing, and must not re-key a cube of single exposures either
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
    # The Doppler shift that registers each spectrum became relativistic
    # (grids.doppler) on 2026-09-10; it was the first-order 1 + v/c, 0.003
    # pixel (1.5 m/s) off at a 30 km/s BERV. A cube registered the old way must
    # not be reused, so the formula is in the key. An observer-frame cube
    # applies no shift at all, and keeps its key and its telluric map.
    if relevant["registration"].get("frame") != "observer":
        relevant["doppler"] = "relativistic"
    blob = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]

