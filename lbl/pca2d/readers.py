"""Read a spectrum from any instrument, through LBL.

Anything that goes through LBL can be sent to PCA2D. LBL already reads every
instrument and data source it supports (lbl.instruments.select.InstDict),
plus a generic class. Reimplementing that reading here would mean
maintaining it twice and diverging, so PCA2D asks LBL's class.

    read(path, instrument, data_source) -> the same dict as tfits.read_tfits

What LBL gives natively and what has to be drawn from it:

    flux    inst.load_science_file(path), always 2D (orders, pixels)
    wave    inst.get_wave_solution(path, data, header), in nm
    blaze   inst.load_blaze_from_science(...), normalised per order
    meta    the keywords by their LBL name (KW_BERV, KW_MJDATE...), which
            each class translates for its instrument

What LBL does not give, and what is done about it:

    recon   the telluric transmission. APERO writes it in the t.fits
            (Recon); other pipelines do not, and LBL computes it itself
            from its TAPAS model (lbl/science/tellu_clean.py, the
            file tapas_lbl.fits that it downloads). `tapas_transmission`
            below evaluates it on the spectrum's grid, which gives pca2d the
            same object as APERO's Recon.
    sky     the sky emission. Most pipelines do not deliver it per
            exposure; quality.max_sky_ratio then does not apply and the sky
            mask is left to the instrument. None, and the rest of the
            pipeline takes that into account.
    snr     the signal-to-noise ratio per order. When the header does not
            carry it, it is measured on the flux itself (robust, per order).

THE TWO PARITIES. pca2d reduces each exposure to two spectra, the even orders
and the odd orders, because on an echelle orders n and n+1 overlap while n and
n+2 never touch: the whole spectrum thus fits on two rows with no redundancy.
The rule is kept for all instruments, but it is CHECKED by `parity_is_safe`,
which measures the actual overlap and warns when the assumption does not
hold.
"""

from __future__ import annotations

import functools
import os
import warnings

import numpy as np
from lbl.core.math import mad_to_sigma

def sources_of(name):
    """The data sources LBL has a class for, for the instrument it calls
    `name` (lbl.instruments.select.InstDict), in its order; () when it knows
    no such instrument."""
    from lbl.instruments import select

    return tuple((select.InstDict.get(str(name).upper()) or {}).keys())


def instrument(name, data_source, config_file=None):
    """The LBL class that reads this instrument, instantiated.

    `parse=False` is what allows it to be instantiated from code rather than
    from a command line; this is already how lbl.check_profile checks a
    profile before a campaign.
    """
    from lbl.instruments import select

    # DATA_TYPE SCIENCE, as a wrap gives it to LBL's recipes: pca2d reads
    #   science spectra only, and several profiles give a BERV for science
    #   alone (get_berv), zero otherwise
    kwargs = dict(instrument=str(name), data_source=str(data_source),
                  data_type='SCIENCE')
    if config_file:
        kwargs["config_file"] = os.path.abspath(config_file)
    args = select.parse_args(
        ["INSTRUMENT", "DATA_SOURCE", "DATA_TYPE", "INPUT_FILE"],
        kwargs, __name__, parse=False)
    return select.load_instrument(args, plogger=None)


def _lbl_root():
    """The folder of the ACTUALLY installed LBL package, or None.

    Three paths, from the most direct to the most stubborn, because the
    repository's `lbl/` data folder can mask the first two:

      1. the ordinary import, when nothing masks it;
      2. the sys.path entries that hold a real lbl/__init__.py;
      3. the installed distribution, through importlib.metadata. It is the
         only one that survives an EDITABLE install (pip install -e), where
         the package is on no sys.path entry but is reached by a finder set
         up by a .pth: a namespace package wins against that finder, and
         only the folder declared by the distribution says where the code
         really is.
    """
    import importlib.metadata as metadata
    import importlib.util
    import sys

    spec = importlib.util.find_spec("lbl")
    if spec is not None and spec.origin:
        return os.path.dirname(spec.origin)
    for entry in sys.path:
        candidate = os.path.join(entry, "lbl", "__init__.py")
        if os.path.exists(candidate):
            return os.path.dirname(candidate)
    try:
        dist = metadata.distribution("lbl")
    except Exception:                                          # noqa: BLE001
        return None
    for item in dist.files or []:
        if item.parts[:2] == ("lbl", "__init__.py"):
            return os.path.dirname(str(dist.locate_file(item)))
    # an editable install: the finder set up by the .pth knows where the code
    # is, and says so in its MAPPING table
    for item in dist.files or []:
        text = str(item)
        if "__editable___" in text and text.endswith("_finder.py"):
            try:
                path = str(dist.locate_file(item))
                source = open(path, encoding="utf-8").read()
            except Exception:                                  # noqa: BLE001
                continue
            import re
            found = re.search(r"['\"]lbl['\"]\s*:\s*['\"]([^'\"]+)['\"]", source)
            if found and os.path.isdir(found.group(1)):
                return found.group(1)
    return None


def _lbl_science(name):
    """A module of lbl.science, even when the data folder masks it.

    THE TRAP: a working folder can hold an `lbl/` folder that is an LBL DATA
    tree (calib, masks, models, science...). As soon as
    the repository's root is on sys.path, `import lbl` finds this folder
    before the installed package and turns it into a namespace package:
    lbl.__file__ is None and `from lbl.science import tellu_clean` fails with
    "unknown location". And this is worse than a plain failure, because
    lbl.instruments is NOT masked (the data folder has no such subfolder):
    the instrument loads, the transmission does not, and nothing says so.

    Loading only the wanted module by its path is not enough: it imports its
    neighbours through `from lbl.science import general`, which falls back
    on the masked folder. So the package is REPAIRED, by adding the real
    folder to the __path__ of lbl and of lbl.science, after which the
    ordinary import finds everything.
    """
    import importlib
    import sys

    full = "lbl.science.%s" % name
    if full in sys.modules:
        return sys.modules[full]
    try:
        root = _lbl_root()
        if root is not None:
            package = importlib.import_module("lbl")
            paths = list(getattr(package, "__path__", []))
            if root not in paths:
                package.__path__ = [root] + paths
                # lbl.science may have been resolved before the repair: it
                # then carries the data folder's path and must be forgotten
                for key in [k for k in sys.modules
                            if k == "lbl.science" or k.startswith("lbl.science.")]:
                    del sys.modules[key]
        return importlib.import_module(full)
    except Exception:                                          # noqa: BLE001
        return None


def tapas_transmission(inst, wave, model_dir, airmass=1.0):
    """The telluric transmission on `wave`, from LBL's TAPAS model.

    APERO writes the transmission it divided out into the t.fits (Recon), and
    pca2d uses it to know where the spectrum is blind. Visible spectrographs
    do not deliver it: LBL models it, and it is that model that is evaluated
    here, raised to the exposure's airmass.

    Returns None when the model is not there, which the caller must accept:
    with no transmission, the cuts that depend on it are simply inactive.
    """
    tellu_clean = _lbl_science("tellu_clean")
    if tellu_clean is None:
        return None
    # LBL copies the TAPAS file into <DATA_DIR>/other/ before reading it, and
    # DATA_DIR is empty on a class instantiated outside a campaign: without
    # it, get_tapas_lbl stops on a realpath(None). The models folder is, by
    # construction, <DATA_DIR>/models, so its parent is the answer. It is
    # never overwritten when a campaign has already set it.
    try:
        if not inst.params.get("DATA_DIR"):
            inst.params.set("DATA_DIR",
                            os.path.dirname(os.path.abspath(model_dir)))
    except Exception:                                          # noqa: BLE001
        pass
    try:
        splines = tellu_clean.get_tapas_spl(inst, model_dir)
    except Exception:                                          # noqa: BLE001
        return None
    if splines is None:
        return None
    # get_tapas_spl returns the splines of the two TAPAS components, water
    # and the rest; the total transmission is their product, each at the
    # exposure's airmass
    out = np.ones_like(np.asarray(wave, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spline in (splines if isinstance(splines, (list, tuple))
                       else [splines]):
            try:
                piece = np.asarray(spline(np.asarray(wave, dtype=float)),
                                   dtype=float)
            except Exception:                                  # noqa: BLE001
                return None
            out *= np.clip(piece, 0.0, 1.0) ** float(airmass)
    return out


def order_snr(flux, header_snr=None, n_orders=None):
    """The signal-to-noise ratio per order.

    The header's when the instrument writes it (APERO does, EXTSNxxx),
    measured on the flux otherwise: the median divided by the robust scatter
    of the differences between neighbours, which assumes nothing about the
    noise model.
    """
    if header_snr is not None and np.size(header_snr):
        return np.asarray(header_snr, dtype=float)
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    out = np.full(flux.shape[0], np.nan)
    for i, row in enumerate(flux):
        good = np.isfinite(row) & (row > 0)
        if good.sum() < 16:
            continue
        level = float(np.median(row[good]))
        step = np.diff(row[good])
        noise = mad_to_sigma * float(np.median(np.abs(step - np.median(step))))
        # the difference of two neighbours has sqrt(2) times the noise of one
        out[i] = level / (noise / np.sqrt(2.0)) if noise > 0 else np.nan
    return out if n_orders is None else out[:n_orders]


def parity_is_safe(wave, tolerance=0.0):
    """(true if orders n and n+2 do not overlap, the worst overlap).

    pca2d puts the even orders on one row and the odd ones on the other, which
    assumes that two orders of the same parity never measure the same
    wavelength. That is true of most echelles, but it has to be checked
    rather than hoped for, because an overlap would write two measurements
    into the same column of the grid.
    """
    wave = np.atleast_2d(np.asarray(wave, dtype=float))
    worst = 0.0
    for i in range(wave.shape[0] - 2):
        a, b = wave[i], wave[i + 2]
        if not (np.isfinite(a).any() and np.isfinite(b).any()):
            continue
        lo, hi = np.nanmax(a), np.nanmin(b)
        if np.isfinite(lo) and np.isfinite(hi):
            worst = max(worst, float(lo - hi))
    return worst <= float(tolerance), worst


def _doppler(velocity_kms):
    """The wavelength ratio for a velocity, RELATIVISTIC.

    A single Doppler in the repository, the one in grids:
    sqrt((1 + v/c) / (1 - v/c)), v > 0 if the source is receding. The first
    order, 1 + v/c, is off by (v/c)^2 / 2, i.e. 1.5 m/s at a 30 km/s BERV,
    which matters here.
    """
    from .grids import doppler

    return float(doppler(float(velocity_kms)))


def berv_of(inst, header, from_header=None):
    """The BERV to apply to this spectrum, in km/s, as LBL decides it.

    NOT THE HEADER'S. The question is not "what was the barycentric motion",
    which the header always gives, but "is it still to be applied", which
    depends on the format. A spectrum on a grid in the observer frame needs
    it applied; a product whose grid is ALREADY barycentric (a
    WAVEDATA_VAC_BARY, for example) does not, and applying it a second time
    shifts the spectrum by the whole BERV.

    LBL has already settled this, class by class: its get_berv reads the
    header for a grid in the observer frame and returns zero for a
    barycentric one. So the question is put to it rather than answered here,
    and the day a format changes convention, LBL is the one that will know.
    """
    try:
        value = inst.get_berv(header)          # LBL returns m/s
    except Exception:                                          # noqa: BLE001
        value = None
    if value is None:
        try:
            return float(from_header)
        except (TypeError, ValueError):
            return float("nan")
    return float(value) / 1000.0


def observer_berv(inst, header):
    """The observer's BERV in km/s, from LBL's instrument profile, as its
    recipes take it: get_berv, what is left to apply to reach the star's
    frame (the whole BERV for a grid in the observer's frame), plus
    get_berv_transmission, what the telluric lines are shifted by on a grid
    that is already barycentric (whose get_berv is zero; lbl_telluclean).
    One of the
    two is zero for every profile. The telluric lines move with it in the
    star's frame, whatever frame the file is in: the river plots' green line
    follows it, and the observer's frame (input.to_observer) and the
    bundles of a night (input.bundle_berv) are made with it. NaN when the
    profile cannot say."""
    try:
        value = (float(inst.get_berv(header))
                 + float(inst.get_berv_transmission(header)))
    except Exception:                                          # noqa: BLE001
        return float("nan")
    return value / 1000.0


def as_science(inst, path):
    """The profile as a recipe runs it on this spectrum: OBJECT_SCIENCE its
    object, the folder it is in, when the profile has none (a get_berv can
    read it, to give no BERV for an FP; without it, it raises, and the BERV
    would be read from the header instead), DATA_TYPE SCIENCE
    (instrument())"""
    try:
        if not inst.params.get("OBJECT_SCIENCE"):
            inst.params.set("OBJECT_SCIENCE", value=os.path.basename(
                os.path.dirname(os.path.abspath(str(path)))),
                source=__name__)
    except Exception:                                          # noqa: BLE001
        pass
    return inst


@functools.lru_cache(maxsize=8)
def profile(name, data_source):
    """LBL's instrument profile, made once per instrument and source"""
    return instrument(name, data_source)


def profile_of(config, inst=None):
    """LBL's instrument profile of a configuration (its `lbl` block, as
    read() is given it), or `inst` when given"""
    if inst is not None:
        return inst
    block = config.get("lbl") or {}
    name = block.get("instrument") or (config.get("input") or {}).get(
        "instrument")
    return profile(str(name), str(block.get("data_source")))


def profile_header(inst, path):
    """A spectrum's header as LBL's template recipe reads it to take its
    BERV (lbl_template._load_template_header): the keys the template uses,
    else the whole header"""
    from lbl.recipes.lbl_template import _load_template_header
    return _load_template_header(inst, path)


def file_observer_berv(config, path, inst=None):
    """observer_berv of a file, its header read as the template recipe
    reads it"""
    try:
        inst = as_science(profile_of(config, inst), path)
        return observer_berv(inst, profile_header(inst, path))
    except Exception:                                          # noqa: BLE001
        return float("nan")


def flux_extension(path, flux):
    """The name (or the number) of the extension this flux comes from.

    RETURN THE CORRECTED SPECTRUM IN THE ORIGINAL FORMAT. The corrected file
    is a copy of the original whose flux is replaced; that still requires
    knowing which extension to replace, and it does not have the same name
    everywhere (FluxAB, SCIDATA...). LBL does not name it: its default class calls fits.getdata(file) with no extension,
    which returns the FIRST one that holds data.

    So this name is not guessed, it is FOUND AGAIN: the extension whose data
    have the shape of the flux read, the first one from the start, exactly as
    astropy chose. Returns a name when there is one, otherwise a number, and
    None if nothing matches, in which case the caller refuses to write rather
    than write in the wrong place.
    """
    from astropy.io import fits

    shape = np.shape(flux)
    try:
        with fits.open(path, memmap=True) as hdulist:
            for index, hdu in enumerate(hdulist):
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                if np.atleast_2d(data).shape == shape:
                    return hdu.name or index
    except Exception:                                          # noqa: BLE001
        return None
    return None


#: the instruments already told that their blaze is taken as 1
_NO_BLAZE = set()


def _blaze(inst, path, flux, header, model_dir):
    """The blaze of a spectrum, as LBL reads it: corrected spectra need none,
    the others the file their header names, in DATA_DIR/calib where LBL
    keeps it (models/ as well, where an older campaign put it), else the
    default blaze of the setup (models/default-blaze, Instrument.blaze_path).
    Without one, the blaze is taken as 1, and said once per instrument and
    plainly, since LBL's own reader would log a red error for a file that is
    simply not there: PCA2D high-passes every spectrum, which takes the
    blaze's shape out; only the weights do without it."""
    if inst.params.get("BLAZE_CORRECTED"):
        return np.ones_like(flux)
    name = None
    try:
        name = header.get_hkey(inst.params["KW_BLAZE_FILE"])
    except Exception:                                           # noqa: BLE001
        pass
    folders = []
    if model_dir:
        folders.append(os.path.join(os.path.dirname(os.path.abspath(
            model_dir)), "calib"))
        folders.append(model_dir)
    for folder in folders:
        if name and os.path.exists(os.path.join(folder, os.path.basename(
                str(name)))):
            try:
                blaze, _ = inst.load_blaze_from_science(
                    path, flux, header, calib_directory=folder,
                    normalize=True)
                return np.atleast_2d(np.asarray(blaze, dtype=float))
            except Exception:                                   # noqa: BLE001
                break
    # the default blaze of the setup (models/default-blaze), which LBL's own
    #   reader takes when the night's is not in calib/; DATA_DIR is the
    #   models folder's parent for a class made outside a campaign
    if model_dir and not inst.params.get("DATA_DIR"):
        try:
            inst.params.set("DATA_DIR",
                            os.path.dirname(os.path.abspath(model_dir)))
        except Exception:                                       # noqa: BLE001
            pass
    default = None
    try:
        default = inst.default_blaze_file()
    except Exception:                                           # noqa: BLE001
        pass
    if default and os.path.exists(default) and folders:
        try:
            blaze, _ = inst.load_blaze_from_science(
                path, flux, header, calib_directory=folders[0],
                normalize=True)
            return np.atleast_2d(np.asarray(blaze, dtype=float))
        except Exception:                                       # noqa: BLE001
            pass
    instrument = str(inst.params.get("INSTRUMENT"))
    if instrument not in _NO_BLAZE:
        _NO_BLAZE.add(instrument)
        from .logger import log
        log("no blaze file for %s (%s, looked for in calib/, models/ and"
            " models/default-blaze/): the"
            " blaze is taken as 1. That is fine for PCA2D: its high pass takes"
            " the blaze's shape out of every spectrum, and only the weights do"
            " without it" % (instrument, os.path.basename(str(name))
                             if name else "none named"), "info")
    return np.ones_like(flux)


def read(path, instrument_name, data_source, config_file=None,
         model_dir=None, inst=None, to_observer=False):
    """A spectrum, in the form pca2d expects, whatever the instrument.

    Same dict as tfits.read_tfits: flux, wave, blaze, recon, sky, snr, meta.
    `inst` avoids re-instantiating the LBL class for every file of a
    campaign.
    """
    from lbl.core import io as lbl_io                           # noqa: F401

    inst = as_science(inst or instrument(instrument_name, data_source,
                                         config_file), path)
    flux, header = inst.load_science_file(path)
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    science_ext = flux_extension(path, flux)
    to_observer = bool(to_observer)
    wave = np.atleast_2d(np.asarray(
        inst.get_wave_solution(path, flux, header), dtype=float))
    blaze = _blaze(inst, path, flux, header, model_dir)

    def key(name, default=None):
        """A keyword by its LBL name, which each class translates for its
        instrument (KW_BERV, KW_AIRMASS...).

        AN ENTRY CAN BE A LIST: an instrument fed by several telescopes
        gives one keyword per telescope, of which only one is filled, and
        `header[list]` would raise. So the first one that is there is taken.
        """
        try:
            wanted = inst.params[name]
        except Exception:                                       # noqa: BLE001
            return default
        for candidate in (wanted if isinstance(wanted, (list, tuple))
                          else [wanted]):
            try:
                value = header[candidate]
            except Exception:                                   # noqa: BLE001
                continue
            if value is not None:
                return value
        return default

    def plain(name, default=None):
        """A keyword by its FITS name, without going through LBL's table."""
        try:
            return header[name]
        except Exception:                                       # noqa: BLE001
            return default

    def number(*candidates):
        """The first of the candidates that is a finite, non-sentinel number.

        An instrument that did not measure something often writes it as an
        agreed value rather than as a blank: -1 for a seeing that ESO does
        not have, -9999.9 for a weather mast that is down. Treating it as a
        number would mean putting a negative seeing into a diagnostic, so it
        becomes NaN, exactly as in tfits.read_tfits.
        """
        for value in candidates:
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(out):
                continue
            if any(abs(out - s) < 1e-6 for s in SENTINELS):
                continue
            return out
        return np.nan

    airmass = key("KW_AIRMASS", 1.0)
    # TIME AND EXPOSURE DO NOT COME FROM LBL'S TABLE FIRST. A class can point
    # KW_MJDATE and KW_EXPTIME at the same keyword (a BJD), because LBL only
    # needs a time axis and does not use the exposure time. pca2d uses them
    # differently: mjdate is a MODIFIED date and exptime weights. The FITS
    # keywords (MJD-OBS, EXPTIME) are therefore read first and LBL's table is
    # only a last resort.
    mjd = number(plain("MJD-OBS"), key("KW_MJDATE"))
    # a header that gives only a BJD: it is brought back to an MJD rather
    # than letting 2.45 million days pass for a modified date
    if np.isfinite(mjd) and mjd > 2.4e6:
        mjd -= 2400000.5
    # THE SAME CONTRACT AS read_tfits, key for key. cube.py reads
    # meta["bjd"], meta["berv"], meta["object"]... without knowing where the
    # spectrum comes from: an incomplete dict does not cause a degraded read
    # but a KeyError in the middle of a campaign. What the header does not
    # carry is NaN, which is the honest answer, and never an invented value.
    bjd = number(key("KW_BJD"), key("KW_MJDATE"))
    if np.isfinite(bjd) and bjd < 1.0e6:
        # a header that gives only an MJD: the BJD is that plus 2400000.5,
        # and the barycentric difference (at most 8 minutes) is not
        # meaningful here
        bjd += 2400000.5
    # THE FRAME IN WHICH THE SPECTRUM IS RETURNED.
    #
    # THE RULE IS LBL'S, AND IT IS UNIFORM. Each LBL class answers a single
    # question with its get_berv: "what is left to apply to reach the star
    # frame". For a grid in the observer frame the answer is the header's
    # BERV; for an already barycentric grid it is zero. LBL does not treat these two cases differently
    # afterwards: it calls get_berv once (science/general.py, "instrument
    # specific berv --> use instrument method") and everything else is
    # common.
    #
    # pca2d gives meta["berv"] EXACTLY that meaning, because it is already
    # the one it gave it: twoframe shifts the star frame by
    # -pixel_shift(meta["berv"]). Going through get_berv rather than through
    # the header therefore adds no rule, it removes one: pca2d no longer has
    # to know which instrument delivers which frame, LBL knows it for it.
    #
    # WHAT `to_observer` CHANGES, AND WHAT IT DOES NOT. With or without it,
    # a stellar line falls at the SAME wavelength in the star's frame, to the
    # pixel, so the STAR frame is the same and the physics is unchanged.
    # What changes is
    # the OBSERVER frame: without the option it is the barycentric grid, the
    # same for all exposures, and the model's observer block then has nothing
    # to hold on to; with it, it is the actual frame of the telescope, and
    # the two blocks become distinguishable again. It is an option of the
    # FIT, not of the reading.
    # THE OBSERVER'S BERV, as LBL's profile gives it
    #   (observer_berv: get_berv + get_berv_transmission), for the observer
    #   frame and the green line of the river plots; never read here from
    #   the header through a keyword list of our own
    obs_berv = observer_berv(inst, header)
    berv = berv_of(inst, header, number(key("KW_BERV")))
    if to_observer:
        if not np.isfinite(obs_berv):
            raise ValueError(
                "%s: input.to_observer asks to undo the barycentric"
                " correction, but LBL's profile gives no BERV"
                % os.path.basename(path))
        if berv not in (0.0,):
            # the file is already in the observer frame: there is nothing
            # to undo, and redoing it would shift everything
            pass
        else:
            # lambda_observer = lambda_bary / doppler(BERV), relativistic
            wave = wave / _doppler(obs_berv)
            berv = float(obs_berv)
    # the start and end of the exposure, by the class's keywords (KW_AIRMASS,
    # KW_AIRMASS_END): an instrument that reports one value gives it for both
    air_start = number(airmass)
    air_end = number(key("KW_AIRMASS_END"), airmass)
    meta = {
        "filename": os.path.basename(path),
        "bjd": float(bjd),
        "mjdmid": float(mjd),
        "mjdate": float(mjd),
        "berv": berv,
        "berv_obs": obs_berv,
        "bervmax": float(number(key("KW_BERVMAX"))),
        "exptime": float(number(plain("EXPTIME"), key("KW_EXPTIME"))),
        "airmass_start": float(air_start),
        "airmass_end": float(air_end),
        "airmass": float(airmass if airmass is not None else np.nan),
        "seeing": float(number(key("KW_SEEING"))),
        "humidity": float(number(key("KW_HUMIDITY"))),
        "sun_elevation": float(number(key("KW_SUN_ELEVATION"))),
        "drs_mode": str(key("KW_DRS_MODE", "") or "").strip(),
        "dprtype": str(key("KW_DPRTYPE", "") or "").strip(),
        "object": str(key("KW_OBJNAME", "") or "").strip(),
        "instrument": str(instrument_name),
        "data_source": str(data_source),
    }
    meta["drsobjn"] = meta["object"]
    # THE TRANSMISSION IS NEVER None. order_quality and _order_sigma index it
    # as an array: a None would make them crash, and making it optional
    # everywhere downstream would mean scattering `if`s for a situation that
    # has an honest answer. Without a model, a transmission of 1 is declared,
    # that is "no known absorption here": the cuts that depend on it then
    # remove nothing, instead of removing anything at all. A campaign's
    # report says which of the two.
    recon = (tapas_transmission(inst, wave, model_dir, meta["airmass"])
             if model_dir else None)
    meta["telluric_model"] = "tapas" if recon is not None else "none"
    if recon is None:
        recon = np.ones_like(flux)
    meta["flux_extension"] = science_ext
    return dict(flux=flux, wave=wave, blaze=blaze, recon=recon, sky=None,
                snr=order_snr(flux), meta=meta)


#: what a header writes when the measurement did not take place, and which
#: must become NaN rather than pass for a number. The first three are those
#: of tfits.read_tfits (a weather mast that is down); -1 is written by some
#: observatories for a missing seeing.
SENTINELS = (-9999.0, -9999.9, -999.0, -1.0)


def reads_natively(config):
    """Should these spectra be read with tfits, or go through LBL?

    An APERO t.fits is read by tfits (input.format 'tfits', from the
    instrument's LBL class): it carries things LBL's class does not return
    (the sky emission, the telluric transmission, the SNR per order in the
    header) and that reader makes use of them. Everything else goes through
    LBL. `input.reader` decides when one or the other is to be forced.
    """
    asked = str((config.get("input") or {}).get("reader") or "auto").lower()
    if asked in ("tfits", "native"):
        return True
    if asked in ("lbl", "bridge"):
        return False
    return str((config.get("input") or {}).get("format") or "") == "tfits"


def star_frame(payload, config, path=None):
    """The star's frame of one exposure: its BERV composed with the star's
    velocity at that exposure, from a free fit of the velocities when there
    is one (registration.velocities, by file name: LBL's template), else the
    Keplerian velocity of the star's companions at its BJD
    (registration.orbits).

    `berv` is what every shift into the star's frame reads (the cube's rows,
    the template, the fit, the correction), so it becomes the velocity of that
    frame, (berv - vstar) / (1 - berv vstar / c2): the star at +vstar is
    where a star at rest would be under that BERV. The BERV itself is kept as
    `berv_bary`, for what is about the barycentre (the diagnostics of a bias
    with BERV), and the companions' velocity as `vstar` [km/s]. Without
    orbits nothing changes. Applied once: a payload that has berv_bary is
    left as it is.
    """
    registration = (config or {}).get("registration") or {}
    orbits = registration.get("orbits") or []
    velocities = registration.get("velocities") or {}
    meta = payload.get("meta")
    if (not orbits and not velocities) or meta is None or "berv_bary" in meta:
        return payload
    from .grids import C_KMS
    berv = float(meta.get("berv", np.nan))
    name = os.path.basename(str(path or meta.get("filename") or ""))
    if name in velocities:
        vstar = float(velocities[name])
    elif orbits:
        from lbl.science import planets
        bjd = float(meta.get("bjd", np.nan))
        vstar = float(planets.star_velocity([bjd], orbits)[0]) / 1000.0
    else:
        return payload
    meta["berv_bary"] = berv
    meta["vstar"] = vstar
    meta["berv"] = (berv - vstar) / (1.0 - berv * vstar / C_KMS ** 2)
    return payload


def read_any(path, config, inst=None):
    """A spectrum, by the reader the configuration names, in the star frame
    the configuration names (star_frame).

    This is the only place where pca2d decides how to read: cube,
    reconstruct and plotting go through here, so adding an instrument only
    touches `reads_natively`.
    """
    return star_frame(_read_any(path, config, inst), config, path)


def _read_any(path, config, inst=None):
    """read_any, without the star's frame"""
    if reads_natively(config):
        from . import tfits as sptf
        inp = config.get("input") or {}
        payload = sptf.read_tfits(path, inp.get("extensions"),
                                  inp.get("header_keys"))
        # the BERV from LBL's instrument profile, as its recipes take it,
        #   rather than as tfits reads the header (the same BERV keyword for
        #   APERO, but one extraction for every reader); tfits' own when the
        #   profile cannot be had (a configuration without its lbl block)
        meta = payload["meta"]
        try:
            prof = as_science(profile_of(config, inst), path)
            header = profile_header(prof, path)
            applied = berv_of(prof, header, meta.get("berv"))
            if np.isfinite(applied):
                meta["berv"] = applied
            meta["berv_obs"] = observer_berv(prof, header)
        except Exception:                                      # noqa: BLE001
            meta.setdefault("berv_obs", meta.get("berv", np.nan))
        return payload
    block = config.get("lbl") or {}
    name = block.get("instrument") or (config.get("input") or {}).get("instrument")
    source = block.get("data_source")
    models = (config.get("output") or {}).get("model_directory")
    observer = bool((config.get("input") or {}).get("to_observer"))
    return read(path, name, source, model_dir=models, inst=inst,
                to_observer=observer)
