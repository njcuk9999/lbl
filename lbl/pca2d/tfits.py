"""Reading APERO `t.fits` spectra: one row per diffraction order.

Layout of a NIRPS t.fits (APERO 0.7.x, the file LBL is fed):

    PRIMARY       ESO + APERO header, no data
    FluxA         (75, 4088)  telluric-CORRECTED extracted flux, blaze still in
    WaveA         (75, 4088)  wavelength solution, nm, OBSERVER frame
    BlazeA        (75, 4088)  blaze
    Recon         (75, 4088)  the telluric transmission that was divided out
    SKYCORR_SCI   (75, 4088)  sky model, not used here
    SKYCORR_CAL   (75, 4088)  sky model, not used here

Three things about this format drive everything below.

1. There is no single wavelength grid. Each order carries its own solution, and
   consecutive orders overlap by 2-5 nm, so part of the spectrum is measured
   twice at two different samplings. Orders n and n+2, on the other hand, never
   overlap (measured: the gap runs from 4 to 22 nm across the array). So the
   whole echellogram collapses onto exactly two non-redundant spectra on a
   common grid -- the even orders and the odd orders -- and that is what we
   build. Two rows per exposure instead of 75, with nothing thrown away.

2. The sampling is not uniform in velocity: within an order the step drifts, and
   between orders it jumps. Anything with a fixed pixel width -- the high-pass,
   above all -- has to be applied *after* resampling onto the log-uniform
   destination grid, never on the native pixels.

3. BERV, BJD and the per-order extraction SNR live in the FluxA header, not in
   PRIMARY. The observing conditions (airmass, humidity) live in PRIMARY. Both
   have to be consulted.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from astropy.io import fits

from . import preprocess as prep
from .grids import doppler
from .io import header_float, robust_open
from .logger import log

C_KMS = 299792.458

# ---------------------------------------------------------------------------
# Which extensions to read, per instrument
# ---------------------------------------------------------------------------
# This table is the one place in the project that names a FITS extension, and
# getting it wrong is the failure mode with no symptom.
#
# A SPIRou t.fits carries FluxA and FluxB as well as FluxAB. Those are the two
# polarimetric fibres taken separately; the science flux, the one LBL works
# with, is AB. Reading "FluxA" from a SPIRou file raises no error at all: it
# returns a real array of half the light, with a different blaze and a different
# SNR, and every number downstream looks plausible and is wrong.
#
# The instrument is read from the FILE, never from the configuration, so a
# mislabelled config cannot select the wrong fibre.
INSTRUMENTS = {
    # `sky` is the pipeline's model of the atmospheric emission that was
    # subtracted: OH airglow for SPIRou, the sky-fibre correction for NIRPS. It
    # is not used to model anything, only to find samples where that emission
    # was so much brighter than the star that whatever fraction of it the
    # subtraction got wrong swamps the spectrum.
    "NIRPS": dict(flux="FluxA", wave="WaveA", blaze="BlazeA", recon="Recon",
                  sky="SKYCORR_SCI"),
    "SPIROU": dict(flux="FluxAB", wave="WaveAB", blaze="BlazeAB", recon="Recon",
                   sky="OHLine"),
}

def register_instruments(table):
    """Add or replace instrument blocks, from the config rather than from here.

    The two below are what this package was written against; a spectrograph is
    added by putting an `instruments:` block in config.yaml, never by editing
    this file. Registering the same name twice replaces it.
    """
    for name, extensions in (table or {}).items():
        INSTRUMENTS[str(name).upper()] = dict(extensions)


#: NIRPS names, kept as module constants so that code and tests written before
#: the table existed still mean what they meant.
EXT_FLUX = INSTRUMENTS["NIRPS"]["flux"]
EXT_WAVE = INSTRUMENTS["NIRPS"]["wave"]
EXT_BLAZE = INSTRUMENTS["NIRPS"]["blaze"]
EXT_RECON = INSTRUMENTS["NIRPS"]["recon"]


def instrument_of(hdulist):
    """The instrument that wrote this file, as a key of INSTRUMENTS.

    INSTRUME lives on the science extension for some products and on PRIMARY
    for others, so both are consulted. Raises rather than guessing: a file whose
    instrument cannot be established must stop the run, because every choice
    below depends on it.
    """
    for hdu in hdulist:
        value = hdu.header.get("INSTRUME")
        if value:
            key = str(value).strip().upper()
            if key in INSTRUMENTS:
                return key
            raise ValueError("unsupported INSTRUME %r; known: %s"
                             % (value, ", ".join(sorted(INSTRUMENTS))))
    raise ValueError("no INSTRUME keyword; cannot tell which extensions to read")


def extensions_for(hdulist):
    """(flux, wave, blaze, recon) extension names for this file's instrument."""
    ext = INSTRUMENTS[instrument_of(hdulist)]
    return ext["flux"], ext["wave"], ext["blaze"], ext["recon"]


def sky_extension_for(hdulist):
    """Name of the sky-emission extension, or None if this file has none."""
    name = INSTRUMENTS[instrument_of(hdulist)].get("sky")
    return name if name and name in [h.name for h in hdulist] else None

# Header keys pulled into the metadata table. Looked up in the FluxA header
# first, then in PRIMARY: APERO writes its own keywords (BERV, BJD, EXTSNxxx)
# only on the science extension.
# Each entry is a list of candidate keywords, tried in order and first hit
# wins. NIRPS is at La Silla and writes the ESO hierarchical set; SPIRou is at
# CFHT and writes its own. Rather than a second per-instrument table, the
# candidates simply sit side by side: a keyword that is absent costs one failed
# dictionary lookup, and a file that carries neither yields NaN, which is the
# honest answer.
META_KEYS = {
    "bjd": ["BJD"],
    "mjdmid": ["MJDMID", "MJDATE"],
    "berv": ["BERV"],
    "bervmax": ["BERVMAX"],
    "exptime": ["EXPTIME"],
    # NIRPS brackets the exposure, SPIRou reports one value for it
    "airmass_start": ["ESO TEL AIRM START", "AIRMASS"],
    "airmass_end": ["ESO TEL AIRM END", "AIRMASS"],
    # SGSSEE is the SPIRou guider's own seeing estimate, which is what the fibre
    # actually saw; the weather-mast values (WMSEE*) describe the site instead
    # and are often flagged -9999.9 when the mast is down
    "seeing": ["ESO TEL AMBI FWHM START", "SGSSEE", "WMSEEMED"],
    "humidity": ["ESO TEL AMBI RHUM", "RELHUMID"],
    "sun_elevation": ["DRSSUNEL"],
    "drs_mode": ["DRSMODE"],
    "dprtype": ["DPRTYPE"],
}

#: values both pipelines use to mean "not measured"; kept out of the metadata
#: rather than allowed to pose as a seeing of -9999 arcsec
_SENTINELS = (-9999.0, -9999.9, -999.0)
_STRING_KEYS = {"drs_mode", "dprtype"}

PARITY_NAMES = ("even", "odd")


def _get(headers, key, default=np.nan):
    for header in headers:
        if key in header:
            return header[key]
    return default


def read_tfits(path: str):
    """Read one t.fits into plain arrays plus a metadata dict.

    Nothing is cropped or resampled here; that is resample_exposure()'s job.
    """
    with robust_open(path) as hdulist:
        e_flux, e_wave, e_blaze, e_recon = extensions_for(hdulist)
        primary = hdulist[0].header
        science = hdulist[e_flux].header
        flux = np.asarray(hdulist[e_flux].data, dtype=np.float64)
        wave = np.asarray(hdulist[e_wave].data, dtype=np.float64)
        blaze = np.asarray(hdulist[e_blaze].data, dtype=np.float64)
        recon = np.asarray(hdulist[e_recon].data, dtype=np.float64)
        e_sky = sky_extension_for(hdulist)
        sky = (np.asarray(hdulist[e_sky].data, dtype=np.float64)
               if e_sky else None)

    headers = (science, primary)
    n_orders = flux.shape[0]
    snr = np.array([header_float(science, "EXTSN%03d" % o) for o in range(n_orders)])
    snr[~(snr > 0)] = np.nan

    meta = {"filename": os.path.basename(path)}
    for name, keys in META_KEYS.items():
        value = None
        for key in keys:
            # default=None, NOT the np.nan _get falls back to: a NaN would end
            # the search on the first candidate that happens to be absent
            value = _get(headers, key, default=None)
            if value is not None:
                break
        if name in _STRING_KEYS:
            meta[name] = str(value if value is not None else "").strip()
        else:
            number = header_float({"k": value}, "k")
            if number is not None and any(abs(number - s) < 1e-6 for s in _SENTINELS):
                number = float("nan")
            meta[name] = number
    meta["object"] = str(_get(headers, "OBJECT", "")).strip()
    # APERO's own name for the target, upper case and without the form's
    # spelling: what cube.same_object falls back on when OBJECT was typed
    # otherwise
    meta["drsobjn"] = str(_get(headers, "DRSOBJN", "")).strip()
    meta["airmass"] = 0.5 * (meta["airmass_start"] + meta["airmass_end"])
    return dict(flux=flux, wave=wave, blaze=blaze, recon=recon, sky=sky,
                snr=snr, meta=meta)


def band_snr(payload: dict, wave_min: float, wave_max: float) -> float:
    """Median per-order extraction SNR over the orders inside the domain."""
    wave, snr = payload["wave"], payload["snr"]
    centre = wave[:, wave.shape[1] // 2]
    keep = np.isfinite(snr) & (centre > wave_min) & (centre < wave_max)
    if not np.any(keep):
        keep = np.isfinite(snr)
    if not np.any(keep):
        return np.nan
    return float(np.median(snr[keep]))


def order_quality(payload: dict, config: dict):
    """Per-order good-pixel masks and the list of orders worth resampling.

    A pixel is usable when flux, wavelength, blaze and transmission are all
    finite, the flux is positive, and the blaze has not collapsed. That last
    cut is what keeps the order edges out: the extracted flux there is a small
    number divided by a small blaze, and splining it is asking for trouble.

    Whole orders drop out of NIRPS in the deep water bands (44-45, ~1370-1400
    nm) and beyond ~1850 nm (71-74), where nothing survives the telluric
    correction. That is expected and is not an error; they are simply skipped
    and the destination grid keeps a gap there.
    """
    inp = config["input"]
    blaze_frac = float(inp.get("blaze_min_frac", 0.0) or 0.0)
    min_fraction = float(inp.get("min_order_finite_fraction", 0.05) or 0.0)

    flux, wave = payload["flux"], payload["wave"]
    blaze, recon = payload["blaze"], payload["recon"]
    n_orders, n_pixels = flux.shape

    good = (
        np.isfinite(flux) & (flux > 0)
        & np.isfinite(wave)
        & np.isfinite(blaze) & (blaze > 0)
        & np.isfinite(recon) & (recon > 0)
    )
    # Samples the sky drowned. On TOI-2120 the OH airglow reaches 1447 per cent
    # of the stellar flux at 1669.23 nm, so a one per cent error in subtracting
    # it leaves fourteen per cent of the star. No fixed basis vector can absorb
    # that: the airglow varies exposure to exposure, on minutes. Down-weighting
    # would still let its noise in, so these samples are removed outright. At a
    # ratio of 4 this costs 0.21 per cent of the samples of a SPIRou spectrum.
    max_sky = config["quality"].get("max_sky_ratio")
    sky = payload.get("sky")
    if max_sky and sky is not None and sky.shape == flux.shape:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.abs(sky) / np.where(flux > 0, flux, np.nan)
        good &= ~(np.isfinite(ratio) & (ratio > float(max_sky)))

    # UNE SOLUTION EN LONGUEUR D'ONDE QUI RECULE N'EST PAS UNE SOLUTION. Le
    # reechantillonnage est un spline, qui exige des abscisses croissantes, et
    # scipy s'arrete sur « x must be increasing ». Mesure le 2026-10-14: 5 des
    # 60 poses ESPRESSO de tau Ceti ont, au bord bleu de leur ordre le plus
    # bleu, 177 pixels dont la longueur d'onde DECROIT avant de repartir. Ce
    # n'est pas une oscillation numerique (le recul vaut 0.00026 nm contre un
    # pas median de 0.00062) mais un bout de solution qui ne tient pas. On
    # rejette ces pixels, comme on rejette un flux negatif, plutot que
    # d'arreter la campagne ou de les reordonner en silence.
    rising = np.ones_like(good)
    forward = np.diff(wave, axis=1) > 0
    rising[:, 1:] &= forward
    rising[:, :-1] &= forward
    if not rising.all():
        lost = int((good & ~rising).sum())
        if lost:
            log("%d samples sit where the wavelength solution runs backwards"
                " and are dropped: a spline cannot resample them" % lost,
                "warn")
    good &= rising

    if blaze_frac > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            peak = np.nanmax(np.where(np.isfinite(blaze), blaze, -np.inf), axis=1)
        good &= blaze > (blaze_frac * peak)[:, None]

    wanted = inp.get("orders")
    if wanted is None:
        wanted = range(n_orders)
    fraction = good.mean(axis=1)
    orders = [o for o in wanted if 0 <= o < n_orders and fraction[o] >= min_fraction]
    return good, orders


def _filled(values: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Linear interpolation over the rejected samples, so the spline is happy.

    The values put back into the gaps are never used: the good-pixel mask is
    resampled alongside and everything it touches is discarded downstream. They
    exist only so that the spline is not fed a NaN.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    if good.all():
        return out
    if not good.any():
        out[:] = 1.0
        return out
    index = np.arange(values.size)
    out[~good] = np.interp(index[~good], index[good], out[good])
    return out


def _order_sigma(flux, recon, good, snr_order):
    """Per-pixel sigma of ln(flux), from photon statistics, per order.

    The extracted flux in a t.fits has already been divided by the telluric
    transmission, so the photon count behind a pixel is flux * recon, not flux.
    Using the corrected flux would understate the noise exactly where the
    correction was largest. Since sigma(f/T)/(f/T) = sigma(f)/f, working from
    the pre-correction counts gives the right *relative* error either way:

        sigma_ln = 1 / sqrt(flux * recon)

    That expression is already close to correct in absolute terms (it agrees
    with APERO's own EXTSN to about 5% at order centre), but it is rescaled per
    order so its median over the central half of the order equals 1 / EXTSN.
    That folds in read noise and the extraction weighting without needing a
    detector model.
    """
    counts = np.where(good, flux * recon, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        shape = 1.0 / np.sqrt(counts)
    if not np.any(np.isfinite(shape)):
        return np.full(flux.shape, np.nan)
    if np.isfinite(snr_order) and snr_order > 0:
        n_pixels = flux.size
        centre = slice(n_pixels // 4, 3 * n_pixels // 4)
        with warnings.catch_warnings():
            # an order can be entirely masked over its central half
            warnings.simplefilter("ignore", RuntimeWarning)
            reference = np.nanmedian(shape[centre])
        if np.isfinite(reference) and reference > 0:
            shape = shape * ((1.0 / snr_order) / reference)
    return shape


def _overlap_blaze(w_shift, blaze_order, ok, grid, start, stop,
                   flux_order=None):
    """Ce que cet ordre porte comme signal sur la zone [start, stop].

    Sert a arbitrer un recouvrement entre deux ordres de meme parite: celui
    qui porte le plus de signal y est au milieu de son ordre, l'autre y est
    sur son bord et son flux y est bruite. Le perdant est mis a NaN.

    LE BLAZE D'ABORD, LE FLUX QUAND IL NE DIT RIEN. Sur un t.fits d'APERO le
    blaze decrit la reponse de l'ordre et repond seul. Sur un S2D de l'ESO il
    vaut UN partout, parce que la classe de LBL ne deblaze pas: il ne
    departage alors rien, et c'est le flux median qui le fait. Mesure le
    2026-10-14 sur un S2D d'ESPRESSO, zone commune aux ordres 0 et 2: 16036
    contre 11749, soit le bord rouge de l'ordre 0 contre le bord bleu de
    l'ordre 2, et le premier gagne. Comparer des flux ne vaut qu'entre deux
    ordres de la MEME pose au MEME endroit, ce qui est exactement le cas ici.
    """
    if stop < start:
        return None
    lo, hi = grid[start], grid[min(stop, grid.size - 1)]
    inside = ok & (w_shift >= lo) & (w_shift <= hi)
    if not inside.any():
        return None
    values = np.asarray(blaze_order, dtype=float)[inside]
    values = values[np.isfinite(values)]
    if values.size and not np.allclose(values, values[0]):
        return float(np.median(values))
    if flux_order is None:
        return float(np.median(values)) if values.size else None
    values = np.asarray(flux_order, dtype=float)[inside]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def resample_exposure(payload: dict, grid: np.ndarray, config: dict):
    """Collapse one exposure's orders onto two rows of the destination grid.

    Row 0 holds the even orders, row 1 the odd ones. Because orders n and n+2
    never overlap, each row is a set of disjoint segments and no sample is ever
    written twice; because orders n and n+1 do overlap, the two rows duplicate
    each other over a few nm at every order boundary. That duplication is the
    point: the two rows sample the same wavelengths at different places on the
    detector, so a cosmic ray, a bad pixel or a blaze-edge artefact shows up in
    one of them and not the other. Downstream they are treated as two
    independent measurements of the same exposure.

    Returns (values, sigma, trans, good, n_orders): the first four are
    (2, n_pixels) with the values already high-passed *on the destination
    grid*, and n_orders counts the orders that made it in.
    """
    reg = config["registration"]
    hip = config["highpass"]
    observer_frame = reg["frame"] == "observer"

    berv = 0.0 if observer_frame else float(payload["meta"]["berv"])
    target = 0.0 if observer_frame else float(reg["target_berv"])

    n_pixels = grid.size
    values = np.zeros((2, n_pixels), dtype=np.float64)
    sigma = np.full((2, n_pixels), np.nan)
    trans = np.zeros((2, n_pixels), dtype=np.float64)
    good = np.zeros((2, n_pixels), dtype=bool)

    order_good, orders = order_quality(payload, config)
    flux, wave = payload["flux"], payload["wave"]
    blaze, recon = payload["blaze"], payload["recon"]

    shift = float(doppler(berv) / doppler(target))
    last_stop = {0: -1, 1: -1}
    #: le dernier ordre pose de chaque parite, garde pour lui redemander ce
    #: qu'il porte sur la zone que le suivant va recouvrir: (longueurs d'onde
    #: decalees, masque, numero d'ordre)
    last_order = {0: None, 1: None}
    n_written = 0

    edge = int(reg.get("edge_pixels", 1))
    #: les bornes des segments ecrits, quand le passe-haut est differe: la
    #: seconde passe doit filtrer segment par segment, jamais a travers le
    #: trou qui separe deux ordres de meme parite
    segments = []
    for order in orders:
        # the last valid pixel beside every gap goes before the spline sees the
        # order: a cubic spline bends toward the straight line the gap is filled
        # with, and beside an OH core that is sky in a sample that reads as valid
        ok = prep.erode_edges(order_good[order], edge)
        if not ok.any():
            continue
        w_obs = wave[order]
        w_shift = w_obs * shift
        low, high = w_shift[ok].min(), w_shift[ok].max()
        start = int(np.searchsorted(grid, low, side="left"))
        stop = int(np.searchsorted(grid, high, side="right"))
        parity = order % 2
        if stop <= start:
            continue
        if start <= last_stop[parity]:
            # DEUX ORDRES DE MEME PARITE SE RECOUVRENT. On garde CELUI DONT
            # LE BLAZE EST LE PLUS HAUT, l'autre etant mis a NaN (poids nul):
            # dans la zone commune l'un finit sur son bord, la ou le blaze
            # s'effondre et le flux est bruite, et l'autre y debute.
            #
            # Mesure le 2026-10-14 sur SPIRou: les longueurs d'onde BRUTES de
            # 14 paires se recouvrent jusqu'a 5.4 nm (ordres 0 a 13), mais une
            # fois les bords eroces et les pixels de mauvaise qualite ecartes
            # (order_quality, erode_edges) il reste 7 nm d'ECART entre l'ordre
            # 0 (964.6-982.2 nm valides) et l'ordre 2 (989.3-1007.7): le
            # recouvrement brut tombe entierement dans ce qui est deja rejete.
            # Cette branche ne se declenche donc pas sur SPIRou ni sur HARPS
            # (recouvrement nul); elle est la pour les geometries a venir.
            overlap_hi = min(stop - 1, last_stop[parity])
            # LES DEUX MESURES SUR LA MEME ZONE, celle du recouvrement. Un
            # ordre moyenne sur toute sa largeur contre un autre pris sur son
            # bord ne se comparent pas: le premier gagnerait toujours. On
            # redemande donc au precedent ce qu'il porte ICI, avec ses
            # longueurs d'onde et son masque a lui (2026-10-14).
            here = _overlap_blaze(w_shift, blaze[order], ok, grid,
                                  start, overlap_hi, flux[order])
            before = None
            previous = last_order.get(parity)
            if previous is not None:
                p_shift, p_ok, p_order = previous
                before = _overlap_blaze(p_shift, blaze[p_order], p_ok, grid,
                                        start, overlap_hi, flux[p_order])
            if before is not None and here is not None and here < before:
                # l'ordre deja pose est le meilleur: celui-ci cede la zone
                start = last_stop[parity] + 1
                if stop <= start:
                    continue
            elif before is not None and here is not None:
                # celui-ci est le meilleur: on efface ce que l'autre a laisse
                lo = max(0, start)
                good[parity][lo:overlap_hi + 1] = False
                sigma[parity][lo:overlap_hi + 1] = np.nan
            log("les ordres de meme parite se recouvrent vers %.2f nm sur %d"
                " echantillons: on garde le blaze le plus haut"
                % (grid[start], overlap_hi - start + 1), "warn")
        last_stop[parity] = stop - 1
        # de quoi redemander a cet ordre ce qu'il porte sur une zone qu'on ne
        # connait pas encore, celle de son recouvrement avec le suivant
        last_order[parity] = (w_shift, ok, order)
        segment = grid[start:stop]

        # blaze out before splining: the raw extracted flux swings by a factor
        # of three across an order, and a cubic spline reproduces a slowly
        # varying function far better than a steeply curved one. The high-pass
        # would remove the blaze anyway; doing it first only makes the
        # interpolation, and the overlap between the two parities, cleaner.
        deblazed = _filled(flux[order] / blaze[order], ok)
        seg_values, seg_good = prep.register(
            w_obs, deblazed, ok, berv, target, segment,
            spline_order=reg["spline_order"], mask_threshold=reg["mask_threshold"],
        )

        sig = _order_sigma(flux[order], recon[order], ok, payload["snr"][order])
        seg_sigma, _ = prep.register(
            w_obs, _filled(sig, ok & np.isfinite(sig)), ok, berv, target, segment,
            spline_order=1, mask_threshold=reg["mask_threshold"],
        )

        seg_trans, trans_good = prep.register(
            w_obs, _filled(recon[order], ok), ok, berv, target, segment,
            spline_order=reg["spline_order"], mask_threshold=reg["mask_threshold"],
        )
        seg_trans = np.where(trans_good, np.clip(seg_trans, 0.0, 1.0), 0.0)

        # the high-pass runs here, on the destination grid, one order at a time.
        # On the grid the pixel width is a fixed 0.5 km/s, so the filter has the
        # same velocity width everywhere -- which it does not on native pixels.
        # And it is applied per order, never across the gap to the next
        # same-parity order, so the filter never interpolates over a hole.
        if hip["method"] == "none":
            with np.errstate(invalid="ignore", divide="ignore"):
                seg_values = np.where(seg_good & (seg_values > 0),
                                      np.log(np.abs(seg_values)), 0.0)
            hp_good = seg_good & np.isfinite(seg_values)
        elif hip.get("against_template"):
            # LE PASSE-HAUT ATTEND LE TEMPLATE. On ne garde ici que le
            # logarithme du flux; le filtre passera dans cube.py, sur le
            # residu a un template median, une fois toutes les poses lues.
            # Voir preprocess.highpass_against_template pour pourquoi.
            with np.errstate(invalid="ignore", divide="ignore"):
                seg_values = np.where(seg_good & (seg_values > 0),
                                      np.log(np.abs(seg_values)), 0.0)
            hp_good = seg_good & np.isfinite(seg_values)
            segments.append((parity, start, stop))
        else:
            seg_values, hp_good = prep.highpass(
                np.where(seg_good, seg_values, np.nan),
                hip["window"], hip["polyorder"], hip["mode"],
            )
            hp_good &= seg_good
        keep = hp_good & np.isfinite(seg_sigma) & (seg_sigma > 0)

        # A sample that survived the cuts but whose neighbourhood did not is
        # not a measurement: its high pass came from a filter window mostly
        # filled by interpolation. Applied per order segment, so the rule never
        # reaches across the gap to the next order, and before the noise below
        # is measured, so that estimate sees the final mask.
        iso = config["quality"].get("isolated_window")
        if iso:
            keep = prep.drop_isolated(keep, int(iso))

        # An empirical noise floor, measured from the sample-to-sample scatter
        # of this very segment, and combined with the photon estimate by taking
        # the LARGER of the two. Never the smaller: a formal error below the
        # scatter you can actually see in the data is not a better measurement,
        # it is a weight that will pull the fit into whatever is making that
        # scatter. See preprocess.empirical_sigma for what this costs and why
        # the red end of SPIRou made it necessary.
        box = config["weights"].get("empirical_noise_box")
        if box:
            emp = prep.empirical_sigma(seg_values, keep, int(box))
            seg_sigma = np.where(np.isfinite(emp), np.maximum(seg_sigma, emp),
                                 seg_sigma)

        values[parity, start:stop] = np.where(keep, seg_values, 0.0)
        sigma[parity, start:stop] = np.where(keep, seg_sigma, np.nan)
        trans[parity, start:stop] = np.where(keep, seg_trans, 0.0)
        good[parity, start:stop] = keep
        n_written += 1

    # LES SEGMENTS SONT TOUJOURS RENDUS, meme vides: un retour dont le nombre
    # d'elements depend d'un reglage se desynchronise au premier appelant
    # qu'on oublie de mettre a jour.
    return values, sigma, trans, good, n_written, segments


def raw_block(payload: dict, grid: np.ndarray, a0: int, b0: int, config: dict):
    """ln(flux) with no high-pass, on grid[a0:b0], one row per order parity.

    The figures draw the spectrum as it sits in the file, which the cube does
    not hold, the cube being high-passed. This is the cube's own resampling,
    onto one contiguous block of the same grid, with the high-pass switched
    off, and it is exact inside the block: the spline is built on the native
    pixels and only evaluated at the block's wavelengths, and the one rule that
    looks at neighbours, drop_isolated, treats the ends of an array as bounded
    rather than as gaps. At most its few samples next to the block's ends can
    differ from a full-grid resampling, inside a margin that exists for the
    shift anyway. float32, NaN where nothing was measured.
    """
    cfg = dict(config, highpass=dict(config["highpass"], method="none"))
    values, _sigma, _trans, good, _n, _seg = resample_exposure(
        payload, grid[a0:b0], cfg)
    return np.where(good, values, np.nan).astype(np.float32)
