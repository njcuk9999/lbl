"""Reading NIRPS s1d_v spectra and the header quantities we care about."""

from __future__ import annotations

import glob
import os
import time

import numpy as np
from astropy.io import fits

from .grids import doppler
from .logger import log


# Header keys pulled into the metadata table. Missing keys become NaN / ''.
META_KEYS = {
    "bjd": "BJD",
    "mjdmid": "MJDMID",
    "berv": "BERV",
    "bervmax": "BERVMAX",
    "exptime": "EXPTIME",
    "airmass_start": "ESO TEL AIRM START",
    "airmass_end": "ESO TEL AIRM END",
    "seeing": "ESO TEL AMBI FWHM START",
    "humidity": "ESO TEL AMBI RHUM",
    "sun_elevation": "DRSSUNEL",
    "drs_mode": "DRSMODE",
    "dprtype": "DPRTYPE",
}

_STRING_KEYS = {"drs_mode", "dprtype"}


def header_float(header, key, default=np.nan) -> float:
    """Read a header key as a float, tolerating APERO's string 'NaN'.

    Some frames have EXTSNxxx (and other numeric keys) written as the *string*
    'NaN' rather than a float. Left alone, that turns an array of SNRs into a
    string array and every downstream isfinite() raises.
    """
    value = header.get(key, default)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float(default) if default is not None else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def robust_open(path: str, attempts: int = 3, pause: float = 2.0):
    """fits.open, retried when the first read comes back malformed.

    On a cloud-synced directory the file may be a placeholder that the provider
    materialises on first access. Under disk pressure that materialisation
    sometimes hands back a partially written file rather than blocking, and
    astropy reports it as "No SIMPLE card found" on a file that is perfectly
    valid a second later. Observed on this dataset with OneDrive Files
    On-Demand on a volume at 99% capacity, on both s1d and t.fits.

    A retry is not a fix for that; moving the data off the cloud folder and off
    the full disk is. It is here so that one flaky file does not cost a
    four-minute read of 782 exposures, and so that the log says plainly when it
    is happening.
    """
    last = None
    for attempt in range(int(attempts)):
        try:
            hdulist = fits.open(path, memmap=False)
            if attempt:
                log("%s read on attempt %d; the storage is not returning files"
                    " reliably" % (os.path.basename(path), attempt + 1), "warn")
            return hdulist
        except Exception as exc:                        # noqa: BLE001
            last = exc
            if attempt + 1 < int(attempts):
                time.sleep(float(pause))
    raise last


def find_spectra(directory: str, pattern: str, max_files=None) -> list[str]:
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if max_files is not None:
        files = files[: int(max_files)]
    return files


_POLY_WARNED = False


def band_snr(header, wave_min: float, wave_max: float) -> float:
    """Median EXTSN over the extracted orders that overlap the requested band.

    RETURNS NaN ON THE s1d PATH, deliberately. Deciding which order falls in the
    band used to be done from the header wavelength polynomials, WAVEORDN /
    WAVEDEGN / WAVE0000..., taking each order's constant term as its starting
    wavelength. Those polynomials are a leftover from earlier pipeline versions
    and are not to be used anywhere, ever: the wavelength solution is read from
    the t.fits extension and from nothing else. An s1d file carries no per-order
    wavelength array, so the mapping cannot be rebuilt without them, and the
    honest answer here is "unknown" rather than a number obtained the forbidden
    way.

    Nothing on the live path reaches this. The t.fits reader has its own
    band_snr in tfits.py, which takes the order centres straight from the WaveA
    array; the s1d reader this belongs to has been retired. It is left in place,
    refusing, rather than removed, so that reviving the s1d path fails loudly
    instead of quietly reintroducing the polynomials.
    """
    global _POLY_WARNED
    if not _POLY_WARNED:
        _POLY_WARNED = True
        log("io.band_snr: the s1d path would need the header wavelength"
              " polynomials, which are forbidden; returning NaN. See the"
              " docstring.")
    return np.nan


def recon_path_for(path: str) -> str | None:
    """Path of the reconstructed-telluric file paired with a tcorr spectrum.

    APERO writes the telluric-corrected spectrum as *_pp_s1d_v_tcorr_A.fits and
    the transmission it divided out as *_pp_s1d_v_recon_A.fits, on the same grid
    and for the same exposure. Returns None if the partner is not on disk.
    """
    candidate = path.replace("_tcorr_", "_recon_")
    if candidate == path or not os.path.exists(candidate):
        return None
    return candidate


def read_transmission(path: str, wave_min: float, wave_max: float,
                      pad_kms: float = 40.0):
    """Telluric transmission from a recon file, cropped like the spectrum.

    The recon file's `flux` column is the transmission itself: ~1 in clean
    stretches, dropping into the absorption bands. It lives in the OBSERVER
    frame, so it has to be carried through exactly the same BERV shift as the
    spectrum before it can be used as a weight in the stellar frame.
    """
    with robust_open(path) as hdulist:
        data = hdulist[1].data
        wave_all = np.asarray(data["wavelength"], dtype=np.float64)
        pad = float(doppler(pad_kms))
        keep = (wave_all > wave_min / pad) & (wave_all < wave_max * pad)
        return np.asarray(data["flux"], dtype=np.float64)[keep]


def read_spectrum(path: str, wave_min: float, wave_max: float, pad_kms: float = 40.0):
    """Read one s1d_v file, cropped to the band plus a velocity margin.

    The margin must cover the BERV shift so the cubic spline never has to
    extrapolate at the edges of the requested domain.

    Returns (wave, flux, s1d_weight, meta) with wave in nm.
    """
    with robust_open(path) as hdulist:
        header = hdulist[0].header
        data = hdulist[1].data
        wave_all = np.asarray(data["wavelength"], dtype=np.float64)
        pad = float(doppler(pad_kms))
        keep = (wave_all > wave_min / pad) & (wave_all < wave_max * pad)
        wave = wave_all[keep]
        flux = np.asarray(data["flux"], dtype=np.float64)[keep]
        s1d_weight = np.asarray(data["weight"], dtype=np.float64)[keep]

        meta = {"filename": os.path.basename(path)}
        for name, key in META_KEYS.items():
            if name in _STRING_KEYS:
                meta[name] = str(header.get(key, "")).strip()
            else:
                meta[name] = header_float(header, key)
        meta["object"] = str(header.get("OBJECT", "")).strip()
        meta["drsobjn"] = str(header.get("DRSOBJN", "")).strip()
        meta["snr_band"] = band_snr(header, wave_min, wave_max)
        meta["airmass"] = 0.5 * (meta["airmass_start"] + meta["airmass_end"])
        with np.errstate(invalid="ignore"):
            finite = np.isfinite(flux)
            meta["median_flux"] = float(np.median(flux[finite])) if finite.any() else np.nan
            meta["nan_fraction"] = float(np.mean(~finite))

    return wave, flux, s1d_weight, meta


def magic_grid(path: str, wave_min: float, wave_max: float) -> np.ndarray:
    """Recycle the NIRPS s1d_v wavelength grid, cropped to the band.

    The s1d_v files are all sampled on the same log-uniform "magic grid": a
    constant 0.5 km/s step running from 965 to 1950 nm. Reusing that grid rather
    than synthesising a fresh one has two virtues. A spectrum with BERV = 0 is
    then resampled onto exactly its own sample positions, so the spline is the
    identity there instead of applying a small systematic smoothing; and the
    output wavelengths stay directly comparable with any other s1d product.
    """
    with fits.open(path, memmap=False) as hdulist:
        wave = np.asarray(hdulist[1].data["wavelength"], dtype=np.float64)
    keep = (wave >= wave_min) & (wave <= wave_max)
    if not np.any(keep):
        raise RuntimeError(
            "requested domain %.1f-%.1f nm lies outside the s1d grid %.1f-%.1f nm"
            % (wave_min, wave_max, wave[0], wave[-1])
        )
    return wave[keep]


def check_common_grid(files: list[str], n_check: int = 5) -> bool:
    """Sanity check that the files really do share one wavelength grid."""
    if len(files) < 2:
        return True
    idx = np.unique(np.linspace(0, len(files) - 1, n_check).astype(int))
    reference = None
    for i in idx:
        with fits.open(files[i], memmap=False) as hdulist:
            wave = np.asarray(hdulist[1].data["wavelength"], dtype=np.float64)
        if reference is None:
            reference = wave
            continue
        if wave.shape != reference.shape or not np.allclose(wave, reference):
            log("input grids differ between files; relying on the spline", "warn")
            return False
    return True
