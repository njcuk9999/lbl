"""Grouping exposures into observing nights, and coadding them.

Averaging the exposures of a night before the PCA is the cheapest speed-up
available: an instrument that takes several exposures per visit makes the
cube that many times smaller, and the decomposition -- whose cost is linear
in the number of rows -- shrinks with it.

It is only legitimate because the coadd happens *after* registration. Every
exposure is splined onto the destination grid using its own BERV, which moves
by up to 0.5 km/s (one destination pixel) within a single night as the Earth
turns. Coadding first and registering afterwards would smear the stellar lines
by that much; registering first and coadding afterwards does not.

The night an exposure belongs to is taken from its timestamp, read out of the
filename when it is there. Opening hundreds of headers costs minutes of pure
I/O, as much as reading the pixels; the filename carries the same
information for free. The rule is the one of all of LBL
(lbl.core.astro.night_number): a night runs from local mean noon to the next
local mean noon at the observatory, whose longitude is the site of the
instrument's EARTH_LOCATION (config_longitude). A cube records the longitude
its nights were cut at (LONGITUDE_KEY in its meta.fits), and its cache key
holds it, so that a cube grouped otherwise is never taken for this one.
"""

from __future__ import annotations

import datetime as _dt
import os
import re

import numpy as np
from astropy.io import fits

from lbl.core.astro import night_number, wrap_longitude  # noqa: F401
from .logger import log

#: the keyword of a cube's meta.fits that records the east longitude [deg]
#:    of the observatory its nights were cut at (local mean noon there)
LONGITUDE_KEY = "NIGHTLON"

# <INSTRUMENT>.2023-01-20T08:42:08.941t.fits or
# <INSTRUMENT>_2023-01-20T08_42_08_941_pp_...
_STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})[:_](\d{2})[:_](\d{2})")


def timestamp_from_name(path: str):
    """UT timestamp parsed out of a filename, or None if it is not there."""
    match = _STAMP.search(os.path.basename(path))
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(v) for v in match.groups())
    try:
        return _dt.datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


#: 2000-01-01 12:00 UT, JD 2451545.0: a UT timestamp's Julian date is counted
#:    from there, and the night_number of a date is its Julian day number
_J2000 = _dt.datetime(2000, 1, 1, 12, 0, 0)
_J2000_JD = 2451545


def julian_date(stamp: _dt.datetime) -> float:
    """The Julian date of a UT timestamp (naive, in UT)."""
    return _J2000_JD + (stamp - _J2000).total_seconds() / 86400.0


def night_label(stamp: _dt.datetime, longitude) -> str:
    """The night a UT timestamp belongs to, as YYYY-MM-DD, the LOCAL date it
    started on at the observatory at east longitude `longitude` [deg].

    It is night_number on the stamp's Julian date, so the boundary is the
    same to the second: anything before local mean noon is credited to the
    night that started the local day before.
    """
    night = int(night_number(julian_date(stamp), longitude))
    return (_J2000.date() + _dt.timedelta(days=night - _J2000_JD)).isoformat()


def config_longitude(config):
    """The east longitude [deg] of the observatory of a PCA2D configuration's
    instrument (lbl.instrument and lbl.data_source, else input.instrument),
    the site of its EARTH_LOCATION (lbl.core.astro.observatory_longitude);
    None when the configuration names no instrument. A plain dict read from
    a resolved_config.yaml is enough."""
    if not config:
        return None
    block = config.get("lbl") or {}
    name = block.get("instrument") or (config.get("input") or {}).get(
        "instrument")
    if name in (None, "", "None"):
        return None
    from lbl.core.astro import observatory_longitude
    source = block.get("data_source")
    return observatory_longitude(str(name), None if source in (None, "", "None")
                                 else str(source))


def cube_longitude(cube):
    """The east longitude [deg] a cube's nights were cut at, as its meta.fits
    records it (LONGITUDE_KEY), or None for a cube that records none (built
    before nights were cut at local noon, or by hand). `cube` is the cube's
    folder or its meta table."""
    meta = cube
    if isinstance(cube, str):
        path = os.path.join(cube, "meta.fits")
        if not os.path.exists(path):
            return None
        value = fits.getheader(path, 1).get(LONGITUDE_KEY)
    else:
        value = (getattr(meta, "meta", None) or {}).get(LONGITUDE_KEY)
    return None if value is None else float(value)


def record_longitude(meta, config):
    """Write the configuration's longitude into a cube's meta table
    (LONGITUDE_KEY), which _save_cache writes with it: what every later
    stage reads the cube's nights by when no configuration is given. Nothing
    when the configuration names no instrument."""
    longitude = config_longitude(config)
    if longitude is not None:
        meta.meta[LONGITUDE_KEY] = float(longitude)
    return meta


def run_longitude(config=None, cube=None, need=None):
    """The longitude a stage after the cube groups exposures into nights by:
    the configuration's instrument's (config_longitude), else what the cube
    records (cube_longitude). Stops when the two disagree, since the cube's
    nights would then not be the ones regrouped here.

    None when neither says, unless `need` says what needs it: then it stops,
    and says so. Never noon UT by default.
    """
    longitude = config_longitude(config)
    recorded = cube_longitude(cube) if cube is not None else None
    if longitude is not None and recorded is not None and abs(
            wrap_longitude(longitude - recorded)) > 1e-6:
        raise SystemExit(
            "the cube %s was cut into nights at local noon at east longitude"
            " %.4f, and the configuration's instrument is at %.4f: they are"
            " not the same observatory" % (cube, recorded, longitude))
    if longitude is not None and recorded is None and isinstance(cube, str) \
            and _stacked(cube):
        log("the cube %s records no night longitude: it was coadded into"
            " nights before LBL cut them at local noon, and its nights may"
            " not be the ones regrouped here; rebuild it" % cube, "warn")
    if longitude is None:
        longitude = recorded
    if longitude is None and need:
        raise SystemExit(
            "%s groups the exposures into nights, cut at local noon at the"
            " observatory, and nothing names it: give the configuration"
            " (--config) whose lbl.instrument names the instrument, or a cube"
            " that records its longitude (%s in its meta.fits)"
            % (need, LONGITUDE_KEY))
    return longitude


def _stacked(cube: str) -> bool:
    """Whether a cube's rows coadd several exposures (n_exposures)."""
    path = os.path.join(cube, "meta.fits")
    if not os.path.exists(path):
        return False
    from astropy.table import Table
    meta = Table.read(path)
    return ("n_exposures" in meta.colnames and len(meta)
            and int(max(meta["n_exposures"])) > 1)


def _stamp_from_header(path: str, extension: int) -> _dt.datetime | None:
    try:
        header = fits.getheader(path, extension)
    except Exception:                                   # noqa: BLE001
        return None
    value = header.get("DATE-OBS") or header.get("DATE_OBS")
    if value is None:
        return None
    try:
        return _dt.datetime.fromisoformat(str(value).strip()[:19])
    except ValueError:
        return None


def header_bjd(path: str):
    """The exposure's BJD, from its headers, or None."""
    try:
        with fits.open(path) as hdulist:
            for hdu in hdulist[:2]:
                value = hdu.header.get("BJD")
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
    except Exception:                                   # noqa: BLE001
        return None
    return None


def frame_bervs(files, bervs, orbits, velocities=None):
    """The BERVs as the star's frame sees them: composed with the star's
    velocity, a free fit's (registration.velocities, by file name) or that
    of its companions (registration.orbits), as readers.star_frame does for
    the cube. A bundle is a span of the star's frame, so with a planet it is
    cut by that and not by the BERV alone."""
    velocities = velocities or {}
    if not orbits and not velocities:
        return bervs
    from lbl.science import planets
    from .grids import C_KMS
    out = []
    for path, berv in zip(files, bervs):
        name = os.path.basename(str(path))
        if berv is None:
            out.append(berv)
            continue
        if name in velocities:
            vstar = float(velocities[name])
        else:
            bjd = header_bjd(path) if orbits else None
            if bjd is None:
                out.append(berv)
                continue
            vstar = float(planets.star_velocity([bjd], orbits)[0]) / 1000.0
        out.append((berv - vstar) / (1.0 - berv * vstar / C_KMS ** 2))
    return out


def bundle_settings(inp: dict):
    """input.bundle_minutes, bundle_berv and bundle_max, or None when none is
    set: the limits a night is cut by, within a night, into bundles."""
    out = {}
    for key, name in (("bundle_minutes", "minutes"), ("bundle_berv", "berv"),
                      ("bundle_max", "max")):
        value = (inp or {}).get(key)
        if value in (None, "", False):
            continue
        value = float(value)
        if value <= 0:
            raise ValueError("input.%s must be positive, not %r" % (key, value))
        out[name] = value
    return out or None


def split_nights(stamps, labels, bundle, bervs=None):
    """Cut each night into bundles, in time order. Returns the new labels.

    A bundle opens at an exposure and takes the next ones of the same night
    while none of the limits is passed: the time from its first exposure's
    start (bundle_minutes), the range of its BERVs (bundle_berv, km/s), its
    number of exposures (bundle_max). A night that fits in one bundle keeps
    its plain label; one that does not gets <night>#00, <night>#01...

    WHY. Coadding a night is what lets the few exposures of a visit be
    fitted as one row. A transit sequence is dozens of them over hours, and
    its night's mean is not any one sky: the airmass, so the water column,
    and the airglow change within it. Bundles keep what changes and still
    coadd what does not.
    """
    if not bundle:
        return list(labels)
    minutes = bundle.get("minutes")
    berv_span = bundle.get("berv")
    most = bundle.get("max")
    out = []
    index = {}
    open_ = None                       # (night, start stamp, bervs, count)
    for i, (stamp, night) in enumerate(zip(stamps, labels)):
        berv = bervs[i] if bervs is not None else None
        fits_in = open_ is not None and open_[0] == night
        if fits_in and minutes is not None:
            fits_in = (stamp - open_[1]).total_seconds() <= 60.0 * minutes
        if fits_in and berv_span is not None:
            values = open_[2] + [berv]
            fits_in = max(values) - min(values) <= berv_span
        if fits_in and most is not None:
            fits_in = open_[3] + 1 <= most
        if fits_in:
            open_ = (night, open_[1], open_[2] + [berv], open_[3] + 1)
        else:
            index[night] = index.get(night, -1) + 1
            open_ = (night, stamp, [berv], 1)
        out.append((night, index[night]))
    split = {night for night, k in out if k > 0}
    return [("%s#%02d" % (night, k)) if night in split else night
            for night, k in out]


def row_unit(source) -> str:
    """What a coadded row of a cube is, in words: "bundles" when its nights
    were cut (their labels carry a #), "nights" otherwise. `source` is the
    cube's meta table, or its folder."""
    meta = source
    if isinstance(source, str):
        path = os.path.join(source, "meta.fits")
        if not os.path.exists(path):
            return "nights"
        from astropy.table import Table
        meta = Table.read(path)
    names = getattr(meta, "colnames", None) or []
    if meta is None or "night" not in names:
        return "nights"
    return "bundles" if any("#" in str(v) for v in meta["night"]) else "nights"


def group_files(files: list[str], stack: bool, extension: int = 0,
                bundle=None, orbits=None, velocities=None, berv_of=None,
                longitude=None):
    """Sort files chronologically and label each with its output group.

    Returns (files, labels). With stack = False every file is its own group, so
    the rest of the pipeline needs no special case. With stack = True the
    group is the night (night_label), cut at local noon at east longitude
    `longitude` [deg] (config_longitude), which stacking then requires. With
    `bundle` (bundle_settings), each night is cut into bundles
    (split_nights), by the star's frame when the star has `orbits`
    (frame_bervs). The BERV of a file is `berv_of(path)`, in km/s: the
    observer's, from LBL's instrument profile (readers.file_observer_berv),
    never read here from the headers.
    """
    if stack and longitude is None:
        raise RuntimeError(
            "nightly stacking cuts the nights at local noon at the"
            " observatory, and nothing names it: lbl.instrument in the"
            " configuration names the instrument, whose EARTH_LOCATION gives"
            " the site")
    stamps = [timestamp_from_name(path) for path in files]
    missing = [i for i, s in enumerate(stamps) if s is None]
    if missing:
        log("%d filenames carry no timestamp; reading their headers instead"
            % len(missing), "warn")
        for i in missing:
            stamps[i] = _stamp_from_header(files[i], extension)

    known = [i for i, s in enumerate(stamps) if s is not None]
    if len(known) == len(files):
        order = np.argsort([s.isoformat() for s in stamps])
        files = [files[i] for i in order]
        stamps = [stamps[i] for i in order]
    elif stack:
        raise RuntimeError(
            "cannot group %d file(s) into nights: no timestamp in the filename "
            "and no DATE-OBS in the header" % (len(files) - len(known))
        )

    if not stack:
        return files, ["%06d" % i for i in range(len(files))]

    labels = [night_label(s, longitude) for s in stamps]
    n_nights = len(set(labels))
    log("nightly stacking: %d exposures over %d nights (%.1f per night),"
        " cut at local noon at east longitude %.2f"
        % (len(files), n_nights, len(files) / max(n_nights, 1),
           wrap_longitude(longitude)), "value")
    if bundle:
        bervs = None
        if bundle.get("berv") is not None:
            if berv_of is None:
                raise RuntimeError("input.bundle_berv: no way to have the"
                                   " BERV of the files (LBL's instrument"
                                   " profile)")
            bervs = [berv_of(path) for path in files]
            bad = [b is None or not np.isfinite(b) for b in bervs]
            if any(bad):
                raise RuntimeError(
                    "input.bundle_berv: LBL's instrument profile gives no BERV"
                    " for %d file(s), first %s" % (sum(bad),
                                                   files[bad.index(True)]))
            bervs = frame_bervs(files, bervs, orbits, velocities)
        labels = split_nights(stamps, labels, bundle, bervs)
        n_groups = len(set(labels))
        log("bundles within the nights (%s): %d exposures in %d bundles over"
            " %d nights (%.1f exposures per bundle)" % (
                ", ".join("%s %g" % (k, v) for k, v in sorted(bundle.items())),
                len(files), n_groups, n_nights, len(files) / max(n_groups, 1)),
            "value")
    return files, labels
