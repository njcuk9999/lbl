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
information for free. The rule is the usual one: everything
before UT noon belongs to the night that started the previous day.
"""

from __future__ import annotations

import datetime as _dt
import os
import re

import numpy as np
from astropy.io import fits

from .logger import log

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


def night_label(stamp: _dt.datetime) -> str:
    """The night a UT timestamp belongs to, as YYYY-MM-DD.

    Anything before UT noon is credited to the previous day, which is the
    convention every observatory log uses and which keeps a whole night in one
    bucket regardless of the site's longitude.
    """
    date = stamp.date()
    if stamp.hour < 12:
        date = date - _dt.timedelta(days=1)
    return date.isoformat()


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
                bundle=None, orbits=None, velocities=None, berv_of=None):
    """Sort files chronologically and label each with its output group.

    Returns (files, labels). With stack = False every file is its own group, so
    the rest of the pipeline needs no special case. With `bundle`
    (bundle_settings), each night is cut into bundles (split_nights), by
    the star's frame when the star has `orbits` (frame_bervs). The BERV of a
    file is `berv_of(path)`, in km/s: the observer's, from LBL's instrument
    profile (readers.file_observer_berv), never read here from the headers.
    """
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

    labels = [night_label(s) for s in stamps]
    n_nights = len(set(labels))
    log("nightly stacking: %d exposures over %d nights (%.1f per night)"
        % (len(files), n_nights, len(files) / max(n_nights, 1)), "value")
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
