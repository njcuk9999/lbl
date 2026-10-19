"""Grouping exposures into observing nights, and coadding them.

Averaging the exposures of a night before the PCA is the cheapest speed-up
available: NIRPS visits a target three or four times in a row, so the cube
shrinks by that factor and the decomposition -- whose cost is linear in the
number of rows -- shrinks with it.

It is only legitimate because the coadd happens *after* registration. Every
exposure is splined onto the destination grid using its own BERV, which moves
by up to 0.5 km/s (one destination pixel) within a single night as the Earth
turns. Coadding first and registering afterwards would smear the stellar lines
by that much; registering first and coadding afterwards does not.

The night an exposure belongs to is taken from its timestamp, read out of the
filename when it is there. Opening 782 t.fits headers on this dataset costs
four minutes of pure I/O, which is as much as reading the pixels; the filename
carries the same information for free. The rule is the usual one: everything
before UT noon belongs to the night that started the previous day.
"""

from __future__ import annotations

import datetime as _dt
import os
import re

import numpy as np
from astropy.io import fits

from .logger import log

# NIRPS.2023-01-20T08:42:08.941t.fits  /  NIRPS_2023-01-20T08_42_08_941_pp_...
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


def group_files(files: list[str], stack: bool, extension: int = 0):
    """Sort files chronologically and label each with its output group.

    Returns (files, labels). With stack = False every file is its own group, so
    the rest of the pipeline needs no special case.
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
    return files, labels
