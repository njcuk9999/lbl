"""The series of a run, read from its .rdb, and what is needed to plot it.

The one place where the report learns the .rdb format: the names of the
columns it uses (time, velocity, error, telluric exponents, BERV, airmass),
how to read the file into a dictionary of columns, and how to find the
spectrum behind a row. Any other code that reads an LBL .rdb can import it
rather than restate the format; it depends only on numpy (and on
lbl.core.simbad for the systemic velocity, imported when asked for).
"""
import functools
import os

import numpy as np

#: the columns that LBL always writes, and their role
TIME = 'rjd'
VELOCITY = 'vrad'
ERROR = 'svrad'
#: the telluric exponents, when the correction has run
WATER = 'TLPEH2O'
DRY = 'TLPEOTR'
#: the name of the original file, to find the spectrum again
FILENAME = 'FILENAME'
#: the header quantities that LBL copies over and that the report uses. A
#: missing key is simply skipped: not every instrument writes them all, and
#: a truncated report is better than a report that refuses to come out.
#: Several names depending on the instrument, the first one present counts:
#: APERO (SPIRou, NIRPS) writes BERV, the HARPS DRS HIERARCH ESO DRS BERV,
#: the ESPRESSO one HIERARCH ESO QC BERV, the HARPS-N and SOPHIE DRS their
#: TNG and OHP equivalents: with a single name, the BERV page would be
#: missing for every instrument but one.
BERV = ('BERV', 'HIERARCH ESO QC BERV', 'HIERARCH ESO DRS BERV',
        'HIERARCH TNG QC BERV', 'HIERARCH TNG DRS BERV',
        'HIERARCH OHP DRS BERV')
AIRMASS = ('AIRMASS', 'HIERARCH ESO TEL AIRM START',
           'HIERARCH TNG TEL AIRM START')


@functools.lru_cache(maxsize=64)
def systemic_velocity(star):
    """SIMBAD's systemic velocity of a star [km/s] (lbl.core.simbad: any
    spelling, LBL's suffixes off), or None when SIMBAD has none, does not
    know the star, or cannot be asked. With it, the velocity series is
    coloured by the star's velocity in the telluric frame rather than by
    the BERV (figures.page_sequence)."""
    if not star:
        return None
    from lbl.core import simbad
    try:
        found = simbad.resolve(star)
    except OSError:
        return None
    if found is None or found.get('rv') is None:
        return None
    return float(found['rv'])


def column(table, name, default=None):
    """A column of the .rdb as floats, or `default` if it is not there.
    `name` can be a sequence of names: the first one present."""
    if isinstance(name, (tuple, list)):
        name = next((one for one in name if one in table), None)
        if name is None:
            return default
    if name not in table:
        return default
    values = np.asarray(table[name])
    try:
        return values.astype(float)
    except (TypeError, ValueError):
        return default


def read_rdb(path):
    """An LBL .rdb, as a dictionary of columns.

    The numeric columns become arrays of floats, the others arrays of
    strings. The rdb format puts a line of dashes under the header, which is
    skipped.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        lines = [line.rstrip('\n') for line in handle if line.strip()]
    if len(lines) < 3:
        raise ValueError('%s holds no rows' % path)
    names = lines[0].split('\t')
    rows = [line.split('\t') for line in lines[2:]]
    out = dict()
    for index, name in enumerate(names):
        values = [row[index] if index < len(row) else '' for row in rows]
        try:
            out[name] = np.array(values, dtype=float)
        except ValueError:
            out[name] = np.array(values, dtype=str)
    return out


def has_telluric(table):
    """Has the telluric correction run on this series?"""
    return WATER in table and np.isfinite(np.asarray(
        table[WATER], dtype=float)).any()


def extremes(values, names=('best', 'median', 'worst')):
    """The indices of the smallest, the median and the largest.

    This is what the report shows of the telluric correction: one exposure
    from each end of the distribution and one from the middle, rather than
    a random exposure that would not tell whether the hard case gets
    through.
    """
    values = np.asarray(values, dtype=float)
    good = np.where(np.isfinite(values))[0]
    if good.size == 0:
        return dict()
    order = good[np.argsort(values[good])]
    middle = order[order.size // 2]
    return {names[0]: int(order[0]), names[1]: int(middle),
            names[2]: int(order[-1])}


def spectrum_path(table, index, science_dir):
    """The spectrum behind a row of the .rdb, or None.

    THE NAME IN THE .rdb IS THAT OF THE LBL FILE, not that of the spectrum:
    LBL appends to it the object, the comparison and the _lbl suffix, for
    example

        r.HARPS.2003-08-08T07:11:56.372_S2D_0-2_A_HD10700_TC_..._lbl.fits

    for a spectrum named r.HARPS.2003-08-08T07:11:56.372_S2D_A.fits. The
    spectrum is therefore found by its TIMESTAMP, which is the only reliable
    part of the name, rather than by guessing which suffix to remove.
    """
    import glob

    if FILENAME not in table or not science_dir:
        return None
    name = str(np.asarray(table[FILENAME])[index])
    if not name or name == 'nan':
        return None
    base = os.path.basename(name)
    direct = os.path.join(science_dir, base)
    if os.path.exists(direct):
        return direct
    # the timestamp: what follows the instrument name, up to the first '_'
    stamp = None
    for piece in base.split('_'):
        if 'T' in piece and piece.count('-') >= 2:
            stamp = piece
            break
    if stamp is None:
        return None
    hits = sorted(glob.glob(os.path.join(science_dir, '*%s*' % stamp)))
    return hits[0] if hits else None
