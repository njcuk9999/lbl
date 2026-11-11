"""The star of a table of the Data Explorer, summed up above its time series:
what the star-o-meter (https://eartigau.github.io/starometer/) shows of a
star, asked by the window's server, with its published planets.

- Who it is, from SIMBAD (lbl.core.simbad: the CDS or its mirrors): main
  identifier, other names, object and spectral type, position, parallax,
  proper motion, systemic velocity, V G J H K, the medians of its Teff, log
  g and [Fe/H] measurements (mesFe_H) and of its v sin i (mesRot).
- Gaia DR3, by the number SIMBAD gives it: BP, RP, astrometry, from ESA's
  archive (with the GSP-Phot and FLAME parameters), else from GAVO's copy
  (without them).
- AllWISE, W1 to W4: the nearest source to the star taken back to the
  AllWISE epoch, from GAVO, else from IRSA.
- Its parameters as the star-o-meter derives them (derive). An M dwarf:
  Teff, R and M from Mann et al. (2015; erratum 2016, ApJ 819, 87), M from
  Delfosse et al. (2000), [Fe/H] from Duque-Arribas et al. (2023, ApJ 944,
  88, Eq. 4), each within the range it was calibrated on. An AFGK dwarf:
  Gaia's GSP-Phot and FLAME, else the dwarf sequence of Pecaut & Mamajek
  (2013). Nothing for a giant, a white dwarf or an O or B star.
- Its motion: tangential and space velocity, heliocentric galactic UVW.
- The days of the year its lines fall on the telluric lines: when its
  velocity seen from the Earth's centre, RV minus the BERV (composed
  relativistically), is within TELLURIC_KMS.
- Its published planets and parameters: the NASA Exoplanet Archive's
  composite table (pscomppars), matched by its Gaia DR3, TIC, HD and HIP
  numbers and its names; when the archive cannot be reached, the planets of
  LBL's catalog (explore.planets_of: period and K only).

Every service is asked with a short timeout, the next one taking over; a
service that failed is left alone for DOWN_FOR seconds. The answer is kept
in DATA_DIR/gui/star_info.json for KEEP_DAYS, or until asked again.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import math
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

#: where the answers are kept, in the data folder
CACHE = os.path.join("gui", "star_info.json")
#: days an answer is kept before it is asked again (planets get published)
KEEP_DAYS = 30
#: the form of an answer: one kept in another form is asked again
VERSION = 3
#: seconds a service may take
TIMEOUT = 20
#: seconds a service that failed is left alone
DOWN_FOR = 300
USER_AGENT = "LBL (github.com/njcuk9999/lbl)"

#: Gaia DR3: ESA's archive (with the astrophysical parameters), then GAVO's
#:   copy of the main table (dc.g-vo.org, gaia.dr3lite)
ESA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
GAVO_TAP = "https://dc.g-vo.org/tap/sync"
#: AllWISE: GAVO (wise.main), then IRSA (allwise_p3as_psd)
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
#: the NASA Exoplanet Archive's TAP service
NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
#: the epochs of the positions [Julian year]: Gaia DR3, AllWISE (the middle
#:   of its survey), SIMBAD's (J2000 when it gives no other)
GAIA_EPOCH = 2016.0
WISE_EPOCH = 2010.5
SIMBAD_EPOCH = 2000.0

#: the star's lines on the telluric lines: |RV - BERV| below this [km/s]
#:   (the star-o-meter's)
TELLURIC_KMS = 5.0

#: the names of a star shown, first those of these catalogs, in this order
ALIASES = ("NAME ", "GJ ", "Gl ", "HD ", "HIP ", "TOI-", "K2-", "Kepler-",
           "WASP-", "HAT-P-", "KELT-", "Wolf ", "Ross ", "LHS ", "LP ",
           "BD", "CD", "TIC ", "Gaia DR3 ", "2MASS ")
#: how many names are shown
ALIAS_COUNT = 8

# the M dwarf relations (Mann et al. 2015, as its erratum gives them,
#   Mann et al. 2016, ApJ 819, 87; Table 1 and 2)
#: R* = a + b M_Ks + c M_Ks^2 (sigma 2.89 %)
MANN_R = (1.9515, -0.3520, 0.01680)
MANN_R_SIGMA = 0.0289
#: R* = (a + b M_Ks + c M_Ks^2) (1 + f [Fe/H]) (sigma 2.70 %)
MANN_R_FEH = (1.9305, -0.3466, 0.01647, 0.04458)
MANN_R_FEH_SIGMA = 0.0270
#: M* = a + b M_Ks + c M_Ks^2 + d M_Ks^3 + e M_Ks^4 (sigma 1.8 %)
MANN_M = (0.5858, 0.3872, -0.1217, 0.0106, -2.7262e-4)
MANN_M_SIGMA = 0.018
#: the range the relations hold in (Mann et al. 2015, Sect. 6)
MANN_MKS = (4.6, 9.8)
MANN_TEFF = (2700.0, 4100.0)
#: Teff/3500 = polynomial of BP-RP (sigma 52 K); + f [Fe/H] (45 K);
#:   + f (J-H) + g (J-H)^2 (49 K)
MANN_TEFF_BPRP = (3.245, -2.4309, 1.043, -0.2127, 0.01649)
MANN_TEFF_BPRP_FEH = (2.835, -1.893, 0.7860, -0.1594, 0.01243, 0.04417)
MANN_TEFF_BPRP_JH = (3.172, -2.475, 1.082, -0.2231, 0.01738, 0.08776,
                     0.04355)
MANN_TEFF_SIGMA = {"bprp": 52.0, "feh": 45.0, "jh": 49.0}
#: the BP-RP of the M dwarfs the Teff relations are applied to (the
#:   star-o-meter's)
MANN_BPRP = (1.5, 4.5)
#: Delfosse et al. (2000): log(M/Msun) = 1e-3 (a + b M_K + ... e M_K^4),
#:   for M_K in [4.5, 9.5]; ~10 % scatter
DELFOSSE_M = (1.8, 6.12, 13.205, -6.2315, 0.37529)
DELFOSSE_MK = (4.5, 9.5)
DELFOSSE_SIGMA = 0.10
#: Duque-Arribas et al. (2023): Eq. 4, Table 4 first row,
#:   [Fe/H] = a + b X + c Y + d Z + e Y^2 + f Z^2, X = W1-W2, Y = BP-RP,
#:   Z = M_G; 0.089 dex (Table 5); calibrated on M0V-M5V with these W1-W2
#:   and G-J
DUQUE_FEH = (5.87, -1.40, 5.08, -2.45, -0.788, 0.1094)
DUQUE_SIGMA = 0.089
DUQUE_W1W2 = (-0.10, 0.24)
DUQUE_GJ = (1.85, 3.10)
#: the Sun: log g [cgs], Teff [K] (the star-o-meter's)
LOGG_SUN = 4.438
TEFF_SUN = 5778.0
#: the M dwarfs by absolute G (the star-o-meter's)
MG_MDWARF = (7.5, 16.5)
#: the AFGK dwarfs whose Gaia GSP-Phot and FLAME parameters are used
AFGK_BPRP = (-0.05, 1.95)
AFGK_MG = (0.5, 7.5)
#: a star this far in M_G from the dwarf sequence at its BP-RP is not
#:   placed on it
AFGK_SEQ_MG = 1.0
#: the dwarf sequence (Pecaut & Mamajek 2013, as the star-o-meter has it):
#:   spectral type, BP-RP, M_G, Teff [K], R [Rsun], M [Msun]
AFGK_SEQUENCE = (
    ("A0V", -0.04, 1.00, 9700.0, 2.193, 2.18),
    ("A1V", 0.01, 1.16, 9300.0, 2.136, 2.05),
    ("A2V", 0.07, 1.35, 8800.0, 2.117, 1.98),
    ("A3V", 0.11, 1.69, 8600.0, 1.861, 1.86),
    ("A5V", 0.19, 1.98, 8100.0, 1.785, 1.88),
    ("A7V", 0.26, 2.19, 7760.0, 1.750, 1.77),
    ("A9V", 0.33, 2.37, 7400.0, 1.747, 1.75),
    ("F0V", 0.377, 2.51, 7220.0, 1.728, 1.61),
    ("F2V", 0.490, 2.89, 6820.0, 1.622, 1.46),
    ("F5V", 0.587, 3.26, 6550.0, 1.473, 1.33),
    ("F8V", 0.694, 3.90, 6180.0, 1.221, 1.18),
    ("G0V", 0.784, 4.33, 5930.0, 1.100, 1.06),
    ("G2V", 0.823, 4.64, 5770.0, 1.012, 1.00),
    ("G5V", 0.850, 4.80, 5660.0, 0.977, 0.98),
    ("G8V", 0.900, 5.10, 5480.0, 0.914, 0.94),
    ("K0V", 0.983, 5.55, 5270.0, 0.813, 0.88),
    ("K1V", 1.010, 5.65, 5170.0, 0.797, 0.86),
    ("K2V", 1.100, 5.83, 5100.0, 0.783, 0.82),
    ("K3V", 1.210, 6.20, 4830.0, 0.755, 0.78),
    ("K4V", 1.340, 6.53, 4600.0, 0.713, 0.73),
    ("K5V", 1.430, 6.83, 4440.0, 0.701, 0.70),
    ("K7V", 1.700, 7.57, 4100.0, 0.630, 0.64),
)

#: the references the rows cite: (text, link)
REFS = {
    "mann15": ("Mann et al. 2015", "https://doi.org/10.1088/0004-637X/804/1/64"),
    "delfosse00": ("Delfosse et al. 2000", "https://doi.org/10.1051/0004-6361:20001166"),
    "duque23": ("Duque-Arribas et al. 2023", "https://doi.org/10.3847/1538-4357/acacf6"),
    "pm13": ("Pecaut & Mamajek 2013", "https://doi.org/10.1088/0067-0049/208/1/9"),
    "gaia": ("Gaia DR3", "https://doi.org/10.1051/0004-6361/202243940"),
}

_LOCK = threading.Lock()
_DOWN = {}


# =============================================================================
# The services
# =============================================================================
def _num(value):
    """A finite float, else None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _usable(base):
    """A service not left alone for having failed."""
    with _LOCK:
        return time.time() - _DOWN.get(base, 0.0) > DOWN_FOR


def _failed(base):
    with _LOCK:
        _DOWN[base] = time.time()


def tap(base, adql, timeout=TIMEOUT):
    """The rows of an ADQL query of a TAP service, as CSV: list of dict.
    The NASA archive takes its own parameters (query, format)."""
    if base == NASA_TAP:
        params = dict(query=adql, format="csv")
    else:
        params = dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv",
                      QUERY=adql)
    url = base + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        raw = answer.read().decode("utf-8", "replace")
    if raw.lstrip().startswith("<"):
        raise OSError("%s answered an error" % base.split("/")[2])
    return list(csv.DictReader(io.StringIO(raw)))


def _first(asks):
    """The rows of the first of [(base, adql)] that answers, and its host:
    (rows, host); ([], None) when none does."""
    for base, adql in asks:
        if not adql or not _usable(base):
            continue
        try:
            return tap(base, adql), base.split("/")[2]
        except (OSError, ValueError, urllib.error.URLError) as exc:
            del exc
            _failed(base)
    return [], None


def _literal(text):
    return "'%s'" % str(text).replace("'", "''")


def ask_simbad(names):
    """The star as SIMBAD knows it, from the first of names it knows: dict
    (main, asked, server, ra, dec, plx, plx_err, pmra, pmdec, sptype,
    otype, rv, rv_err, mags {V G J H K}, ids, fe_h rows, vsini rows), or
    None when SIMBAD knows none. Raises OSError when SIMBAD cannot be
    reached."""
    from lbl.core import simbad
    row = asked = None
    for one in names:
        rows = simbad.tap(
            "SELECT b.oid, b.main_id, b.ra, b.dec, b.plx_value, b.plx_err,"
            " b.pmra, b.pmdec, b.sp_type, b.otype_txt, b.rvz_radvel,"
            " b.rvz_err FROM ident AS i JOIN basic AS b ON b.oid = i.oidref"
            " WHERE i.id = %s" % _literal(one))
        if rows:
            row, asked = rows[0], one
            break
    if row is None:
        return None
    oid = int(row["oid"])
    fluxes = simbad.tap("SELECT filter, flux FROM flux WHERE oidref = %d"
                        " AND filter IN ('V', 'G', 'J', 'H', 'K')" % oid)
    ids = simbad.tap("SELECT id FROM ident WHERE oidref = %d" % oid)
    fe_h = simbad.tap("SELECT teff, log_g, fe_h FROM mesFe_h"
                      " WHERE oidref = %d" % oid)
    rot = simbad.tap("SELECT vsini FROM mesRot WHERE oidref = %d" % oid)
    return dict(
        main=re.sub(r"\s+", " ", str(row["main_id"])).strip(), asked=asked,
        server=simbad.SERVERS[0], ra=_num(row["ra"]), dec=_num(row["dec"]),
        plx=_num(row["plx_value"]), plx_err=_num(row["plx_err"]),
        pmra=_num(row["pmra"]), pmdec=_num(row["pmdec"]),
        sptype=(row["sp_type"] or "").strip() or None,
        otype=(row["otype_txt"] or "").strip() or None,
        rv=_num(row["rvz_radvel"]), rv_err=_num(row["rvz_err"]),
        mags={r["filter"]: _num(r["flux"]) for r in fluxes},
        ids=[re.sub(r"\s+", " ", str(r["id"])).strip() for r in ids],
        fe_h=fe_h, vsini=[_num(r["vsini"]) for r in rot])


GAIA_COLUMNS = ("source_id, ra, dec, parallax, parallax_error, pmra, pmdec,"
                " phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,"
                " radial_velocity, radial_velocity_error, ruwe")


def ask_gaia(source_id):
    """A Gaia DR3 source: its row (with teff_gspphot, logg_gspphot,
    mh_gspphot, radius_flame, lum_flame, mass_flame when ESA answers) and
    where it came from, or None."""
    where = " WHERE source_id = %d" % int(source_id)
    rows, host = _first([
        (ESA_TAP, "SELECT " + GAIA_COLUMNS + " FROM gaiadr3.gaia_source"
         + where),
        (GAVO_TAP, "SELECT " + GAIA_COLUMNS + " FROM gaia.dr3lite" + where)])
    if not rows:
        return None
    out = {key: _num(value) for key, value in rows[0].items()}
    out["source_id"] = int(source_id)
    out["from"] = host
    if host == ESA_TAP.split("/")[2]:
        params, _ = _first([(ESA_TAP, (
            "SELECT teff_gspphot, logg_gspphot, mh_gspphot, radius_flame,"
            " lum_flame, mass_flame FROM gaiadr3.astrophysical_parameters"
            + where))])
        if params:
            out.update({key: _num(value) for key, value in params[0].items()})
    return out


def _moved(ra, dec, pmra, pmdec, years):
    """A position moved by its proper motion over years [deg]."""
    if None in (pmra, pmdec):
        return ra, dec
    cos_dec = max(math.cos(math.radians(dec)), 1e-6)
    return ((ra + pmra * years / 3.6e6 / cos_dec) % 360.0,
            max(-90.0, min(90.0, dec + pmdec * years / 3.6e6)))


def ask_wise(ra, dec, pmra, pmdec, epoch):
    """The AllWISE source nearest to the star taken back to the AllWISE
    epoch: dict(W1, W1_err ... W4_err, sep [arcsec], from), or None. The
    search radius grows with the proper motion (AllWISE's exposures span a
    year)."""
    if ra is None or dec is None:
        return None
    ra0, dec0 = _moved(ra, dec, pmra, pmdec, WISE_EPOCH - epoch)
    total = math.hypot(pmra or 0.0, pmdec or 0.0) / 1000.0
    radius = min(30.0, 3.0 + total)
    cone = ("1 = CONTAINS(POINT('ICRS', %s, %s), CIRCLE('ICRS', %.7f, %.7f,"
            " %.6f))")
    rows, host = _first([
        (GAVO_TAP, "SELECT raj2000 AS ra, dej2000 AS dec, w1mpro, w1sigmrpo"
         " AS w1sigmpro, w2mpro, w2sigmrpo AS w2sigmpro, w3mpro, w3sigmrpo"
         " AS w3sigmpro, w4mpro, w4sigmrpo AS w4sigmpro FROM wise.main WHERE "
         + cone % ("raj2000", "dej2000", ra0, dec0, radius / 3600.0)),
        (IRSA_TAP, "SELECT ra, dec, w1mpro, w1sigmpro, w2mpro, w2sigmpro,"
         " w3mpro, w3sigmpro, w4mpro, w4sigmpro FROM allwise_p3as_psd WHERE "
         + cone % ("ra", "dec", ra0, dec0, radius / 3600.0))])
    best = None
    for row in rows:
        r_ra, r_dec = _num(row.get("ra")), _num(row.get("dec"))
        if r_ra is None or r_dec is None:
            continue
        sep = 3600.0 * math.degrees(math.acos(max(-1.0, min(1.0, (
            math.sin(math.radians(dec0)) * math.sin(math.radians(r_dec))
            + math.cos(math.radians(dec0)) * math.cos(math.radians(r_dec))
            * math.cos(math.radians(ra0 - r_ra)))))))
        if best is None or sep < best[0]:
            best = (sep, row)
    if best is None:
        return None
    sep, row = best
    out = dict(sep=round(sep, 2), source=host)
    for band in (1, 2, 3, 4):
        out["W%d" % band] = _num(row.get("w%dmpro" % band))
        out["W%d_err" % band] = _num(row.get("w%dsigmpro" % band))
    return out


PLANET_COLUMNS = (
    "pl_name, hostname, pl_orbper, pl_orbpererr1, pl_rvamp, pl_bmasse,"
    " pl_bmassprov, pl_rade, pl_orbsmax, pl_orbeccen, pl_eqt,"
    " discoverymethod, disc_year, disc_refname, pl_orbper_reflink,"
    " pl_controv_flag, st_teff, st_teff_reflink, st_rad, st_rad_reflink,"
    " st_mass, st_mass_reflink, st_met, st_met_reflink, st_logg,"
    " st_logg_reflink, st_age, st_age_reflink, st_rotp, st_rotp_reflink,"
    " st_vsin, st_vsin_reflink, st_spectype")
#: a reference of the NASA archive: <a ... href=URL ...>TEXT</a>
_REFLINK = re.compile(r"href=['\"]?([^\s'\">]+)[^>]*>\s*([^<]*?)\s*</a>")


def reference(link):
    """A reference link of the NASA archive as dict(text, url), or None."""
    match = _REFLINK.search(str(link or ""))
    if not match:
        return None
    return dict(text=match.group(2).strip(), url=match.group(1))


def planet_keys(ids, names):
    """What the NASA archive is asked by: its Gaia DR3, TIC, HD and HIP
    numbers as it writes them, and the star's names (SIMBAD's NAME taken
    off) as its host names."""
    gaia = [i for i in ids if i.startswith("Gaia DR3 ")]
    tic = [i for i in ids if i.startswith("TIC ")]
    hd = [i for i in ids if re.match(r"HD \d+[A-Z]?$", i)]
    hip = [i for i in ids if re.match(r"HIP \d+[A-Z]?$", i)]
    hosts = []
    for one in list(ids) + list(names):
        one = re.sub(r"^NAME ", "", str(one)).strip()
        if one and one not in hosts:
            hosts.append(one)
    return dict(gaia_dr3_id=gaia, tic_id=tic, hd_name=hd, hip_name=hip,
                hostname=hosts)


def ask_planets(ids, names):
    """The star's planets and published parameters in the NASA Exoplanet
    Archive's composite table: dict(planets [dict], host {key: (value,
    reference)}), or None when the archive cannot be reached."""
    keys = planet_keys(ids, names)
    where = " OR ".join("%s IN (%s)" % (column, ", ".join(
        _literal(v) for v in values)) for column, values in keys.items()
        if values)
    if not where:
        return dict(planets=[], host={})
    rows, host = _first([(NASA_TAP, "select %s from pscomppars where %s"
                          % (PLANET_COLUMNS, where))])
    if host is None:
        return None
    planets = []
    for row in rows:
        planets.append(dict(
            name=row["pl_name"], period=_num(row["pl_orbper"]),
            period_err=_num(row["pl_orbpererr1"]), k=_num(row["pl_rvamp"]),
            mass=_num(row["pl_bmasse"]), mass_kind=row["pl_bmassprov"] or None,
            radius=_num(row["pl_rade"]), a=_num(row["pl_orbsmax"]),
            e=_num(row["pl_orbeccen"]), teq=_num(row["pl_eqt"]),
            method=row["discoverymethod"] or None,
            year=int(_num(row["disc_year"])) if _num(row["disc_year"]) else None,
            discovery=reference(row["disc_refname"]),
            orbit=reference(row["pl_orbper_reflink"]),
            controversial=row.get("pl_controv_flag") == "1"))
    planets.sort(key=lambda one: (one["period"] is None, one["period"] or 0))
    published = {}
    if rows:
        first = rows[0]
        for key in ("teff", "rad", "mass", "met", "logg", "age", "rotp",
                    "vsin"):
            value = _num(first.get("st_" + key))
            if value is not None:
                published[key] = dict(
                    value=value, ref=reference(first.get("st_%s_reflink" % key)))
        if first.get("st_spectype"):
            published["sptype"] = dict(value=first["st_spectype"], ref=None)
    return dict(planets=planets, host=published)


# =============================================================================
# What is derived (the star-o-meter's relations)
# =============================================================================
def _poly(coeffs, x):
    return sum(c * x ** i for i, c in enumerate(coeffs))


def _family(sptype):
    """The class of a spectral type: (letter, subtype or None)."""
    text = (sptype or "").upper().strip()
    if not text or text[0] not in "OBAFGKM":
        return None, None
    match = re.match(r"[OBAFGKM](\d+(?:\.\d+)?)", text)
    return text[0], float(match.group(1)) if match else None


def _luminosity(sptype):
    """'dwarf', 'other' (a subgiant, giant, subdwarf...) or None (no
    luminosity class): the star-o-meter's reading."""
    text = (sptype or "").upper().replace(" ", "")
    if "VI" in text or "VII" in text:
        return "other"
    if "IV" in text or "III" in text or "II" in text or "I" in text:
        return "dwarf" if ("V" in text and "IV" not in text) else "other"
    if "V" in text:
        return "dwarf"
    return None


def _row(label, value, err=None, unit="", how="", ref=None, digits=2,
         note=None):
    """A line of a group: its value as text."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        text = "%.*f" % (digits, value)
        if err is not None:
            text += " ± %.*f" % (digits, err)
    out = dict(label=label, value=text, unit=unit, how=how)
    if ref:
        out["ref"] = dict(text=REFS[ref][0], url=REFS[ref][1]) \
            if isinstance(ref, str) else ref
    if note:
        out["note"] = note
    return out


def _note(key, *args):
    return dict(key=key, args=[str(a) for a in args])


def mdwarf(data):
    """The parameters of an M dwarf (Mann et al. 2015, Delfosse et al. 2000,
    Duque-Arribas et al. 2023), each within the range of its calibration:
    (rows, notes)."""
    rows, notes = [], []
    m_ks, m_g = data.get("M_Ks"), data.get("M_G")
    bp, rp = data.get("BP"), data.get("RP")
    bp_rp = bp - rp if None not in (bp, rp) else None
    j, h, g = data.get("J"), data.get("H"), data.get("G")
    w1, w2 = data.get("W1"), data.get("W2")
    # [Fe/H] first: the Teff and the radius use it
    feh = None
    if None in (w1, w2, bp_rp, m_g):
        notes.append(_note("star_note_duque_missing"))
    else:
        x = w1 - w2
        g_j = g - j if None not in (g, j) else None
        if not DUQUE_W1W2[0] <= x <= DUQUE_W1W2[1]:
            notes.append(_note("star_note_out", "W1−W2", "%.2f" % x,
                               "%.2f" % DUQUE_W1W2[0], "%.2f" % DUQUE_W1W2[1],
                               REFS["duque23"][0]))
        elif g_j is not None and not DUQUE_GJ[0] <= g_j <= DUQUE_GJ[1]:
            notes.append(_note("star_note_out", "G−J", "%.2f" % g_j,
                               "%.2f" % DUQUE_GJ[0], "%.2f" % DUQUE_GJ[1],
                               REFS["duque23"][0]))
        else:
            a, b, c, d, e, f = DUQUE_FEH
            feh = a + b * x + c * bp_rp + d * m_g + e * bp_rp ** 2 \
                + f * m_g ** 2
            rows.append(_row("[Fe/H]", feh, DUQUE_SIGMA, "dex",
                             "W1−W2, BP−RP, M_G", "duque23"))
    # Teff from BP-RP, with J-H, with [Fe/H]
    teffs = []
    if bp_rp is None:
        notes.append(_note("star_note_no_bprp"))
    elif not MANN_BPRP[0] <= bp_rp <= MANN_BPRP[1]:
        notes.append(_note("star_note_out", "BP−RP", "%.2f" % bp_rp,
                           "%.1f" % MANN_BPRP[0], "%.1f" % MANN_BPRP[1],
                           REFS["mann15"][0]))
    else:
        asks = []
        if None not in (j, h):
            *poly, f, gg = MANN_TEFF_BPRP_JH
            asks.append(("jh", "BP−RP, J−H", 3500.0 * (
                _poly(poly, bp_rp) + f * (j - h) + gg * (j - h) ** 2)))
        if feh is not None:
            *poly, f = MANN_TEFF_BPRP_FEH
            asks.append(("feh", "BP−RP, [Fe/H]",
                         3500.0 * (_poly(poly, bp_rp) + f * feh)))
        asks.append(("bprp", "BP−RP",
                     3500.0 * _poly(MANN_TEFF_BPRP, bp_rp)))
        for kind, how, teff in asks:
            if MANN_TEFF[0] <= teff <= MANN_TEFF[1]:
                rows.append(_row("Teff", teff, MANN_TEFF_SIGMA[kind], "K",
                                 how, "mann15", digits=0))
                teffs.append(teff)
            else:
                notes.append(_note("star_note_out", "Teff (%s)" % how,
                                   "%.0f" % teff, "%.0f" % MANN_TEFF[0],
                                   "%.0f" % MANN_TEFF[1], REFS["mann15"][0]))
    # radius and mass from M_Ks
    radius = mass = None
    if m_ks is None:
        notes.append(_note("star_note_no_mk"))
    elif not MANN_MKS[0] <= m_ks <= MANN_MKS[1]:
        notes.append(_note("star_note_out", "M_Ks", "%.2f" % m_ks,
                           "%.1f" % MANN_MKS[0], "%.1f" % MANN_MKS[1],
                           REFS["mann15"][0]))
    else:
        radius = _poly(MANN_R, m_ks)
        rows.append(_row("R★", radius, MANN_R_SIGMA * radius, "R☉",
                         "M_Ks", "mann15", digits=3))
        if feh is not None:
            *poly, f = MANN_R_FEH
            r_feh = _poly(poly, m_ks) * (1.0 + f * feh)
            rows.append(_row("R★", r_feh, MANN_R_FEH_SIGMA * r_feh,
                             "R☉", "M_Ks, [Fe/H]", "mann15", digits=3))
        mass = _poly(MANN_M, m_ks)
        rows.append(_row("M★", mass, MANN_M_SIGMA * mass, "M☉",
                         "M_Ks", "mann15", digits=3))
    if m_ks is not None and DELFOSSE_MK[0] <= m_ks <= DELFOSSE_MK[1]:
        m_del = 10.0 ** (1e-3 * _poly(DELFOSSE_M, m_ks))
        rows.append(_row("M★", m_del, DELFOSSE_SIGMA * m_del, "M☉",
                         "M_K", "delfosse00", digits=3))
    if radius is not None and mass is not None:
        logg = LOGG_SUN + math.log10(mass) - 2.0 * math.log10(radius)
        err = math.hypot(MANN_M_SIGMA, 2.0 * MANN_R_SIGMA) / math.log(10.0)
        rows.append(_row("log g", logg, err, "cgs", "M★, R★",
                         "mann15", digits=2))
        if teffs:
            lum = radius ** 2 * (teffs[0] / TEFF_SUN) ** 4
            rows.append(_row("L★", lum, None, "L☉",
                             "R★, Teff", "mann15", digits=4))
    return [r for r in rows if r], notes


def afgk(data, gaia):
    """The parameters of an AFGK dwarf: Gaia's GSP-Phot and FLAME, else the
    dwarf sequence of Pecaut & Mamajek (2013) at its BP-RP (its M_G near
    the sequence's), else at its spectral type: (mode, rows, notes)."""
    bp, rp, m_g = data.get("BP"), data.get("RP"), data.get("M_G")
    bp_rp = bp - rp if None not in (bp, rp) else None
    in_box = (bp_rp is not None and m_g is not None
              and AFGK_BPRP[0] <= bp_rp <= AFGK_BPRP[1]
              and AFGK_MG[0] <= m_g <= AFGK_MG[1])
    gaia = gaia or {}
    teff, logg = gaia.get("teff_gspphot"), gaia.get("logg_gspphot")
    mh, radius = gaia.get("mh_gspphot"), gaia.get("radius_flame")
    lum, mass = gaia.get("lum_flame"), gaia.get("mass_flame")
    if in_box and any(v is not None for v in (teff, logg, mh, radius, lum,
                                              mass)):
        if radius is None and None not in (lum, teff):
            radius = math.sqrt(lum) * (TEFF_SUN / teff) ** 2
        if logg is None and None not in (mass, radius):
            logg = LOGG_SUN + math.log10(mass) - 2.0 * math.log10(radius)
        rows = [_row("Teff", teff, None, "K", "GSP-Phot", "gaia", digits=0),
                _row("[M/H]", mh, None, "dex", "GSP-Phot", "gaia"),
                _row("R★", radius, None, "R☉", "FLAME", "gaia",
                     digits=3),
                _row("M★", mass, None, "M☉", "FLAME", "gaia",
                     digits=3),
                _row("log g", logg, None, "cgs", "GSP-Phot", "gaia"),
                _row("L★", lum, None, "L☉", "FLAME", "gaia",
                     digits=3)]
        return "afgk_gaia", [r for r in rows if r], []
    lower = upper = None
    weight = 0.0
    how = ""
    seq = AFGK_SEQUENCE
    if bp_rp is not None and m_g is not None \
            and seq[0][1] <= bp_rp <= seq[-1][1]:
        for left, right in zip(seq[:-1], seq[1:]):
            if left[1] <= bp_rp <= right[1]:
                lower, upper = left, right
                break
        span = upper[1] - lower[1]
        weight = (bp_rp - lower[1]) / span if span > 0 else 0.0
        if abs(m_g - (lower[2] + weight * (upper[2] - lower[2]))) \
                > AFGK_SEQ_MG:
            lower = upper = None
        else:
            how = "BP−RP, M_G"
    family, sub = _family(data.get("sptype"))
    if lower is None and family in ("A", "F", "G", "K") and sub is not None \
            and _luminosity(data.get("sptype")) == "dwarf":
        rows_f = [r for r in seq if r[0][0] == family]
        subs = [float(re.match(r"[AFGK](\d+(?:\.\d+)?)", r[0]).group(1))
                for r in rows_f]
        if subs and subs[0] <= sub <= subs[-1]:
            for (left, s_l), (right, s_r) in zip(
                    list(zip(rows_f, subs))[:-1], list(zip(rows_f, subs))[1:]):
                if s_l <= sub <= s_r:
                    lower, upper = left, right
                    weight = (sub - s_l) / (s_r - s_l) if s_r > s_l else 0.0
                    break
            if lower is None:
                lower = upper = rows_f[-1]
            how = data.get("sptype")
    if lower is None:
        return None, [], []

    def lerp(i):
        return lower[i] + weight * (upper[i] - lower[i])

    teff, radius, mass = lerp(3), lerp(4), lerp(5)
    logg = LOGG_SUN + math.log10(mass) - 2.0 * math.log10(radius)
    lum = radius ** 2 * (teff / TEFF_SUN) ** 4
    near = lower[0] if lower[0] == upper[0] else "%s-%s" % (lower[0],
                                                                 upper[0])
    rows = [_row("Teff", teff, None, "K", how, "pm13", digits=0),
            _row("R★", radius, None, "R☉", how, "pm13", digits=3),
            _row("M★", mass, None, "M☉", how, "pm13", digits=3),
            _row("log g", logg, None, "cgs", how, "pm13"),
            _row("L★", lum, None, "L☉", "R★, Teff", "pm13",
                 digits=3)]
    return "afgk_seq", [r for r in rows if r], [_note("star_note_sequence",
                                                      near)]


def derive(data, gaia=None):
    """What kind of star it is, and its parameters as the star-o-meter
    derives them: (mode, rows, notes). mode: mdwarf, afgk_gaia, afgk_seq,
    wd (a white dwarf), other (not a dwarf), early (O, B), None."""
    sptype, otype = data.get("sptype") or "", (data.get("otype") or "")
    if sptype.upper().startswith("D") or "WD" in otype.upper() \
            or "WHITE DWARF" in otype.upper():
        return "wd", [], []
    if _luminosity(sptype) == "other":
        return "other", [], []
    family, _ = _family(sptype)
    m_g = data.get("M_G")
    if (m_g is not None and MG_MDWARF[0] <= m_g <= MG_MDWARF[1]) \
            or family == "M":
        rows, notes = mdwarf(data)
        return "mdwarf", rows, notes
    mode, rows, notes = afgk(data, gaia)
    if mode:
        return mode, rows, notes
    if family in ("O", "B"):
        return "early", [], []
    return None, [], []


# =============================================================================
# The motion and the tellurics
# =============================================================================
def _speed_of_light_kms():
    from astropy import constants
    return constants.c.to("km/s").value


def compose(rv, berv):
    """The star's velocity seen by an observer whose barycentric correction
    is berv [km/s]: RV minus BERV, composed relativistically (the Doppler
    factor of rv over that of berv)."""
    c = _speed_of_light_kms()
    ratio = math.sqrt((1.0 + rv / c) / (1.0 - rv / c)) \
        / math.sqrt((1.0 + berv / c) / (1.0 - berv / c))
    return c * (ratio ** 2 - 1.0) / (ratio ** 2 + 1.0)


def bervs(ra, dec, year):
    """The barycentric correction of the Earth's centre toward the star on
    each day of a year (0 h UTC) [km/s]: the Earth's barycentric velocity
    on the star's direction, positive when the Earth moves toward it (as
    astropy's radial_velocity_correction, without the Earth's rotation,
    under 0.5 km/s). (dates [datetime.date], values)."""
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
    from astropy.time import Time
    first = datetime.date(year, 1, 1)
    days = (datetime.date(year + 1, 1, 1) - first).days
    times = Time(first.isoformat()) + np.arange(days) * u.day
    _, velocity = get_body_barycentric_posvel("earth", times)
    star = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).cartesian
    berv = (velocity.dot(star)).to(u.km / u.s).value
    return [first + datetime.timedelta(days=int(i)) for i in range(days)], \
        [float(v) for v in berv]


def tellurics(ra, dec, rv, year=None):
    """The days of a year when the star's lines sit within TELLURIC_KMS of
    the telluric lines: dict(year, threshold, windows [[first, last] ISO
    dates; one across the new year joined], berv [min, max], rv)."""
    if None in (ra, dec, rv):
        return None
    year = year or datetime.date.today().year
    dates, berv = bervs(ra, dec, year)
    near = [abs(compose(rv, b)) < TELLURIC_KMS for b in berv]
    windows, start = [], None
    for i, flag in enumerate(near + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            windows.append([start, i - 1])
            start = None
    if len(windows) > 1 and windows[0][0] == 0 \
            and windows[-1][1] == len(dates) - 1:
        windows = [[windows[-1][0], windows[0][1]]] + windows[1:-1]
    return dict(year=year, threshold=TELLURIC_KMS, rv=rv,
                berv=[round(min(berv), 2), round(max(berv), 2)],
                windows=[[dates[a].isoformat(), dates[b].isoformat()]
                         for a, b in windows])


def motion(ra, dec, plx, pmra, pmdec, rv):
    """Tangential and space velocity, heliocentric galactic U (toward the
    Galactic centre), V (along its rotation), W (toward its north pole)
    [km/s]: dict, the values that can be had."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    out = {}
    if None in (ra, dec, plx, pmra, pmdec) or plx <= 0:
        return out
    distance = 1000.0 / plx * u.pc
    total = math.hypot(pmra, pmdec) * u.mas / u.yr
    out["vtan"] = float((total * distance).to(
        u.km / u.s, equivalencies=u.dimensionless_angles()).value)
    if rv is None:
        return out
    out["vspace"] = math.hypot(out["vtan"], rv)
    star = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=distance,
                    pm_ra_cosdec=pmra * u.mas / u.yr,
                    pm_dec=pmdec * u.mas / u.yr,
                    radial_velocity=rv * u.km / u.s)
    velocity = star.galactic.velocity
    out.update(U=float(velocity.d_x.to(u.km / u.s).value),
               V=float(velocity.d_y.to(u.km / u.s).value),
               W=float(velocity.d_z.to(u.km / u.s).value))
    return out


# =============================================================================
# The summary
# =============================================================================
def sexagesimal(ra, dec):
    """A position as hh:mm:ss.ss +dd:mm:ss.s, or None."""
    if None in (ra, dec):
        return None
    from astropy import units as u
    from astropy.coordinates import Angle
    return "%s %s" % (
        Angle(ra, u.deg).to_string(unit=u.hour, sep=":", precision=2,
                                   pad=True),
        Angle(dec, u.deg).to_string(unit=u.deg, sep=":", precision=1,
                                    alwayssign=True, pad=True))


def title(main):
    """SIMBAD's main identifier as a title: without its NAME, *, V*..."""
    return re.sub(r"^(NAME|\*+|V\*|EM\*|Em\*)\s+", "", main or "").strip()


def spelled(name, ids):
    """A name as SIMBAD writes it among the star's identifiers (GJ581 as
    GJ 581), else as it is."""
    def bare(text):
        return re.sub(r"[\s_]+", "", str(text)).lower()
    return next((re.sub(r"^NAME ", "", one) for one in ids
                 if bare(one) == bare(name)), name)


def _median(values):
    values = [v for v in values if v is not None]
    return (statistics.median(values), len(values)) if values else (None, 0)


def _aliases(main, ids, asked):
    """The star's names worth showing, the catalogs of ALIASES first."""
    out = []
    for prefix in ALIASES:
        for one in ids:
            if one.startswith(prefix) and one != main and one not in out:
                out.append(re.sub(r"^NAME ", "", one))
                break
    return out[:ALIAS_COUNT]


def summarise(names, fallback_planets=None, today=None):
    """Everything about a star, from the first of names SIMBAD knows: the
    answer the page draws (see the module's docstring)."""
    try:
        star = ask_simbad(names)
    except Exception as exc:                              # noqa: BLE001
        return dict(state="failed", asked=list(names),
                    error="SIMBAD: %s: %s" % (type(exc).__name__,
                                              str(exc)[:200]))
    if star is None:
        return dict(state="unknown", asked=list(names))
    gaia_id = next((int(i.split()[-1]) for i in star["ids"]
                    if re.match(r"Gaia DR3 \d+$", i)), None)
    with ThreadPoolExecutor(max_workers=3) as pool:
        gaia_job = pool.submit(ask_gaia, gaia_id) if gaia_id else None
        planets_job = pool.submit(ask_planets, star["ids"], names)
        wise_job = pool.submit(ask_wise, star["ra"], star["dec"],
                               star["pmra"], star["pmdec"], SIMBAD_EPOCH)
        gaia = gaia_job.result() if gaia_job else None
        nasa = planets_job.result()
        wise = wise_job.result()
    # what is measured: Gaia's astrometry and photometry when there,
    #   SIMBAD's otherwise
    plx = (gaia or {}).get("parallax") or star["plx"]
    plx_err = (gaia or {}).get("parallax_error") if (gaia or {}).get(
        "parallax") else star["plx_err"]
    pmra = (gaia or {}).get("pmra") if (gaia or {}).get("pmra") is not None \
        else star["pmra"]
    pmdec = (gaia or {}).get("pmdec") if (gaia or {}).get("pmdec") is not \
        None else star["pmdec"]
    rv, rv_from = star["rv"], "SIMBAD"
    if rv is None and (gaia or {}).get("radial_velocity") is not None:
        rv, rv_from = gaia["radial_velocity"], "Gaia DR3"
    mags = dict(star["mags"])
    data = dict(sptype=star["sptype"], otype=star["otype"],
                G=(gaia or {}).get("phot_g_mean_mag") or mags.get("G"),
                BP=(gaia or {}).get("phot_bp_mean_mag"),
                RP=(gaia or {}).get("phot_rp_mean_mag"),
                V=mags.get("V"), J=mags.get("J"), H=mags.get("H"),
                K=mags.get("K"))
    for band in ("W1", "W2", "W3", "W4"):
        data[band] = (wise or {}).get(band)
    distance = 1000.0 / plx if plx and plx > 0 else None
    modulus = 5.0 * math.log10(distance) - 5.0 if distance else None
    if modulus is not None:
        data["M_G"] = data["G"] - modulus if data["G"] is not None else None
        data["M_Ks"] = data["K"] - modulus if data["K"] is not None else None
    mode, derived, notes = derive(data, gaia)

    groups = []
    moving = motion(star["ra"], star["dec"], plx, pmra, pmdec, rv)
    groups.append(dict(key="position", rows=[r for r in (
        _row("RA, Dec", sexagesimal(star["ra"], star["dec"]), how="J2000"),
        _row("ϖ", plx, plx_err, "mas", digits=3,
             how="Gaia DR3" if (gaia or {}).get("parallax") else "SIMBAD"),
        _row("d", distance, distance * plx_err / plx if distance and plx_err
             else None, "pc", digits=3 if distance and distance < 10 else 2),
        _row("μα*, μδ", "%.2f, %.2f" % (pmra, pmdec)
             if None not in (pmra, pmdec) else None, unit="mas/yr"),
        _row("RV", rv, star["rv_err"] if rv_from == "SIMBAD" else
             (gaia or {}).get("radial_velocity_error"), "km/s", how=rv_from,
             digits=3),
        _row("v_tan", moving.get("vtan"), None, "km/s"),
        _row("v_space", moving.get("vspace"), None, "km/s"),
        _row("U, V, W", "%.1f, %.1f, %.1f" % (moving["U"], moving["V"],
                                              moving["W"])
             if "U" in moving else None, unit="km/s",
             how="heliocentric")) if r]))
    bp_rp = data["BP"] - data["RP"] if None not in (data["BP"], data["RP"]) \
        else None
    w1_w2 = data["W1"] - data["W2"] if None not in (data["W1"], data["W2"]) \
        else None
    groups.append(dict(key="photometry", rows=[r for r in (
        _row("V", data["V"], unit="mag"),
        _row("G", data["G"], unit="mag", digits=3),
        _row("BP−RP", bp_rp, unit="mag", digits=3),
        _row("J, H, Ks", ", ".join("%.2f" % v for v in (data["J"], data["H"],
                                                        data["K"]))
             if None not in (data["J"], data["H"], data["K"]) else None,
             unit="mag"),
        _row("W1, W2", "%.2f, %.2f" % (data["W1"], data["W2"])
             if w1_w2 is not None else None, unit="mag",
             how="AllWISE, %s″" % wise["sep"] if wise else ""),
        _row("M_G", data.get("M_G"), unit="mag"),
        _row("M_Ks", data.get("M_Ks"), unit="mag"),
        _row("RUWE", (gaia or {}).get("ruwe"), how="Gaia DR3")) if r]))
    groups.append(dict(key="derived", mode=mode, rows=derived, notes=notes))
    published = []
    host = (nasa or {}).get("host") or {}
    teff_m, n_teff = _median([_num(r["teff"]) for r in star["fe_h"]])
    feh_m, n_feh = _median([_num(r["fe_h"]) for r in star["fe_h"]])
    logg_m, n_logg = _median([_num(r["log_g"]) for r in star["fe_h"]])
    vsini_m, n_vsini = _median(star["vsini"])
    for label, key, value, n, unit, digits in (
            ("Teff", "teff", teff_m, n_teff, "K", 0),
            ("[Fe/H]", "met", feh_m, n_feh, "dex", 2),
            ("log g", "logg", logg_m, n_logg, "cgs", 2),
            ("v sin i", "vsin", vsini_m, n_vsini, "km/s", 1)):
        if value is not None:
            published.append(_row(label, value, None, unit, digits=digits,
                                  how="SIMBAD", note=_note("star_median", n)
                                  if n > 1 else None))
        if key in host:
            published.append(_row(label, host[key]["value"], None, unit,
                                  digits=digits, how="NASA",
                                  ref=host[key]["ref"]))
    for label, key, unit, digits in (("R★", "rad", "R☉", 3),
                                     ("M★", "mass", "M☉", 3),
                                     ("age", "age", "Gyr", 1),
                                     ("P_rot", "rotp", "d", 1)):
        if key in host:
            published.append(_row(label, host[key]["value"], None, unit,
                                  digits=digits, how="NASA",
                                  ref=host[key]["ref"]))
    groups.append(dict(key="published", rows=[r for r in published if r]))

    if nasa is not None:
        planets = dict(source="nasa", rows=nasa["planets"])
    else:
        rows = []
        if fallback_planets is not None:
            try:
                rows = [dict(name=p["name"], period=p.get("period"),
                             k=p.get("k"), discovery=dict(
                                 text=p["reference"], url=None)
                             if p.get("reference") else None)
                        for p in fallback_planets().get("planets") or []]
            except Exception:                             # noqa: BLE001
                rows = []
        planets = dict(source="lbl", rows=rows)
    server = star["server"]
    return dict(
        state="done", version=VERSION, asked=list(names), main=star["main"],
        title=title(star["main"]), asked_as=spelled(star["asked"],
                                                    star["ids"]),
        simbad="https://%s/simbad/sim-id?Ident=%s" % (
            server, urllib.parse.quote(star["main"])),
        sptype=star["sptype"], otype=star["otype"],
        aliases=_aliases(star["main"], star["ids"], star["asked"]),
        distance=distance, teff=next((r["value"] for r in derived
                                      if r["label"] == "Teff"), None),
        groups=groups, planets=planets,
        tellurics=tellurics(star["ra"], star["dec"], rv,
                            (today or datetime.date.today()).year),
        sources=dict(simbad=server, gaia=(gaia or {}).get("from"),
                     wise=(wise or {}).get("source"),
                     nasa=nasa is not None))


def _read(data_dir):
    try:
        with open(os.path.join(data_dir, CACHE), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def summary(data_dir, key, names, refresh=False, fallback_planets=None):
    """The star of a table, from the folder's record when it has it (less
    than KEEP_DAYS old, of this VERSION, and not asked again), from the
    services otherwise; an answer that failed is not kept."""
    with _LOCK:
        known = _read(data_dir).get(key)
    if known and not refresh and known.get("version") == VERSION:
        try:
            when = datetime.datetime.fromisoformat(known["when"])
            if (datetime.datetime.now() - when).days < KEEP_DAYS:
                return known
        except (KeyError, ValueError):
            pass
    answer = summarise(names, fallback_planets)
    answer["when"] = datetime.datetime.now().isoformat(timespec="seconds")
    answer["key"] = key
    if answer["state"] == "failed":
        return answer
    with _LOCK:
        everything = _read(data_dir)
        everything[key] = answer
        path = os.path.join(data_dir, CACHE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", "w", encoding="utf-8") as handle:
            json.dump(everything, handle, indent=1)
        os.replace(path + ".part", path)
    return answer
