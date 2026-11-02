"""The vetted templates: in each 100 K bin of 3000-4200 K and for each
instrument, the star whose APERO template holds the most spectra, leaving
out
  - young accreting stars (SIMBAD TT*, Or*, Y*O, Y*?, Ae*);
  - fast rotators: v_eq = 2 pi R / P_rot above 2 km/s, from PUBLISHED
    ROTATION PERIODS
    (prot_match.py: Shan+ 2024 best periods first, then MEarth, Diez Alonso+
    2019, TESS), R from the same catalog or from Teff (Mann+ 2015, solar
    metallicity). v_eq is an upper bound on v sin i;
  - without a period, the median of SIMBAD's high-resolution v sin i above
    4 km/s (APOGEE, R ~ 22 500, and the old compilations left out: they put
    GJ 411 at 50 to 58 km/s).
  - without a period, a template whose autocorrelation (linewidth.py) is
    more than 1 km/s wider than that of slow rotators of the same
    instrument at its Teff: its lines are broadened. The width grows with
    Teff (9 km/s at 3300 K, 15 at 4050 K for slow rotators), so the slow
    rotators' widths (a published period with v_eq under 2 km/s, or a v sin
    i of 2 km/s or less) are interpolated in Teff; beyond the last of them
    there is no verdict, and the star is kept.
A candidate without a period whose width is not measured yet stops the
choice (fetch its template, run linewidth.py again).

Reads, in the current folder: template_catalog.json (one row per APERO
template: instrument, file, object, Teff and its source, number of spectra,
APERO tree and version), candidates_simbad.json (vet_simbad.py),
vsini_rows.json (vsini_rows.py), prot_match.json (prot_match.py) and
linewidth.json (linewidth.py). Writes selection.json: the stars chosen
("chosen") and every candidate passed over with the reason ("dropped").

    python select_vetted_stars.py
"""
import json

import numpy as np
from astropy import constants
from astropy import units as u

# the candidates (the ten with the most spectra in each bin) and their
#   SIMBAD identity, object type and v sin i (vet_simbad.py)
d = json.load(open("candidates_simbad.json"))
cands, sim = d["cands"], d["simbad"]
# every v sin i measurement of SIMBAD's mesRot, with its bibcode
rows_v = json.load(open("vsini_rows.json"))
# the published rotation periods, by catalog (prot_match.py)
prot = json.load(open("prot_match.json"))["prot"]
#: SIMBAD object types of young (accreting) stars, never chosen
YOUNG = {"TT*", "Or*", "Y*O", "Y*?", "Ae*"}
#: the v sin i sources left out, by bibcode prefix: APOGEE DR16 and DR17
#:    (Jonsson+ 2020, Abdurro'uf+ 2022), whose R ~ 22 500 cannot resolve a
#:    slow M dwarf's lines, and two older compilations that give fast
#:    rotation to known slow rotators
LOW_RES = ("2020AJ....160..120J", "2022ApJS..259...35A", "2005yCat.3244", "1970CoKwa")
V_MAX = 2.0          # v_eq from a published period
VSINI_MAX = 4.0      # median high-resolution v sin i, without a period
WIDTH_EXCESS = 1.0   # km/s of autocorrelation FWHM over slow rotators, without a period
# the H-band autocorrelation FWHM of each template fetched (linewidth.py)
widths = json.load(open("linewidth.json"))
#: the rotation catalogs in order of preference (keys of prot_match.py);
#:    the first that has a period for a star gives it
ORDER = ("Shan24", "MEarthN", "MEarthS", "DiezAlonso19", "TESS25")
#: objects whose template is a second name of another candidate's star
#:    (or of a star already in another bin), not candidates of their own
SAME_STAR = {"GL905", "GL617B", "GL382", "V374PEG", "MHO_3"}
#: 2 pi R_sun / 1 day [km/s]: v_eq = KMS_PER_RSUN_PER_DAY * R [R_sun] / P [d]
KMS_PER_RSUN_PER_DAY = (2 * np.pi * constants.R_sun / u.day).to(u.km / u.s).value


def radius(teff):
    """The radius [R_sun] of an M dwarf from its Teff [K], ignoring its
    metallicity: R = a + b X + c X^2 + d X^3 with X = Teff / 3500 K.

    Mann et al. 2015, ApJ 804, 64, Table 1 (R_* against Teff/3500, Eq. 4),
    as reprinted in the erratum, Mann et al. 2016, ApJ 819, 87:
    https://doi.org/10.1088/0004-637X/804/1/64
    https://doi.org/10.3847/0004-637X/819/1/87
    Valid for 2700 < Teff < 4100 K (K7 to M7, 0.1 to 0.7 R_sun); scatter
    13.4% in R (reduced chi^2 2.35), the [Fe/H] spread of the sample being
    most of it. Used only for the stars whose rotation catalog gives no
    radius."""
    x = teff / 3500.0
    return 10.5440 - 33.7546 * x + 35.1909 * x ** 2 - 11.5928 * x ** 3


def rotation(name, teff):
    """The period [d], radius [R_sun], v_eq = 2 pi R / P [km/s] and catalog
    of the first catalog of ORDER with a period for this star; None
    without any"""
    for key in ORDER:
        found = (prot.get(name) or {}).get(key)
        if found and found.get("P"):
            r = found.get("R") or radius(teff)
            return dict(P=found["P"], R=r, veq=KMS_PER_RSUN_PER_DAY * r / found["P"], ref=key)
    return None


def vsini(name):
    """(median v sin i [km/s], number of measurements) of SIMBAD's
    high-resolution measurements of this star; (None, 0) without any"""
    vals = [m["v"] for m in rows_v.get(name, []) if not m["bib"].startswith(LOW_RES) and m["v"] >= 0]
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)


# the Teff of every measured template (catalog), to compare widths
catalog = {("%s_%s" % (r["inst"], r["file"])): float(r["teff"])
           for r in json.load(open("template_catalog.json")) if r["teff"] is not None}
teff_of = {key: catalog[key] for key in widths if key in catalog}


def slow(name, teff):
    """A slow rotator, whose width is a reference: v_eq under V_MAX from a
    published period or, without one, a v sin i of 2 km/s or less"""
    rot = rotation(name, teff)
    if rot:
        return rot["veq"] < V_MAX
    vs, _ = vsini(name)
    return vs is not None and vs <= 2.0


def reference_width(inst, teff):
    """The slow rotators' autocorrelation width at this Teff, interpolated
    between the nearest below and above; None outside them."""
    refs = []
    for key, w in widths.items():
        if not key.startswith(inst + "_") or key not in teff_of:
            continue
        # the object name, from <INSTRUMENT>_Template_s1dv_<OBJECT>_sc1d_v_file...
        name = key[len(inst) + len("_Template_s1dv_"):].split("_sc1d_v_file")[0]
        if slow(name, teff_of[key]):
            refs.append((teff_of[key], w["fwhm_acf"]))
    refs.sort()
    below = [x for x in refs if x[0] <= teff]
    above = [x for x in refs if x[0] >= teff]
    if not below or not above:
        return None
    (t0, w0), (t1, w1) = below[-1], above[0]
    return w0 if t1 == t0 else w0 + (w1 - w0) * (teff - t0) / (t1 - t0)


chosen, dropped = [], []
for inst in ("NIRPS_HE", "SPIROU"):
    for lo in range(3000, 4200, 100):
        # the candidates of this bin, the most spectra first
        rows = sorted([r for lst in cands.values() for r in lst if r["inst"] == inst and r["bin"] == lo
                       and r["name"] not in SAME_STAR and (r["ntemp"] or 0) > 0], key=lambda r: -(r["ntemp"] or 0))
        pick = None
        # the first candidate that passes every cut is the bin's star; each
        #   one passed over is recorded with its reason
        for r in rows:
            s = sim.get(r["name"]) or {}
            rot = rotation(r["name"], float(r["teff"]))
            vs, nvs = vsini(r["name"])
            why = None
            if s.get("otype") in YOUNG:
                why = "young star (SIMBAD %s)" % s["otype"]
            # a published period decides alone when there is one
            elif rot and rot["veq"] > V_MAX:
                why = "P_rot %.2f d (%s), v_eq %.1f km/s" % (rot["P"], rot["ref"], rot["veq"])
            elif not rot and vs is not None and vs > VSINI_MAX:
                why = "no period; v sin i %.1f km/s (median of %d)" % (vs, nvs)
            elif not rot:
                mine = widths.get("%s_%s" % (inst, r["file"]))
                if mine is None:
                    raise SystemExit("measure the width of %s %s first" % (inst, r["file"]))
                ref = reference_width(inst, float(r["teff"]))
                if ref is not None and mine["fwhm_acf"] - ref > WIDTH_EXCESS:
                    why = ("no period; autocorrelation %.1f km/s wider than slow rotators at %d K"
                           % (mine["fwhm_acf"] - ref, float(r["teff"])))
            if why:
                dropped.append(dict(instrument=inst, bin=lo, object=r["name"], nfiles=r["ntemp"], why=why))
                continue
            pick = dict(r, rot=rot, vsini=vs, vsini_n=nvs)
            break
        if pick is None:
            dropped.append(dict(instrument=inst, bin=lo, object=None, why="no star left in the bin"))
            continue
        s = sim.get(pick["name"]) or {}
        rot = pick["rot"]
        # everything make_vetted.py writes in the depot file's header
        chosen.append(dict(instrument=inst, bin=lo, object=pick["name"], teff=pick["teff"], teff_ref=pick["teffsrc"],
                           nfiles=pick["ntemp"], tree=pick["tree"], file=pick["file"], apero=pick["version"],
                           simbad=s.get("main_id"), otype=s.get("otype"), sp_type=s.get("sp_type"),
                           prot=rot["P"] if rot else None, prot_ref=rot["ref"] if rot else None,
                           veq=rot["veq"] if rot else None, vsini=pick["vsini"], vsini_n=pick["vsini_n"],
                           fwhm_acf=(widths.get("%s_%s" % (inst, pick["file"])) or {}).get("fwhm_acf")))
json.dump(dict(chosen=chosen, dropped=dropped), open("selection.json", "w"), indent=1)
for c in chosen:
    print("%-8s %d  %-15s %5.0f K  n=%-5d %-22s P=%-7s v_eq=%-5s vsini=%s" % (
        c["instrument"], c["bin"], c["object"], c["teff"], c["nfiles"], (c["simbad"] or "")[:22],
        "%.1f" % c["prot"] if c["prot"] else "-", "%.1f" % c["veq"] if c["veq"] else "-",
        "%.1f" % c["vsini"] if c["vsini"] is not None else "-"))
print("passed over:")
for r in dropped:
    print("  ", r["instrument"], r["bin"], r["object"], r.get("nfiles"), r["why"])
