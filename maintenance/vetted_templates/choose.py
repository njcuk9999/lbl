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
choice (fetch its template, run linewidth.py again)."""
import json
import numpy as np
d = json.load(open("candidates_simbad.json"))
cands, sim = d["cands"], d["simbad"]
rows_v = json.load(open("vsini_rows.json"))
prot = json.load(open("prot_match.json"))["prot"]
YOUNG = {"TT*", "Or*", "Y*O", "Y*?", "Ae*"}
LOW_RES = ("2020AJ....160..120J", "2022ApJS..259...35A", "2005yCat.3244", "1970CoKwa")
V_MAX = 2.0          # v_eq from a published period
VSINI_MAX = 4.0      # median high-resolution v sin i, without a period
WIDTH_EXCESS = 1.0   # km/s of autocorrelation FWHM over slow rotators, without a period
widths = json.load(open("linewidth.json"))
ORDER = ("Shan24", "MEarthN", "MEarthS", "DiezAlonso19", "TESS25")
SAME_STAR = {"GL905", "GL617B", "GL382", "V374PEG", "MHO_3"}


def radius(teff):
    x = teff / 3500.0
    return 10.5440 - 33.7546 * x + 35.1909 * x ** 2 - 11.5928 * x ** 3


def rotation(name, teff):
    for key in ORDER:
        found = (prot.get(name) or {}).get(key)
        if found and found.get("P"):
            r = found.get("R") or radius(teff)
            return dict(P=found["P"], R=r, veq=50.58 * r / found["P"], ref=key)
    return None


def vsini(name):
    vals = [m["v"] for m in rows_v.get(name, []) if not m["bib"].startswith(LOW_RES) and m["v"] >= 0]
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)


# the Teff of every measured template (catalog), to compare widths
catalog = {("%s_%s" % (r["inst"], r["file"])): float(r["teff"])
           for r in json.load(open("template_catalog.json")) if r["teff"] is not None}
teff_of = {key: catalog[key] for key in widths if key in catalog}


def slow(name, teff):
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
        rows = sorted([r for lst in cands.values() for r in lst if r["inst"] == inst and r["bin"] == lo
                       and r["name"] not in SAME_STAR and (r["ntemp"] or 0) > 0], key=lambda r: -(r["ntemp"] or 0))
        pick = None
        for r in rows:
            s = sim.get(r["name"]) or {}
            rot = rotation(r["name"], float(r["teff"]))
            vs, nvs = vsini(r["name"])
            why = None
            if s.get("otype") in YOUNG:
                why = "young star (SIMBAD %s)" % s["otype"]
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
