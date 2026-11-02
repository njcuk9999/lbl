"""SIMBAD identity, object type, spectral type and v sin i of the template
candidates: the ten objects with the most files in their template in each
100 K bin of 3000-4200 K, for NIRPS_HE and SPIROU.

Reads template_catalog.json (one row per APERO template: inst, file, name,
drsobjn, teff, teffsrc, ntemp, tree, version); writes candidates_simbad.json:
"cands" (per object, its candidate rows, each with its bin) and "simbad"
(per object, the name SIMBAD resolved, main_id, otype, sp_type and a v sin i
summary; None when no form of the name resolved).

    python vet_simbad.py
"""
import json, re
import numpy as np
from astroquery.simbad import Simbad

# the templates with a Teff, filed under APERO's own object name (DRSOBJN)
rows = [r for r in json.load(open("template_catalog.json"))
        if r["drsobjn"] and r["name"] == r["drsobjn"] and r["teff"] is not None]
#: the IAU constellation abbreviations, in upper case as APERO writes them,
#:    to recognise variable-star names (AU_MIC, YZ_CMI)
CONSTELLATIONS = ("AND ANT APS AQR AQL ARA ARI AUR BOO CAE CAM CNC CVN CMA CMI CAP CAR CAS CEN CEP CET CHA CIR COL "
                  "COM CRA CRB CRV CRT CRU CYG DEL DOR DRA EQU ERI FOR GEM GRU HER HOR HYA HYI IND LAC LEO LMI LEP "
                  "LIB LUP LYN LYR MEN MIC MON MUS NOR OCT OPH ORI PAV PEG PER PHE PIC PSC PSA PUP PYX RET SGE SGR "
                  "SCO SCL SCT SER SEX TAU TEL TRI TRA TUC UMA UMI VEL VIR VOL VUL").split()


def variants(n):
    """The forms of an APERO object name to try in SIMBAD, in order: APERO
    writes '-' as M, '+' as P and spaces as '_' (GJ1214, TOIM270, L_98M59)"""
    s = n.replace("_", " ")
    s = re.sub(r"(?<=\d)M(?=\d)", "-", s)
    s = re.sub(r"(?<=\d)P(?=\d)", "+", s)
    s = re.sub(r"^GL(?=\d)", "Gl ", s)
    s = re.sub(r"^GJ(?=\d)", "GJ ", s)
    s = re.sub(r"^TOIM?(?=\d)", "TOI-", s)
    out = [s]
    # a variable-star name: the constellation in SIMBAD's capitalisation
    m = re.match(r"^([A-Z]{1,2}|V\d+)\s?([A-Z]{3})$", s)
    if m and m.group(2) in CONSTELLATIONS:
        out.append("%s %s" % (m.group(1), m.group(2).capitalize()))
    # common names SIMBAD only knows with their punctuation
    if s.startswith("TEEGARDENS"):
        out.append("Teegarden's Star")
    if s == "PROXIMA":
        out.append("Proxima Cen")
    return out


# the candidates: per instrument and 100 K bin, each object once (its
#   template with the most spectra), the ten with the most spectra kept
cands = {}
for inst in ("NIRPS_HE", "SPIROU"):
    for lo in range(3000, 4200, 100):
        best = {}
        for r in rows:
            if r["inst"] == inst and lo <= float(r["teff"]) < lo + 100:
                if r["name"] not in best or (r["ntemp"] or 0) > (best[r["name"]]["ntemp"] or 0):
                    best[r["name"]] = r
        for r in sorted(best.values(), key=lambda r: -(r["ntemp"] or 0))[:10]:
            cands.setdefault(r["name"], []).append(dict(r, bin=lo))

s = Simbad()
s.add_votable_fields("otype", "sp_type", "mesrot")
info = {}
for name in cands:
    # the first form of the name that SIMBAD resolves
    for v in variants(name):
        try:
            t = s.query_objects([v])
        except Exception:
            t = None
        if t is not None and len(t) and str(t["main_id"][0]).strip():
            # one row per mesRot measurement: their v sin i summarised
            vs = np.array([float(x) for x in t["mesrot.vsini"] if np.isfinite(float(x))]) if len(t) else np.array([])
            info[name] = dict(query=v, main_id=str(t["main_id"][0]).strip(), otype=str(t["otype"][0]),
                              sp_type=str(t["sp_type"][0]),
                              vsini_median=float(np.median(vs)) if len(vs) else None,
                              vsini_min=float(vs.min()) if len(vs) else None,
                              vsini_n=int(len(vs)),
                              vsini_refs=sorted(set(str(b) for b in t["mesrot.bibcode"] if str(b).strip())))
            break
    else:
        info[name] = None
json.dump(dict(cands=cands, simbad=info), open("candidates_simbad.json", "w"), indent=1)
print(len(cands), "candidates,", sum(v is None for v in info.values()), "unresolved:",
      [k for k, v in info.items() if v is None])
