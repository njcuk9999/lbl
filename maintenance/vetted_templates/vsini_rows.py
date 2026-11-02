"""Per-measurement v sin i of the top five candidates of each bin, from
SIMBAD's mesRot, with their bibcodes and upper-limit flags.

Reads candidates_simbad.json (vet_simbad.py); writes vsini_rows.json: per
object, a list of dict(v [km/s], up (the upper-limit flag), bib (bibcode)).
select_vetted_stars.py takes their median, leaving out the low-resolution
sources.

    python vsini_rows.py
"""
import json
import numpy as np
from astroquery.simbad import Simbad
d = json.load(open("candidates_simbad.json"))
cands, sim = d["cands"], d["simbad"]
# the five candidates with the most spectra in each bin and instrument
names = set()
for inst in ("NIRPS_HE", "SPIROU"):
    for lo in range(3000, 4200, 100):
        rows = sorted([r for lst in cands.values() for r in lst if r["inst"] == inst and r["bin"] == lo],
                      key=lambda r: -(r["ntemp"] or 0))[:5]
        names |= {r["name"] for r in rows}
s = Simbad()
s.add_votable_fields("mesrot")
out = {}
for name in sorted(names):
    info = sim.get(name)
    # not resolved by SIMBAD (vet_simbad.py): nothing to ask
    if not info:
        continue
    # one row per mesRot measurement, queried by the name that resolved
    t = s.query_objects([info["query"]])
    meas = []
    for row in t:
        v = row["mesrot.vsini"]
        if np.ma.is_masked(v):
            continue
        meas.append(dict(v=float(v), up=str(row["mesrot.upvsini"]).strip(), bib=str(row["mesrot.bibcode"]).strip()))
    out[name] = meas
json.dump(out, open("vsini_rows.json", "w"), indent=1)
print(len(out))
