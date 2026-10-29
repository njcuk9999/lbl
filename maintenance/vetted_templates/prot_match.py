"""Published rotation periods of the template candidates (the five with the
most spectra in each bin), matched by position (SIMBAD's J2000 coordinates,
nearest within 30 arcsec) in Shan+2024 (CARMENES, best periods), Newton+
2016 and 2018 (MEarth, photometric), Diez Alonso+ 2019 (CARMENES,
photometric), Lu+ 2022 (ZTF, photometric) and Lambier+ 2025 (TESS).

Reads candidates_simbad.json (vet_simbad.py) and vsini_rows.json
(vsini_rows.py), whose objects are the candidates; writes prot_match.json:
"coords" (per object, SIMBAD's RA, Dec [deg]) and "prot" (per object and
catalog, P [d] or None, R [R_sun] when the catalog gives one, the catalog's
note or flag, and the match separation [arcsec]). select_vetted_stars.py
reads the catalogs in its own order of preference.

    python prot_match.py
"""
import json
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

sim = json.load(open("candidates_simbad.json"))["simbad"]
rows_v = json.load(open("vsini_rows.json"))
# the candidates SIMBAD resolved, and their coordinates
names = [n for n in rows_v if sim.get(n)]
s = Simbad()
t = s.query_objects([sim[n]["query"] for n in names])
coords = {}
for n, row in zip(names, t):
    coords[n] = (float(row["ra"]), float(row["dec"]))
# every row of each catalog, with VizieR's computed J2000 position
v = Vizier(row_limit=-1, columns=["**", "_RAJ2000", "_DEJ2000"])
# per catalog: (VizieR table, period column, radius column or None, note
#   column or None); ZTF22 is matched for comparison only, the selection
#   (select_vetted_stars.ORDER) does not use it
cats = {
    "Shan24": ("J/A+A/684/A9/tablec1", "Pbest", "Rstar", "r_Pbest"),
    "MEarthN": ("J/ApJ/821/93/table1", "Per", "Rad", "Type"),
    "MEarthS": ("J/AJ/156/217/table1", "Per", "R*", "Type"),
    "DiezAlonso19": ("J/A+A/621/A126/photom", "Per", None, "Survey"),
    "ZTF22": ("J/AJ/164/251/table1", "Prot", None, None),
    "TESS25": ("J/AJ/170/168/table2", "Per", None, "Status"),
}
# Diez Alonso+ 2019: the periods are in rotper, by Karmn; photom has the positions
rp = v.get_catalogs("J/A+A/621/A126/rotper")[0]
rotper = {}
for row in rp:
    for col in ("Prot", "ProtLit"):
        try:
            val = float(row[col])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0:
            rotper.setdefault(str(row["Karmn"]).strip(), val)
cand = SkyCoord([coords[n][0] for n in names] * u.deg, [coords[n][1] for n in names] * u.deg)
out = {n: {} for n in names}
# each candidate matched to its nearest catalog row, kept within 30 arcsec
for key, (table, pcol, rcol, ncol) in cats.items():
    try:
        tab = v.get_catalogs(table)[0]
    except Exception as exc:
        print(key, "failed", exc)
        continue
    cc = SkyCoord(np.array(tab["_RAJ2000"], float) * u.deg, np.array(tab["_DEJ2000"], float) * u.deg)
    idx, sep, _ = cand.match_to_catalog_sky(cc)
    for n, i, d in zip(names, idx, sep.arcsec):
        if d > 30:
            continue
        p = tab[pcol][i]
        if key == "DiezAlonso19":
            p = rotper.get(str(tab["Karmn"][i]).strip(), np.nan)
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = np.nan
        # in the catalog but without a period: recorded, with its note
        if np.ma.is_masked(p) or not np.isfinite(p) or p <= 0:
            out[n][key] = dict(P=None, note=str(tab[ncol][i]) if ncol else "", sep=round(float(d), 1))
            continue
        rad = None
        if rcol and not np.ma.is_masked(tab[rcol][i]):
            rad = float(tab[rcol][i])
        out[n][key] = dict(P=float(p), R=rad, note=str(tab[ncol][i]) if ncol else "", sep=round(float(d), 1))
    print(key, len(tab), "rows,", sum(key in out[n] for n in names), "matched")
json.dump(dict(coords=coords, prot=out), open("prot_match.json", "w"), indent=1)
