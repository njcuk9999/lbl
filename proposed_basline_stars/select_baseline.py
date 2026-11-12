"""Proposed baseline stars for frozen PCA2D bases: the 4 best stars of SPIRou
and of NIRPS HE in each SNR bin, from the survey of APERO's t.fits on rali
(survey_rali.py).

Per star (APERO's object name):
- n: its exposures with a BERV and an SNR; nights: distinct nights, from
  local noon to local noon (LONGITUDE);
- span: max - min BERV [km/s]; a star needs span > MIN_SPAN to be considered;
- gini: the Gini coefficient of its BERV histogram in BIN_KMS bins from its
  lowest to its highest BERV: 0 when every bin holds as many exposures,
  towards 1 when they crowd into a few (an empty bin inside the span counts);
- snr: the median of its exposures' SNR (each the median of the file's
  EXTSNxxx, as lbl_gui shows it), which places the star in an SNR bin;
- score: n x (1 - gini), the number of exposures discounted by how unevenly
  they cover the BERV;
- hot stars (Teff >= HOT_TEFF or type O, B, A in APERO's astrometric
  database) left out: telluric standards, whose residuals are not those of
  the cool stars the bases are for.

Writes, in this folder: candidates_<INSTRUMENT>.csv (every star with its
numbers and why it is out), proposed_baseline_stars.csv (the 4 best per
instrument and bin) and berv_<INSTRUMENT>.pdf (their BERV histograms).

    python select_baseline.py <exposures.csv>

On rali, beside survey_rali.py's output:

    cd ~/proposed_basline_stars
    nohup ~/miniconda3/envs/lbl-pca2d/bin/python survey_rali.py > exposures.csv 2> survey.log &
    ~/miniconda3/envs/lbl-pca2d/bin/python select_baseline.py exposures.csv
"""
import csv
import glob
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
#: the SNR bins, [low, high)
BINS = (("20-40", 20.0, 40.0), ("40-80", 40.0, 80.0), (">80", 80.0, np.inf))
MIN_SPAN = 30.0
BIN_KMS = 2.0
HOT_TEFF = 7000.0
PER_BIN = 4
#: the observatories' longitudes [deg, east positive]: a night runs from
#:   local noon to local noon (at La Silla, 0 h UT falls in the night)
LONGITUDE = {"SPIROU": -155.47, "NIRPS_HE": -70.73}
#: APERO's astrometric database: $APERO_ASTROMETRICS, else a copy beside this
#:   script (apero_astrometrics/, as on rali), else the one lbl_gui keeps
APERO_DB = (os.environ.get("APERO_ASTROMETRICS")
            or next((p for p in (os.path.join(HERE, "apero_astrometrics"),)
                     if os.path.isdir(p)), None)
            or os.path.expanduser("~/.cache/lbl/gui/apero_astrometrics"))


def gini(counts):
    """The Gini coefficient of non-negative counts: 0 equal, near 1
    concentrated."""
    x = np.asarray(counts, dtype=float)
    # every exposure in one bin: as concentrated as can be. Not NaN: a NaN
    #   in the sort key leaves the whole ranking in disorder
    if x.size < 2 or x.sum() <= 0:
        return 1.0
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff / (2.0 * x.size * x.sum()))


def berv_gini(berv):
    lo = np.floor(berv.min() / BIN_KMS) * BIN_KMS
    hi = np.ceil(berv.max() / BIN_KMS) * BIN_KMS
    edges = np.arange(lo, hi + BIN_KMS, BIN_KMS)
    counts, _ = np.histogram(berv, bins=edges)
    return gini(counts)


def apero_db():
    """{APERO name: (Teff, spectral type)} from APERO's astrometric database
    (the copy lbl_gui keeps)."""
    out = {}
    for path in glob.glob(os.path.join(APERO_DB, "**", "*.yaml"),
                          recursive=True):
        try:
            with open(path, encoding="utf-8") as handle:
                entry = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            continue
        name = str(entry.get("APERO_NAME") or "")
        teff = (entry.get("TEFF") or {}).get("value")
        spt = (entry.get("SPT") or {}).get("value")
        if name:
            out[name] = (float(teff) if teff is not None else np.nan,
                         str(spt) if spt else "")
    return out


def read(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("error"):
                continue
            try:
                berv, snr = float(row["berv"]), float(row["snr"])
                mjd = float(row["mjdmid"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(berv) and np.isfinite(snr)):
                continue
            key = (row["instrument"], row["object"])
            rows.setdefault(key, []).append((mjd, berv, snr))
    return rows


def main(path):
    db = apero_db()
    stars = []
    for (inst, obj), values in sorted(read(path).items()):
        mjd, berv, snr = (np.array(v) for v in zip(*values))
        teff, spt = db.get(obj, (np.nan, ""))
        span = float(berv.max() - berv.min())
        star = dict(instrument=inst, object=obj, n=len(berv),
                    nights=len(np.unique(np.floor(
                        mjd + LONGITUDE.get(inst, 0.0) / 360.0 - 0.5))),
                    berv_min=round(float(berv.min()), 2),
                    berv_max=round(float(berv.max()), 2),
                    span=round(span, 2), gini=round(berv_gini(berv), 3),
                    snr=round(float(np.median(snr)), 1), teff=teff, spt=spt)
        star["score"] = round(star["n"] * (1.0 - star["gini"]), 1)
        star["snr_bin"] = next((name for name, lo, hi in BINS
                                if lo <= star["snr"] < hi), "")
        why = []
        if span <= MIN_SPAN:
            why.append("BERV span %.1f <= %.0f km/s" % (span, MIN_SPAN))
        if (np.isfinite(teff) and teff >= HOT_TEFF) or spt[:1] in ("O", "B", "A"):
            why.append("hot star (Teff %s, %s)" % (
                "%.0f" % teff if np.isfinite(teff) else "?", spt or "?"))
        if not star["snr_bin"]:
            why.append("median SNR %.1f below 20" % star["snr"])
        star["out"] = "; ".join(why)
        star["_berv"] = berv
        stars.append(star)
    columns = ("instrument", "object", "snr_bin", "snr", "n", "nights",
               "berv_min", "berv_max", "span", "gini", "score", "teff", "spt",
               "out")
    proposed = []
    for inst in sorted({s["instrument"] for s in stars}):
        mine = sorted((s for s in stars if s["instrument"] == inst),
                      key=lambda s: -s["score"] if np.isfinite(s["score"])
                      else np.inf)
        with open(os.path.join(HERE, "candidates_%s.csv" % inst), "w",
                  newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(mine)
        for name, _, _ in BINS:
            chosen = [s for s in mine if not s["out"]
                      and s["snr_bin"] == name][:PER_BIN]
            for rank, s in enumerate(chosen, 1):
                proposed.append(dict(s, rank=rank))
            print("%-9s SNR %-6s %d eligible: %s" % (
                inst, name, sum(1 for s in mine if not s["out"]
                                and s["snr_bin"] == name),
                ", ".join("%s (n %d, span %.0f, gini %.2f)" % (
                    s["object"], s["n"], s["span"], s["gini"])
                    for s in chosen)))
    with open(os.path.join(HERE, "proposed_baseline_stars.csv"), "w",
              newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank",) + columns,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(proposed)
    plot(proposed)


def plot(proposed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for inst in sorted({s["instrument"] for s in proposed}):
        fig, axes = plt.subplots(len(BINS), PER_BIN,
                                 figsize=(3.0 * PER_BIN, 2.3 * len(BINS)),
                                 squeeze=False, sharex=True)
        for row, (name, _, _) in enumerate(BINS):
            chosen = [s for s in proposed if s["instrument"] == inst
                      and s["snr_bin"] == name]
            for col in range(PER_BIN):
                ax = axes[row, col]
                if col >= len(chosen):
                    ax.set_axis_off()
                    continue
                s = chosen[col]
                ax.hist(s["_berv"], bins=np.arange(-32, 33, BIN_KMS),
                        color="#1f77b4")
                ax.set_title("%s  (SNR %s)\nn %d, span %.0f km/s, Gini %.2f"
                             % (s["object"], name, s["n"], s["span"],
                                s["gini"]), fontsize=8)
                ax.tick_params(labelsize=7)
                if row == len(BINS) - 1:
                    ax.set_xlabel("BERV (km/s)", fontsize=8)
        fig.suptitle("%s: proposed baseline stars, BERV coverage" % inst,
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "berv_%s.pdf" % inst))
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
