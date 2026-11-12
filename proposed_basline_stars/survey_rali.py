"""The exposures of every star APERO reduced for SPIRou (spirou_offline) and
NIRPS HE (nirps_he_offline) on rali, read from the headers of their t.fits:
one CSV row per exposure on stdout, progress on stderr. The data are only
read: what it writes is the CSV, in the home folder it is run from
(~/proposed_basline_stars on rali), never beside APERO's files.

Run on rali, detached (rali's lbl-pca2d environment has astropy):

    cd ~/proposed_basline_stars
    nohup ~/miniconda3/envs/lbl-pca2d/bin/python survey_rali.py > exposures.csv 2> survey.log &

Columns: instrument, object (APERO's folder name), file, mjdmid, berv
[km/s, BERV of the science extension], snr (the median of its EXTSNxxx, the
SNR lbl_gui shows for an APERO file), snr_min, n_orders (EXTSN keys found),
drsobjn, exptime, error. Objects with fewer than MIN_FILES t.fits are left
out (they cannot be baseline stars).
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
from astropy.io import fits

ROOTS = {"SPIROU": "/cosmos99/spirou/apero-data/spirou_offline/objects",
         "NIRPS_HE": "/cosmos99/nirps/apero-data/nirps_he_offline/objects"}
MIN_FILES = 20
WORKERS = 12
COLUMNS = ("instrument", "object", "file", "mjdmid", "berv", "snr", "snr_min",
           "n_orders", "drsobjn", "exptime", "error")


def one(task):
    inst, obj, path = task
    row = dict(instrument=inst, object=obj, file=os.path.basename(path))
    try:
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
            primary = hdul[0].header
            science = hdul[1].header
            snr = []
            for key, value in science.items():
                if str(key).startswith("EXTSN"):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value) and value > 0:
                        snr.append(value)
            row.update(
                mjdmid=science.get("MJDMID", primary.get("MJDMID")),
                berv=science.get("BERV", primary.get("BERV")),
                snr=round(float(np.median(snr)), 2) if snr else None,
                snr_min=round(float(np.min(snr)), 2) if snr else None,
                n_orders=len(snr),
                drsobjn=science.get("DRSOBJN", primary.get("DRSOBJN")),
                exptime=primary.get("EXPTIME", science.get("EXPTIME")))
    except Exception as exc:                                  # noqa: BLE001
        row["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:80])
    return row


def tasks():
    out = []
    for inst, root in ROOTS.items():
        for entry in sorted(os.scandir(root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            files = sorted(f.path for f in os.scandir(entry.path)
                           if f.name.endswith("t.fits"))
            if len(files) >= MIN_FILES:
                out += [(inst, entry.name, f) for f in files]
    return out


def main():
    todo = tasks()
    sys.stderr.write("%d t.fits to read\n" % len(todo))
    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()
    with Pool(WORKERS) as pool:
        for k, row in enumerate(pool.imap_unordered(one, todo, chunksize=40)):
            writer.writerow(row)
            if k % 5000 == 0:
                sys.stdout.flush()
                sys.stderr.write("%d / %d\n" % (k, len(todo)))
    sys.stdout.flush()
    sys.stderr.write("done\n")


if __name__ == "__main__":
    main()
