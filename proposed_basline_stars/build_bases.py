"""Build the proposed baseline bases on rali: for each instrument and SNR
bin, one joint PCA2D fit of its 4 stars (proposed_baseline_stars.csv), whose
observer components are saved as the basis baseline_snr<bin> and kept in
DATA_DIR/models/pca2d_<INSTRUMENT>_CADC/, where another star reads them
frozen with PCA2D input_name='baseline_snr<bin>'.

DATA_SOURCE is CADC although the files are read on rali: they are APERO's
t.fits (out/), the very files CADC distributes, and LBL reads that layout
with its CADC classes, which take the BERV from the flux extension's header.
Its APERO classes are for APERO's e2dsff_tcorr files (red/, which rali's
own LBL runs read), whose BERV is in the primary header: on a t.fits, whose
primary header has none, the observer's BERV is NaN for every file (the
bundles of a night refuse it, the river plots lose their line). A basis
filed as CADC is also the one a run on CADC data takes.

    python build_bases.py [root]          # links, wraps, runners
    cd <root> && for s in run_*_baseline_*.sh; do
        nohup setsid bash $s > /dev/null 2>&1 < /dev/null & done

What it writes, all under root (~/pca2d_bases by default), never beside
APERO's files:
- <root>/<INSTRUMENT>/: an LBL DATA_DIR, its science/<OBJECT>/ made of
  symbolic links to APERO's t.fits (the exposures the selection counted:
  read without error, with a BERV and an SNR), no copy;
- <root>/wraps/<INSTRUMENT>_<basis>.py: the wraps, as the LBL window writes
  them (lbl.gui.wrapgen.render): PCA2D joint, n_star=1, the instrument's
  n_earth, LBL's own steps off (the bases are the product; the corrected
  spectra of the training stars are written by the fit as always);
- <root>/run_<INSTRUMENT>_<basis>.sh: one wrap, its basis kept once made;
  the six run side by side (a SPIRou fit is about 14 min an iteration, the
  bases one after the other took days; each SPIRou fit holds 25 to 50 GB,
  rali has 376), logs in <root>/logs/, one line per step in
  <root>/logs/status.txt.
"""
import csv
import math
import os
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from survey_rali import ROOTS                                  # noqa: E402

DATA_SOURCE = "CADC"
#: the basis of each SNR bin, by name (output_name, then input_name)
NAMES = {"20-40": "baseline_snr20-40", "40-80": "baseline_snr40-80",
         ">80": "baseline_snr80plus"}
#: the substitutions accepted on 2026-11-12 (SUGGESTIONS.md, 2 and 3): quiet
#:   M dwarfs in place of two young, very active stars (AU Mic, V1298 Tau)
#:   and of a K2 with 73 nights (eps Eri). Gl 15 A rather than Gl 880: the
#:   higher score (699 against 674) and no emission flag in its type
SWAPS = {("SPIROU", ">80"): {"AUMIC": "GJ411", "V1298TAU": "GL15A"},
         ("NIRPS_HE", ">80"): {"EPSILONERI": "GJ1"}}
#: BLAS threads of one run, and its priority: rali is shared
THREADS = 12
NICE = 10


def proposed(path):
    """{(instrument, bin): [(object, teff), ...]} in rank order, SWAPS made
    (the Teff of a substitute from candidates_<INSTRUMENT>.csv)"""
    out = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["instrument"], row["snr_bin"])
            out.setdefault(key, []).append(
                (int(row["rank"]), row["object"], row["teff"]))
    teffs = {}
    for inst in {inst for inst, _ in SWAPS}:
        with open(os.path.join(HERE, "candidates_%s.csv" % inst),
                  encoding="utf-8") as handle:
            teffs.update({(inst, row["object"]): row["teff"]
                          for row in csv.DictReader(handle)})
    plan = {}
    for key, rows in out.items():
        swap = SWAPS.get(key, {})
        plan[key] = [(swap.get(obj, obj),
                      teffs[(key[0], swap[obj])] if obj in swap else teff)
                     for _, obj, teff in sorted(rows)]
    return plan


def exposures(path, wanted):
    """{(instrument, object): [file, ...]}: the files the selection counted"""
    out = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["instrument"], row["object"])
            if key not in wanted or row.get("error"):
                continue
            try:
                ok = all(math.isfinite(float(row[k]))
                         for k in ("berv", "snr", "mjdmid"))
            except (TypeError, ValueError):
                ok = False
            if ok:
                out.setdefault(key, []).append(row["file"])
    return out


def link(inst, obj, files, data_dir):
    """science/<obj>/<file> -> APERO's file (its real path); returns how
    many links there are"""
    folder = os.path.join(data_dir, "science", obj)
    os.makedirs(folder, exist_ok=True)
    for name in files:
        target = os.path.realpath(os.path.join(ROOTS[inst], obj, name))
        dest = os.path.join(folder, name)
        if os.path.lexists(dest):
            continue
        if not os.path.exists(target):
            sys.stderr.write("missing: %s\n" % target)
            continue
        os.symlink(target, dest)
    return len(os.listdir(folder))


def teff_of(value):
    try:
        teff = float(value)
    except (TypeError, ValueError):
        return 3500
    return int(round(teff)) if math.isfinite(teff) else 3500


def wrap(inst, name, stars, data_dir):
    from lbl.gui import wrapgen
    objects = [obj for obj, _ in stars]
    rparams = dict(
        INSTRUMENT=inst, DATA_SOURCE=DATA_SOURCE, DATA_DIR=data_dir,
        INPUT_FILE="*t.fits", DATA_TYPES=["SCIENCE"] * len(objects),
        OBJECT_SCIENCE=objects, OBJECT_COMPARISON=list(objects),
        OBJECT_TEFF=[teff_of(teff) for _, teff in stars],
        RUN_LBL_RESET=False, RUN_LBL_TELLUCLEAN=False,
        RUN_LBL_TEMPLATE=False, RUN_LBL_MASK=False, RUN_LBL_COMPUTE=False,
        RUN_LBL_COMPILE=False, REPORT=True,
        PCA2D=dict(n_star=1, joint=True, compare=False, output_name=name))
    return wrapgen.render(rparams, name="%s %s" % (inst, name),
                          python=sys.executable)


def runner(inst, name, wrap_path, data_dir, root):
    """<root>/run_<INSTRUMENT>_<basis>.sh: the wrap of one basis, then the
    basis kept (basis.keep) when the fit wrote it. With a PID as argument,
    a run of that wrap already going is waited for instead of starting
    one. One line per step in <root>/logs/status.txt."""
    logs = os.path.join(root, "logs")
    status = os.path.join(logs, "status.txt")
    log = os.path.join(logs, "%s_%s.log" % (inst, name))
    made = os.path.join(data_dir, "pca2d", "components",
                        "PCA2D_Observer_%s_%s.fits" % (name, inst.lower()))
    kept = os.path.join(data_dir, "models", "pca2d_%s_%s" % (
        inst, DATA_SOURCE), os.path.basename(made))
    def say(what):
        return 'echo "$(date "+%%F %%T") %s %s %s" >> %s' % (inst, name, what,
                                                           status)
    lines = [
        "#!/bin/bash",
        "# the basis %s of %s (build_bases.py); with a PID, that run of"
        " its wrap is waited for instead of starting one" % (name, inst),
        "export OMP_NUM_THREADS=%d OPENBLAS_NUM_THREADS=%d"
        " MKL_NUM_THREADS=%d MPLBACKEND=Agg" % ((THREADS,) * 3),
        "mkdir -p %s" % logs,
        "code=0",
        'if [ -n "$1" ]; then',
        "  " + say("waiting for its run (PID $1)"),
        '  while kill -0 "$1" 2>/dev/null; do sleep 60; done',
        "else",
        "  " + say("started"),
        "  nice -n %d %s %s > %s 2>&1; code=$?" % (NICE, sys.executable,
                                                  wrap_path, log),
        "fi",
        'if [ "$code" = 0 ] && [ -f %s ]; then' % made,
        "  where=$(%s -c 'from lbl.pca2d import basis; print(basis.keep("
        '"%s", "%s", "%s"))\' 2>&1 | tail -1)' % (sys.executable, name,
                                                   data_dir, inst),
        "  " + say("done: $where"),
        'elif [ "$code" = 0 ] && [ -f %s ]; then' % kept,
        "  " + say("done, updated where it is kept: %s" % kept),
        "else",
        "  " + say("FAILED (exit $code, no basis written): see %s" % log),
        "fi"]
    path = os.path.join(root, "run_%s_%s.sh" % (inst, name))
    text = "\n".join(lines) + "\n"
    # not rewritten when it is the same: bash reads a running script as it
    #   goes, and a file truncated under it ends the run
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            if handle.read() == text:
                return path
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def main(root):
    root = os.path.abspath(os.path.expanduser(root))
    plan = proposed(os.path.join(HERE, "proposed_baseline_stars.csv"))
    wanted = {(inst, obj) for (inst, _), stars in plan.items()
              for obj, _ in stars}
    files = exposures(os.path.join(HERE, "exposures.csv"), wanted)
    os.makedirs(os.path.join(root, "wraps"), exist_ok=True)
    order = list(NAMES)
    for inst in sorted({inst for inst, _ in plan}):
        data_dir = os.path.join(root, inst)
        for snr_bin in order:
            stars = plan.get((inst, snr_bin))
            if not stars:
                continue
            for obj, _ in stars:
                n = link(inst, obj, files.get((inst, obj), []), data_dir)
                print("%-9s %-6s %-16s %5d links" % (inst, snr_bin, obj, n))
            path = os.path.join(root, "wraps", "%s_%s.py" % (inst,
                                                             NAMES[snr_bin]))
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(wrap(inst, NAMES[snr_bin], stars, data_dir))
            print("runner:", runner(inst, NAMES[snr_bin], path, data_dir,
                                    root))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/pca2d_bases")
