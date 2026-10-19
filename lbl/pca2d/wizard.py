"""Check a configuration before it costs twenty minutes, and fill it in if not.

Two entry points. `validate` returns every problem it can find without reading
more than one spectrum: paths that do not exist, a glob that matches nothing,
a wavelength range that misses the data, component counts that outnumber the
spectra. `run_wizard` asks for the missing values one at a time, re-checking
each answer as it is given, and writes the file back.

The checks are deliberately concrete. `pattern: "*t.fits"` is not validated by
looking at the string; it is validated by expanding it and counting files, then
opening the first one. A configuration that passes here has been tried against
the actual data, not merely parsed.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from .config import spectra_dir
from .logger import log

PLACEHOLDER = "REPLACE_ME"

# what a NIRPS-like near-infrared setup can plausibly ask for; a value outside
# these is not forbidden, it is questioned
SANE = {
    "wave_nm": (300.0, 30000.0),
    "dv_kms": (0.01, 20.0),
    "components": (0, 50),
    "iters": (1, 500),
    "max_mad": (0.0, 100.0),
}


def _is_placeholder(value):
    return isinstance(value, str) and PLACEHOLDER in value


def _files(config):
    root = config["input"].get("directory")
    pattern = config["input"].get("pattern") or "*.fits"
    if not root or _is_placeholder(root):
        return None, None
    directory = spectra_dir(config)
    return directory, sorted(glob.glob(os.path.join(directory, pattern)))


def probe_files(config, max_open=1):
    """Open the first matching file and report what it actually contains.

    Returns a dict, or None if nothing could be opened. This is what turns
    "the pattern looks fine" into "the pattern points at 782 readable files
    covering 972 to 1919 nm".
    """
    from .io import robust_open

    directory, paths = _files(config)
    if not paths:
        return None
    out = {"n_files": len(paths), "first": paths[0], "directory": directory}
    from . import readers as _readers
    # UN T.FITS N'EST PAS LE SEUL SPECTRE VALIDE. Les instruments du visible
    # passent par le pont de LBL et n'ont ni WaveA ni WaveAB: leur solution
    # peut etre une table d'un autre nom. Demander la grille au lecteur que la
    # configuration designe verifie donc la meme chose pour tous, et bien
    # davantage qu'un nom d'extension: que le fichier s'ouvre, que la classe
    # de LBL le reconnaisse et qu'il en sorte des longueurs d'onde
    # (2026-10-14).
    out["bridged"] = not _readers.reads_natively(config)
    try:
        with robust_open(paths[0]) as hdulist:
            out["extensions"] = [h.name for h in hdulist]
            out["object"] = hdulist[0].header.get("OBJECT")
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    try:
        if out["bridged"]:
            wave = np.asarray(_readers.read_any(paths[0], config)["wave"],
                              dtype=float)
        else:
            names = set(out["extensions"])
            if not ({"WaveA", "WaveAB"} & names):
                out["no_wave"] = True
                return out
            from .tfits import extensions_for
            _, e_wave, _, _ = extensions_for(hdulist)
            with robust_open(paths[0]) as hdulist:
                _, e_wave, _, _ = extensions_for(hdulist)
                wave = np.asarray(hdulist[e_wave].data, dtype=float)
        wave = np.atleast_2d(wave)
        finite = wave[np.isfinite(wave) & (wave > 0)]
        if finite.size:
            out["wave_min"] = float(finite.min())
            out["wave_max"] = float(finite.max())
        out["n_orders"] = wave.shape[0]
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    return out


def validate(config, probe=True):
    """Every problem worth stopping for. Returns a list of (key, message)."""
    problems = []

    def bad(key, message):
        problems.append((key, message))

    directory, paths = _files(config)
    if directory is None:
        bad("input.directory", "not set (still %s)" % PLACEHOLDER)
    elif not os.path.isdir(directory):
        bad("input.directory", "%r is not a directory" % directory)
    elif not paths:
        bad("input.pattern", "%r matches no file in %s"
            % (config["input"].get("pattern"), directory))

    info = probe_files(config) if (probe and paths) else None
    if info and info.get("error"):
        bad("input.pattern", "the first match does not open: %s" % info["error"])
    elif info and info.get("no_wave"):
        bad("input.pattern", "%s has no WaveA extension; is it really a t.fits?"
            % os.path.basename(info["first"]))

    dom = config["domain"]
    lo, hi = dom.get("wave_min"), dom.get("wave_max")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        bad("domain.wave_min/max", "not both numbers")
    elif lo >= hi:
        bad("domain.wave_min", "%.1f is not below wave_max %.1f" % (lo, hi))
    elif not (SANE["wave_nm"][0] <= lo < hi <= SANE["wave_nm"][1]):
        bad("domain.wave_min/max", "%.1f-%.1f nm is outside %g-%g nm"
            % (lo, hi, *SANE["wave_nm"]))
    elif info and "wave_min" in info:
        # the domain must actually intersect what the files cover
        if hi <= info["wave_min"] or lo >= info["wave_max"]:
            bad("domain.wave_min/max",
                "%.1f-%.1f nm does not overlap the data, which covers %.1f-%.1f nm"
                % (lo, hi, info["wave_min"], info["wave_max"]))

    dv = dom.get("dv")
    if dom.get("smart_dv"):
        # measured from the first spectrum at load time, so whatever is in
        # domain.dv is not read and is not this function's business
        pass
    elif not isinstance(dv, (int, float)) or not (SANE["dv_kms"][0] <= dv <= SANE["dv_kms"][1]):
        bad("domain.dv", "%r is not a sane sampling in km/s (%g-%g)"
            % (dv, *SANE["dv_kms"]))

    frame = config["registration"].get("frame")
    if frame != "observer":
        bad("registration.frame",
            "%r; the two-frame fit does its own shifting and needs 'observer'" % frame)

    tf = config.get("twoframe", {})
    m, n = tf.get("n_star"), tf.get("n_earth")
    for key, value in (("twoframe.n_star", m), ("twoframe.n_earth", n)):
        if not isinstance(value, int) or not (SANE["components"][0] <= value <= SANE["components"][1]):
            bad(key, "%r is not an integer in %d-%d" % (value, *SANE["components"]))
    if isinstance(m, int) and isinstance(n, int):
        if m + n < 1:
            bad("twoframe.n_star", "n_star + n_earth is 0; nothing would be fitted")
        elif paths and m + n >= len(paths):
            bad("twoframe.n_star", "n_star + n_earth = %d, but only %d files match;"
                " the fit would be underdetermined" % (m + n, len(paths)))

    it = tf.get("iters")
    if not isinstance(it, int) or not (SANE["iters"][0] <= it <= SANE["iters"][1]):
        bad("twoframe.iters", "%r is not an integer in %d-%d" % (it, *SANE["iters"]))

    mad = tf.get("max_mad")
    if not isinstance(mad, (int, float)) or not (SANE["max_mad"][0] <= mad <= SANE["max_mad"][1]):
        bad("twoframe.max_mad", "%r is not a number in %g-%g (0 disables the cut)"
            % (mad, *SANE["max_mad"]))
    elif 0 < mad < 3:
        bad("twoframe.max_mad", "%g is very tight; below about 5 the cut starts"
            " removing whole observing epochs rather than outliers" % mad)

    out = config["output"].get("directory")
    if _is_placeholder(out) or not out:
        bad("output.directory", "not set")

    return problems, info


def report(config):
    """Print what the configuration points at. Returns True if it is usable."""
    problems, info = validate(config)
    if info and not info.get("error"):
        log("input: %d files in %s" % (info["n_files"], info["directory"]), "value")
        detail = []
        if info.get("object"):
            detail.append("OBJECT=%s" % info["object"])
        if info.get("n_orders"):
            detail.append("%d orders" % info["n_orders"])
        if "wave_min" in info:
            detail.append("%.1f-%.1f nm" % (info["wave_min"], info["wave_max"]))
        if detail:
            log("first file: %s" % ", ".join(detail), "value")
    for key, message in problems:
        log("%s: %s" % (key, message), "warn")
    return not problems


# --------------------------------------------------------------------------
# the interactive half
# --------------------------------------------------------------------------
def _ask(question, default, cast, check=None):
    """One question, repeated until the answer passes `check`."""
    while True:
        shown = "" if default is None else " [%s]" % default
        try:
            raw = input("  %s%s: " % (question, shown)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit("cancelled; nothing was written")
        if not raw and default is not None:
            raw = str(default)
        try:
            value = cast(raw)
        except Exception:                                     # noqa: BLE001
            log("     not a %s, try again" % cast.__name__)
            continue
        if check is not None:
            problem = check(value)
            if problem:
                log("     %s" % problem)
                continue
        return value


def run_wizard(config, path):
    """Ask for what is missing, check each answer, write the file back."""
    import yaml

    print()
    log("This configuration is not ready. Answering the questions below fills")
    log("it in and writes %s; press Ctrl-C to stop." % path)
    print()

    obj = config["input"].get("object")

    def dir_check(value):
        # the answer is the input ROOT; what has to exist is the object's
        # folder under it, which is what every stage will read
        probe = spectra_dir({"input": dict(config["input"], directory=value)})
        if not os.path.isdir(probe):
            return "%r is not a directory" % probe
        return None

    question = ("input root, holding %s/ with the t.fits files in it" % obj
                if obj else "directory holding the t.fits files")
    config["input"]["directory"] = _ask(
        question, config["input"].get("directory"), str, dir_check)

    def pattern_check(value):
        hits = glob.glob(os.path.join(spectra_dir(config), value))
        if not hits:
            return "matches no file"
        log("     %d files match, e.g. %s" % (len(hits), os.path.basename(sorted(hits)[0])))
        return None

    config["input"]["pattern"] = _ask(
        "filename pattern", config["input"].get("pattern") or "*t.fits",
        str, pattern_check)

    info = probe_files(config) or {}
    if info.get("error"):
        raise SystemExit("the first matching file does not open: %s" % info["error"])
    if info.get("object"):
        log("     OBJECT in the first file: %s" % info["object"])
    if "wave_min" in info:
        log("     the data covers %.1f to %.1f nm over %d orders"
              % (info["wave_min"], info["wave_max"], info.get("n_orders", 0)))

    lo_default = round(info.get("wave_min", config["domain"]["wave_min"]))
    hi_default = round(info.get("wave_max", config["domain"]["wave_max"]))

    def wave_lo_check(value):
        if not SANE["wave_nm"][0] <= value <= SANE["wave_nm"][1]:
            return "outside %g-%g nm" % SANE["wave_nm"]
        if "wave_max" in info and value >= info["wave_max"]:
            return "above everything the data covers (%.1f nm)" % info["wave_max"]
        return None

    config["domain"]["wave_min"] = _ask("domain start, nm", lo_default, float, wave_lo_check)

    def wave_hi_check(value):
        if value <= config["domain"]["wave_min"]:
            return "must be above wave_min (%.1f)" % config["domain"]["wave_min"]
        if "wave_min" in info and value <= info["wave_min"]:
            return "below everything the data covers (%.1f nm)" % info["wave_min"]
        return None

    config["domain"]["wave_max"] = _ask("domain end, nm", hi_default, float, wave_hi_check)
    config["domain"]["wave0"] = config["domain"]["wave_min"]

    def dv_check(value):
        if not SANE["dv_kms"][0] <= value <= SANE["dv_kms"][1]:
            return "outside %g-%g km/s" % SANE["dv_kms"]
        return None

    if config["domain"].get("smart_dv"):
        log("     grid sampling: measured from the data, domain.smart_dv is on."
            " Nothing to ask and nothing in domain.dv would be read.")
    else:
        config["domain"]["dv"] = _ask("grid sampling, km/s",
                                      config["domain"].get("dv", 0.5),
                                      float, dv_check)

    n_files = info.get("n_files", 0)

    def comp_check(value):
        if not SANE["components"][0] <= value <= SANE["components"][1]:
            return "outside %d-%d" % SANE["components"]
        if n_files and value >= n_files:
            return "more components than files (%d)" % n_files
        return None

    print()
    log("  Two component counts, one per reference frame. They need not be equal.")
    config["twoframe"]["n_star"] = _ask(
        "M, components in the STELLAR rest frame", config["twoframe"].get("n_star", 5),
        int, comp_check)
    config["twoframe"]["n_earth"] = _ask(
        "N, components in the OBSERVER frame", config["twoframe"].get("n_earth", 5),
        int, comp_check)
    if config["twoframe"]["n_star"] + config["twoframe"]["n_earth"] < 1:
        raise SystemExit("M + N is 0; there would be nothing to fit")

    config["output"]["tag"] = _ask("output tag (a subdirectory name)",
                                   config["output"].get("tag", "run"), str,
                                   lambda v: None if v.strip() else "cannot be empty")

    config["registration"]["frame"] = "observer"

    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    print()
    log("wrote %s" % path, "value")

    problems, _ = validate(config)
    for key, message in problems:
        log("%s: %s" % (key, message), "warn")
    if problems:
        raise SystemExit("still not usable; edit %s by hand" % path)
    return config
