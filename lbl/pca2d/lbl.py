"""Hand both sets of spectra to LBL, the uncorrected and the corrected.

The point of correcting a spectrum is the velocity that comes out of it, and
the only honest way to know whether the correction helped is to measure both.
So this stage never runs LBL on the corrected files alone. It sets up two
objects side by side in one LBL tree, from the same instrument profile:

    lbl/science/TOI-2120/               symlinks to the spectra as delivered
    lbl/science/TOI-2120_PCA2D_2-3v/    symlinks to what this package wrote

and LBL's own products then sit next to each other under the same names, so the
comparison that matters is two columns of one table rather than two runs
someone has to remember to line up.

The delivered spectra are measured against the template LBL builds from them.
The corrected ones are measured against the star as the fit sees it: its first
star component at its mean amplitude, in LBL's template format and written by
LBL's own writer (lbltemplate.py), so LBL's template step finds it in place and
has nothing to do. With two or more star components, the ones past the first
also go to LBL, as RESPROJ tables STRPCA2..N: LBL projects every line on them
the way it does on its DTEMP gradients, and the rdb gets each exposure's
amplitude along them.

Symlinks by default (lbl.link), not copies: a few hundred t.fits are a few
tens of gigabytes, they already exist twice (delivered and corrected), and a
third copy would buy nothing. A disk that cannot hold a link at all, exFAT as
the data disks here are, gets copies instead, and the run says so and how much
before it starts rather than failing at this stage after the fit (link_mode).
Nothing here writes into the input tree.

What it leaves beside the run's other outputs:

    lbl_config.yaml     LBL's configuration, in LBL's keys and LBL's spelling,
                        which `lbl_compute --config lbl_config.yaml` reads
    run_lbl.py          the wrap script, one runparams dict per object, the
                        dict LBL users know, ready to be read and run by hand
    star_template.fits  the fit's star template, copied from here to where
                        LBL looks for the corrected object's template

Running LBL takes hours, so the stage prepares by default and runs only when
asked: `lbl.run: true` in the config, or --run-lbl on the command line.

Which LBL runs it is `lbl.environment`: a conda environment by name, and
`lbl-rapide` by default, LBL's speed branch in an environment of its own
(FAST_RECIPE). `current` is the LBL installed beside this package by
environment.yml, which is what every run used until 2026-09-18. The template
writer and the profile check still use that one, in this process: they read
and write files, and the speed branch changed neither format.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

from .logger import log

#: LBL's name for a spectrograph is not always its INSTRUME, and the pairing
#: with a data source is what selects the class that reads the files: NIRPS is
#: NIRPS_HA or NIRPS_HE depending on the mode it was observed in, which LBL
#: expects the caller to know rather than reading it from the header. So the
#: pairing lives in config.yaml, per instrument, and this is only the fallback
#: for a spectrograph whose block does not say.
FALLBACK = {
    # CADC, and not APERO, even for an APERO reduction: LBL's CADC classes are
    # the ones that read the named fibre extensions (FluxAB / WaveAB / BlazeAB
    # for SPIRou, FluxA / WaveA / BlazeA for NIRPS), which is where the
    # wavelength solution of a t.fits is. Its APERO classes read the primary
    # header's wavelength polynomials, which these files do not carry there.
    "SPIROU": ("SPIROU", "CADC"),
    "NIRPS": ("NIRPS_HE", "CADC"),
}

STEPS = ("telluclean", "template", "mask", "compute", "compile")

#: Where the target's effective temperature is written, in the order they are
#: tried. APERO puts OBJTEMP on the primary header and PP_TEFF on the science
#: extension, with PP_TEFFS saying where the DRS got it. LBL needs the number
#: and does not read it from anywhere itself: it stops with "Teff is require.
#: Please add OBJECT_TEFF to config". It is in the file, so read it there.
TEFF_KEYS = ("OBJTEMP", "PP_TEFF", "OBJ_TEMP", "TEFF")


def teff_from_header(path: str):
    """(teff, keyword) from the first of TEFF_KEYS that carries a number."""
    from .io import robust_open

    try:
        with robust_open(path) as hdulist:
            for key in TEFF_KEYS:
                for hdu in hdulist:
                    value = hdu.header.get(key)
                    if value in (None, "", "None"):
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if number > 0:
                        return number, key
    except Exception:                                         # noqa: BLE001
        return None, None
    return None, None


def resolve_teff(config: dict, files) -> tuple:
    """The Teff LBL will be given, and where it came from.

    `auto`, the default, reads it from the spectra, for the same reason the
    instrument is read from them: it is written in the file, by the pipeline
    that reduced it, and a number typed into a config is a number that can be
    typed wrong. A number in the config wins over the header, and is checked
    against it out loud, because disagreeing with the file is a thing worth
    doing deliberately and not by accident.
    """
    asked = (config.get("lbl") or {}).get("teff", "auto")
    header, key = (teff_from_header(files[0]) if files else (None, None))

    if isinstance(asked, str) and asked.strip().lower() == "auto":
        if header is None:
            log("lbl.teff is `auto` and none of %s is in the header of %s."
                " LBL needs a Teff to choose the stellar model its mask comes"
                " from: put a number in lbl.teff."
                % (", ".join(TEFF_KEYS), os.path.basename(files[0]) if files
                   else "the spectra"), "warn")
            return None, "nowhere"
        return header, "%s in the header" % key
    if asked is None:
        return None, "unset"
    value = float(asked)
    if header is not None and abs(header - value) > 1.0:
        log("lbl.teff says %.0f K and %s in the header says %.0f K. The config"
            " wins, which is what it is for, but one of the two is wrong."
            % (value, key, header), "warn")
    return value, "lbl.teff in the config"


def available():
    """(True, version) if the LBL package can be imported, (False, why) if not.

    This module is itself called lbl, so whenever the pca2d/ directory gets onto
    sys.path the name `lbl` resolves to this file rather than to LBL. It did:
    a module of the figures stage used to put it there, and the error that came
    out, "attempted relative import with no known parent package", named
    neither the file nor the cause. Now it says which file answered.
    """
    try:
        import lbl
    except Exception as exc:                                  # noqa: BLE001
        where = _what_lbl_resolves_to()
        if where and os.path.basename(where) == "lbl.py":
            return False, ("the name `lbl` resolves to %s, which is not the LBL"
                           " package: its directory is on sys.path ahead of"
                           " site-packages" % where)
        return False, str(exc)
    if not hasattr(lbl, "__path__"):
        return False, ("the name `lbl` resolves to %s, a single module and not"
                       " the LBL package" % getattr(lbl, "__file__", "?"))
    return True, getattr(lbl, "__version__", "unknown")


def _what_lbl_resolves_to():
    """The file the name `lbl` would be imported from on the current path."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("lbl")
    except (ImportError, ValueError):
        return None
    return getattr(spec, "origin", None) if spec else None


#: lbl.environment's word for the interpreter this package runs in, and the
#: LBL environment.yml installs beside it (LBL main)
CURRENT = "current"
#: lbl.environment when nothing says otherwise
DEFAULT_ENVIRONMENT = "lbl-rapide"
#: how that environment is made, said wherever it is missing. LBL's speed
#: branch: main plus [LBL.SPEED] commits, numba kernels and single-open FITS
#: I/O, every one of which says in its message that the outputs do not change
FAST_RECIPE = ("git clone -b test-speed-260918-110104"
               " https://github.com/njcuk9999/lbl.git lbl-rapide\n"
               "conda create -n lbl-rapide python=3.12\n"
               "conda activate lbl-rapide\n"
               "pip install -e ./lbl-rapide")

#: What another interpreter is asked, from a folder with no `lbl` in it.
#: `python -c` puts its working directory first on the path, and the root of
#: this repository holds an `lbl/` folder, the examples' LBL tree: an LBL
#: installed in place (pip install -e) is found through a finder hook that
#: comes after the path, so that folder won, as an empty namespace package,
#: the first time this was asked from there.
PROBE = ("import os, lbl\n"
         "where = getattr(lbl, '__file__', None)\n"
         "if not where:\n"
         "    raise SystemExit('the name lbl is %s, a folder and not the LBL"
         " package' % list(lbl.__path__))\n"
         "print(getattr(lbl, '__version__', 'unknown'))\n"
         "print(os.path.dirname(os.path.abspath(where)))\n")


def asked_environment(config: dict) -> str:
    """lbl.environment, as written, or the default."""
    asked = (config.get("lbl") or {}).get("environment")
    return str(asked).strip() if asked not in (None, "") else DEFAULT_ENVIRONMENT


def conda_prefix(name: str):
    """The folder of the conda environment called `name`, or None.

    Looked for beside the environment this runs in first, which is where
    `conda create -n` puts it and costs nothing, then in conda's own envs
    folder, and only then asked of conda, which takes a second.
    """
    if os.path.basename(sys.prefix) == name:
        return sys.prefix
    roots = []
    parent = os.path.dirname(sys.prefix)
    if os.path.basename(parent) == "envs":
        roots.append(parent)
    roots.append(os.path.join(sys.prefix, "envs"))     # when this is base
    conda = os.environ.get("CONDA_EXE")
    if conda:
        roots.append(os.path.join(os.path.dirname(os.path.dirname(conda)),
                                  "envs"))
    roots.append(os.path.expanduser(os.path.join("~", ".conda", "envs")))
    for root in roots:
        if os.path.isdir(os.path.join(root, name)):
            return os.path.join(root, name)
    conda = conda or shutil.which("conda")
    if not conda:
        return None
    try:
        out = subprocess.run([conda, "env", "list", "--json"],
                             capture_output=True, text=True, timeout=60)
        prefixes = json.loads(out.stdout).get("envs", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return next((p for p in prefixes if os.path.basename(p) == name), None)


def python_in(prefix: str):
    """The python of the environment at `prefix`, or None if it has none."""
    for rel in (os.path.join("bin", "python"), "python.exe"):
        path = os.path.join(prefix, rel)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def interpreter(config: dict) -> tuple:
    """(python, what it is) for LBL's script, from lbl.environment.

    `current` is the interpreter this package runs in. A path is a python, or
    the folder of an environment. Any other word is a conda environment by
    name. python is None when it names nothing on this machine, and `what`
    then says what was looked for.
    """
    asked = asked_environment(config)
    if asked.lower() == CURRENT:
        return sys.executable, "the environment pca2d-preclean runs in"
    if os.sep in asked or asked.startswith("~"):
        path = os.path.expanduser(asked)
        python = python_in(path) if os.path.isdir(path) else (
            path if os.path.isfile(path) and os.access(path, os.X_OK) else None)
        return python, ("%s" % path if python else
                        "%s, which is not a python nor an environment's folder"
                        % path)
    prefix = conda_prefix(asked)
    if prefix is None:
        return None, "the conda environment %s, which is not on this machine" \
            % asked
    python = python_in(prefix)
    return python, ("the conda environment %s" % asked if python else
                    "the conda environment %s, which has no python (%s)"
                    % (asked, prefix))


def environment_for(python: str) -> dict:
    """The process environment `python` is started with: this one, less
    PYTHONPATH and PYTHONHOME when it is another environment's.

    The window starts the pipeline with this repository first on PYTHONPATH
    (gui.App), and the repository holds an `lbl/` folder, the examples' LBL
    tree. An LBL installed in place is found through a finder hook that
    comes after the path, so in lbl-rapide that folder was `lbl`, as an empty
    namespace package, and `from lbl.recipes import lbl_wrap` could not work.
    The runner finds pca2d by itself (write_runner), after LBL.
    """
    env = dict(os.environ)
    if python != sys.executable:
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
    return env


@functools.lru_cache(maxsize=None)
def probe(python: str) -> tuple:
    """(True, LBL's version, its package folder) as `python` imports LBL, or
    (False, why, None).

    This process's own interpreter is answered by available(), as it always
    was; another is asked, which costs the two seconds of importing LBL once
    per run.
    """
    if python == sys.executable:
        ok, detail = available()
        if not ok:
            return False, detail, None
        import lbl
        return True, detail, os.path.dirname(os.path.abspath(lbl.__file__))
    try:
        out = subprocess.run([python, "-c", PROBE], capture_output=True,
                             text=True, timeout=300, cwd=tempfile.gettempdir(),
                             env=environment_for(python))
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc), None
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or len(lines) < 2:
        tail = (out.stderr.strip().splitlines() or ["exit %d" % out.returncode])
        return False, tail[-1], None
    return True, lines[-2], lines[-1]


def chosen(config: dict) -> dict:
    """Which LBL measures: the python lbl.environment names and what it imports.

    {asked, python, what, ok, detail, where}: detail is LBL's version when ok
    and why not otherwise; where is the folder of the LBL package it imports.
    """
    asked = asked_environment(config)
    python, what = interpreter(config)
    if python is None:
        return dict(asked=asked, python=None, what=what, ok=False,
                    detail=what, where=None)
    ok, detail, where = probe(python)
    return dict(asked=asked, python=python, what=what, ok=ok, detail=detail,
                where=where)


def say_chosen(code: dict) -> str:
    """One line for the log: which LBL, from where, run by which python."""
    if code["ok"]:
        return ("LBL %s from %s, run by %s: %s (lbl.environment: %s)"
                % (code["detail"], code["where"], code["python"], code["what"],
                   code["asked"]))
    return ("lbl.environment is %s, and %s" % (code["asked"], (
        "%s cannot import LBL: %s" % (code["python"], code["detail"])
        if code["python"] else "that is %s" % code["what"])))


def how_to_get(code: dict) -> str:
    """What to do about a chosen LBL that is not there."""
    fix = ("--lbl-env %s (lbl.environment: %s) runs the LBL installed beside"
           " this package instead" % (CURRENT, CURRENT))
    if code["asked"] == DEFAULT_ENVIRONMENT:
        return ("%s is LBL's speed branch, in an environment of its own, made"
                " once with\n\n%s\n\nor %s."
                % (DEFAULT_ENVIRONMENT, "\n".join("    " + line for line in
                                                   FAST_RECIPE.splitlines()),
                   fix))
    return fix[0].upper() + fix[1:] + "."


def profile(config: dict) -> tuple:
    """(instrument, data_source) as LBL spells them, and where they came from.

    Read from the instrument's own block in config.yaml, because that is where
    everything else about a spectrograph is written and because choosing this
    wrongly is not an error: LBL would read the files with another instrument's
    class and return velocities.
    """
    block = config.get("lbl") or {}
    instrume = str(config["input"].get("instrument") or "").upper()
    name = block.get("instrument")
    source = block.get("data_source")
    if name and source:
        return str(name), str(source), "config.yaml"
    fallback = FALLBACK.get(instrume)
    if not fallback:
        raise SystemExit(
            "no LBL profile for INSTRUME %r. Add `lbl: {instrument: ..., "
            "data_source: ...}` to its block in config.yaml: LBL's name for a"
            " spectrograph is not always the one in the header, and NIRPS is"
            " NIRPS_HA or NIRPS_HE by the mode it was observed in."
            % (instrume or "unset"))
    return str(name or fallback[0]), str(source or fallback[1]), "the fallback table"


def object_names(config: dict, object_name: str, tag: str) -> tuple:
    """The names the two runs go under in the LBL tree, before and after."""
    suffix = str((config.get("lbl") or {}).get("suffix") or "_PCA2D_{tag}")
    return object_name, object_name + suffix.format(tag=tag)


def corrected_files(corrdir: str) -> list:
    """The corrected spectra in a folder, and nothing that merely looks like one.

    A name starting with a dot is skipped. The corrected folders can live on an
    exFAT disk, where macOS writes a `._name` AppleDouble file beside anything
    that carries extended attributes: it ends in .fits too, holds none of a
    spectrum, and would otherwise be linked into LBL's science folder.
    """
    if not os.path.isdir(corrdir):
        return []
    return sorted(os.path.join(corrdir, name) for name in os.listdir(corrdir)
                  if name.endswith(".fits") and not name.startswith("."))


def can_link(folder: str) -> bool:
    """Whether `folder` can hold a symbolic link, found out by making one.

    Not read off the filesystem's name: macOS mounts exFAT through fskit and
    a network share can be anything, and the only answer that is certainly
    right is the one the disk gives. The probe is made in the folder itself,
    which is created if it has to be, and removed whatever happens.
    """
    import tempfile

    os.makedirs(folder, exist_ok=True)
    # a folder that cannot even take a file is not this function's question:
    # a copy would fail there too, and mkstemp says so in its own words
    handle, source = tempfile.mkstemp(prefix=".pca2d_probe_", dir=folder)
    os.close(handle)
    probe = source + ".link"
    try:
        os.symlink(source, probe)
        return os.path.islink(probe)
    except OSError:
        return False
    finally:
        for path in (probe, source):
            if os.path.lexists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def link_mode(data_dir: str, asked: str = "symlink") -> tuple:
    """(mode, reason) for putting spectra in `data_dir`/science.

    `asked` is lbl.link. A copy is always possible; a link only where the disk
    can hold one, and where it cannot the answer is a copy and the reason says
    why. The reason is None when the answer is what was asked.
    """
    asked = str(asked or "symlink")
    if asked == "copy":
        return "copy", None
    science = os.path.join(data_dir, "science")
    if can_link(science):
        return "symlink", None
    return "copy", ("%s cannot hold a symbolic link (an exFAT disk cannot), so"
                    " the spectra are COPIED into it: each one will be on that"
                    " disk twice. An LBL folder on a disk that takes links, the"
                    " internal one for instance, avoids that" % science)


def link_spectra(files, target: str, mode: str = "symlink") -> tuple:
    """Put `files` in `target` as symlinks (or copies).

    Returns (linked, kept, strangers): how many were made, how many were
    already right, and which .fits in the folder are none of ours.

    Idempotent, and never destructive: a name already in the folder is left
    exactly as it is, whether it is one of our symlinks from a previous run or
    a file someone else put there. LBL globs the folder, so a name that is
    already right needs nothing done to it.

    The strangers matter because LBL takes the whole folder. Anything in there
    that this run did not put there is going to be measured with everything
    else, and no error will be raised about it.
    """
    import shutil

    os.makedirs(target, exist_ok=True)
    ours = {os.path.basename(path) for path in files}
    linked = kept = 0
    for path in files:
        destination = os.path.join(target, os.path.basename(path))
        if os.path.lexists(destination):
            # ... unless it is a link to something that is no longer there. The
            # rule below is for a file that is already right; a DANGLING link is
            # not right, and keeping it makes LBL die on it much later. One
            # such folder, left by a run whose corrected spectra had since been
            # deleted, cost 8 h 40 of LBL before it was reached (2026-09-14).
            if os.path.islink(destination) and not os.path.exists(destination):
                os.unlink(destination)
            else:
                kept += 1
                continue
        if mode == "copy":
            shutil.copy2(path, destination)
        else:
            os.symlink(os.path.abspath(path), destination)
        linked += 1
    strangers = sorted(name for name in os.listdir(target)
                       if name.endswith(".fits") and name not in ours)
    return linked, kept, strangers


def config_document(config: dict, data_dir: str, instrument: str,
                    data_source: str, teff=None) -> dict:
    """The default LBL configuration, as LBL's own keys.

    Only keys LBL knows go in: it validates every one of them against its
    parameter table and refuses a file with a name it does not recognise, which
    is a good reason to write this from the code rather than by hand.
    """
    block = config.get("lbl") or {}
    document = {
        "INSTRUMENT": instrument,
        "DATA_SOURCE": data_source,
        "DATA_DIR": os.path.abspath(data_dir),
        "DATA_TYPE": "SCIENCE",
        # LBL's per-instrument default is already '*.fits'. It is written out
        # because ours must match BOTH sets, and the corrected files are named
        # <stem>t_<M>-<N>.fits: a '*t.fits' glob would find the delivered
        # spectra and silently none of the corrected ones.
        "INPUT_FILE": str(block.get("input_file") or "*.fits"),
        # re-running must not redo the hours that are already on disk
        "SKIP_DONE": True,
    }
    if teff is not None:
        document["OBJECT_TEFF"] = teff
    return document


#: the temperatures LBL has gradient tables for, temperature_gradient_<T>.fits
#: in its models folder, which LBL downloads itself with its MODEL_FILES
DTEMP_GRID = (3000, 3500, 4000, 4500, 5000, 5500, 6000)


def dtemp_table(block: dict, teff=None) -> dict:
    """{'DTEMP<T>': 'temperature_gradient_<T>.fits'} for this star, or {}.

    lbl.dtemp 'auto' takes the grid temperature nearest the star's Teff, a
    number the one nearest that number, and false (or no Teff to go by)
    none at all.
    """
    asked = block.get("dtemp", "auto")
    if asked in (None, False) or str(asked).lower() in ("false", "none", "no"):
        return {}
    if str(asked).lower() == "auto":
        target = teff
    else:
        try:
            target = float(asked)
        except (TypeError, ValueError):
            raise SystemExit("lbl.dtemp is %r: 'auto', a temperature in K, or"
                             " false" % (asked,))
    if target is None:
        return {}
    grid = min(DTEMP_GRID, key=lambda t: abs(t - float(target)))
    return {"DTEMP%d" % grid: "temperature_gradient_%d.fits" % grid}


def runparams(config: dict, data_dir: str, instrument: str, data_source: str,
              objects: list, config_file: str, teff=None) -> dict:
    """The dict LBL's wrapper takes, with both objects in it.

    Every object is its own comparison, which is to say each builds its own
    template. Sharing one template between the corrected and uncorrected
    spectra would be measuring two things with one ruler that fits neither:
    what the correction removes is exactly what would differ between them.
    `lbl.template` overrides that for anyone who wants it anyway.
    """
    block = config.get("lbl") or {}
    steps = [str(s).strip().lower() for s in (block.get("steps") or [])]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit("unknown lbl.steps %s; known ones are %s"
                         % (", ".join(unknown), ", ".join(STEPS)))
    template = block.get("template")
    params = {
        "INSTRUMENT": instrument,
        "DATA_SOURCE": data_source,
        "DATA_DIR": os.path.abspath(data_dir),
        "CONFIG_FILE": os.path.abspath(config_file),
        "DATA_TYPES": ["SCIENCE"] * len(objects),
        "OBJECT_SCIENCE": list(objects),
        "OBJECT_COMPARISON": [template or name for name in objects],
        "OBJECT_TEFF": [teff] * len(objects),
    }
    for step in STEPS:
        params["RUN_LBL_%s" % step.upper()] = step in steps
    # SKIP_LBL_<step> is LBL's "leave what is already done alone", which is
    # what a second run of this stage wants: the template and the mask are
    # hours, and neither depends on anything this package changed.
    for step in ("template", "mask", "compute", "compile"):
        params["SKIP_LBL_%s" % step.upper()] = True
    # the same temperature table on both objects, so that the report can set
    # DTEMP delivered beside DTEMP corrected
    tables = dtemp_table(block, teff)
    if tables:
        params["RESPROJ_TABLES"] = tables
    # LBL'S REPORT, the one lbl/report writes after compile. It is true by
    # default in LBL as here: it costs a few seconds at the end of a run that
    # took minutes, and it is what says whether the run worked. Set to false
    # in the config when dozens of variants are chained and only the .rdb
    # files matter.
    params["REPORT"] = bool(block.get("report", True))
    return params


RUNNER = '''#!%(shebang)s
"""Run LBL on %(object)s, as delivered and as pca2d-preclean corrected it (%(tag)s).

Written by pca2d-preclean, and left here to be read, edited and re-run by
hand: an ordinary LBL wrap script, with one of the runparams dicts LBL users
know per object. The settings that are not about which object is which live
in %(config)s. It is run by the python of the LBL it was written for
(lbl.environment):

    %(python)s %(script)s
"""

from lbl.recipes import lbl_wrap
%(blocks)s


if __name__ == "__main__":
%(main)s
'''

#: what run_lbl.py says about the STRPCA dict, above it
STRPCA_NOTE = ("The star components past the first, as LBL RESPROJ tables"
               " STRPCA2..N. They are written in the star's rest frame, which"
               " LBL's mask step measures, so between the mask and the"
               " velocities.")


def _block(name, comment, params, upper=True):
    """One NAME = dict(...) of the runner, with its comment above it."""
    import textwrap
    lines = ["", "", *("# " + line for line in textwrap.wrap(comment, 76)),
             "%s = dict(" % name]
    for key, value in params.items():
        lines.append("    %s=%r," % (key.upper() if upper else key, value))
    lines.append(")")
    return "\n".join(lines)


def write_runner(path: str, runs: list, strpca, object_name: str, tag: str,
                 config_file: str, python: str = None) -> str:
    """The wrap script: one runparams dict per object, spelled one key a line.

    `runs` is [(NAME, comment, params)], in the order they are run. `strpca`,
    when not None, is lbltemplate.strpca_from's arguments: the object named
    AFTER then runs its mask alone first, the tables are written in the rest
    frame that mask measured, and its velocities are measured with them.
    `python` is the interpreter it is written for, its first line, so that
    ./run_lbl.py is run by the LBL it was meant for.
    """
    # UPPER CASE, and not a style choice: lbl_wrap reads runparams['INSTRUMENT']
    # and every other key by that exact spelling, so a lower-case dict fails on
    # the first line of its own checking with "Must define key INSTRUMENT".
    blocks = [_block(name, comment, params) for name, comment, params in runs]
    if strpca:
        blocks.append(_block("STRPCA", STRPCA_NOTE, strpca, upper=False))
    main = []
    for name, _, _ in runs:
        if name == "AFTER" and strpca:
            main += ["    # its mask first: STRPCA is written in the rest frame"
                     " the mask measures",
                     "    lbl_wrap.main(dict(AFTER, RUN_LBL_COMPUTE=False,"
                     " RUN_LBL_COMPILE=False))",
                     "    # pca2d is part of LBL (lbl.pca2d): nothing to add"
                     " to the path",
                     "    from lbl.pca2d.lbltemplate import strpca_from",
                     "    # after DTEMP, which has to stay first (see the"
                     " config's lbl.dtemp)",
                     '    AFTER["RESPROJ_TABLES"] = dict(AFTER.get('
                     '"RESPROJ_TABLES") or {}, **strpca_from(**STRPCA))']
        main.append("    lbl_wrap.main(%s)" % name)
    body = RUNNER % {
        "shebang": python or "/usr/bin/env python",
        "python": python or "python",
        "object": object_name,
        "tag": tag,
        "script": os.path.abspath(path),
        "config": os.path.basename(config_file),
        "blocks": "".join(blocks),
        "main": "\n".join(main) if main else "    pass",
    }
    with open(path, "w") as handle:
        handle.write(body)
    os.chmod(path, 0o755)
    return path


def check_profile(instrument: str, data_source: str, config_file: str,
                  sample: str) -> tuple:
    """Ask LBL to read one spectrum with the chosen profile. (ok, why not).

    Two seconds against hours. The pairing of instrument and data source picks
    the class that reads the files, and the wrong one does not fail on the
    first line: it downloads a few hundred megabytes of stellar models, sorts
    every exposure, and dies at the reference file, which is what happened the
    first time this was pointed at SPIRou with LBL's APERO class instead of its
    CADC one. What is read here is the wavelength solution, because that is the
    thing the two classes disagree about.
    """
    try:
        from lbl.instruments import select
    except Exception as exc:                                  # noqa: BLE001
        return True, "LBL is not importable here (%s), so nothing was checked" % exc
    try:
        # THE INSTRUMENT AND THE SOURCE ARE PASSED EXPLICITLY, NOT THROUGH
        # THE YAML. LBL's YAML reader skips 'None' values (select.py: "skip
        # Nones"), and that is precisely the source name of ESPRESSO, SOPHIE,
        # CARMENES and CORALIE in its own table. Read through the file, it
        # disappeared, and the check failed on "Data source must be defined"
        # for the instruments that work best. Seen on 2026-10-14 on
        # ESPRESSO.
        args = select.parse_args(
            ["INSTRUMENT", "DATA_DIR", "DATA_SOURCE", "DATA_TYPE", "INPUT_FILE"],
            dict(config_file=os.path.abspath(config_file),
                 instrument=str(instrument), data_source=str(data_source)),
            __name__, parse=False)
        inst = select.load_instrument(args, plogger=None)
        image, header = inst.load_science_file(sample)
        inst.get_wave_solution(sample, image, header)
    except Exception as exc:                                  # noqa: BLE001
        return False, str(exc)
    return True, "reads %s" % os.path.basename(sample)


def fit_components(outdir: str) -> int:
    """How many star components the run's fit has; 0 when there is no fit."""
    import numpy as np
    path = os.path.join(outdir, "fit.npz")
    if not os.path.exists(path):
        return 0
    with np.load(path) as blob:
        return int(blob["P"].shape[0])


def star_template(plan, config_file: str, name: str, files, place=True) -> dict:
    """The fit's star template for `name`, made once per fit, put where LBL looks.

    Made beside the run's other outputs as star_template.fits, and made again
    only when the fit's stamp has changed (lbltemplate.fit_stamp). Copied from
    there to LBL's template folder unless LBL built one there itself, which is
    never replaced (lbltemplate.place). Returns the file made, LBL's paths for
    the object's template and mask and its models folder, and what was done.
    """
    from . import lbltemplate as lt

    # the fit is the run's own, or the one a variant reuses (plan["fitdir"])
    fit = os.path.join(plan.get("fitdir") or plan["outdir"], "fit.npz")
    made = os.path.join(plan["outdir"], "star_template.fits")
    if lt.stamp(made) != lt.template_stamp(fit):
        log("making the star template: the fit's star at its mean amplitude,"
            " each order parity with its own residual mean, in LBL's format and"
            " by LBL's own writer", "info")
        seen, _, split = lt.build(plan["cube"], fit, config_file, name, files,
                                  made, run=plan["tag"])
        log("star template %s, covering %.1f%% of the grid; its even and odd"
            " orders differ by %.4f rms in ln f" % (made, 100 * seen, split),
            "value")
    inst = lt.lbl_instrument(config_file, name)
    slot, mask, models = lt.lbl_paths(inst)
    status = lt.place(made, slot) if place else "not placed"
    if status in ("copied", "same"):
        log("%s is measured against the fit's star template: %s"
            % (name, slot), "value")
    elif status == "replaced":
        log("the star template at %s came from an earlier fit and now holds this"
            " one. LBL's mask for %s and any velocity it already measured were"
            " made against the old one, and LBL keeps what is done: remove %s"
            " and lbl/lblrv/%s_%s to measure against this one."
            % (slot, name, mask, name, name), "warn")
    elif status == "theirs":
        log("LBL built a template for %s itself (%s) and it stays: this package"
            " never replaces one of LBL's. Remove it to measure %s against the"
            " fit's star template." % (name, slot, name), "warn")
    return {"made": made, "slot": slot, "mask": mask, "models": models,
            "status": status}


#: the folders of the LBL tree where a product carries the object's name, and
#: where a case collision therefore makes LBL reread the product of ANOTHER
#: run
_NAMED_PRODUCTS = ("templates", "masks", "lblreftable", "lblrdb")


def _warn_case_collisions(data_dir, objects):
    """Say which products of a previous run LBL will reread through a case
    collision, because LBL itself will not say so.

    WHAT HAPPENED on 2026-10-15. The macOS filesystem does not distinguish
    case, while LBL names its products after the object. A previous run on
    HD10700_tc thus left a template, a mask and an .rdb that the run on
    HD10700_TC took for its own:

        Template ... exists. Skipping template creation.
        File lbl_HD10700_TC_HD10700_TC.rdb exists, we will read it.

    The second line is the worst: LBL had just computed 99 fresh velocities,
    and it compiled a 74-line table from another run's file, without
    flagging anything abnormal.

    Nothing is touched: erasing another run's product would be worse. The
    file, the object that claims it and the one that wrote it are named, so
    that the decision is made with full knowledge.
    """
    import os

    wanted = {str(name) for name in objects if name}
    lowered = {name.lower(): name for name in wanted}
    found = []
    for sub in _NAMED_PRODUCTS:
        folder = os.path.join(data_dir, sub)
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            continue
        for entry in entries:
            low = entry.lower()
            for key, asked in lowered.items():
                if key not in low:
                    continue
                # the name as it is written on disk, in place of the name
                # asked for: if they differ only by case, this is a
                # collision and not this run's product
                start = low.index(key)
                written = entry[start:start + len(key)]
                if written != asked:
                    found.append((sub, entry, asked, written))
                break
    if not found:
        return
    log("%d product(s) of the LBL tree differ from this run's object name ONLY"
        " BY CASE, and this filesystem does not tell them apart. LBL will"
        " reuse them as if they were its own:" % len(found), "warn")
    for sub, entry, asked, written in found:
        log("    %-12s %s   (asked %r, on disk %r)"
            % (sub, entry, asked, written), "warn")
    log("    move them aside if they come from another run; LBL rebuilds what"
        " it cannot find.", "warn")


def prepare(plan, code=None) -> dict:
    """Stage both objects, make the star template, write LBL's config and the
    script that runs it. `code` is chosen(config), found here if not given."""
    config = plan["config"]
    block = config.get("lbl") or {}
    object_name = config["input"]["object"]
    instrument, data_source, where = profile(config)
    from .config import lbl_directory
    data_dir = lbl_directory(config)
    before, after = object_names(config, object_name, plan["tag"])
    code = code or chosen(config)

    log("LBL profile: instrument %s, data source %s, from %s"
        % (instrument, data_source, where), "value")
    _warn_case_collisions(data_dir, [before] + ([after] if after else []))

    wanted = []
    corrected = []
    if block.get("before", True):
        wanted.append((before, plan["directory"], plan["files"]))
    if block.get("after", True):
        corrected = corrected_files(plan["corrdir"])
        if not corrected:
            log("no corrected spectra in %s yet, so LBL is set up on the"
                " delivered ones only. Run the correct stage and this stage"
                " again to get the other half of the comparison."
                % plan["corrdir"], "warn")
        else:
            wanted.append((after, plan["corrdir"], corrected))

    objects = []
    mode, why = link_mode(data_dir, block.get("link"))
    if why:
        log(why, "warn")
    for name, source, files in wanted:
        target = os.path.join(data_dir, "science", name)
        linked, kept, strangers = link_spectra(files, target, mode)
        log("%-24s %d spectra (%d new %s, %d already there) from %s"
            % (name, linked + kept, linked, mode + "s", kept, source), "value")
        if strangers:
            log("  %d other .fits are in %s and LBL takes the whole folder, so"
                " they will be measured with the rest: %s%s"
                % (len(strangers), target, ", ".join(strangers[:3]),
                   ", ..." if len(strangers) > 3 else ""), "warn")
        objects.append(name)

    teff, whence = resolve_teff(config, plan["files"])
    if teff is not None:
        log("Teff %.0f K, from %s" % (teff, whence), "value")
    tables = dtemp_table(block, teff)
    if tables:
        log("LBL measures %s on both objects, the temperature table nearest"
            " %s" % (", ".join(tables), "%.0f K" % teff if str(
                block.get("dtemp", "auto")).lower() == "auto" and teff
                else block.get("dtemp")), "value")
    elif str(block.get("dtemp", "auto")).lower() == "auto":
        log("no Teff, so no DTEMP table for LBL to measure", "warn")

    document = config_document(config, data_dir, instrument, data_source, teff)
    config_file = os.path.join(plan["outdir"], "lbl_config.yaml")
    with open(config_file, "w") as handle:
        handle.write("# LBL's own configuration, written by pca2d-preclean.\n"
                     "# Every key here is one LBL knows; it refuses any other.\n")
        yaml.safe_dump(document, handle, sort_keys=False,
                       default_flow_style=False)

    ok, why = check_profile(instrument, data_source, config_file,
                            plan["files"][0] if plan["files"] else None)
    if ok:
        log("profile check: %s/%s %s" % (instrument, data_source, why), "value")
    else:
        log("%s/%s cannot read %s: %s"
            % (instrument, data_source, os.path.basename(plan["files"][0]), why),
            "error")
        log("the instrument and data source together pick the class that reads"
            " the files, and it is set per spectrograph in config.yaml. For a"
            " t.fits with named fibre extensions the reader is the CADC one.",
            "error")

    def params_for(name):
        return runparams(config, data_dir, instrument, data_source, [name],
                         config_file, teff)

    runs, star, strpca = [], None, None
    if before in objects:
        runs.append(("BEFORE", "The spectra as delivered, measured against the"
                     " template LBL builds from them.", params_for(before)))
    if after in objects:
        fitdir = plan.get("fitdir") or plan["outdir"]
        n_star = fit_components(fitdir)
        # a comparison object named in lbl.template is the template, not ours
        use_star = bool(block.get("star_template", True)) and not block.get("template")
        use_strpca = bool(block.get("strpca", True)) and n_star >= 2
        if (use_star or use_strpca) and not n_star:
            log("no fit in %s, so the corrected object gets the template LBL"
                " builds" % fitdir, "warn")
        elif (use_star or use_strpca) and not (ok and available()[0]):
            log("the star template is written by LBL's own writer, and LBL"
                " cannot read these spectra here; the corrected object gets"
                " the template LBL builds", "warn")
        elif use_star or use_strpca:
            linked = [os.path.join(data_dir, "science", after,
                                   os.path.basename(path)) for path in corrected]
            star = star_template(plan, config_file, after, linked, place=use_star)
        placed = bool(star) and star["status"] in ("copied", "same", "replaced")
        runs.append(("AFTER", (
            "The corrected spectra, measured against the fit's star template,"
            " which pca2d-preclean put where LBL looks for it (%s): LBL's"
            " template step finds it there and skips."
            % os.path.basename(star["slot"])) if placed else (
            "The corrected spectra, measured against the template LBL builds"
            " from them."), params_for(after)))
        if star and use_strpca:
            strpca = dict(fit=os.path.abspath(os.path.join(fitdir, "fit.npz")),
                          template=os.path.abspath(star["made"]),
                          mask=star["mask"], models_dir=star["models"],
                          prefix=after, run=plan["tag"])
            log("%d star components: %s go to LBL as RESPROJ tables, written"
                " once its mask has measured the rest frame"
                % (n_star, ", ".join("STRPCA%d" % k for k in range(2, n_star + 1))),
                "value")
            from .lbltemplate import resproj_divides_in_place
            # the LBL that will run, which is not this process's own
            aliased = resproj_divides_in_place(code["where"])
            if dtemp_table(block, teff) and aliased:
                log("DTEMP is the first RESPROJ table and the STRPCA ones come"
                    " after it, and the LBL that runs divides the residual"
                    " in place for each table: DTEMP is right and STRPCA2..%d"
                    " are not. lbl.dtemp: false puts STRPCA2 first again"
                    % n_star, "warn")
            if n_star >= 3 and aliased:
                log("the LBL that runs divides the residual in place for"
                    " each RESPROJ table (frac_diff_seg = diff_seg in"
                    " lbl/science/general.py, diff_seg[i] /= in the speed"
                    " branch's lbl/core/fastmath.py), so every table after the"
                    " first is projected on a residual divided twice: STRPCA2"
                    " is right and STRPCA3..%d are not, until that is a copy"
                    % n_star, "warn")

    script = write_runner(os.path.join(plan["outdir"], "run_lbl.py"), runs,
                          strpca, object_name, plan["tag"], config_file,
                          python=code["python"])

    if runs and runs[0][2]["RUN_LBL_MASK"] and teff is None:
        log("no Teff, and the mask step is on. LBL stops on that rather than"
            " guessing, so either the spectra have to carry one of %s or"
            " lbl.teff has to be a number." % ", ".join(TEFF_KEYS), "warn")

    log("LBL config   %s" % config_file, "value")
    log("LBL script   %s" % script, "value")
    return {"config_file": config_file, "script": script, "objects": objects,
            "data_dir": data_dir, "readable": ok, "star": star,
            "strpca": strpca is not None, "python": code["python"],
            "code": code}


def run(script: str, code: dict = None) -> None:
    """Run the script that was just written, by the LBL lbl.environment chose.

    `code` is chosen(config); without one it is this interpreter's LBL, as it
    was before there was a choice.
    """
    code = code or chosen({"lbl": {"environment": CURRENT}})
    if not code["ok"]:
        raise SystemExit("lbl.run is on, but %s. %s"
                         % (say_chosen(code), how_to_get(code)))
    log("running LBL %s from %s, with %s. This is hours, and LBL prints its"
        " own progress." % (code["detail"], code["where"], code["python"]),
        "info")
    result = subprocess.run([code["python"], script],
                            env=environment_for(code["python"]))
    if result.returncode != 0:
        raise SystemExit("LBL exited %d; the script that ran is %s"
                         % (result.returncode, script))
