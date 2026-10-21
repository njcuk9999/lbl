"""The one command.

    pca2d-preclean --object TOI-2120

That is the whole interface. The object names a folder under the input root,
the instrument is read from the INSTRUME keyword of the files there, and
`config.yaml` supplies everything else in three layers: what is general, what
belongs to that spectrograph, and what belongs to that target.

Two roots, both in the config and both overridable for one run:

    input.directory    read from <root>/<object>/, never written to
    output.directory   written to <root>/<object>/<M>-<N>/, never read from

Four stages, in order, each skippable and each announcing how long it took:

    cube      the spectra registered onto one log-uniform grid
    fit       the two-frame decomposition, star frame and observer frame
    figures   one multipage PDF, parameters first, then every plot
    correct   the observer block divided out, written as t.fits
    lbl       both sets of spectra handed to LBL, delivered and corrected, as
              two objects in one tree. Prepared by default and run when asked.

Nothing is silent. A run takes tens of minutes and the difference between a
working one and a wedged one has to be visible from across the room.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import glob
import os
import re
import shlex
import sys
import time

from .config import (VARIANT_META, cache_key, lbl_directory, load_config,
                     read_yaml, resolve_highpass, spectra_dir)
from .logger import log
from .progress import human, stage

STAGES = ("cube", "fit", "figures", "correct", "lbl")


#: The settings the window can change that are CONFIG keys, and the flag each
#: one travels under. Without these, a value typed in the window reached
#: nothing: only n_star and n_earth had flags, so changing the high pass, the
#: shrinkage or the metric and pressing Run ran the configuration's own values
#: and said nothing about it (2026-09-15). A flag also means a run's own log
#: records what it was asked, which a config file edited afterwards does not.
SETTING_FLAGS = (
    ("--weight", "correct.weight", str,
     "the metric the correction's amplitudes are measured in: 'flux', every"
     " sample as the fit saw it, or 'velocity', each weighted by the star's own"
     " derivative there, (dT/dv)^2. It implies a refit of the amplitudes"),
    ("--velocity-term", "twoframe.velocity_term", "bool",
     "fit one velocity per exposure beside the components"),
    ("--iters", "twoframe.iters", int, "sweeps at most"),
    ("--shrink", "correct.shrink", "bool",
     "divide each observer component out only where it is significant"),
    # the residual clip travelled under no flag at all until 2026-09-25, so a
    # run asked for one on the command line took the configuration's value
    ("--nsig-cut", "correct.nsig_cut", float,
     "NaN every sample whose residual, panel 5, is beyond this many running"
     " robust sigmas; 0 or nothing leaves them all in"),
    ("--excursion-nsig", "correct.excursion_nsig", float,
     "flag every window of the residual, up to --excursion-elements wide,"
     " whose aggregate significance is beyond this; 6 is four independent"
     " pixels at 3 sigma"),
    ("--excursion-elements", "correct.excursion_elements", float,
     "how wide those windows go, in resolution elements of the instrument"),
    ("--weight-floor", "correct.weight_floor", float,
     "NaN la ou la boucle a deprecie l'echantillon sous cette fraction de son"
     " poids nominal: le poids du fit est le masque"),
    ("--bias-nsig", "correct.bias_nsig", float,
     "retirer de toutes les poses les colonnes dont le residu MOYEN sur les"
     " poses depasse ce nombre de sigmas: ce qui est ancre dans le referentiel"
     " de l'observateur et a un signe"),
    ("--bias-elements", "correct.bias_elements", float,
     "la largeur sur laquelle cette moyenne est sommee, en elements de"
     " resolution"),
    ("--excess-nsig", "correct.excess_nsig", float,
     "drop from every exposure the observer columns whose scatter over the"
     " exposures is this many sigmas above the noise"),
    ("--excess-chi2", "correct.excess_chi2", float,
     "and whose windowed chi2 is above this"),
    ("--excess-elements", "correct.excess_elements", float,
     "the width that scatter is averaged over, in resolution elements"),
    ("--column-frac", "correct.column_frac", float,
     "with --column-chi2: drop an observer column from every exposure when"
     " more than this fraction of them is clipped there"),
    ("--column-chi2", "correct.column_chi2", float,
     "and the survivors' reduced chi2 is still above this; noise gives 0.97"
     " after a 3 sigma clip"),
    ("--high-pass", "highpass.width_kms", float,
     "the Savitzky-Golay high pass, in km/s"),
    ("--dv", "domain.dv", float, "the grid step, in km/s"),
    ("--nightly-stack", "input.nightly_stack", str,
     "coadd each night: true, false, or auto"),
    ("--bundle-minutes", "input.bundle_minutes", float,
     "cut each night into bundles of at most this many minutes, each its own"
     " row of the fit; turns stacking on unless --nightly-stack false"),
    ("--bundle-berv", "input.bundle_berv", float,
     "or of at most this BERV range, in km/s"),
    ("--bundle-max", "input.bundle_max", int,
     "or of at most this many exposures"),
    # The LBL page of the window, which travelled under no flag at all: a copy
    # asked for there, or another LBL folder, reached nothing, and the run
    # linked into the configuration's own tree (2026-09-16).
    ("--lbl-dir", "lbl.directory", str,
     "LBL's own tree (its DATA_DIR). Default: <output root>/lbl"),
    ("--lbl-link", "lbl.link", ("symlink", "copy"),
     "how the spectra get into LBL's science folders"),
    ("--lbl-run", "lbl.run", "bool", "run LBL, which is hours"),
    ("--lbl-env", "lbl.environment", str,
     "which LBL runs: a conda environment by name (default lbl-rapide, LBL's"
     " speed branch), 'current' for the one installed beside this package,"
     " or a path to a python"),
    ("--lbl-prepare", "lbl.prepare", "bool",
     "write LBL's config and its run script"),
    ("--lbl-before", "lbl.before", "bool", "measure the delivered spectra too"),
    ("--lbl-after", "lbl.after", "bool", "measure the corrected spectra"),
    ("--lbl-star-template", "lbl.star_template", "bool",
     "measure the corrected spectra against the fit's star"),
    ("--lbl-strpca", "lbl.strpca", "bool",
     "star components past the first as RESPROJ tables"),
    ("--lbl-suffix", "lbl.suffix", str,
     "the corrected object's name after the target's; {tag} is the counts"),
    ("--lbl-teff", "lbl.teff", str, "auto, or a temperature in K"),
    ("--lbl-template", "lbl.template", str,
     "another object's template for both (OBJECT_COMPARISON)"),
    ("--lbl-steps", "lbl.steps", "list",
     "LBL's steps, comma separated: template,mask,compute,compile"),
)


def add_setting_flags(parser):
    """Give the parser one flag per config key the window can change."""
    for flag, path, kind, help_text in SETTING_FLAGS:
        if kind == "bool" or isinstance(kind, tuple):
            parser.add_argument(flag, dest=flag[2:].replace("-", "_"),
                                choices=("true", "false") if kind == "bool"
                                else kind, default=None,
                                help="%s (config %s)" % (help_text, path))
        elif kind == "list":
            parser.add_argument(flag, dest=flag[2:].replace("-", "_"),
                                default=None, metavar="A,B",
                                help="%s (config %s)" % (help_text, path))
        else:
            parser.add_argument(flag, dest=flag[2:].replace("-", "_"),
                                type=kind, default=None,
                                help="%s (config %s)" % (help_text, path))
    return parser


def apply_setting_flags(config, args):
    """Put what the command line said into the resolved configuration, and
    leave that configuration resolved.

    The second half is not decoration. load_config DERIVES highpass.window from
    highpass.width_kms and the grid step; two of these flags, --dv and
    --high-pass, are exactly the two numbers it derived it from. Written in
    here and left at that, the window would keep the value the file's dv gave
    it, while every stage reads the config this run writes and derives it
    again, at the dv that ran. The window is hashed into the cube's cache key
    and the width is not, so the run then had TWO keys: the cube stage built
    cache/cube_..._3e01c018993b and the fit opened cache/cube_..._4e793f8df25f,
    which nothing had ever written. A resolved configuration has to be a fixed
    point of reading it back, or the stages are not looking at one run.
    """
    said = []
    for flag, path, kind, _help in SETTING_FLAGS:
        value = getattr(args, flag[2:].replace("-", "_"), None)
        if value is None:
            continue
        if kind == "bool":
            value = str(value).lower() == "true"
        elif kind == "list":
            value = [part for part in str(value).replace(",", " ").split()]
        section, key = path.split(".")
        config.setdefault(section, {})[key] = value
        said.append("%s = %s" % (path, value))
    if said:
        log("from the command line: %s" % ", ".join(said), "value")
        # what a stage reading this configuration back would derive, derived
        # here instead: the file's own highpass section is what it would see
        resolve_highpass(config, dict(config["highpass"]))
    return config


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="pca2d-preclean", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", default=None, metavar="NAME",
                   help="the target, and the name of its directory under"
                        " --data-dir. Case matters: it is also the key looked"
                        " up under `objects:` in the config. Required unless"
                        " --objects names several")
    p.add_argument("--config", default="config.yaml",
                   help="default: config.yaml beside the repository root")
    p.add_argument("--data-dir", default=None, metavar="DIR",
                   help="override input.directory from the config. It is the"
                        " input ROOT: spectra are read from <DIR>/<object>/")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="override output.directory from the config. It is the"
                        " output ROOT: everything a run writes lands in"
                        " <DIR>/<object>/<M>-<N>/")
    p.add_argument("--fits-dir", default=None, metavar="DIR",
                   help="override output.fits_directory: the disk a run's"
                        " products are KEPT on, each run folder being one link"
                        " to it. Empty keeps everything under the output root."
                        " A path that is not there stops the run rather than"
                        " quietly filling the internal disk")
    add_setting_flags(p)
    p.add_argument("--no-fits-dir", action="store_true",
                   help="keep everything under the output root, whatever the"
                        " config says: the window sends this when its products"
                        " field is empty, so that an empty field is a decision"
                        " and not a silence")
    p.add_argument("--stages", default=",".join(STAGES),
                   help="comma-separated subset of %s, in this order"
                        % ",".join(STAGES))
    p.add_argument("--instrument", default=None,
                   help="override the INSTRUME keyword. Almost never right:"
                        " reading the wrong extensions raises no error, it"
                        " returns different photons")
    p.add_argument("--n-star", type=int, default=None)
    p.add_argument("--n-earth", type=int, default=None)
    p.add_argument("--min-rjd", type=float, default=None, metavar="RJD",
                   help="keep only exposures from this date on (reduced Julian"
                        " date, BJD - 2400000). quality.min_rjd for one run:"
                        " what is excluded is neither fitted nor corrected")
    p.add_argument("--max-rjd", type=float, default=None, metavar="RJD",
                   help="and only up to this one. The pair cuts a campaign to"
                        " a season, which is what the barycentric coverage"
                        " sometimes asks for (docs/options.md)")
    p.add_argument("--windows", nargs="+", default=None,
                   help="centre:width in nm; default is the list in the config")
    p.add_argument("--rebuild-cube", action="store_true",
                   help="ignore any cached cube for this configuration")
    p.add_argument("--clean-cache", action="store_true",
                   help="empty the cache first: cubes, figure snippets and"
                        " telluric maps, everything this package rebuilds on"
                        " its own. With --dry-run it only lists what would go")
    p.add_argument("--run-lbl", action="store_true",
                   help="have the lbl stage run LBL, not only prepare it."
                        " Hours. Same as lbl.run: true in the config")
    p.add_argument("--objects", default=None, metavar="A,B,C",
                   help="fit these objects TOGETHER against one observer basis,"
                        " each keeping its own star spectrum per order parity"
                        " (pca2d.joint). The atmosphere and the instrument are"
                        " the same for all of them; the stars, their BERV"
                        " coverage and their systemic velocities are not, which"
                        " is what makes the shared basis cleaner. Everything"
                        " lands under <output root>/joint/<A+B+C>/, and each"
                        " object is measured by LBL on its own")
    p.add_argument("--name", default=None, metavar="NAME",
                   help="name this run. Its products go to <out root>/_NAME/"
                        " and its LBL object is <object>_PCA2D_<M-N>_NAME, so"
                        " two runs of the same objects at different settings"
                        " never write into one folder nor under one LBL name."
                        " A run whose folder already holds a fit says so."
                        " `--name auto` names it after its targets and a hash"
                        " of this whole command line, so that anything the"
                        " command says - the roots, the counts, the stages,"
                        " every LBL setting - makes it a different run")
    p.add_argument("--variant", default=None, metavar="NAME",
                   help="variants/NAME.yaml, beside the config, on top of it:"
                        " the nominal plus what the variant changes. Its"
                        " products go to <output root>/_NAME and its LBL object"
                        " is <object>_PCA2D_<M-N>_NAME, unless the file says"
                        " otherwise; with reuse_fit it corrects an existing fit")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve everything, print the plan, touch nothing")
    # LBL's own call (lbl.recipes.lbl_pca2d): the corrected spectra go
    # straight into one of its science folders, under their own names, and
    # the folder they are read from is the object whatever the headers say
    p.add_argument("--corrected-dir", default=None, metavar="DIR",
                   help="write the corrected spectra here rather than into"
                        " <run>/corrected; {object} is each object's name")
    p.add_argument("--keep-names", action="store_true",
                   help="each corrected file under its source's own name")
    p.add_argument("--any-object", action="store_true",
                   help="read every file of the object's folder, whatever"
                        " its OBJECT and DRSOBJN say")
    args = p.parse_args(argv)
    # NO config.yaml HERE, which is the case of anyone who installed LBL alone:
    # the nominal configuration that ships with it, rather than a stop
    if args.config == "config.yaml" and not os.path.exists(args.config):
        args.config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "nominal.yaml")
        log("no config.yaml here, so the nominal one: %s" % args.config, "warn")
    return args


def load_variant(config_path, name):
    """variants/<name>.yaml, beside the config file: the nominal plus what the
    variant changes. None when no variant is named."""
    if not name:
        return None
    path = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                        "variants", "%s.yaml" % name)
    if not os.path.exists(path):
        log("no variant %s: there is no %s" % (name, path), "error")
        raise SystemExit(2)
    with open(path, "r") as handle:
        return read_yaml(handle) or {}


def name_variant(config, name, variant, out_dir=None):
    """Where a variant's products go and what its LBL object is called, unless
    its file says: beside the nominal's, under the variant's name, the way
    outputs/_star_spl and TOI2120_PCA2D_1-3_star_spl were named. Returns the
    output root the nominal's runs are under, where a reused fit is found."""
    root = config["output"]["directory"]
    if variant is None:
        return root
    if not out_dir and not (variant.get("output") or {}).get("directory"):
        config["output"]["directory"] = os.path.join(root, "_" + name)
    if not (variant.get("lbl") or {}).get("suffix"):
        config["lbl"]["suffix"] = "%s_%s" % (config["lbl"].get("suffix")
                                             or "_PCA2D_{tag}", name)
    config["variant"] = dict({"name": name},
                             **{k: variant[k] for k in VARIANT_META if k in variant})
    return root


def window_label(args):
    """The name a date window gives itself when none was typed."""
    lo = getattr(args, "min_rjd", None)
    hi = getattr(args, "max_rjd", None)
    if lo is None and hi is None:
        return None
    return "rjd%s-%s" % ("%.0f" % lo if lo is not None else "",
                         "%.0f" % hi if hi is not None else "")


def name_run(config, args):
    """Give this run its own folder and its own LBL object, if it needs one.

    `--name` when it was typed, otherwise the date window's own label when one
    was given, and in both cases the hash of the command after it: two runs
    called `earth3` at different settings are two results, and two of them
    must not write into one folder nor under one LBL object name, since LBL
    globs its science folder and would measure the mixture without a word
    (the {tag} comment in config.yaml). `--name auto` is the hash with the
    targets in front of it and nothing typed.

    A folder `_<name>` that is already there is that run, and is used as it
    is: runs made before the names carried a hash stay where they are, and a
    resume lands where it left off. It is refused only when what is in it was
    made by another command (its own provenance says so).
    """
    from .naming import run_hash, run_name

    label = getattr(args, "name", None) or window_label(args)
    if not label:
        return None
    stamp = run_hash(sys.argv[1:])
    if str(label).strip().lower() == "auto":
        # the command itself, which is what makes this run a different result
        label = run_name(sys.argv[1:], getattr(args, "objects", None)
                         or ([args.object] if getattr(args, "object", None)
                             else []))
    label = re.sub(r"[^0-9A-Za-z._+-]", "_", str(label)).strip("_") or "run"
    from .naming import label_with_hash
    root = config["output"]["directory"]
    label = older_run(root, label, stamp) or label_with_hash(label, stamp)
    config["output"]["directory"] = os.path.join(config["output"]["directory"],
                                                 "_" + label)
    config["lbl"]["suffix"] = "%s_%s" % (config["lbl"].get("suffix")
                                         or "_PCA2D_{tag}", label)
    return label


def older_run(root, label, stamp):
    """`label` when a run of that exact name is already on disk and is this
    one, else None.

    This one means: made by a command that hashed the same, or made before
    2026-09-18, when names began to carry a hash and nothing recorded one.
    Anything else is another run that happened to be given the same name,
    and the hash is what keeps the two apart.
    """
    from . import runs as _runs

    folder = os.path.join(root, "_" + label)
    if not os.path.isdir(folder):
        return None
    found = [_runs.one(f) for f in _runs.folders(folder)]
    hashes = {(run["config"].get("provenance") or {}).get("run_hash")
              for run in found}
    if not hashes or hashes <= {None, "", stamp}:
        log("the folder %s is already there and is this run: it is used as it"
            " is%s" % (folder, ", and was made before the names carried a"
                       " hash" if hashes <= {None, ""} and hashes else ""),
            "value")
        return label
    return None


def reused_fit(root, base, object_name, tag):
    """The run folder a correction-only variant takes its fit from.

    `reuse_fit: nominal` is the unnamed root; anything else names a run, and
    a named run's folder is `_<name>`, which since 2026-09-18 carries a hash
    after the name it was given. So a name matches the folder called exactly
    that, or the one called that plus a hash; a hash on its own matches the
    folder that ends with it; and a path is taken as given. The newest of
    what matches, since a scenario run twice is the same scenario.
    """
    base = str(base)
    if os.sep in base:                       # a folder, said in full
        here = base if os.path.isabs(base) else os.path.join(root, base)
        return here if os.path.exists(os.path.join(here, "fit.npz")) \
            else os.path.join(here, object_name, tag)
    if base == "nominal" and os.path.isdir(os.path.join(root, object_name)):
        return os.path.join(root, object_name, tag)
    here = []
    for name in sorted(os.listdir(root) if os.path.isdir(root) else []):
        if not name.startswith("_"):
            continue
        label = name[1:]
        if label == base or label.startswith(base + "_") \
                or label.endswith("_" + base):
            folder = os.path.join(root, name, object_name, tag)
            if os.path.exists(os.path.join(folder, "fit.npz")):
                here.append((os.path.getmtime(os.path.join(folder, "fit.npz")),
                             folder))
    if here:
        return max(here)[1]
    # nothing yet: the name check_reused_fit will report as missing
    parent = root if base == "nominal" else os.path.join(root, "_" + base)
    return os.path.join(parent, object_name, tag)


def check_reused_fit(plan):
    """Refuse a reused fit that is not there, or that was made from another
    cube than the one this variant's settings give."""
    fitdir = plan["fitdir"]
    for name in ("fit.npz", "twoframe_components.fits", "resolved_config.yaml"):
        if not os.path.exists(os.path.join(fitdir, name)):
            log("this variant reuses the fit in %s, which has no %s: run the"
                " variant it names first" % (fitdir, name), "error")
            raise SystemExit(2)
    base = cache_key(load_config(os.path.join(fitdir, "resolved_config.yaml")))
    if base != plan["key"]:
        log("the fit in %s was made from cube %s and this variant's settings"
            " give cube %s: a variant repeats its base's cube settings"
            % (fitdir, base, plan["key"]), "error")
        raise SystemExit(2)


def run_tag(config):
    """<M>-<N>, and `v` when the velocity term is on: the name of a run."""
    return "%d-%d%s" % (config["twoframe"]["n_star"], config["twoframe"]["n_earth"],
                        "v" if config["twoframe"].get("velocity_term", False) else "")


def without_object(config):
    """The config with the object taken out, so two objects built the same way
    give the same cube key.

    BOTH names of it: the folder and the name matched in the headers. They say
    which object a cube is of, never how it is built, so a target whose folder
    is named otherwise than its headers (GL699_SPIROU for OBJECT = Gl699) is
    built the same way as one whose folder is not. Left in, it gave that target
    a different shared key and a joint run refused the whole set.
    """
    other = copy.deepcopy(config)
    other["input"]["object"] = ""
    other["input"]["object_header"] = None
    return other


def joint_members(args, variant):
    """One entry per object of a joint run: its config, its spectra, its cube.

    Every object is resolved exactly as a solo run resolves it, so each keeps
    its own folder and its own cube; the joint fit adds the one basis they
    share. They must agree on the domain, the step and the high pass, or their
    rows could not sit on one grid.
    """
    members = []
    for name in args.objects:
        cfg = load_config(args.config, object_name=name, data_dir=args.data_dir,
                          out_dir=args.out_dir, instrument=args.instrument,
                          variant=variant)
        if getattr(args, "no_fits_dir", False):
            cfg["output"]["fits_directory"] = None
        elif getattr(args, "fits_dir", None):
            cfg["output"]["fits_directory"] = args.fits_dir
        if getattr(args, "any_object", False):
            cfg["input"]["object_header"] = "*"
        apply_setting_flags(cfg, args)
        # the joint config is a copy of the first member's, taken before a run
        # name moves its output directory, so the tree is resolved here
        cfg["lbl"]["directory"] = lbl_directory(cfg)
        # the date window decides which exposures are IN THE CUBE, so it has to
        # be on every member's own configuration, not only on the joint copy
        # made from the first of them: set there alone, the member cubes would
        # have been built without it and the cut would have done nothing
        for key in ("min_rjd", "max_rjd"):
            if getattr(args, key, None) is not None:
                cfg["quality"][key] = float(getattr(args, key))
        directory = spectra_dir(cfg)
        if not os.path.isdir(directory):
            log("no directory %s: %s has no spectra under the input root"
                % (directory, name), "error")
            raise SystemExit(2)
        key = cache_key(cfg)
        members.append({
            "object": name, "config": cfg, "directory": directory,
            "files": sorted(glob.glob(os.path.join(
                directory, cfg["input"].get("pattern", "*t.fits")))),
            "key": key,
            "shared_key": cache_key(without_object(cfg)),
            "cube": os.path.join(cfg["output"]["cache_directory"],
                                 "cube_%s_%s" % (cfg["input"]["format"], key)),
        })
    return members


def joint_plan(args, variant):
    """The plan of a run that fits several objects against one observer basis."""
    from . import joint as _joint

    members = joint_members(args, variant)
    if len({m["shared_key"] for m in members}) > 1:
        log("these objects are not built the same way, so their rows cannot"
            " share a grid: %s" % ", ".join("%s %s" % (m["object"], m["shared_key"])
                                            for m in members), "error")
        raise SystemExit(2)
    config = copy.deepcopy(members[0]["config"])
    # the same command-line overrides a solo run takes
    apply_setting_flags(config, args)
    config["lbl"]["directory"] = lbl_directory(config)
    if args.n_star is not None:
        config["twoframe"]["n_star"] = args.n_star
    if args.n_earth is not None:
        config["twoframe"]["n_earth"] = args.n_earth
    if args.windows:
        config["output"]["windows"] = list(args.windows)
    if args.rebuild_cube:
        config["output"]["reuse_cache"] = False
        for member in members:
            member["config"]["output"]["reuse_cache"] = False
    if args.run_lbl:
        config.setdefault("lbl", {})["run"] = True
    named = name_run(config, args)
    if named:
        log("run named %s: its own folder and its own LBL object, since it fits"
            " and corrects its own set of exposures. The six characters are"
            " the hash of this command, with --stages, --dry-run,"
            " --rebuild-cube, --clean-cache and --lbl-before left out of it,"
            " so a run resumed keeps its name" % named, "value")
    name = _joint.joint_name(args.objects)
    config["input"]["object"] = name
    # a variant names its own folder and its own LBL objects here, exactly as it
    # does for a solo run (resolve). Without this, two joint variants wrote into
    # ONE folder and their corrected spectra into ONE LBL science folder, where
    # LBL globs the folder and would have measured the mixture of two different
    # corrections without a word.
    # every member's known periods, not the first member's: one shared basis
    # must not vary at ANY of these stars' planet periods, and config is a copy
    # of member 0's, which would have marked only Proxima's
    periods = []
    for member in members:
        for p in ((member["config"].get("target") or {}).get("planets") or []):
            if p not in periods:
                periods.append(float(p))
    config.setdefault("target", {})["planets"] = sorted(periods)
    variant_name = getattr(args, "variant", None)
    name_variant(config, variant_name, variant, args.out_dir)
    if variant and variant.get("reuse_fit"):
        log("variant %s asks to reuse a fit, which a joint run does not do: it"
            " will fit these objects together from the cube" % variant_name,
            "warn")
    # Its own LBL objects, so a joint measurement is never taken for a solo one
    # AND never for another joint one: the tag is the component counts, which
    # two joint runs of different object sets share. PROXIMA+GJ1+GJ3090 and
    # PROXIMA+GJ1+GJ3090+GL699_NIRPS both came out as PROXIMA_PCA2D_0-3_joint,
    # so the second was handed the first's science folder, and would have been
    # measured on the first's spectra had they still been there (2026-09-14).
    import hashlib as _hash
    stamp = _hash.sha1("+".join(sorted(args.objects)).encode()).hexdigest()[:4]
    config["lbl"]["suffix"] = "%s_joint%d%s" % (config["lbl"].get("suffix")
                                                or "_PCA2D_{tag}",
                                                len(args.objects), stamp)
    tag = run_tag(config)
    outdir = os.path.join(config["output"]["directory"], "joint", name, tag)
    plan = {
        "config": config, "objects": list(args.objects), "members": members,
        "directory": ", ".join(m["directory"] for m in members),
        "files": [f for m in members for f in m["files"]],
        "key": members[0]["shared_key"], "tag": tag, "outdir": outdir,
        "corrdir": os.path.join(outdir, "corrected"),
        "cube": _joint.cube_path(config["output"]["cache_directory"],
                                 config["input"]["format"],
                                 members[0]["shared_key"], args.objects),
    }
    plan["keep_names"] = bool(getattr(args, "keep_names", False))
    for k, member in enumerate(members):
        member["star_group"] = 2 * k
        member["corrdir"] = (corrected_dir(args, member["object"], None)
                             if getattr(args, "corrected_dir", None)
                             else os.path.join(plan["corrdir"], member["object"]))
    return plan


def corrected_dir(args, object_name, outdir):
    """Where an object's corrected spectra go: --corrected-dir, {object} being
    its name, or the run's own corrected/ folder."""
    asked = getattr(args, "corrected_dir", None)
    if asked:
        return os.path.abspath(str(asked).replace("{object}", str(object_name)))
    return os.path.join(outdir, "corrected")


def resolve(args):
    """The config for this object, and where its pieces will land."""
    name = getattr(args, "variant", None)
    variant = load_variant(args.config, name)
    config = load_config(args.config, object_name=args.object,
                         data_dir=args.data_dir, out_dir=args.out_dir,
                         instrument=args.instrument, variant=variant)
    if getattr(args, "no_fits_dir", False):
        config["output"]["fits_directory"] = None
    elif getattr(args, "fits_dir", None):
        config["output"]["fits_directory"] = args.fits_dir
    if getattr(args, "any_object", False):
        config["input"]["object_header"] = "*"
    # the root as the command line and the config gave it, before a variant or
    # a run name adds its own level: LBL's tree hangs off THIS one
    out_root = config["output"]["directory"]
    root = name_variant(config, name, variant, args.out_dir)
    directory = spectra_dir(config)
    if not os.path.isdir(directory):
        log("no directory %s. The object names a folder under the input root,"
            " which is input.directory in %s (%s) unless --data-dir overrides"
            " it." % (directory, args.config, config["input"]["directory"]),
            "error")
        raise SystemExit(2)
    # what the command line said about the settings, before anything reads them
    apply_setting_flags(config, args)
    config["lbl"]["directory"] = lbl_directory(config, out_root)
    if args.n_star is not None:
        config["twoframe"]["n_star"] = args.n_star
    if args.n_earth is not None:
        config["twoframe"]["n_earth"] = args.n_earth
    for key in ("min_rjd", "max_rjd"):
        if getattr(args, key, None) is not None:
            config["quality"][key] = float(getattr(args, key))
    named = name_run(config, args)
    if named:
        log("run named %s: its own folder and its own LBL object, since it fits"
            " and corrects its own set of exposures. The six characters are"
            " the hash of this command, with --stages, --dry-run,"
            " --rebuild-cube, --clean-cache and --lbl-before left out of it,"
            " so a run resumed keeps its name" % named, "value")
    if args.windows:
        config["output"]["windows"] = list(args.windows)
    if args.rebuild_cube:
        # build it again AND keep it: the fit reads the cube from the cache,
        # so a rebuild that wrote nothing left the next stage with nothing
        config["output"]["reuse_cache"] = False
    if args.run_lbl:
        config.setdefault("lbl", {})["run"] = True

    # `v` when the velocity term is on, so a run with it and a run without it
    # do not write into the same folder, and their LBL objects do not either.
    # Comparing the two is the reason the knob exists at all.
    tag = "%d-%d%s" % (config["twoframe"]["n_star"],
                       config["twoframe"]["n_earth"],
                       "v" if config["twoframe"].get("velocity_term", False) else "")
    # Everything this run writes hangs off one root, the corrected spectra
    # included: they used to land in a subfolder of the input directory, which
    # made the input tree both read and written and meant a shared or
    # read-only archive of spectra could not be used as it stands.
    outdir = os.path.join(config["output"]["directory"], args.object, tag)
    plan = {
        "config": config,
        "directory": directory,
        "files": sorted(glob.glob(os.path.join(
            directory, config["input"].get("pattern", "*t.fits")))),
        "key": cache_key(config),
        "tag": tag,
        "outdir": outdir,
        "corrdir": corrected_dir(args, args.object, outdir),
        "keep_names": bool(getattr(args, "keep_names", False)),
    }
    plan["cube"] = os.path.join(config["output"]["cache_directory"],
                                "cube_%s_%s" % (config["input"]["format"],
                                                plan["key"]))
    if variant and variant.get("reuse_fit"):
        plan["fitdir"] = reused_fit(root, variant["reuse_fit"], args.object, tag)
        refuse_to_overwrite_the_reused_fit(
            plan["outdir"], plan["fitdir"],
            (config.get("variant") or {}).get("name"), root)
    return plan


def refuse_to_overwrite_the_reused_fit(outdir, fitdir, name, root):
    """Stop a variant that borrows a fit and would write over that run.

    Its corrected spectra would replace the ones the older run was measured
    on, under the same names, while its own LBL object stays separate: that
    run's velocities would no longer be the velocities of the files on disk.
    It happens with an explicit --out-dir, which overrides the `_<name>`
    folder name_variant gives a variant (2026-09-25: variants/0-7chi2 with
    --out-dir outputs overwrote 19 of TOI2120 0-7's corrected spectra before
    it was stopped).
    """
    if os.path.abspath(outdir) != os.path.abspath(fitdir):
        return
    raise SystemExit(
        "variant %s reuses the fit in %s and would write its own corrected"
        " spectra into that same folder, over the ones the run there was"
        " measured on. Leave --out-dir out, so the variant writes to %s"
        " instead, or give the variant its own output.directory."
        % (name or "?", fitdir,
           os.path.join(root or "outputs", "_" + str(name or "variant"))))


def announce(args, plan):
    """Say what is about to happen, in the terms the config used."""
    cfg = plan["config"]
    log("object      %s" % args.object, "value")
    log("instrument  %s   (read from INSTRUME, not chosen)"
        % cfg["input"].get("instrument", "?"), "value")
    log("spectra     %d files in %s" % (len(plan["files"]), plan["directory"]),
        "value")
    source = ("measured, %.0f%% of the %.2f km/s finest pixel; domain.dv in the"
              " config is not used" % (100 * cfg["domain"]["dv"]
                                       / cfg["domain"]["pixel_dv"],
                                       cfg["domain"]["pixel_dv"])
              if cfg["domain"].get("pixel_dv") else "domain.dv")
    log("domain      %.1f to %.1f nm at %.2f km/s per sample   (%s)"
        % (cfg["domain"]["wave_min"], cfg["domain"]["wave_max"],
           cfg["domain"]["dv"], source), "value")
    log("components  %d in the star's frame, %d in the observer's"
        % (cfg["twoframe"]["n_star"], cfg["twoframe"]["n_earth"]), "value")
    log("cube        %s" % plan["cube"], "value")
    log("outputs     %s" % plan["outdir"], "value")
    log("corrected   %s" % plan["corrdir"], "value")
    if plan.get("objects"):
        log("objects     %s, fitted together against ONE observer basis, each"
            " with its own star spectrum per order parity"
            % ", ".join(plan["objects"]), "value")
    if cfg.get("variant"):
        log("variant     %s: variants/%s.yaml on top of the config%s"
            % (cfg["variant"]["name"], cfg["variant"]["name"],
               "; the fit is %s's" % plan["fitdir"] if plan.get("fitdir") else ""),
            "value")


def reuses_cache(plan):
    """Whether a cube already on disk is read back rather than built again."""
    out = plan["config"]["output"]
    return bool(out["use_cache"] and out.get("reuse_cache", True))


def build_member_cube(plan, member, build_main):
    """One object's own cube, from its own configuration written beside the run.

    The member's config is a file on disk before it is built, because that file
    is the recipe the cube carries (build.main copies it into the cube folder)
    and because a stage is handed a file, never a dictionary.
    """
    import yaml

    path = os.path.join(plan["outdir"],
                        "cube_config_%s.yaml" % member["object"])
    with open(path, "w") as handle:
        yaml.safe_dump(member["config"], handle, sort_keys=False,
                       default_flow_style=False)
    log("  %-10s %d spectra of its own" % (member["object"],
                                           len(member["files"])), "info")
    build_main([path] + ([] if plan["config"]["output"]["use_cache"]
                         else ["--no-cache"]))


def run_joint_cube(plan, build_main):
    """Each object's own cube, then the one cube their rows share."""
    from . import joint as _joint

    if reuses_cache(plan):
        why = cube_ready(plan["cube"])
        if why is None:
            log("a joint cube for these objects and this configuration is"
                " already on disk, reusing it: %s" % plan["cube"], "warn")
            return
        if os.path.isdir(plan["cube"]):
            log("the joint cube at %s is %s, so it is built again rather than"
                " reused" % (plan["cube"], why), "warn")
    for member in plan["members"]:
        if reuses_cache(plan):
            why = cube_ready(member["cube"])
            if why is None:
                log("  %-10s its own cube is already built: %s"
                    % (member["object"], member["cube"]))
                continue
            if os.path.isdir(member["cube"]):
                log("  %-10s its cube is %s, building it again"
                    % (member["object"], why), "warn")
        build_member_cube(plan, member, build_main)
    # Checked NOW, and not only when the loop decided what to build: the loop
    # skips a member whose cube is already there, and an hour later, when the
    # last member is finally built, that cube can be gone. It was, on
    # 2026-09-15: a cache emptied from the window's cleanup page while the run
    # was going took two members' cubes with it, and the joint build died in
    # np.load on GL205's grid.npy. There is one thing to do about it, which is
    # to build it again.
    for member in plan["members"]:
        why = cube_ready(member["cube"])
        if why is None:
            continue
        log("  %-10s its cube is %s, and it was there when this stage began:"
            " something emptied the cache while the run was going. Building it"
            " again" % (member["object"], why), "warn")
        build_member_cube(plan, member, build_main)
    _joint.build([m["cube"] for m in plan["members"]],
                 [m["object"] for m in plan["members"]],
                 plan["cube"], plan["written_config"])


#: the files a cube folder has to hold for anything to read it. `snippets` is
#: not one: it is a convenience for the figures and they redraw without it
CUBE_FILES = ("data.npy", "grid.npy", "sigma.npy", "meta.fits")
#: the stages that open a cube. lbl works from the corrected files instead
NEEDS_CUBE = ("fit", "figures", "correct")
#: what the fit leaves behind, and which stage opens which. Deleting either is
#: the same trap as deleting a cube, one stage further along: reconstruct.py
#: raises a bare FileNotFoundError on twoframe_components.fits, and a missing
#: fit.npz puts sequence.py on the report's "what did not build" page instead
#: of drawing anything
FIT_FILES = {"figures": "fit.npz", "correct": "twoframe_components.fits"}


def cubes_of(plan):
    """[(what it is for, path)] of every cube a run reads, in that order.

    A joint run builds one cube per member and then the cube their rows share,
    and the joint one is the only thing the fit opens; the members' are read to
    make it. All of them are checked, because a member's cube missing is the
    same kind of surprise one stage later.
    """
    cubes = [("%s, its own" % m["object"], m["cube"])
             for m in plan.get("members") or []]
    cubes.append((str(plan["config"]["input"].get("object") or "the run"),
                  plan["cube"]))
    return cubes


def cube_ready(path):
    """None when a cube can be read, otherwise why it cannot.

    The one place that decides whether a cube on disk counts as a cube. Every
    caller that was about to reuse one asks here: "is the folder there" is not
    the same question, and a cache emptied while a run was not looking can
    leave the folder and take data.npy with it. Reusing THAT reports a cube hit
    and then dies in np.load, which is worse than no cube at all.
    """
    if not os.path.isdir(path):
        return "not there"
    gone = [f for f in CUBE_FILES if not os.path.exists(os.path.join(path, f))]
    return "incomplete, no " + ", ".join(gone) if gone else None


def missing_cubes(plan):
    """[(what it is for, path, why)] of the cubes that are not usable."""
    out = []
    for what, path in cubes_of(plan):
        why = cube_ready(path)
        if why:
            out.append((what, path, why))
    return out


def missing_fit(plan, wanted):
    """[(stage, file, path)] of the fit products a stage would open and cannot.

    Nothing for a variant that reuses another run's fit: that fit is somewhere
    else and check_reused_fit is the one that looks for it.
    """
    if plan.get("fitdir"):
        return []
    out = []
    for stage, name in FIT_FILES.items():
        if stage not in wanted:
            continue
        path = os.path.join(plan["outdir"], name)
        if not os.path.exists(path):
            out.append((stage, name, path))
    return out


def check_cubes(plan, wanted):
    """The cube half of check_inputs, kept apart because it is the older half.

    Returns the stage list to run. A cube that is not there is put back on the
    list of things to build, loudly: a user deleted their cache on 2026-09-15,
    started a run with the cube stage unticked because it had always been
    cached before, and got a FileNotFoundError out of np.load after the fit
    stage had already announced itself. There is exactly one thing to do about
    a missing cube, and it is to build it; the only choice is whether that is
    said out loud or found out from a traceback.
    """
    if not any(stage in wanted for stage in NEEDS_CUBE):
        return wanted
    gone = missing_cubes(plan)
    if not gone:
        return wanted
    for what, path, why in gone:
        log("the cube for %s is %s: %s" % (what, why, path), "warn")
    if "cube" in wanted:
        log("the cube stage is in this run, so %d will be built before"
            " anything reads them" % len(gone), "info")
        return wanted
    log("the cube stage was not asked for, and %s would open %s. Building"
        " %s first: it reads every spectrum once and takes a few minutes"
        % (", ".join(s for s in NEEDS_CUBE if s in wanted),
           "them" if len(gone) > 1 else "it",
           "them" if len(gone) > 1 else "it"), "warn")
    return ["cube"] + list(wanted)


def rebuild_missing_cubes(plan, stage_name):
    """A cube the next stage will open, gone since the run began: built again.

    check_inputs looks before anything runs, and the cube stage builds what was
    missing then. This is about the interval SINCE: a cache emptied while the
    run was going, from the window's cleanup page or --clean-cache or another
    run, takes cubes out from under the stage about to read them. It happened
    on 2026-09-15, between a joint run's member loop and its own joint build,
    and came out as a FileNotFoundError in np.load with 20 minutes of building
    behind it. There is one thing to do about a missing cube and it is to build
    it; the only choice is whether that is said or found out from a traceback.
    """
    gone = missing_cubes(plan)
    if not gone:
        return
    for what, path, why in gone:
        log("%s is about to open the cube for %s and it is %s: %s"
            % (stage_name, what, why, path), "warn")
    log("it was there when this run started, so something emptied the cache"
        " while the run was going. Building it again rather than stopping:"
        " that reads every spectrum once", "warn")
    run_cube(plan)
    gone = missing_cubes(plan)
    if not gone:
        return
    if not plan["config"]["output"]["use_cache"]:
        log("output.use_cache is false, so the cube stage keeps nothing on"
            " disk and there is nothing for %s to open. Set it to true, or run"
            " every stage in one go" % stage_name, "error")
    else:
        log("the cube is still not there after building it: %s. Nothing else"
            " here can fix that" % ", ".join(p for _w, p, _y in gone), "error")
    raise SystemExit(2)


def check_inputs(plan, wanted):
    """Before anything runs: everything a remaining stage will open, checked.

    The stages are a chain, cube to fit to figures and correct, and each link
    is skippable precisely so that work already done is not redone. That makes
    every link a place where a file can be gone and nobody notices until the
    stage that needs it opens it. Backwards along the chain: the fit's products
    first, so that putting the fit back is then itself checked for a cube.

    Not lbl: the one thing IT reads from an earlier stage is the corrected
    spectra, and lbl.prepare already says, in words, that it found none and
    what to run. A missing file that is already explained is not this
    function's business.
    """
    wanted = list(wanted)
    gone = missing_fit(plan, wanted)
    if gone and "fit" not in wanted:
        for stage, name, path in gone:
            log("%s would open %s and it is not there: %s"
                % (stage, name, path), "warn")
        log("the fit stage was not asked for. Running it first: this is the"
            " long one", "warn")
        wanted = ["fit"] + wanted
    elif gone:
        for stage, name, _path in gone:
            log("%s needs %s, which this run's fit stage will write"
                % (stage, name), "info")
    return check_cubes(plan, wanted)


def say_lbl_tree(plan, dry_run=False):
    """Where LBL will write, and whether the spectra will be linked or copied.

    Said at the top, because the LBL stage comes after the fit, and finding
    out there that a disk takes no links, or that the tree is not the one
    holding the delivered object's hours of LBL, is finding out too late. A
    dry run touches nothing, so it names the folder without probing it.
    """
    from . import lbl as splbl

    block = plan["config"].get("lbl") or {}
    tree = lbl_directory(plan["config"])
    log("LBL tree    %s" % tree, "value")
    # which LBL, here and not after the fit: an environment that is not on
    # this machine found hours in is hours lost. Asking touches no file, so
    # a dry run is told too, and only a real run is stopped.
    code = splbl.chosen(plan["config"])
    if code["ok"]:
        log(splbl.say_chosen(code), "value")
    elif block.get("run", False):
        log(splbl.say_chosen(code), "error")
        log(splbl.how_to_get(code), "error")
        if not dry_run:
            raise SystemExit(2)
    else:
        log("%s. LBL is only prepared by this run, so it goes on; %s"
            % (splbl.say_chosen(code), splbl.how_to_get(code)), "warn")
    if dry_run:
        return
    mode, why = splbl.link_mode(tree, block.get("link"))
    if why:
        log(why, "warn")
    else:
        log("the spectra go into it as %ss" % mode, "info")


def warn_if_run_exists(plan, wanted=STAGES):
    """Say so when this run's folder already holds a fit.

    Not a refusal: re-running is how a fit is redone with a changed setting,
    and the stages are skippable precisely so that one can be. But finding
    somebody else's fit under your own name, silently, is how two experiments
    become one set of numbers.
    """
    fit = os.path.join(plan["outdir"], "fit.npz")
    if not os.path.exists(fit):
        return False
    # only the stages that write over it: resuming LBL on a finished fit is
    # what the stages are skippable for, and "overwrites it" would be untrue
    if not {"fit", "correct"} & set(wanted):
        return False
    import datetime as _dt
    when = _dt.datetime.fromtimestamp(os.path.getmtime(fit))
    n = len(glob.glob(os.path.join(plan["corrdir"], "**", "*.fits"),
                      recursive=True))
    log("this run already exists: %s holds a fit from %s%s. Running again"
        " overwrites it; --name gives this one a folder of its own."
        % (plan["outdir"], when.strftime("%d %b %H:%M"),
           " and %d corrected spectra" % n if n else ""), "warn")
    return True


def run_cube(plan):
    from .build import main as build_main

    if plan.get("members"):
        return run_joint_cube(plan, build_main)
    if reuses_cache(plan):
        why = cube_ready(plan["cube"])
        if why is None:
            log("a cube for this exact configuration is already on disk,"
                " reusing it: %s" % plan["cube"], "warn")
            log("--rebuild-cube forces it to be built again", "warn")
            return
        if os.path.isdir(plan["cube"]):
            log("the cube at %s is %s, so it is built again rather than"
                " reused" % (plan["cube"], why), "warn")
    cfg = plan["config"]
    log("building the cube: %d spectra onto %.0f-%.0f nm at %.2f km/s"
        % (len(plan["files"]), cfg["domain"]["wave_min"],
           cfg["domain"]["wave_max"], cfg["domain"]["dv"]), "info")
    log("this reads every file once and takes a few minutes", "info")
    argv = [plan["written_config"]]
    if not cfg["output"]["use_cache"]:
        argv.append("--no-cache")
    # LE REPERE DEMANDE EST RESPECTE. build.main ramene registration.frame a
    # 'observer' sauf si on le lui interdit, parce que c'est le reglage de
    # presque tous les runs et qu'un frame laisse par megarde dans un fichier
    # de config coutait vingt minutes de cube inutile. Mais quand la config
    # resolue demande vraiment un autre repere, c'est un choix et non un
    # oubli: le passe-haut contre un template median, par exemple, n'a de
    # sens que si les raies stellaires sont co-incidentes.
    if (cfg.get("registration") or {}).get("frame", "observer") != "observer":
        argv.append("--allow-any-frame")
    build_main(argv)


def run_fit(plan):
    from .twoframe import main as fit_main

    cfg = plan["config"]
    log("fitting %d star and %d observer components by block coordinate"
        " descent, at most %d sweeps"
        % (cfg["twoframe"]["n_star"], cfg["twoframe"]["n_earth"],
           cfg["twoframe"]["iters"]), "info")
    log("this is the long one: it prints an R2 after every sweep, and stops"
        " when chi2 has clearly turned over", "info")
    fit_main(["--config", plan["written_config"], "--cube", plan["cube"],
              "--outdir", plan["outdir"]])


def run_figures(plan):
    from .config import check_windows
    from .figures.bundle import main as bundle_main

    check_windows(plan["config"]["output"]["windows"], plan["config"]["domain"])
    log("drawing %d windows and binding everything into one PDF"
        % len(plan["config"]["output"]["windows"]), "info")
    log("each figure script reads the cube once; expect a few minutes", "info")
    argv = ["--config", plan["written_config"], "--cube", plan["cube"],
            "--outdir", plan["outdir"],
            "--windows", *[str(w) for w in plan["config"]["output"]["windows"]]]
    # a joint run's object is a NAME, not a folder: input.object is
    # PROXIMA+GJ1+GJ3090 and there is no such directory. The figures that open
    # real spectra are pointed at the first member's folder instead, which is
    # where the sky they draw was recorded.
    if plan.get("members"):
        argv += ["--source-dir", plan["members"][0]["directory"]]
    bundle_main(argv)
    render_report(plan)


def render_report(plan):
    """The run's LaTeX report, written again from what is on disk.

    At the figures stage, and again after LBL with the velocities. Never
    fatal: without pdflatex, or when it fails, the PDF the figures stage bound
    stays the report, and texreport says why. Returns the report, or None.
    """
    from . import texreport
    stars = ([m["object"] for m in plan["members"]] if plan.get("members")
             else None)
    try:
        return texreport.render(plan["outdir"], config=plan["config"],
                                stars=stars)
    except Exception as exc:                                  # noqa: BLE001
        log("the LaTeX report could not be written (%s: %s); the PDF the"
            " figures stage bound stays the report"
            % (type(exc).__name__, exc), "warn")
        return None


def run_correct(plan):
    from .reconstruct import main as apply_main

    cfg = plan["config"]
    n_star = cfg["correct"]["n_star"]
    n_earth = cfg["correct"]["n_earth"]
    if n_earth is None:
        n_earth = cfg["twoframe"]["n_earth"]
    log("dividing out the observer block, its per-parity mean and %d"
        " components%s, and setting to NaN every sample the fit gave no weight:"
        " exactly panel 3 of the sequence figure"
        % (n_earth, " with %d star components" % n_star if n_star else ""), "info")
    mode = correct_mode(plan)
    if str((cfg.get("correct") or {}).get("weight") or "flux") == "velocity" \
            and "--refit" not in mode:
        # measuring the amplitudes in another metric means measuring them again,
        # whatever the cube's rows are
        mode = list(mode) + ["--refit"]
    if "--config" not in mode:
        # LA CONFIGURATION PART TOUJOURS, PAS SEULEMENT POUR REAJUSTER. Elle
        # dit par quel lecteur les spectres ont ete lus, donc quelle extension
        # porte le flux a remplacer: sans elle, un instrument qui passe par le
        # pont de LBL retombe sur la table d'APERO et la correction s'arrete
        # sur « unsupported INSTRUME 'HARPS' ». Vu le 2026-10-14 avec
        # correct.weight: flux, le seul chemin qui ne reajuste pas.
        mode = list(mode) + ["--config", plan["written_config"]]
    if "--refit" in mode:
        log("the fit's rows are nights: every exposure of every fitted night is"
            " corrected with its own coefficients, solved for that file against"
            " the fixed basis", "info")
    # one target, or each object of a joint fit with its own star spectra
    targets = plan.get("members") or [{"object": cfg["input"].get("object"),
                                       "directory": plan["directory"],
                                       "corrdir": plan["corrdir"],
                                       "files": plan["files"], "star_group": 0}]
    for target in targets:
        log("writing %d corrected spectra to %s%s"
            % (len(target["files"]), target["corrdir"],
               " (star spectra %d and %d of the joint fit)"
               % (target["star_group"], target["star_group"] + 1)
               if len(targets) > 1 else ""), "info")
        apply_main([
            "--correct", *mode,
            "--fits", os.path.join(plan.get("fitdir") or plan["outdir"],
                                   "twoframe_components.fits"),
            "--n-star", str(n_star), "--n-earth", str(n_earth),
            "--max-sky-ratio", str(cfg["quality"]["max_sky_ratio"] or 0),
            "--source-dir", target["directory"], "--cube", plan["cube"],
            "--corrected-dir", target["corrdir"], "--overwrite",
            *(["--keep-name"] if plan.get("keep_names") else []),
            "--star-group", str(target.get("star_group", 0)),
            *clip_args(cfg), *shrink_args(cfg), *diagnostic_args(cfg)])


def nightly_stacked(cube):
    """Whether a cube's rows are nights, several exposures coadded, rather than
    one exposure each: its meta.fits counts them in n_exposures."""
    from astropy.table import Table
    path = os.path.join(cube, "meta.fits")
    if not os.path.exists(path):
        return False
    meta = Table.read(path)
    return "n_exposures" in meta.colnames and int(max(meta["n_exposures"])) > 1


def correct_mode(plan):
    """How the correct stage picks the files and their coefficients.

    A fit of single exposures has a coefficient row for every file: --all.
    A fit of nights does not, and the night's row is not any one exposure's:
    the sky of a night is not one sky. Then every t.fits of a fitted night is
    corrected, each with its own amplitudes solved against the fixed basis
    (reconstruct --by-night --refit), from the configuration the cube was
    built with.
    """
    if nightly_stacked(plan["cube"]):
        return ["--by-night", "--refit", "--config", plan["written_config"]]
    return ["--all"]


def clip_args(cfg):
    """The correct stage's residual clip, when correct.nsig_cut asks for one:
    panel 5 beyond that many running robust sigmas over the high pass's window."""
    from .outliers import element_samples

    corr = cfg.get("correct") or {}
    nsig = corr.get("nsig_cut")
    excursion = corr.get("excursion_nsig")
    elements = corr.get("excursion_elements")
    floor = corr.get("weight_floor")
    if (not nsig and not (excursion and elements)
            and not corr.get("excess_nsig") and not corr.get("bias_nsig")
            and not floor):
        return []
    out = ["--clip-window", str(int(cfg["highpass"]["window"]))]
    if excursion and elements:
        resolution = (cfg.get("twoframe") or {}).get("resolution") or 70000
        samples = element_samples(resolution, cfg["domain"]["dv"], elements)
        log("and flagging every excursion of the residual, panel 5, up to %.1f"
            " resolution elements wide (%d samples at R = %d and %.2f km/s),"
            " whose aggregate significance is beyond %.1f sigma"
            % (float(elements), samples, int(resolution), cfg["domain"]["dv"],
               float(excursion)), "info")
        out += ["--excursion-nsig", str(float(excursion)),
                "--excursion-samples", str(int(samples))]
    if floor:
        log("et mettant a NaN ce que la boucle a deprecie sous %.2f de son"
            " poids nominal: le poids du fit est le masque" % float(floor),
            "info")
        out += ["--weight-floor", str(float(floor))]
    bias = corr.get("bias_nsig")
    if bias:
        resolution = (cfg.get("twoframe") or {}).get("resolution") or 70000
        wide = element_samples(resolution, cfg["domain"]["dv"],
                               corr.get("bias_elements") or 2.0)
        log("et retirant de toutes les poses les colonnes dont le residu MOYEN"
            " sur les poses, somme sur %d echantillons, depasse %.0f sigmas:"
            " ce qui est ancre dans le referentiel de l'observateur"
            % (wide, float(bias)), "info")
        out += ["--bias-nsig", str(float(bias)),
                "--bias-samples", str(int(wide))]
    excess = corr.get("excess_nsig")
    if excess:
        resolution = (cfg.get("twoframe") or {}).get("resolution") or 70000
        wide = element_samples(resolution, cfg["domain"]["dv"],
                               corr.get("excess_elements") or 2.0)
        log("and dropping from every exposure the observer columns whose scatter"
            " over the exposures, averaged over %d samples, is %.0f sigmas above"
            " the noise and above a chi2 of %.2f: what stands still in the"
            " observer's frame"
            % (wide, float(excess), float(corr.get("excess_chi2") or 0)), "info")
        out += ["--excess-nsig", str(float(excess)),
                "--excess-samples", str(int(wide))]
        if corr.get("excess_chi2"):
            out += ["--excess-chi2", str(float(corr["excess_chi2"]))]
    if nsig:
        log("and setting to NaN every sample whose residual, panel 5, is beyond"
            " %.1f running robust sigmas" % float(nsig), "info")
        out += ["--nsig-cut", str(float(nsig))]
    frac, chi2 = corr.get("column_frac"), corr.get("column_chi2")
    if frac and chi2:
        log("and dropping from every exposure each observer column where more"
            " than %.0f%% of them is clipped and the survivors' reduced chi2 is"
            " still above %.2f" % (100 * float(frac), float(chi2)), "info")
        out += ["--column-frac", str(float(frac)),
                "--column-chi2", str(float(chi2))]
    return out


def diagnostic_args(cfg):
    """Les tests de diagnostic de l'etape correct (pca2d.diagnostics): le
    support tellurique et la permutation des amplitudes entre nuits.

    A part de shrink_args, qui part aussi vers la figure de sequence: ces deux
    options ne sont pas des corrections a dessiner mais des questions posees
    aux fichiers, et la figure ne les connait pas."""
    corr = cfg.get("correct") or {}
    out = []
    if corr.get("q_support"):
        out += ["--q-support", str(float(corr["q_support"]))]
        log("DIAGNOSTIC: the observer basis divided out only where the"
            " telluric transmission falls below %.4f on some exposure; the"
            " sequence figure does not show this" % float(corr["q_support"]),
            "warn")
    if corr.get("permute_b") is not None:
        out += ["--permute-b", str(int(corr["permute_b"]))]
        log("DIAGNOSTIC: each night corrected with the observer amplitudes of"
            " ANOTHER night (seed %d); a null test, not a correction"
            % int(corr["permute_b"]), "warn")
    return out


def shrink_args(cfg):
    """The correct stage's own options, which the sequence figure needs too so
    that its panels 3 and 6 show what the files hold: the shrinkage of the
    observer components by their significance and its two Savitzky-Golay
    variants (pca2d.shrink), both measured in the instrument's resolution
    element, twoframe.resolution, and which samples a corrected file blanks."""
    corr = cfg.get("correct") or {}
    out = []
    mask = str(corr.get("mask") or "exposure")
    share = corr.get("mask_share")
    if mask != "exposure":
        out += ["--mask", mask]
        if mask == "common" and share is not None and float(share) < 1.0:
            out += ["--mask-share", str(float(share))]
            log("corrected files blank every sample fewer than %.0f%% of the"
                " exposures weighted, so all of them carry the same set of"
                " lines, decided by a majority rather than unanimously"
                % (100 * float(share)), "info")
        else:
            log("corrected files blank %s"
                % ("every sample any exposure left unweighted, so each of them"
                   " carries the same set of lines" if mask == "common"
                   else "nothing: an unweighted sample keeps its delivered flux"),
                "info")
    if corr.get("shrink"):
        out.append("--shrink")
        if corr.get("shrink_smooth"):
            out.append("--shrink-smooth")
    which = corr.get("smooth_components") or []
    if which:
        out += ["--smooth-components", ",".join(str(int(j)) for j in which)]
    # the metric the refitted amplitudes are measured in (pca2d.reconstruct)
    weight = str(corr.get("weight") or "flux")
    if weight == "velocity":
        out += ["--weight", "velocity",
                "--velocity-floor", str(float(corr.get("velocity_floor") or 0))]
        log("the amplitudes are measured on (dT/dv)^2 weights rather than on"
            " the flux: a contaminant moves a line only through its overlap"
            " with the star's derivative. The correction itself is unchanged",
            "info")
    tf = cfg.get("twoframe") or {}
    resolution = tf.get("resolution") or tf.get("star_resolution")
    if out and resolution:
        out += ["--resolution", str(float(resolution))]
    if out:
        log("each observer component divided out only where it is significant,"
            " all exposures together%s%s"
            % (", its significance smoothed over a resolution element"
                         if "--shrink-smooth" in out else "",
                         ", components %s smoothed first" % ",".join(map(str, which))
                         if which else ""), "info")
    return out


def _velocity_pages(plan):
    """The run's own velocities, on the end of the run's own report.

    A correction is worth what it does to the velocities, so they go in the
    document rather than waiting for somebody to run lblscan by hand. Never
    fatal: LBL has already finished by the time this runs, and a report that
    could not be appended to is not a reason to lose the velocities.
    """
    # the whole report again, velocities included; the pages below are what
    # a machine without LaTeX gets instead, on the end of the bound PDF
    if render_report(plan):
        return
    from .rvpages import velocity_pages
    try:
        velocity_pages(plan)
    except Exception as exc:                                  # noqa: BLE001
        log("the RV pages could not be added to the report (%s: %s). The"
            " velocities themselves are in lbl/lblrdb."
            % (type(exc).__name__, exc), "warn")


def _check_lbl_tree(directory):
    """Refuser de lancer LBL sur un arbre dont un dossier est un lien MORT.

    CE QUI S'EST PASSE le 2026-10-15 a 22h55. outputs/lbl/ portait dix liens
    vers /Volumes/irrisor, un disque externe qui n'etait pas monte. LBL s'est
    arrete au bout de 1.9 seconde sur une FileExistsError en voulant creer son
    dossier de log, et le journal de la nuit ne disait rien de plus: les
    quatre etapes d'avant avaient reussi, donc la chaine avait l'air d'etre
    allee au bout.

    Le piege est que Path.exists() SUIT le lien: un lien casse n'existe pas
    pour lui, donc io.make_dir de LBL passe dans sa branche "creer", et le
    mkdir echoue parce que le nom, lui, est bien pris. Les deux moities du
    test portent sur des choses differentes.

    On regarde donc ici, une fois, avant de donner l'arbre a LBL, et on dit
    quel lien et vers quoi plutot que de laisser LBL mourir sur un message qui
    ne nomme pas la cause.
    """
    broken = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return
    for name in names:
        path = os.path.join(directory, name)
        if os.path.islink(path) and not os.path.exists(path):
            broken.append((name, os.readlink(path)))
    if not broken:
        return
    log("%d folder(s) of the LBL tree are DEAD LINKS, and LBL would stop on"
        " one of them without naming it:" % len(broken), "error")
    for name, target in broken:
        log("    %-14s -> %s" % (name, target), "error")
    log("    the disk they point at is not mounted. Mount it, or point them"
        " somewhere that exists, and run the lbl stage again.", "error")
    raise SystemExit(2)


def run_lbl(plan):
    from . import lbl as splbl

    cfg = plan["config"]
    block = cfg.get("lbl") or {}
    if not block.get("prepare", True) and not block.get("run", False):
        log("lbl.prepare and lbl.run are both off, so nothing to do here",
            "warn")
        return
    # the LBL that measures (lbl.environment), asked once for every object
    code = splbl.chosen(cfg)
    log(splbl.say_chosen(code) if code["ok"] else
        "%s. The files below are still written. %s"
        % (splbl.say_chosen(code), splbl.how_to_get(code)),
        "value" if code["ok"] else "warn")
    # With output.fits_directory set, every folder LBL writes in is a link to
    # the external disk before it starts; lbl/science stays here (storage.py)
    from . import storage as _storage
    for sub in _storage.LBL_FOLDERS:
        _storage.link_dir(plan["config"],
                          os.path.join(lbl_directory(plan["config"]), sub))
    _check_lbl_tree(lbl_directory(plan["config"]))
    if plan.get("members"):
        # one LBL object per star, each measured on its own; what they share is
        # the observer basis that corrected them, not their velocities
        for member in plan["members"]:
            theirs = dict(plan)
            theirs["config"] = copy.deepcopy(plan["config"])
            theirs["config"]["input"]["object"] = member["object"]
            theirs["directory"] = member["directory"]
            theirs["files"] = member["files"]
            theirs["corrdir"] = member["corrdir"]
            theirs["fitdir"] = plan.get("fitdir") or plan["outdir"]
            theirs["outdir"] = os.path.join(plan["outdir"], "lbl",
                                            member["object"])
            os.makedirs(theirs["outdir"], exist_ok=True)
            log("LBL for %s, corrected by the joint fit" % member["object"],
                "info")
            prepared = splbl.prepare(theirs, code)
            if block.get("run", False):
                if not prepared.get("readable", True):
                    log("not running LBL for %s: the profile above cannot read"
                        " its spectra" % member["object"], "error")
                    continue
                splbl.run(prepared["script"], code)
        _velocity_pages(plan)
        return
    prepared = splbl.prepare(plan, code)
    if block.get("run", False):
        if not prepared.get("readable", True):
            log("not running LBL: the profile above cannot read these spectra,"
                " and it would take a few hundred megabytes of downloads and a"
                " template to find that out again", "error")
            raise SystemExit(2)
        splbl.run(prepared["script"], code)
        _velocity_pages(plan)
        return
    log("LBL is not run by this stage unless asked. Both objects are staged"
        " and everything it needs is written; to run it:", "info")
    log("    %s %s" % (prepared["python"] or "python", prepared["script"]),
        "value")
    log("or set lbl.run: true in the config, or pass --run-lbl", "info")


def say_if_code_moved(snapshot, stage_name, said):
    """Once per run: the package's code changed on disk since the run began.

    What the run does itself is safe from that (provenance.freeze loaded it all
    at the start). What it starts as another process is not: the figures are
    scripts and LBL is a script, and a new process reads the files as they are.
    """
    from . import provenance as _provenance

    if said:
        return
    moved = _provenance.drift(snapshot)
    if not moved:
        return
    said.append(stage_name)
    log("the code of this package changed on disk while this run was going"
        " (%s%s). Everything this run does itself keeps the version it started"
        " with; the figure scripts and LBL are processes of their own, and from"
        " %s on they run the new one"
        % (", ".join(moved[:4]), ", ..." if len(moved) > 4 else "", stage_name),
        "warn")


def main(argv=None):
    args = parse_args(argv)
    # One version of the code for the whole run: every module now, not each
    # when its stage starts, hours later and perhaps after an edit
    from . import provenance as _provenance
    snapshot = _provenance.freeze()
    moved_said = []
    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        log("unknown stage %s; known ones are %s"
            % (", ".join(unknown), ", ".join(STAGES)), "error")
        raise SystemExit(2)

    started = time.time()
    log("pca2d-preclean", "info")
    if args.objects:
        args.objects = [o.strip() for o in str(args.objects).split(",") if o.strip()]
        args.object = args.object or args.objects[0]
        plan = joint_plan(args, load_variant(args.config,
                                             getattr(args, "variant", None)))
    elif not args.object:
        log("--object NAME, or --objects A,B,C to fit several of them against"
            " one observer basis", "error")
        raise SystemExit(2)
    else:
        plan = resolve(args)
    announce(args, plan)
    warn_if_run_exists(plan, wanted)
    from . import storage as _storage
    where = _storage.check(plan["config"], dry_run=args.dry_run)
    if where:
        log("kept on     %s   (output.fits_directory; %s links there)"
            % (_storage.external(plan["config"], plan["outdir"]),
               plan["outdir"]), "value")
    if args.clean_cache:
        from . import cache as _cache
        _cache.clean(plan["config"], dry_run=args.dry_run)
        # what that leaves the run without is said by check_cubes below, which
        # says it for a cache emptied by anything and not only by this flag
    if not plan["files"]:
        log("no spectra matched %s in %s"
            % (plan["config"]["input"].get("pattern"), plan["directory"]),
            "error")
        raise SystemExit(2)
    if plan.get("fitdir"):
        # a variant that only changes the correction: the fit, and the cube
        # and figures that go with it, are its base's
        check_reused_fit(plan)
        skipped = [s for s in ("cube", "fit", "figures") if s in wanted]
        wanted = [s for s in wanted if s not in skipped]
        if skipped:
            log("the fit is %s's, so %s not run for this variant"
                % (plan["fitdir"], ", ".join(skipped)), "info")
    # everything a remaining stage will open, checked here and not found
    # missing by np.load three stages in
    wanted = check_inputs(plan, wanted)
    if "lbl" in wanted:
        say_lbl_tree(plan, dry_run=args.dry_run)
    if args.dry_run:
        log("dry run: stopping here, nothing written", "warn")
        return None

    # The resolved config is written out and every stage is handed THAT file,
    # not the one on the command line: the three layers have been merged and
    # the instrument resolved, so this is what actually ran, and it sits next
    # to the outputs it produced.
    # With output.fits_directory set, the run folder is a link to the external
    # disk, made now, before the first file goes into it (storage.py).
    if where:
        _storage.link_dir(plan["config"], plan["outdir"])
    os.makedirs(plan["outdir"], exist_ok=True)
    plan["written_config"] = os.path.join(plan["outdir"], "resolved_config.yaml")
    # and the code that ran it, so the result can be traced and made again
    from .provenance import code_version, stamp as code_stamp
    plan["config"]["provenance"] = dict(code_version())
    # and what it was ASKED, word for word. The resolved config says what every
    # key ended up being, which is not the same thing: it cannot tell a value
    # typed in the window from a value that was in the file all along, and a
    # file edited after the run says whatever it says now.
    plan["config"]["provenance"]["command"] = " ".join(
        shlex.quote(a) for a in sys.argv)
    # and when it started, so that a list of past runs can be read in order
    # without going by the file times of what each one happened to write
    plan["config"]["provenance"]["started"] = datetime.datetime.now(
    ).astimezone().isoformat(timespec="seconds")
    # and which run this is: the hash of the command, as the name carries it
    from .naming import run_hash
    plan["config"]["provenance"]["run_hash"] = run_hash(sys.argv[1:])
    log("code        pca2d %s" % code_stamp(plan["config"]["provenance"]), "value")
    import yaml
    with open(plan["written_config"], "w") as fh:
        yaml.safe_dump(plan["config"], fh, sort_keys=False,
                       default_flow_style=False)
    log("resolved configuration written to %s" % plan["written_config"], "info")

    runners = {"cube": run_cube, "fit": run_fit, "figures": run_figures,
               "correct": run_correct, "lbl": run_lbl}
    log("%d stages to run: %s" % (len([s for s in STAGES if s in wanted]),
                                  ", ".join(s for s in STAGES if s in wanted)),
        "info")
    log("on a few hundred exposures the whole thing takes tens of minutes,"
        " most of it in the fit", "info")
    for name in STAGES:
        if name not in wanted:
            log("stage %s: skipped" % name, "warn")
            continue
        say_if_code_moved(snapshot, "stage " + name, moved_said)
        # between two stages a cache can be emptied, and the second of them
        # would open a cube that was there when the first one ended
        if name in NEEDS_CUBE:
            rebuild_missing_cubes(plan, "stage " + name)
        with stage("stage %s" % name):
            runners[name](plan)

    log("done in %s" % human(time.time() - started), "info")
    # the run's own name for its target: A+B+C for a joint run, as the report
    # and the RV pages name their file
    from .naming import report_name
    name = plan["config"]["input"].get("object") or args.object
    bundle = os.path.join(plan["outdir"], report_name(
        name, plan["tag"],
        (plan["config"].get("provenance") or {}).get("run_hash")) + ".pdf")
    if os.path.exists(bundle):
        log("everything this run produced: %s" % bundle, "value")
    return None


if __name__ == "__main__":
    sys.exit(main())
