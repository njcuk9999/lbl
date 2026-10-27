"""From what the window holds to an LBL wrap: the run parameters, checked,
and the wrap.py that runs them.

The wrap is the record of a run and the run itself: it is written to disk
before anything starts, runs by hand as it is, and holds nothing the window
alone could read. The PCA2D options a run leaves at their defaults are left
out of it (n_star and n_earth are written whenever they were chosen), so that
a wrap says what was chosen and the rest is LBL's nominal of the day.
"""
from __future__ import annotations

import datetime
import os
import re
import sys

from . import i18n, lblinfo

#: the PCA2D options a wrap writes whenever they are chosen, even at their
#:    default (the rest only when they differ from it); one left unchosen is
#:    not written, since its default may not be what a run uses (n_earth is
#:    the frozen basis's own count when input_name names one)
ALWAYS = ("n_star", "n_earth")
#: the temperature handed to LBL when the headers give none [K]
TEFF_FALLBACK = 3500


def _coerce(value, kind: str):
    """A value as the option's kind, from what a page sends (strings, empty
    for unset)."""
    if value is None or value == "":
        return None
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind == "int":
        return int(float(value))
    if kind == "float":
        return float(value)
    return str(value)


#: the instruments whose spectra come telluric-corrected (APERO): LBL's
#:    telluric cleaning is always off for them
NO_TELLUCLEAN = ("NIRPS", "SPIROU")


def _family(name) -> str:
    """An instrument's family, the headers' way or LBL's: NIRPS for NIRPS,
    NIRPS_HE and NIRPS_HA, HARPSN for HARPS-N. Empty when unknown."""
    name = str(name or "").upper()
    if not name or name == "?" or "/" in name:
        return ""
    return re.sub(r"[^A-Z0-9]", "", name.split("_")[0])


def build(state: dict) -> dict:
    """The run parameters of a wrap from the window's state: dict(rparams,
    errors, warnings). Nothing is written."""
    errors, warnings = [], []
    # the window's language, for what it says (i18n's err_* and warn_*)
    words = i18n.strings(state.get("lang") or "en")

    def say(key, *args):
        return words[key] % tuple(str(a) for a in args)
    data_dir = os.path.abspath(os.path.expanduser(state.get("data_dir") or ""))
    instrument = state.get("instrument") or ""
    source = state.get("source") or "None"
    targets = [t for t in state.get("targets") or [] if t]
    if not data_dir or not os.path.isdir(data_dir):
        errors.append(say("err_no_folder", data_dir))
    if not instrument:
        errors.append(say("err_no_instrument"))
    if not targets:
        errors.append(say("err_no_target"))
    known = {row["instrument"]: row for row in lblinfo.instruments()}
    sources = {s["source"]: s for s in
               (known.get(instrument) or {}).get("sources", [])}
    if instrument and instrument not in known:
        errors.append(say("err_unknown_instrument", instrument))
    elif instrument and str(source) not in sources:
        errors.append(say("err_no_source", instrument, source,
                          ", ".join(sources)))
    # THE SPECTRA ARE THE INSTRUMENT'S: SPIRou files run as NIRPS_HE made a
    #   NIRPS template of SPIRou spectra and stopped on the first lblrv file
    #   (2026-10-23). The headers' instrument against the one chosen, by
    #   family (NIRPS for NIRPS_HE and NIRPS_HA)
    for target, header in (state.get("instruments") or {}).items():
        if target in targets and instrument and _family(header) \
                and _family(header) != _family(instrument):
            errors.append(say("err_wrong_instrument", target, header,
                              instrument))
    rparams = dict()
    rparams["INSTRUMENT"] = instrument
    rparams["DATA_SOURCE"] = source
    rparams["DATA_DIR"] = data_dir
    rparams["DATA_TYPES"] = ["SCIENCE"] * len(targets)
    rparams["OBJECT_SCIENCE"] = list(targets)
    comparison = state.get("comparison") or {}
    teff = state.get("teff") or {}
    rparams["OBJECT_COMPARISON"] = [comparison.get(t) or t for t in targets]
    teffs = []
    for target in targets:
        value = teff.get(target)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is None:
            warnings.append(say("warn_no_teff", target, TEFF_FALLBACK))
            value = TEFF_FALLBACK
        teffs.append(int(round(value)))
    rparams["OBJECT_TEFF"] = teffs
    templates = {row["name"] for row in lblinfo.templates(data_dir, instrument)}
    vetted = {row["name"]: row for row in lblinfo.vetted(instrument)}
    for target, other in zip(targets, rparams["OBJECT_COMPARISON"]):
        # a vetted template comes from LBL's depot when the run needs it:
        #   one the depot does not have stops the run on its first step
        if str(other).upper().startswith("VETTED_"):
            if other not in vetted:
                errors.append(say("err_vetted_unknown", other, instrument))
                continue
            where = lblinfo.vetted_where(data_dir, vetted[other])
            if where == "missing":
                errors.append(say("err_vetted_missing", other,
                                  vetted[other]["file"],
                                  os.path.join(data_dir, "models",
                                               "template_%s_%s" % (
                                                   instrument.upper(),
                                                   str(source).upper()))))
            elif where == "unknown":
                warnings.append(say("warn_vetted_unknown", other))
            continue
        if other != target and other not in templates and not os.path.isdir(
                os.path.join(data_dir, "science", other)):
            errors.append(say("err_no_comparison", target, other))
    # FOLDERS FETCHED FROM AN ARCHIVE (archives.py): their reader, the
    #   files LBL reads in them and its parameters, from what they came
    #   from; a source other than theirs would read them wrongly
    from .archives import provenance
    origins = [provenance(os.path.join(data_dir, "science", t))
               for t in targets]
    found = [o for o in origins if o]
    for target, origin in zip(targets, origins):
        if origin and (origin.get("lbl") != instrument
                       or origin.get("source") != str(source)):
            errors.append(say("err_archive_source", target, origin["lbl"],
                              origin["source"]))
    # THE OTHERS: the data source their files' layout says
    #   (targets.layout_source). The CADC's t.fits run as APERO stopped LBL
    #   on the first one, WAVEORDN not in the header it reads (2026-10-24)
    from .targets import layout_of
    for target, origin in zip(targets, origins):
        if origin:
            continue
        header, said = layout_of(data_dir, target)
        if said and said != str(source) and said in sources \
                and _family(header) == _family(instrument):
            errors.append(say("err_layout_source", target, said, source,
                              said))
    if found and not errors:
        rparams.setdefault("INPUT_FILE", found[0].get("input"))
        for key, value in (found[0].get("params") or {}).items():
            rparams.setdefault(key, value)
    # LOCAL FOLDERS, where LBL's default says nothing of the files ('*.fits'
    #   for HARPS): the end their spectra's names share, as LBL's own HARPS
    #   wraps write it. Without it PCA2D took its HARPS default, the ESO
    #   archive's *_S2D_A.fits, and found no e2ds (GJ 1214, 2026-10-25)
    default = (sources.get(str(source)) or {}).get("input_file")
    if not found and "INPUT_FILE" not in rparams \
            and default in (None, "", "*", "*.fits"):
        from .targets import files_of
        pattern = files_pattern([path for target in targets
                                 for path in files_of(data_dir, target)])
        if pattern:
            rparams["INPUT_FILE"] = pattern
    # the LBL switches
    lbl = state.get("lbl") or {}
    telluclean = (sources.get(str(source)) or {}).get("telluclean", False)
    for row in lblinfo.lbl_switches():
        key = row["key"]
        value = lbl.get(key, row["default"])
        if key == "RUN_LBL_TELLUCLEAN" and value is None:
            value = telluclean
        # NIRPS's and SPIRou's spectra come telluric-corrected: never
        #   LBL's own cleaning on them, whatever the state says
        if key == "RUN_LBL_TELLUCLEAN" and _family(instrument) in NO_TELLUCLEAN:
            value = False
        rparams[key] = bool(value)
    if state.get("resproj"):
        rparams["RESPROJ_TABLES"] = {
            "DTEMP3500": "temperature_gradient_3500.fits"}
    # what a loaded wrap held that the window does not: written back as it
    #   was (wrapload.to_state)
    from .wrapload import MANAGED
    for key, value in (state.get("extra") or {}).items():
        if key not in MANAGED and key not in rparams:
            rparams[key] = value
    # PCA2D
    if state.get("pca2d_on", True):
        options = {row["key"]: row for row in
                   lblinfo.pca2d_options(instrument)} if instrument else {}
        chosen = state.get("pca2d") or {}
        pca2d = dict()
        for key, row in options.items():
            if key not in chosen:
                # n_star and n_earth are always written, their defaults when
                #   the window left them: the fit's folder is named after
                #   them (a fresh window named its run s<None>-o<None>)
                if key in ALWAYS and row["default"] is not None:
                    pca2d[key] = row["default"]
                continue
            try:
                value = _coerce(chosen[key], row["kind"])
            except (TypeError, ValueError):
                errors.append(say("err_pca2d_kind", key, repr(chosen[key]),
                                  row["kind"]))
                continue
            if key in ALWAYS or value != row["default"]:
                if value is not None or row["default"] is not None:
                    pca2d[key] = value
        for key in ("min_rjd", "max_rjd"):
            value = state.get(key)
            if value not in (None, ""):
                pca2d[key] = float(value)
        for key, value in (state.get("pca2d_extra") or {}).items():
            pca2d.setdefault(key, value)
        # THE JOINT FIT: offered for several targets only, and on by default
        #   there (one observer basis for stars observed together); for one
        #   target it is not offered, and a value left from an earlier
        #   choice of several is dropped
        if len(targets) > 1:
            if "joint" not in chosen:
                pca2d["joint"] = True
        else:
            pca2d.pop("joint", None)
        if pca2d.get("input_name"):
            found = [b for b in lblinfo.bases(data_dir, instrument, source)
                     if b["name"] == pca2d["input_name"]]
            if not found:
                errors.append(say("err_no_basis", pca2d["input_name"],
                                  os.path.join(data_dir, "models"),
                                  os.path.join(data_dir, lblinfo.BASES)))
            elif not found[0]["compatible"]:
                errors.append(say("err_basis_other", pca2d["input_name"],
                                  found[0]["instrument"], found[0]["source"],
                                  instrument, source))
        try:
            from lbl.recipes import lbl_pca2d
            lbl_pca2d.settings(dict(pca2d), report=rparams.get("REPORT", True))
        except Exception as exc:                          # noqa: BLE001
            errors.append(say("err_pca2d", exc))
        rparams["PCA2D"] = pca2d
        skip = state.get("skip_pca2d")
        if skip is not None:
            rparams[lblinfo.SKIP_PCA2D] = bool(skip)
    return dict(rparams=rparams, errors=errors, warnings=warnings)


#: the wrap's comments, so that one reads it without the window: its
#:    sections, in order, and a line or two above each setting (English, as
#:    LBL's own code and wraps are)
WRAP_SECTIONS = (
    ("instrument and data", (
        ("INSTRUMENT", "the spectrograph, as LBL names it"),
        ("DATA_SOURCE", "the pipeline its files come from: LBL reads each "
         "pipeline's file layout with a class of its own"),
        ("DATA_DIR", "the data folder: the spectra in science/<object>/, and "
         "LBL's products beside them (templates/, masks/, lblrv/, lblrdb/, "
         "...)"),
        ("INPUT_FILE", "the files of science/<object>/ that LBL reads"),
        ("BLAZE_CORRECTED", "whether the spectra come with the blaze already "
         "divided out"),
        ("BLAZE_FILE", "the blaze, in calib/"),
    )),
    ("targets: one entry per object, in the same order in each list", (
        ("DATA_TYPES", "SCIENCE for a star (FP or LFC for a calibration "
         "source)"),
        ("OBJECT_SCIENCE", "the folders of science/ to measure"),
        ("OBJECT_COMPARISON", "the template each one is measured against: its "
         "own, another star's, or a vetted template (VETTED_<star>, fetched "
         "from LBL's model repository the first time)"),
        ("OBJECT_TEFF", "effective temperature [K]: it picks the line mask"),
        ("RESPROJ_TABLES", "the residual projections measured on every "
         "spectrum, here the temperature-change indicator (dTEMP)"),
    )),
    ("LBL's steps: RUN_ runs one, SKIP_ keeps an earlier run's", (
        ("RUN_LBL_TELLUCLEAN", "LBL's own telluric correction (off for "
         "spectra that come corrected, as NIRPS's and SPIRou's do)"),
        ("RUN_LBL_TEMPLATE", "build the template from the spectra"),
        ("RUN_LBL_MASK", "build the line mask from the template"),
        ("RUN_LBL_COMPUTE", "measure every line's velocity in every spectrum "
         "(lblrv/)"),
        ("RUN_LBL_COMPILE", "combine the lines into one velocity per spectrum "
         "(lblrdb/)"),
        ("SKIP_LBL_TEMPLATE", "True: a template already made is kept"),
        ("SKIP_LBL_MASK", "True: a mask already made is kept"),
        ("SKIP_LBL_COMPUTE", "True: the spectra already measured are not "
         "measured again"),
        ("SKIP_LBL_COMPILE", "True: a table already compiled is kept"),
        (lblinfo.SKIP_PCA2D, "True: a PCA2D fit and corrected spectra "
         "already made are kept"),
        ("REPORT", "write a PDF report of each run's velocities"),
        ("PLANETS", "build the template on the BERV plus the Keplerian "
         "velocity of the star's known companions, when LBL's planet "
         "catalogs have them (the velocities stay BERV-corrected only)"),
    )),
)
#: above the PCA2D settings
WRAP_PCA2D = ("the telluric and instrumental residuals are fitted in two "
              "frames (the star's and the observer's) and taken out of the "
              "spectra before LBL measures them again; only what differs from "
              "the nominal settings is written")
#: a comment line's width
WRAP_WIDTH = 79
#: the PCA2D options whose help starts from another's (the window lays them
#:    one under the other): their comment in the wrap, which has no "above"
WRAP_OPTIONS = dict(
    column_frac="the column test, observer's frame: a wavelength where more "
    "than this fraction of the exposures was clipped, and whose survivors "
    "still have a reduced chi2 above column_chi2, is blanked in every "
    "exposure",
    column_chi2="the column test's second condition: the reduced chi2 of the "
    "exposures left after clipping",
    star_column_frac="the column test (column_frac) in the star's frame",
    star_column_chi2="the column test's chi2 (column_chi2) in the star's "
    "frame",
    star_bias_nsig="the bias test (bias_nsig) in the star's frame",
)


def _comment(text: str, indent: int) -> list:
    """A comment, wrapped, its continued lines indented as LBL's are."""
    import textwrap
    pad = " " * indent
    return textwrap.wrap(text, width=WRAP_WIDTH, initial_indent=pad + "# ",
                         subsequent_indent=pad + "#   ")


def _rule(title: str) -> str:
    """A section's heading: its title in a rule as wide as the comments."""
    head = "    # ---- %s " % title
    return head + "-" * max(3, WRAP_WIDTH - len(head))


def _option_comment(key: str) -> str:
    """A PCA2D option's comment: the first sentence of its help, as the
    other comments are written (no capital, no full stop)."""
    if key in WRAP_OPTIONS:
        return WRAP_OPTIONS[key]
    known = i18n.OPTIONS.get(key)
    if not known or not known[1]:
        return ""
    text = known[1][0].split("\n")[0].strip()
    first = re.split(r"(?<=[.:])\s", text, maxsplit=1)[0].rstrip(".:")
    # a capital that is not an acronym's (ONE, LBL) goes down
    if len(first) > 1 and not first[1].isupper():
        first = first[0].lower() + first[1:]
    return first


def window_text(posted: str, rparams: dict, name: str = "",
                python: str = "") -> str:
    """The wrap to write, run or saved: the window's own, commented and
    current, when the text the page posts reads back to these very run
    parameters (the page can hold a wrap written before the comments were:
    a run saved without them on 2026-10-24); the posted text as it is when
    it was edited by hand."""
    import json
    from .wrapload import read_text
    fresh = render(rparams, name, python)
    if not posted:
        return fresh
    try:
        found = read_text(posted)["rparams"]
    except (SyntaxError, ValueError):
        return posted
    same = lambda d: json.dumps(d, sort_keys=True, default=str)
    return fresh if same(found) == same(rparams) else posted


def render(rparams: dict, name: str = "", python: str = "") -> str:
    """The wrap.py of these run parameters, commented."""
    python = python or sys.executable
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    targets = ", ".join(rparams.get("OBJECT_SCIENCE") or [])
    lines = [
        "#!%s" % python,
        "# -*- coding: utf-8 -*-",
        '"""',
        "%s: LBL%s on %s (%s, %s), written by the LBL window (lbl_gui)"
        " on %s."
        % (name or "run", " with the PCA2D precleaning"
           if "PCA2D" in rparams else "", targets,
           rparams.get("INSTRUMENT"), rparams.get("DATA_SOURCE"), when),
        "",
        "    %s %s" % (python, "wrap.py"),
        "",
        "It runs by hand as it is: what the window ran is all here. Every "
        "setting",
        "is explained above it; LBL's documentation lists the others a wrap "
        "can hold.",
        '"""',
        "from lbl.recipes import lbl_wrap",
        "",
        'if __name__ == "__main__":',
        "    rparams = dict()",
    ]
    done = {"PCA2D"}
    for title, keys in WRAP_SECTIONS:
        present = [(key, text) for key, text in keys if key in rparams]
        if not present:
            continue
        lines.append("")
        lines.append(_rule(title))
        for key, text in present:
            lines += _comment(text, 4)
            lines.append("    rparams[%r] = %r" % (key, rparams[key]))
            done.add(key)
    others = [key for key in rparams if key not in done]
    if others:
        lines.append("")
        lines.append(_rule("other settings, as a loaded wrap held them"))
        for key in others:
            lines.append("    rparams[%r] = %r" % (key, rparams[key]))
    if "PCA2D" in rparams:
        lines.append("")
        lines.append(_rule("PCA2D precleaning"))
        lines += _comment(WRAP_PCA2D, 4)
        lines.append("    rparams['PCA2D'] = dict(")
        for key, value in rparams["PCA2D"].items():
            lines += _comment(_option_comment(key), 8)
            lines.append("        %s=%r," % (key, value))
        lines.append("    )")
    lines.append("")
    lines.append("    lbl_wrap.main(rparams)")
    return "\n".join(lines) + "\n"


#: LBL's recipes, in the order _lbl_steps runs them, and their switch
LBL_STEPS = (("template", "RUN_LBL_TEMPLATE"), ("mask", "RUN_LBL_MASK"),
             ("compute", "RUN_LBL_COMPUTE"), ("compile", "RUN_LBL_COMPILE"))


def files_pattern(paths: list):
    """'*' and the end the files' names share, its leading digits off (a
    date's or a number's): *_e2ds_A.fits for HARPS read as ORIG; None when
    they share no more than .fits (as lbl_pca2d.files_pattern)."""
    names = [os.path.basename(path) for path in paths]
    if not names:
        return None
    tail = os.path.commonprefix([name[::-1] for name in names])[::-1]
    tail = tail.lstrip("0123456789")
    if tail.lower() in ("", "fits", ".fits", "s", "ts", "its"):
        return None
    return "*" + tail


def steps(rparams: dict) -> list:
    """The steps of a run, in the order lbl_wrap goes through them:
    [dict(kind, obj)], kind one of telluclean, template, mask, compute,
    compile, report, pca2d_cube, pca2d_fit, pca2d_figures, pca2d_correct.
    The window ticks them off as the log names them (runner.Job)."""
    out = []
    targets = list(rparams.get("OBJECT_SCIENCE") or [])
    pca2d = rparams.get("PCA2D")
    on = pca2d not in (None, False)
    pca2d = pca2d if isinstance(pca2d, dict) else {}
    compare = on and pca2d.get("compare", True)
    joint = on and pca2d.get("joint")
    figures = pca2d.get("figures", rparams.get("REPORT", True))
    name = pca2d.get("name")
    tellu = bool(rparams.get("RUN_LBL_TELLUCLEAN"))
    measured = set()

    def lbl(obj, template=True):
        for kind, switch in LBL_STEPS:
            if kind == "template" and not template:
                continue
            if rparams.get(switch, True):
                out.append(dict(kind=kind, obj=obj))
        if rparams.get("REPORT", True) and rparams.get("RUN_LBL_COMPILE", True):
            out.append(dict(kind="report", obj=obj))

    fitted = False
    for target in targets:
        obj = target
        # the telluric cleaning in its four steps (lbl_wrap): a first pass
        #   blind, a template of its cleaned spectra, a second pass of the
        #   spectra as delivered divided by that template, the template made
        #   again from them; all under the target's name (group), each with
        #   its own label. LBL's own template step then finds it on disk
        if tellu:
            for kind, obj, label in (
                    ("telluclean", target, "step_telluclean_1"),
                    ("template", target + "_tc", "step_template_tc"),
                    ("telluclean", target, "step_telluclean_2"),
                    ("template", target + "_tc", "step_template_tc2")):
                out.append(dict(kind=kind, obj=obj, group=target,
                                label=label))
            obj = target + "_tc"
        if not on:
            lbl(obj, template=not tellu)
            continue
        if compare and obj not in measured:
            lbl(obj, template=not tellu)
            measured.add(obj)
        if joint and not fitted and compare and not tellu:
            for other in targets:
                if other not in measured:
                    lbl(other)
                    measured.add(other)
        if not (joint and fitted):
            stages = ("cube", "fit") + (("figures",) if figures else ()) \
                + ("correct",)
            for stage in stages:
                out.append(dict(kind="pca2d_" + stage,
                                obj="+".join(targets) if joint else obj))
            fitted = True
        lbl(obj + "_pca2d" + ("_" + name if name else ""))
    return out
