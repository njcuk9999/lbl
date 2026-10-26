"""What the window knows about LBL, read from LBL itself.

Nothing here is a copy of LBL's tables: the instruments and their data
sources come from lbl.instruments.select, the PCA2D switches and their short
names from lbl.recipes.lbl_pca2d, their defaults from lbl/pca2d/nominal.yaml
merged over the code's own, their descriptions from the tables of
lbl/doc/PCA2D.md, and the observer bases from DATA_DIR/pca2d/components. A
new option in LBL shows up here without a line changed.
"""
from __future__ import annotations

import copy
import functools
import glob
import os
import re

#: the folder the bases live in, under DATA_DIR (lbl.pca2d.basis.FOLDER)
BASES = os.path.join("pca2d", "components")

#: the LBL switches of a wrap the window offers, in the order shown, with the
#: default the window starts from (not lbl_wrap's, which leaves every step
#: off: a window that runs nothing by default would be a puzzle)
LBL_SWITCHES = (
    ("RUN_LBL_TELLUCLEAN", None),       # None: the instrument decides
    ("RUN_LBL_TEMPLATE", True),
    ("RUN_LBL_MASK", True),
    ("RUN_LBL_COMPUTE", True),
    ("RUN_LBL_COMPILE", True),
    ("SKIP_LBL_TEMPLATE", True),
    ("SKIP_LBL_MASK", True),
    ("SKIP_LBL_COMPUTE", True),
    ("SKIP_LBL_COMPILE", False),
    ("REPORT", True),
    ("PLANETS", False),
)

#: the switch lbl_wrap reads for the PCA2D correction on disk: None (redo it
#: only if the request changed), True (keep it), False (redo it)
SKIP_PCA2D = "SKIP_LBL_PCA2D"


@functools.lru_cache(maxsize=1)
def _select():
    from lbl.instruments import select
    return select


@functools.lru_cache(maxsize=1)
def lbl_home() -> str:
    """The folder of the lbl package the window reads."""
    import lbl
    return os.path.dirname(os.path.abspath(lbl.__file__))


def instruments() -> list:
    """Every instrument LBL knows, its data sources, its default input files
    and whether it cleans its own tellurics: [dict(instrument, sources:
    [dict(source, input_file, telluclean)])]."""
    select = _select()
    out = []
    for name, sources in select.InstDict.items():
        rows = []
        for source in sources:
            inst = instrument_object(name, source)
            rows.append(dict(
                source=source,
                input_file=inst.params["INPUT_FILE"] if inst else "*.fits",
                telluclean=_cleans_tellurics(name, inst)))
        out.append(dict(instrument=name, sources=rows))
    return out


#: the instruments whose spectra come telluric-corrected (APERO)
CORRECTED = ("NIRPS", "SPIROU")


def _cleans_tellurics(name: str, inst) -> bool:
    """Whether a wrap cleans the tellurics by default: as LBL's class says
    (DO_TELLUCLEAN), and for every optical spectrograph (its domain ending
    short of 1 micron: HARPS, HARPS-N, ESPRESSO, SOPHIE...), whose pipelines
    leave the telluric lines in; never for NIRPS and SPIRou."""
    if str(name).upper().split("_")[0] in CORRECTED:
        return False
    if inst is None:
        return False
    if inst.params.get("DO_TELLUCLEAN"):
        return True
    wavemax = getattr(inst, "wavemax", None)
    return wavemax is not None and float(wavemax) < 1000.0


@functools.lru_cache(maxsize=64)
def instrument_object(instrument: str, source: str):
    """An instance of LBL's class for this instrument and data source, with
    LBL's default parameters: what reads its headers. None when the class
    cannot be made without more (Generic)."""
    from lbl.core.base_classes import ParamDict
    from lbl.core.parameters import params
    select = _select()
    try:
        cls = select.InstDict[instrument][str(source)]
        return cls(params.copy(), ParamDict())
    except Exception:                                     # noqa: BLE001
        return None


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def nominal_block(instrument: str) -> str:
    """nominal.yaml's instrument block for an LBL instrument (NIRPS_HE and
    NIRPS_HA share NIRPS's)."""
    upper = str(instrument).upper()
    if upper.startswith("NIRPS"):
        return "NIRPS"
    if upper == "HARPSN":
        return "HARPN"
    return upper


@functools.lru_cache(maxsize=16)
def pca2d_defaults(instrument: str) -> dict:
    """The PCA2D configuration as it stands before a wrap changes it: the
    code's defaults, nominal.yaml's general block, then its instrument's."""
    import yaml
    from lbl.pca2d.config import DEFAULTS as CODE_DEFAULTS
    path = os.path.join(lbl_home(), "pca2d", "nominal.yaml")
    with open(path) as handle:
        nominal = yaml.safe_load(handle) or {}
    merged = _merge(CODE_DEFAULTS, nominal.get("general") or {})
    block = (nominal.get("instruments") or {}).get(nominal_block(instrument))
    return _merge(merged, block or {})


def _value_at(config: dict, path: str):
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


@functools.lru_cache(maxsize=1)
def doc_descriptions() -> dict:
    """{key: description} from the tables of lbl/doc/PCA2D.md, whose rows
    are | `key` | default | what it does | (or | `key` | what |)."""
    path = os.path.join(lbl_home(), "doc", "PCA2D.md")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as handle:
        for line in handle:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2 or not cells[0].startswith("`"):
                continue
            key = cells[0].strip("`").strip()
            # `nightly_stack=True` describes one value, not the key
            if "=" in key:
                continue
            text = cells[-1]
            if key and text and key not in out:
                out[key] = re.sub(r"`([^`]*)`", r"\1", text)
    return out


#: what an option left at None takes when it is set, by its name
NONE_KINDS = (("_frac", "float"), ("_chi2", "float"), ("_nsig", "float"),
              ("_share", "float"), ("_minutes", "float"), ("_max", "int"),
              ("_name", "str"), ("name", "str"))


def _kind(value, key: str = "") -> str:
    if value is None:
        for ending, kind in NONE_KINDS:
            if key.endswith(ending):
                return kind
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


#: the switches and short names grouped as the window shows them; anything
#: LBL adds later lands in "more"
GROUPS = (
    ("model", ("n_star", "n_earth", "joint", "velocity_term", "iters",
               "width_kms", "dv", "nightly_stack", "bundle_berv",
               "bundle_minutes", "bundle_max")),
    ("bases", ("input_name", "output_name", "name")),
    ("correction", ("shrink", "weight", "mask", "mask_share", "nsig_cut",
                    "column_frac", "column_chi2", "bias_nsig",
                    "star_column_frac", "star_column_chi2", "star_bias_nsig",
                    "min_pixel_snr")),
    ("runs", ("compare", "figures", "overwrite")),
)


def pca2d_options(instrument: str) -> list:
    """The options of a wrap's PCA2D dict, as the window lays them out:
    [dict(key, path, default, kind, group, description)]. `path` is the
    configuration key a short name stands for (None for the recipe's own
    switches), `default` what the run does when the window leaves it out."""
    from lbl.recipes import lbl_pca2d
    config = pca2d_defaults(instrument)
    described = doc_descriptions()
    from lbl.pca2d.config import WINDOW_SETTINGS
    window = dict(WINDOW_SETTINGS)
    rows = {}
    for key, default in lbl_pca2d.DEFAULTS.items():
        rows[key] = dict(key=key, path=None, default=default)
    for key, path in lbl_pca2d.ALIASES.items():
        if key in rows or key in ("cache_directory", "windows", "high_pass",
                                  "min_rjd", "max_rjd", "drs_modes"):
            continue
        rows[key] = dict(key=key, path=path, default=_value_at(config, path))
    group_of = {key: name for name, keys in GROUPS for key in keys}
    order = [key for _, keys in GROUPS for key in keys]
    out = []
    for key, row in rows.items():
        description = (described.get(key) or described.get(row["path"] or "")
                       or window.get(row["path"] or "") or "")
        row.update(kind=_kind(row["default"], key),
                   group=group_of.get(key, "more"),
                   description=description)
        out.append(row)
    out.sort(key=lambda r: (order.index(r["key"]) if r["key"] in order
                            else len(order), r["key"]))
    return out


def lbl_switches() -> list:
    """The LBL switches of a wrap the window offers: [dict(key, default)]."""
    return [dict(key=key, default=default) for key, default in LBL_SWITCHES]


def bases(data_dir: str, instrument: str = "", source: str = "") -> list:
    """The observer bases saved under DATA_DIR/pca2d/components, those of
    this instrument and data source first: [dict(name, file, instrument,
    source, n_earth, objects, compatible)]. LBL refuses a basis of another
    instrument or source (lbl.pca2d.basis.check_compatible), so the window
    offers the compatible ones only, and says why the others are not."""
    from astropy.io import fits
    out = []
    folder = os.path.join(data_dir or "", BASES)
    for path in sorted(glob.glob(os.path.join(folder,
                                              "PCA2D_Observer_*.fits"))):
        try:
            head = fits.getheader(path)
        except Exception:                                 # noqa: BLE001
            continue
        inst = str(head.get("INSTRUME", "")).upper()
        src = str(head.get("DATASRC", "")).upper()
        objects = []
        try:
            from astropy.table import Table
            objects = [str(o) for o in Table.read(path, "TRAINING")["object"]]
        except Exception:                                 # noqa: BLE001
            pass
        out.append(dict(
            name=str(head.get("BASISNAM", os.path.basename(path))),
            file=os.path.basename(path), instrument=inst, source=src,
            n_earth=head.get("NEARTH"), objects=sorted(set(objects)),
            compatible=(inst == str(instrument).upper()
                        and src == str(source).upper())))
    out.sort(key=lambda b: (not b["compatible"], b["name"]))
    return out


#: the header keys a template's effective temperature is read from, in order
TEFF_KEYS = ("PP_TEFF", "OBJTEMP", "TEFF")


def template_teff(path: str):
    """A template's effective temperature [K] from its header, or None."""
    from astropy.io import fits
    try:
        header = fits.getheader(path)
    except Exception:                                     # noqa: BLE001
        return None
    for key in TEFF_KEYS:
        try:
            value = float(header[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def templates(data_dir: str, instrument: str) -> list:
    """The objects whose LBL template is in DATA_DIR/templates for this
    instrument, with the temperature its header gives: what
    OBJECT_COMPARISON can name (another star's template is then the star's
    spectrum in PCA2D). [dict(name, teff)]"""
    folder = os.path.join(data_dir or "", "templates")
    suffix = "_%s.fits" % str(instrument).lower()
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "LBL_Template_*"))):
        name = os.path.basename(path)
        if name.lower().endswith(suffix):
            out.append(dict(name=name[len("LBL_Template_"):-len(suffix)],
                            teff=template_teff(path)))
    return out


def vetted(instrument: str) -> list:
    """LBL's vetted templates for this instrument (lbl.core.vetted): what a
    wrap can name as VETTED_<object>, coolest first, each with its Teff,
    its number of spectra and its rotation period. Empty for an instrument
    without them, or an LBL that has none."""
    try:
        from lbl.core import vetted as lbl_vetted
    except ImportError:
        return []
    keys = ("name", "object", "teff", "nfiles", "prot", "simbad", "file")
    return [{key: entry.get(key) for key in keys}
            for entry in lbl_vetted.entries(instrument)]


def vetted_where(data_dir: str, entry: dict, timeout: float = 4.0) -> str:
    """Where a run would get a vetted template from: 'local' (its depot
    file in DATA_DIR/models, or the template already in templates/),
    'repo' (LBL's model repository has it), 'missing' (the repository
    answers it has not) or 'unknown' (the repository does not answer)."""
    import urllib.error
    import urllib.request
    if os.path.exists(os.path.join(data_dir, "models", entry["file"])):
        return "local"
    if glob.glob(os.path.join(data_dir, "templates",
                              "LBL_Template_%s_*.fits" % entry["name"])):
        return "local"
    try:
        from lbl.core import parameters
        url = "%s/%s" % (parameters.params["MODEL_REPO_URL"], entry["file"])
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout):
            return "repo"
    except urllib.error.HTTPError as exc:
        return "missing" if exc.code == 404 else "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"
