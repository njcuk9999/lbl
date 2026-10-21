"""Build the registered data and weight cubes, from s1d or from t.fits.

The cube has one row per *measurement*, not per exposure. With s1d input that
is the same thing. With t.fits input each exposure contributes two rows, the
even orders and the odd orders resampled separately onto the same destination
grid (see pca2d/tfits.py for why two is the right number). Rows carry
`exposure` and `parity` columns so the two halves can be recombined afterwards.

With input.nightly_stack the "measurement" is a whole night rather than a
single exposure: every exposure is still registered individually, with its own
BERV, and only then are they coadded. See pca2d/nights.py.
"""

from __future__ import annotations

import os
import re
import shutil

import numpy as np
from astropy.table import Table

from . import grids
from . import io as spio
from . import nights as spnights
from . import preprocess as prep
from . import readers as _readers
from . import tfits as sptf
from .config import cache_key, spectra_dir
from .logger import log
from .progress import bar as _bar


def object_key(name) -> str:
    """An object name reduced to what identifies it: upper case, letters and
    digits only, so 'Proxima', 'PROXIMA', 'TOI-2120' and 'TOI 2120' are two
    names and not four."""
    return re.sub(r"[^0-9A-Z]", "", str(name or "").upper())


def one_object(meta: dict, inp: dict) -> dict:
    """UNE CAMPAGNE EST UN OBJET, DONC UNE VITESSE.

    `meta["object"]` sort de l'en-tete, et un observateur ecrit ce qu'il veut:
    les cent poses HARPS de HD 10700 portent HD10700, tau-Cet ou HR509 selon
    l'annee, pour la MEME etoile. Tout ce qui groupe par objet les prenait
    alors pour trois cibles, et chacune recevait sa propre vitesse systemique,
    son propre masque commun et son propre seuil de SNR.

    Mesure le 2026-10-15 sur la raie stellaire a 685.7 nm, decalage median par
    graphie: HD10700 -0.93 km/s, tau-Cet -16.55, HR509 -89.04. La raie
    sautait d'un bloc d'observations a l'autre dans le river plot, sans aucun
    rapport avec le BERV (pente -0.17, correlation -0.09), uniquement parce
    que le nom change.

    Le nom de la campagne est celui que l'utilisateur a demande, pas celui que
    l'en-tete porte: c'est lui qui decide ce qui est une cible.
    """
    wanted = inp.get("object")
    if wanted:
        meta = dict(meta)
        meta["object"] = str(wanted)
    return meta


def same_object(meta: dict, wanted) -> bool:
    """Whether a file's OBJECT, or APERO's DRSOBJN, names the object asked for.

    Compared as object_key, never as typed. OBJECT is what somebody typed into
    an observing form: 'Proxima' on the files APERO calls PROXIMA, which an
    exact comparison with the folder's name skipped one by one.

    PLUSIEURS NOMS SONT ACCEPTES. `wanted` peut etre une liste, parce qu'une
    archive ne couvre pas cinq ans sans changer de graphie: les poses HARPS de
    tau Ceti sont OBJECT = HD10700 en 2003 et tau-Cet en 2004, et aucune
    normalisation ne rapproche un nom de catalogue d'un nom de Bayer. Les
    donner tous les deux est la seule reponse honnete; la deviner en serait
    une mauvaise, parce qu'elle melangerait un jour deux etoiles.
    """
    # "*": tout fichier du dossier est la cible. C'est le cas dans LBL, ou le
    # dossier science/<objet> definit l'objet et ou son nom n'a pas a etre
    # celui des en-tetes (lbl.recipes.lbl_pca2d)
    if wanted == "*":
        return True
    keys = {object_key(w) for w in
            (wanted if isinstance(wanted, (list, tuple, set)) else [wanted])
            if w}
    return any(object_key(meta.get(name)) in keys
               for name in ("object", "drsobjn") if meta.get(name))

C_KMS = 299792.458

# Storage dtype of the cube, and not a knob. float64 doubles four arrays of
# (rows x samples) for nothing the fit can use: the data are photon counts
# carrying five or six significant digits, every weighted sum accumulates in
# float64 whatever this says, and on a few hundred exposures the difference is
# gigabytes and whether the fit holds in memory at all. It was in the config
# long enough to be asked for; nobody has a reason to change it.
CUBE_DTYPE = np.dtype("float32")


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def _cache_dir(config: dict) -> str:
    """Where this configuration's cube lives on disk.

    The input format is in the name, not only in the hash, so that a directory
    listing of the cache says what each cube is. The hash already depends on it
    (input.format is part of the hashed config), so this is labelling rather
    than disambiguation, and that is the point.
    """
    from . import cache as _cache
    directory = _cache.ensure(config)
    fmt = str(config["input"].get("format", "tfits"))
    return os.path.join(directory, "cube_%s_%s" % (fmt, cache_key(config)))


def _load_cache(path: str):
    """Read a cached cube back. Returns None if it is absent or incomplete."""
    needed = ["grid.npy", "data.npy", "sigma.npy", "meta.fits"]
    if not all(os.path.exists(os.path.join(path, name)) for name in needed):
        return None
    log("loading cached cube from %s" % path)
    grid = np.load(os.path.join(path, "grid.npy"))
    data = np.load(os.path.join(path, "data.npy"))
    sigma = np.load(os.path.join(path, "sigma.npy"))
    trans_path = os.path.join(path, "trans.npy")
    trans = np.load(trans_path) if os.path.exists(trans_path) else None
    meta = Table.read(os.path.join(path, "meta.fits"))
    log("cube: %d rows x %d pixels" % data.shape, "value")
    return grid, data, sigma, trans, meta


def _save_cache(path: str, grid, data, sigma, trans, meta):
    """Write the cube as separate .npy files rather than one compressed npz.

    A full-domain NIRPS cube is several GB. np.savez_compressed on that spends
    minutes in zlib for a file that barely compresses -- the payload is float32
    noise -- and has to hold the whole thing in memory twice. Plain .npy files
    write at disk speed and can be memory-mapped later if it ever comes to that.
    """
    tmp = path + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, "grid.npy"), grid)
    np.save(os.path.join(tmp, "data.npy"), data)
    np.save(os.path.join(tmp, "sigma.npy"), sigma)
    if trans is not None:
        np.save(os.path.join(tmp, "trans.npy"), trans)
    meta.write(os.path.join(tmp, "meta.fits"), overwrite=True)
    if _PENDING_TEMPLATE is not None:
        # le template du passe-haut voyage avec le cube: `correct` doit
        # filtrer les spectres exactement comme le fit l'a fait
        np.save(os.path.join(tmp, TEMPLATE_FILE), _PENDING_TEMPLATE)
    if _PENDING_MEMBERS is not None:
        # which exposure went into which row: what `correct` matches an
        # exposure to its row by, once a night is cut into bundles
        Table(rows=_PENDING_MEMBERS, names=("filename", "row", "group")
              ).write(os.path.join(tmp, MEMBERS_FILE), format="ascii.csv",
                      overwrite=True)
    shutil.rmtree(path, ignore_errors=True)
    os.rename(tmp, path)
    log("cached cube -> %s" % path)


# --------------------------------------------------------------------------
# destination grid
# --------------------------------------------------------------------------
def destination_grid(config: dict, files: list[str]):
    """The common wavelength grid every spectrum is resampled onto."""
    dom = config["domain"]
    source = dom.get("grid_source", "magic")
    if source == "native":
        if config["input"].get("format", "tfits") == "tfits":
            raise ValueError(
                "domain.grid_source = 'native' needs an s1d file to read the grid"
                " from; t.fits carry no common grid. Use 'magic'."
            )
        grid = spio.magic_grid(files[0], dom["wave_min"], dom["wave_max"])
        log("recycling the native s1d_v grid from %s" % os.path.basename(files[0]))
    else:
        if dom.get("dv") is None:
            raise ValueError(
                "domain.dv is null and nothing has measured it. It is measured"
                " from the data only when domain.smart_dv is on AND the run"
                " names an object, as `pca2d-preclean --object NAME` does;"
                " reading a config straight into this stage does not. Put a"
                " number in domain.dv or go through the one command.")
        grid = grids.magic_grid(dom["wave0"], dom["dv"], dom["wave_min"], dom["wave_max"])
        log("magic grid: anchor %.3f nm, step %.4f km/s"
            % (float(dom["wave0"]), float(dom["dv"])))
    dv = grids.grid_dv(grid)
    log("domain %.1f-%.1f nm -> %d pixels at %.4f km/s"
        % (dom["wave_min"], dom["wave_max"], grid.size, dv), "value")
    return grid, dv


def resolve_stacking(config, n_files, n_pixels):
    """Whether to coadd each night, and why. Returns (stack, reason).

    Coadding a night is not a modelling choice and never was: it exists so the
    fit holds in memory. Every exposure is registered with its own BERV before
    it is added, so nothing is smeared either way, and fitting the individual
    spectra is strictly more information. So the default is `auto`: fit every
    file unless the fit would not fit.

    The budget compared against is `output.max_memory_gb`, a CONFIG value, and
    deliberately not the machine's actual RAM. The decision has to be a pure
    function of the configuration, because the configuration is what the cube
    cache key hashes: if it depended on the machine, the same key would name a
    stacked cube on one computer and an unstacked one on another, and nothing
    downstream would notice.

    What the budget is spent on is the FIT, not the cube. The cube is float32
    and its own limit is checked separately; the fit promotes to float64 and
    holds four arrays of (rows, pixels) at its peak: the data, the weights,
    the star block carried into every frame, and the observer block.
    """
    inp = config["input"]
    asked = inp.get("nightly_stack", "auto")
    rows = 2 * int(n_files)                       # even and odd order parities
    # two arrays at the fit's own storage dtype, the data and the weights, plus
    # the two float64 temporaries the model costs: the bases stay float64, so
    # the star block carried into every frame and the observer block are
    # float64 whatever the cube is stored as
    item = np.dtype(config["twoframe"].get("dtype", "float64")).itemsize \
        if "twoframe" in config else 8
    need = rows * int(n_pixels) * (2 * item + 2 * 8) / 1e9
    budget = float(config["output"].get("max_memory_gb") or 0.0)
    if asked is True or str(asked).lower() == "true":
        return True, "asked for"
    if asked is False or str(asked).lower() == "false":
        if spnights.bundle_settings(inp):
            log("input.bundle_* are set and nightly_stack is false: every"
                " exposure is fitted on its own, and the bundles are not"
                " used", "warn")
        return False, "asked for; the fit needs about %.1f GB" % need
    if spnights.bundle_settings(inp):
        # a bundle IS a coadd: asking for bundles is asking for stacking
        return True, "bundles within the nights asked for"
    if not budget:
        return False, "auto, and no output.max_memory_gb to weigh it against"
    if need <= budget:
        return False, ("auto: the fit needs about %.1f GB of the %.1f GB budget,"
                       " so every file is fitted on its own" % (need, budget))
    return True, ("auto: fitting every file would need about %.1f GB, above the"
                  " %.1f GB budget, so nights are coadded" % (need, budget))


def _check_memory(config, n_rows, n_pixels, n_arrays=4):
    """Refuse to allocate a cube that will not fit, with a useful message.

    Four arrays of that shape are live at the peak: data, sigma, transmission
    and, once cube.build_weights has run, the weights. The mean subtraction is
    done in place, but the band-restricted block handed to the decomposition is
    an extra copy of a fraction of it in pca.fit_dtype.
    """
    gigabytes = n_rows * n_pixels * CUBE_DTYPE.itemsize * n_arrays / 1e9
    log("cube footprint: %d rows x %d pixels x %d arrays (%s) = %.2f GB"
        % (n_rows, n_pixels, n_arrays, CUBE_DTYPE.name, gigabytes), "value")
    limit = config["output"].get("max_memory_gb")
    if limit and gigabytes > float(limit):
        raise MemoryError(
            "the cube would need %.1f GB, above output.max_memory_gb = %.1f. "
            "Narrow domain.wave_min/wave_max, set input.max_files, or raise the "
            "limit if the machine really has the memory." % (gigabytes, float(limit))
        )
    return gigabytes


# --------------------------------------------------------------------------
# inverse-variance coadder
# --------------------------------------------------------------------------
class _Coadder:
    """Accumulate registered rows into their output group.

    A group is one exposure by default, or one night when input.nightly_stack
    is on. Within a group the rows are combined by inverse variance, per parity
    and per grid column:

        y = sum(w y) / sum(w),  sigma = 1 / sqrt(sum w),  w = 1 / sigma^2

    which is the right answer for the mean of independent measurements and,
    just as importantly, is the operation that lets a pixel be rejected in one
    exposure and kept in the next without leaving a hole.

    Groups are assumed to arrive contiguously (the file list is time-ordered
    upstream), so only the open group's accumulators are ever in memory.
    """

    def __init__(self, n_groups, n_parity, n_pixels, dtype, with_trans):
        self.n_parity = n_parity
        self.n_pixels = n_pixels
        self.data = np.zeros((n_groups * n_parity, n_pixels), dtype=dtype)
        self.sigma = np.full((n_groups * n_parity, n_pixels), np.nan, dtype=dtype)
        self.trans = (np.zeros((n_groups * n_parity, n_pixels), dtype=dtype)
                      if with_trans else None)
        self.rows = []
        self.n_groups = 0
        self._label = None
        self._sw = np.zeros((n_parity, n_pixels))
        self._swy = np.zeros((n_parity, n_pixels))
        self._swt = np.zeros((n_parity, n_pixels))
        self._meta = []
        #: (exposure, the name of the rows it went into, the group's label):
        #: which row an exposure is, once several are coadded into one
        self.members = []

    def add(self, label, values, sigma, trans, good, meta):
        if self._label is not None and label != self._label:
            self.flush()
        self._label = label
        self._meta.append(meta)
        for parity in range(self.n_parity):
            keep = good[parity] & np.isfinite(sigma[parity]) & (sigma[parity] > 0)
            if not np.any(keep):
                continue
            weight = np.zeros(self.n_pixels)
            weight[keep] = 1.0 / np.asarray(sigma[parity], dtype=np.float64)[keep] ** 2
            self._sw[parity] += weight
            self._swy[parity] += weight * np.where(keep, values[parity], 0.0)
            if trans is not None:
                self._swt[parity] += weight * np.where(keep, trans[parity], 0.0)

    def flush(self):
        """Close the open group and write its rows out."""
        if self._label is None:
            return
        merged = _merge_meta(self._meta)
        merged["night"] = self._label
        merged["n_exposures"] = len(self._meta)
        for meta in self._meta:
            self.members.append((os.path.basename(str(meta["filename"])),
                                 os.path.basename(str(merged["filename"])),
                                 str(self._label)))
        for parity in range(self.n_parity):
            slot = self.n_groups * self.n_parity + parity
            # UN POIDS NON NUL N'EST PAS UN POIDS UTILISABLE. sigma vaut
            # 1/sqrt(somme des poids), et le cube est en float32: sous un
            # poids de 1e-77 la sigma depasse 3.4e38 et devient inf a la
            # conversion. Un inf n'est pas faux (il pese zero) mais il
            # contamine toute somme qui le rencontre, et numpy criait 45 fois
            # par campagne. Vu le 2026-10-14 sur ESPRESSO, dont les ordres se
            # recouvrent assez pour qu'un echantillon ne recoive que le bord
            # mourant d'un ordre. On declare donc mort ce qui ne tient pas
            # dans le type du cube, ce qui est la meme chose dite franchement.
            biggest = float(np.finfo(self.sigma.dtype).max)
            live = self._sw[parity] > 1.0 / (biggest * biggest)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.data[slot] = np.where(live, self._swy[parity] / self._sw[parity], 0.0)
                self.sigma[slot] = np.where(live, 1.0 / np.sqrt(self._sw[parity]), np.nan)
                if self.trans is not None:
                    self.trans[slot] = np.where(
                        live, self._swt[parity] / self._sw[parity], 0.0)
            row = dict(merged)
            row["exposure"] = self.n_groups
            row["parity"] = parity
            row["parity_name"] = (sptf.PARITY_NAMES[parity] if self.n_parity == 2
                                  else "all")
            row["n_good_columns"] = int(live.sum())
            self.rows.append(row)
        self.n_groups += 1
        self._label = None
        self._meta = []
        self._sw[:] = 0.0
        self._swy[:] = 0.0
        self._swt[:] = 0.0

    def result(self):
        self.flush()
        set_pending_members(self.members)
        n_rows = self.n_groups * self.n_parity
        trans = self.trans[:n_rows] if self.trans is not None else None
        return self.data[:n_rows], self.sigma[:n_rows], trans, Table(self.rows)


_MEAN_KEYS = ("bjd", "mjdmid", "berv", "bervmax", "airmass", "airmass_start",
              "airmass_end", "seeing", "humidity", "sun_elevation",
              "median_flux", "nan_fraction")


def _merge_meta(rows):
    """Collapse the metadata of the exposures that went into one group.

    Times and conditions are averaged, exposure time adds up, and the band SNR
    adds in quadrature -- which is what coadding n exposures actually does to
    it, and what makes the quality cuts mean the same thing before and after
    stacking.
    """
    if len(rows) == 1:
        return dict(rows[0])
    merged = dict(rows[0])
    for key in _MEAN_KEYS:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        with np.errstate(invalid="ignore"):
            merged[key] = float(np.nanmean(values)) if np.any(np.isfinite(values)) \
                else np.nan
    merged["exptime"] = float(np.nansum([row.get("exptime", np.nan) for row in rows]))
    snr = np.array([row.get("snr_band", np.nan) for row in rows], dtype=float)
    snr = snr[np.isfinite(snr)]
    merged["snr_band"] = float(np.sqrt(np.sum(snr ** 2))) if snr.size else np.nan
    merged["filename"] = rows[0]["filename"]
    return merged


#: les formats ou chaque ORDRE est une ligne, que _build_tfits sait lire: un
#: t.fits d'APERO, un S2D de l'archive ESO (HARPS, HARPS-N, ESPRESSO), un e2ds
#: du DRS d'origine (HARPS, SOPHIE). Ce qu'ils ont en commun n'est pas le nom
#: mais la forme, et c'est la forme qui decide du constructeur.
ORDER_FORMATS = frozenset({"tfits", "s2d", "e2ds"})

#: les formats deja recolles sur une grille unique, que _build_s1d sait lire
FLAT_FORMATS = frozenset({"s1d"})


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def build_cube(config: dict):
    """Read every spectrum, high-pass it, register it and stack.

    Returns (grid, data, sigma, trans, meta). `data` is the high-passed log
    spectrum on the common grid, `sigma` its per-pixel uncertainty with NaN
    marking unusable samples, `trans` the telluric transmission carried through
    the identical resampling, and `meta` one row per cube row.
    """
    cache = _cache_dir(config)
    out = config["output"]
    if out["use_cache"] and out.get("reuse_cache", True):
        cached = _load_cache(cache)
        if cached is not None:
            return cached

    inp = config["input"]
    source = spectra_dir(config)
    files = spio.find_spectra(source, inp["pattern"], inp["max_files"])
    if len(files) == 0:
        raise RuntimeError("no files matched %s/%s" % (source, inp["pattern"]))
    log("found %d files in %s" % (len(files), source), "value")

    grid, _ = destination_grid(config, files)
    # CE QUE `format` DECIDE: un spectre par ORDRE ou un spectre deja recolle.
    # Les noms sont ceux des fichiers (t.fits d'APERO, S2D d'ESO, e2ds du DRS
    # de HARPS et de SOPHIE), mais la seule chose qui change ici est la
    # dimension: _build_tfits parcourt les ordres, _build_s1d une grille
    # unique. Le nom sert aussi a etiqueter le cube en cache, pour qu'un
    # listing du dossier dise de quel instrument il vient (2026-10-14).
    fmt = str(inp.get("format", "tfits"))
    snippets = {}
    if fmt in ORDER_FORMATS:
        result = _build_tfits(config, files, grid, snippets)
    elif fmt in FLAT_FORMATS:
        result = _build_s1d(config, files, grid)
    else:
        raise ValueError("unknown input.format: %r (expected one of %s)"
                         % (fmt, ", ".join(sorted(ORDER_FORMATS | FLAT_FORMATS))))

    grid, data, sigma, trans, meta = result
    # written whenever the cache is on, rebuild or not: what a rebuild rebuilds
    # is the cube on disk, and the stages after this one read it from there
    if out["use_cache"]:
        _save_cache(cache, grid, data, sigma, trans, meta)
        # after _save_cache, which replaces the whole directory
        _write_snippets(cache, snippets)
    return grid, data, sigma, trans, meta


def _figure_blocks(config: dict, grid: np.ndarray) -> list:
    """The grid blocks around output.windows, as grids.window_block cuts them."""
    blocks = []
    dv = grids.grid_dv(grid)
    for spec in config["output"].get("windows") or []:
        try:
            centre, width = grids.parse_window(spec)
        except ValueError:
            continue
        block = grids.window_block(grid, centre, width, dv)
        if block is not None and block not in blocks:
            blocks.append(block)
    return blocks


def _write_snippets(cube_dir: str, snippets: dict) -> None:
    """The raw flux kept during the build, one file per window block."""
    from . import cache as _cache

    for (a0, b0), by_file in sorted(snippets.items()):
        _cache.write_snippet(cube_dir, a0, b0, by_file)
    if snippets:
        log("kept the raw flux around %d figure windows, %d spectra each, in %s"
            % (len(snippets), max(len(v) for v in snippets.values()),
               os.path.join(cube_dir, "snippets")), "value")


# --------------------------------------------------------------------------
# t.fits path
# --------------------------------------------------------------------------
def _build_tfits(config: dict, files: list[str], grid: np.ndarray,
                 snippets=None):
    dom = config["domain"]
    inp = config["input"]
    n_pixels = grid.size

    stack, why = resolve_stacking(config, len(files), n_pixels)
    log("nightly stacking: %s (%s)" % ("on" if stack else "off", why), "value")
    files, labels = spnights.group_files(
        files, stack, extension=1,
        bundle=spnights.bundle_settings(inp) if stack else None)
    n_groups = len(set(labels))
    _check_memory(config, 2 * n_groups, n_pixels)

    use_recon = config["weights"].get("telluric_mode") == "recon"
    coadd = _Coadder(n_groups, 2, n_pixels, CUBE_DTYPE, use_recon)

    n_rejected = 0
    #: les poses qui attendent le template, quand le passe-haut est differe
    held = []
    n_used = 0
    parity_ratio = []
    blocks = _figure_blocks(config, grid) if snippets is not None else []

    for i, (path, label) in enumerate(_bar(zip(files, labels), total=len(files),
                                          desc="reading spectra", unit="file")):
        try:
            payload = _readers.read_any(path, config)
        except Exception as exc:                        # noqa: BLE001
            log("skipping %s (%s)" % (os.path.basename(path), exc), "warn")
            continue

        meta = dict(payload["meta"])
        if inp["object"] is not None and not same_object(
                meta, inp.get("object_header") or inp["object"]):
            log("skipping %s (OBJECT=%s, DRSOBJN=%s, not %s)"
                % (meta["filename"], meta["object"], meta.get("drsobjn"),
                   inp.get("object_header") or inp["object"]), "warn")
            continue
        meta = one_object(meta, inp)
        meta["snr_band"] = sptf.band_snr(payload, dom["wave_min"], dom["wave_max"])
        usable = np.isfinite(payload["flux"]) & (payload["flux"] > 0)
        meta["median_flux"] = (float(np.median(payload["flux"][usable]))
                               if usable.any() else np.nan)
        meta["nan_fraction"] = float(1.0 - usable.mean())

        # LES LONGUEURS D'ONDE DE LA POSE, et non la grille: c'est la seule
        # branche qui en dispose, et le critere qui rejette une solution qui
        # recule a besoin de les voir ordre par ordre.
        reason = _quality_reason(meta, payload["wave"], config["quality"])
        if reason is not None:
            log("rejecting %s: %s" % (meta["filename"], reason), "warn")
            n_rejected += 1
            continue

        # each exposure is registered with its OWN BERV before it is coadded:
        # within a night the BERV moves by up to 0.5 km/s, a whole destination
        # pixel, and stacking first would smear every stellar line by that much
        values, sig, tr, good, n_orders, segments = sptf.resample_exposure(
            payload, grid, config)
        deferred_hp = bool(config["highpass"].get("against_template"))
        if n_orders == 0 or not good.any():
            log("rejecting %s: no order survived in the domain" % meta["filename"],
                "warn")
            n_rejected += 1
            continue

        # The raw flux around every figure window, from the spectrum already in
        # memory, so that drawing the figures never reopens the files. A few
        # thousand samples per exposure; see cache.py.
        for a0, b0 in blocks:
            snippets.setdefault((a0, b0), {})[meta["filename"]] = sptf.raw_block(
                payload, grid, a0, b0, config)

        ratio = _parity_agreement(values, sig, good)
        if np.isfinite(ratio):
            parity_ratio.append(ratio)

        if deferred_hp:
            held.append((label, values, sig, tr if use_recon else None,
                         good, meta, segments))
            continue
        coadd.add(label, values, sig, tr if use_recon else None, good, meta)
        n_used += 1

    if held:
        n_used += _highpass_against_template(coadd, held, grid, config)

    if n_used == 0:
        raise RuntimeError("every exposure was rejected")
    if n_rejected:
        log("%d exposures failed the quality cuts" % n_rejected, "warn")
    data, sigma, trans, meta_table = coadd.result()
    log("kept %d exposures -> %d groups -> %d cube rows (even + odd orders)"
        % (n_used, coadd.n_groups, data.shape[0]), "value")
    if parity_ratio:
        median_ratio = float(np.median(parity_ratio))
        log("odd vs even disagreement in the order overlaps: %.2f x the photon"
            " noise (median over exposures)" % median_ratio, "value")
        if median_ratio > 1.3:
            log("  the two parities disagree by more than photon noise. That is"
                " expected -- the overlaps sit at the order edges, where the"
                " blaze has collapsed and the effective resolution differs"
                " between the two orders -- but it means the photon model is"
                " optimistic there.", "warn")
    return grid, data, sigma, trans, meta_table


def _parity_agreement(values, sigma, good):
    """Observed odd-even scatter over the order overlaps, in units of sigma.

    Free diagnostic: wherever both parities cover a column they measured the
    same photons at two different places on the detector, so their difference
    is an error estimate that owes nothing to any noise model.
    """
    both = good[0] & good[1]
    if both.sum() < 100:
        return np.nan
    difference = values[0][both] - values[1][both]
    expected = np.sqrt(sigma[0][both] ** 2 + sigma[1][both] ** 2)
    expected = np.median(expected[np.isfinite(expected)])
    if not np.isfinite(expected) or expected <= 0:
        return np.nan
    return float(np.std(difference) / expected)


# --------------------------------------------------------------------------
# s1d path (unchanged in substance; rows are exposures, parity is always 0)
# --------------------------------------------------------------------------
def _build_s1d(config: dict, files: list[str], grid: np.ndarray):
    dom = config["domain"]
    inp = config["input"]
    reg = config["registration"]
    hip = config["highpass"]
    wcf = config["weights"]

    spio.check_common_grid(files)
    stack, why = resolve_stacking(config, len(files), grid.size)
    log("nightly stacking: %s (%s)" % ("on" if stack else "off", why), "value")
    files, labels = spnights.group_files(
        files, stack, extension=0,
        bundle=spnights.bundle_settings(config["input"]) if stack else None)
    n_groups = len(set(labels))
    n_pixels = grid.size
    _check_memory(config, n_groups, n_pixels)

    use_recon = wcf.get("telluric_mode") == "recon"
    coadd = _Coadder(n_groups, 1, n_pixels, CUBE_DTYPE, use_recon)

    n_missing_recon = 0
    n_used = 0
    n_rejected = 0
    observer_frame = reg["frame"] == "observer"
    pad = max(40.0, abs(reg["target_berv"]) + 40.0)

    for i, (path, label) in enumerate(_bar(zip(files, labels), total=len(files),
                                          desc="reading spectra", unit="file")):
        try:
            wave, flux, s1d_w, meta = spio.read_spectrum(
                path, dom["wave_min"], dom["wave_max"], pad_kms=pad,
            )
        except Exception as exc:                        # noqa: BLE001
            log("skipping %s (%s)" % (os.path.basename(path), exc), "warn")
            continue

        if inp["object"] is not None and not same_object(
                meta, inp.get("object_header") or inp["object"]):
            log("skipping %s (OBJECT=%s, DRSOBJN=%s, not %s)"
                % (meta["filename"], meta["object"], meta.get("drsobjn"),
                   inp.get("object_header") or inp["object"]), "warn")
            continue
        meta = one_object(meta, inp)
        reason = _quality_reason(meta, wave, config["quality"])
        if reason is not None:
            log("rejecting %s: %s" % (meta["filename"], reason), "warn")
            n_rejected += 1
            continue

        if wcf["mode"] == "photon":
            sig_obs = prep.photon_sigma(flux, s1d_w, meta["snr_band"],
                                        wcf["snr_pixel_scale"])
        else:
            sig_obs = np.ones_like(flux)

        if hip["method"] == "none":
            positive = np.isfinite(flux) & (flux > 0)
            values = np.where(positive, np.log(np.where(positive, flux, 1.0)), 0.0)
            good = positive
        elif hip["frame"] == "observer":
            values, good = prep.highpass(flux, hip["window"], hip["polyorder"],
                                         hip["mode"])
        else:
            positive = np.isfinite(flux) & (flux > 0)
            values, good = np.where(positive, flux, np.nan), positive

        good &= np.isfinite(sig_obs)
        berv = 0.0 if observer_frame else float(meta["berv"])
        target = 0.0 if observer_frame else float(reg["target_berv"])

        reg_values, reg_good = prep.register(
            wave, values, good, berv, target, grid,
            spline_order=reg["spline_order"], mask_threshold=reg["mask_threshold"],
        )
        reg_sigma, _ = prep.register(
            wave, np.where(good, sig_obs, np.nanmedian(sig_obs)), good, berv, target,
            grid, spline_order=1, mask_threshold=reg["mask_threshold"],
        )
        if hip["method"] != "none" and hip["frame"] != "observer":
            reg_values, hp_good = prep.highpass(
                np.where(reg_good, reg_values, np.nan),
                hip["window"], hip["polyorder"], hip["mode"],
            )
            reg_good &= hp_good

        reg_trans = None
        if use_recon:
            recon = spio.recon_path_for(path)
            trans_obs = None
            if recon is not None:
                try:
                    trans_obs = spio.read_transmission(
                        recon, dom["wave_min"], dom["wave_max"], pad_kms=pad)
                except Exception as exc:                # noqa: BLE001
                    # an unreadable partner must not end a 700-file run. It
                    # happens: on a cloud-synced directory a file can be a
                    # placeholder that materialises as garbage, which astropy
                    # reports as "No SIMPLE card found"
                    log("cannot read %s (%s); telluric weight left at 1"
                        % (os.path.basename(recon), exc), "warn")
            if trans_obs is None:
                if recon is None:
                    log("no recon partner for %s; telluric weight left at 1"
                        % meta["filename"], "warn")
                n_missing_recon += 1
                reg_trans = np.ones(n_pixels)
            else:
                finite_trans = np.isfinite(trans_obs)
                reg_trans, trans_good = prep.register(
                    wave, np.where(finite_trans, trans_obs, 1.0), finite_trans,
                    berv, target, grid, spline_order=reg["spline_order"],
                    mask_threshold=reg["mask_threshold"],
                )
                reg_trans = np.where(trans_good, reg_trans, 0.0)

        reg_good &= np.isfinite(reg_sigma) & (reg_sigma > 0)
        coadd.add(label, reg_values[None, :], reg_sigma[None, :],
                  None if reg_trans is None else reg_trans[None, :],
                  reg_good[None, :], meta)
        n_used += 1

    if n_used == 0:
        raise RuntimeError("every spectrum was rejected")
    if n_rejected:
        log("%d spectra failed the quality cuts" % n_rejected, "warn")
    if n_missing_recon:
        log("%d spectra had no recon partner" % n_missing_recon, "warn")
    data, sigma, trans, meta_table = coadd.result()
    log("kept %d / %d spectra -> %d cube rows" % (n_used, len(files), data.shape[0]),
        "value")
    return grid, data, sigma, trans, meta_table


#: le nom du template, a cote du cube
TEMPLATE_FILE = "hp_template.npy"


def template_path(config):
    """Ou le template du passe-haut est range, ou None s'il n'y en a pas."""
    from .config import cache_key

    if not (config.get("highpass") or {}).get("against_template"):
        return None
    fmt = str(config["input"].get("format", "tfits"))
    folder = os.path.join(config["output"].get("cache_directory", "cache"),
                          "cube_%s_%s" % (fmt, cache_key(config)))
    return os.path.join(folder, TEMPLATE_FILE)


#: le template que la seconde passe vient de construire, en attente d'etre
#: ecrit avec le cube. IL NE PEUT PAS ETRE ECRIT TOUT DE SUITE: _save_cache
#: batit le cube dans un dossier .partial puis efface le dossier final avant
#: de renommer, donc un fichier depose la avant serait emporte (vu le
#: 2026-10-16: le template etait construit, annonce dans le journal, et
#: absent du disque).
_PENDING_TEMPLATE = None
#: the coadder's members, [(exposure, row name, group)], saved with the cube
_PENDING_MEMBERS = None
#: where they are saved, in the cube's folder
MEMBERS_FILE = "members.csv"


def set_pending_members(members):
    global _PENDING_MEMBERS
    _PENDING_MEMBERS = list(members) if members else None


def _stash_template(config, template):
    """Mettre le template de cote jusqu'a l'ecriture du cube."""
    global _PENDING_TEMPLATE
    _PENDING_TEMPLATE = np.asarray(template, dtype=np.float32)


def load_template(config):
    """Le template du passe-haut, ou None."""
    path = template_path(config)
    if path is None or not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None


def _highpass_against_template(coadd, held, grid, config):
    """La seconde passe: le passe-haut pris sur le residu a un template median.

    LE PROBLEME QU'ELLE RESOUT. Un Savitzky-Golay passe sur un spectre entier
    est tire vers le bas par chaque raie profonde: le lisse n'est pas le
    continu, c'est le continu moins une fraction de la raie. Tant que la raie
    est la meme d'une pose a l'autre, ce biais est constant et la PCA ne le
    voit pas. Mais une tellurique qui balaie le coeur d'une raie stellaire
    avec le BERV change ce biais D'UNE POSE A L'AUTRE, et laisse un residu qui
    suit le BERV sans etre stellaire.

    Mesure le 2026-10-16 sur HD 69830, raie a 688.85 nm: quand le BERV couvre
    ses 51 km/s, la tellurique parcourt 0.117 nm, plusieurs largeurs de raie,
    et sa profondeur au coeur passe de 3% a 89%. L'effet etant intermittent,
    une correlation simple avec le BERV ne le voit pas (0.036 sur la
    campagne).

    LE TEMPLATE EST BATI DANS LE REPERE DE L'ETOILE, puis redescendu dans le
    repere de chaque pose. Le cube, lui, ne change pas de repere: l'ajustement
    a deux reperes fait son propre decalage du bloc stellaire et wizard.py
    refuse un cube deja registre. Seul le template voyage.

    LE FILTRE RESTE PAR SEGMENT D'ORDRE, jamais a travers le trou qui separe
    deux ordres de meme parite, exactement comme la premiere passe.
    """
    from . import preprocess as prep

    reg = config["registration"]
    hip = config["highpass"]
    bervs = np.array([float(m["berv"]) for _, _, _, _, _, m, _ in held])

    # le template: la mediane des poses ramenees dans le repere de l'etoile
    rows = []
    for (_, values, _, _, good, _, _), berv in zip(held, bervs):
        row = np.where(good, values, np.nan)
        # chaque parite normalisee par sa mediane, pour qu'une pose brillante
        # ne pese pas plus qu'une pose faible
        with np.errstate(invalid="ignore"):
            row = row - np.nanmedian(row, axis=1, keepdims=True)
        moved = np.empty_like(row)
        for parity in range(row.shape[0]):
            finite = np.isfinite(row[parity])
            shifted, shifted_good = prep.register(
                grid, np.where(finite, row[parity], 0.0), finite, berv, 0.0,
                grid, spline_order=reg["spline_order"],
                mask_threshold=reg["mask_threshold"])
            moved[parity] = np.where(shifted_good, shifted, np.nan)
        rows.append(moved)
    with np.errstate(invalid="ignore"):
        template = np.nanmedian(np.array(rows), axis=0)
    covered = int(np.sum(np.isfinite(template)))
    log("high pass against a median template built in the star's frame:"
        " %d of %d samples covered" % (covered, template.size), "value")
    # LE TEMPLATE EST ECRIT A COTE DU CUBE, parce que l'etape `correct` doit
    # appliquer EXACTEMENT le meme passe-haut: elle resample les memes
    # spectres et retranche les composantes ajustees ici, donc un filtre
    # different de part et d'autre compare des choses qui ne sont pas a la
    # meme echelle. Vu le 2026-10-16: sans cela `correct` recevait du
    # ln(flux) brut (mediane +10.14) la ou le cube portait un residu
    # passe-haut (mediane +0.017), et le bruit des spectres corriges doublait.
    _stash_template(config, template)

    n_used = 0
    for (label, values, sig, tr, good, meta, segments), berv in zip(held,
                                                                   bervs):
        here = np.empty_like(template)
        for parity in range(template.shape[0]):
            finite = np.isfinite(template[parity])
            shifted, shifted_good = prep.register(
                grid, np.where(finite, template[parity], 0.0), finite, 0.0,
                berv, grid, spline_order=reg["spline_order"],
                mask_threshold=reg["mask_threshold"])
            here[parity] = np.where(shifted_good, shifted, np.nan)

        out = np.zeros_like(values)
        keep = np.zeros(good.shape, dtype=bool)
        for parity, start, stop in segments:
            seg, seg_good = prep.highpass_against_template(
                values[parity, start:stop], good[parity, start:stop],
                here[parity, start:stop], hip["window"], hip["polyorder"],
                already_log=True)
            out[parity, start:stop] = seg
            keep[parity, start:stop] = seg_good
        keep &= np.isfinite(sig) & (sig > 0)
        coadd.add(label, np.where(keep, out, 0.0), sig, tr, keep, meta)
        n_used += 1
    return n_used


def _quality_reason(meta, wave, quality):
    """Return why this exposure should be rejected, or None to keep it.

    The four 2025-04-11 frames in this dataset motivated making these checks
    explicit: they carry a *negative* median flux and negative or 'NaN'
    per-order SNR across the whole band, i.e. the extraction failed. Before
    these cuts existed they were dropped only by accident, when the string
    'NaN' in EXTSN crashed a numpy call.
    """
    if np.size(wave) < 10:
        return "no coverage in the requested band"
    # UNE SOLUTION EN LONGUEUR D'ONDE QUI RECULE REND LA POSE INUTILISABLE, et
    # pas seulement pour pca2d. Mesure le 2026-10-14: 5 des 60 poses ESPRESSO
    # de tau Ceti ont, au bord bleu de leur ordre le plus bleu, 177 pixels
    # dont la longueur d'onde decroit avant de repartir. pca2d sait les
    # ecarter du masque et trier ses abscisses, mais LBL lit les MEMES
    # fichiers et s'arrete dessus dans lbl_template, donc une pose corrigee
    # par pca2d et refusee par LBL n'avance a rien. On rejette la pose ici,
    # une fois, et les deux cotes la voient partir pour la meme raison.
    if np.ndim(wave) == 2 and wave.shape[1] > 1:
        steps = np.diff(wave, axis=1)
        backward = np.isfinite(steps) & (steps <= 0)
        if backward.any():
            orders = int(backward.any(axis=1).sum())
            return ("the wavelength solution runs backwards at %d samples in"
                    " %d order(s), which no spline can resample"
                    % (int(backward.sum()), orders))
    if not np.isfinite(meta["berv"]):
        return "no valid BERV"
    if not np.isfinite(meta["bjd"]):
        return "no valid BJD"
    snr = meta["snr_band"]
    if quality["min_snr"] is not None:
        if not np.isfinite(snr):
            return "no valid per-order SNR in the band"
        if snr < float(quality["min_snr"]):
            return "band SNR %.2f below the %.1f threshold" % (snr, quality["min_snr"])
    if quality["require_positive_flux"] and not (meta["median_flux"] > 0):
        return "median band flux %.4g is not positive" % meta["median_flux"]
    if quality["max_nan_fraction"] is not None:
        if meta["nan_fraction"] > float(quality["max_nan_fraction"]):
            return "%.1f%% of the band is NaN" % (100 * meta["nan_fraction"])
    # the epoch window, in rjd. meta["bjd"] is a full Julian date, so the
    # conversion happens here once rather than in every config
    rjd = float(meta["bjd"]) - 2400000.0
    if quality.get("min_rjd") is not None and rjd < float(quality["min_rjd"]):
        return "rjd %.2f is before the %.0f window start" % (rjd, quality["min_rjd"])
    if quality.get("max_rjd") is not None and rjd > float(quality["max_rjd"]):
        return "rjd %.2f is after the %.0f window end" % (rjd, quality["max_rjd"])
    # A LIST of nights, where the window above is a range. What it is for: a
    # joint fit assumes the atmosphere belongs to the night, which only holds for
    # stars observed on the SAME nights, and the nights two campaigns share are
    # not a contiguous range (GJ 1 and GJ 3090 share 47, scattered through two
    # seasons of 147 and 99). Integer rjd, the same night label pca2d.scan uses.
    nights = quality.get("nights")
    if nights:
        if int(np.floor(rjd)) not in {int(n) for n in nights}:
            return "rjd %.2f is not one of the %d nights asked for" % (
                rjd, len(nights))
    # LE MODE DU SPECTROGRAPHE, quand une campagne en melange deux. Mesure le
    # 2026-10-15 sur les 100 poses HARPS de HD 10700: 8 sont en mode EGGS, la
    # fibre de 1.4" a haute efficacite, et ce sont EXACTEMENT les 8 dont
    # l'ajustement tellurique echoue, l'exposant d'eau y tombant entre 0.004
    # et 0.047 contre 0.30 pour les poses HARPS.
    #
    # CE N'EST PAS UN DECALAGE EN VITESSE, contrairement a ce que ce
    # commentaire a d'abord dit: le modele TAPAS tombe a sa place sur ces
    # poses, minimum local entre 0 et +3 km/s pour 7 des 8. Ce n'est pas non
    # plus la resolution (de 115000 a 60000, l'eau ne monte que de 0.0039 a
    # 0.0089), et les telluriques y sont bien presentes, aussi profondes que
    # dans les poses HARPS. Ce qui cloche reste a trouver; on les ecarte parce
    # que leur correction est manifestement fausse, pas parce qu'on a compris.
    modes = quality.get("drs_modes")
    if modes:
        mode = str(meta.get("drs_mode", "") or "").strip()
        if mode not in {str(m).strip() for m in modes}:
            return ("instrument mode %r is not one of the %d asked for (%s)"
                    % (mode or "(unset)", len(modes),
                       ", ".join(str(m) for m in modes)))
    return None


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
def build_weights(config: dict, grid, data, sigma, meta, trans=None, chunk=64):
    """Turn per-pixel sigmas into PCA weights, applying every rejection rule.

    Done in row blocks. At full NIRPS coverage a single float64 temporary the
    size of the cube is 5 GB, so `variance = sigma.astype(float) ** 2` written
    the obvious way is the difference between a run that works and one that
    swaps.
    """
    wcf = config["weights"]
    weights = np.zeros(data.shape, dtype=np.float32)
    n_total = weights.size
    counters = dict(dead=0, low=0, high=0, telluric_killed=0, telluric_sum=0.0)

    mode = wcf.get("telluric_mode", "recon")
    use_ramp = mode == "recon" and trans is not None
    if mode == "recon" and trans is None:
        log("telluric_mode is 'recon' but no transmission cube was built;"
            " no telluric weighting applied", "warn")

    for start in range(0, data.shape[0], chunk):
        stop = min(start + chunk, data.shape[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = np.asarray(sigma[start:stop], dtype=np.float64) ** 2
            if wcf["sigma_floor_frac"]:
                variance = variance + float(wcf["sigma_floor_frac"]) ** 2
            block = np.where(np.isfinite(variance) & (variance > 0), 1.0 / variance, 0.0)
        values = data[start:stop]
        bad = ~np.isfinite(values)
        block[bad] = 0.0
        counters["dead"] += int(np.sum(block == 0))
        # LE SEUIL PORTE SUR LE SNR DU PIXEL, PAS SUR SA PROFONDEUR. Un fond de
        # raie profonde porte moins de photons, donc un RMS fractionnaire plus
        # grand, mais ce n'est pas une raison de le jeter: c'est au contraire
        # l'endroit ou le contenu en vitesse est le plus dense. Ce qu'il faut
        # ecarter, c'est le pixel dont le bruit depasse ce que les photons
        # expliquent, pas celui qui est simplement sombre.
        #
        # sigma est le sigma photonique du log-flux (preprocess.photon_sigma),
        # donc 1/sigma EST le SNR local, et le seuil se lit en SNR.
        #
        # CE QUE L'ANCIEN CRITERE COUTAIT, mesure le 2026-10-16 sur HD 10700:
        # ln_clip_low = -0.5 tuait tout pixel sous exp(-0.5) = 0.607 du
        # continu, c'est-a-dire le fond de TOUTE raie plus profonde que 39%.
        # Au fond du Na D2 a 589.15 nm, les pixels perdus avaient une
        # profondeur mediane de 0.340 contre 0.712 pour les gardes. Les
        # spectres corrigeaient 13.7% de NaN contre 0.07% dans les livres, et
        # LBL y perdait le tiers de ses raies (7860 au lieu de 11134) avec une
        # erreur par raie de 340 m/s au lieu de 145.
        if wcf.get("min_pixel_snr") is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                snr = 1.0 / np.sqrt(variance)
            cut = ~(snr >= float(wcf["min_pixel_snr"]))
            block[cut] = 0.0
            counters["low"] += int(cut.sum())
        elif wcf.get("ln_clip_low") is not None:
            cut = values < float(wcf["ln_clip_low"])
            block[cut] = 0.0
            counters["low"] += int(cut.sum())
        if wcf["ln_clip_high"] is not None:
            cut = values > float(wcf["ln_clip_high"])
            block[cut] = 0.0
            counters["high"] += int(cut.sum())
        if use_ramp:
            ramp = prep.telluric_ramp(trans[start:stop], wcf["telluric_ramp_zero"],
                                      wcf["telluric_ramp_one"])
            block *= ramp
            counters["telluric_killed"] += int(np.sum(ramp <= 0))
            counters["telluric_sum"] += float(ramp.sum())
        weights[start:stop] = block

    log("zero-weight from non-finite / failed registration: %.3f%%"
        % (100.0 * counters["dead"] / n_total), "value")
    if wcf.get("min_pixel_snr") is not None:
        log("zero-weight from pixel SNR < %.1f: %.3f%%"
            % (wcf["min_pixel_snr"], 100.0 * counters["low"] / n_total), "value")
    elif wcf.get("ln_clip_low") is not None:
        log("zero-weight from y < %.2f (deep lines): %.3f%%"
            % (wcf["ln_clip_low"], 100.0 * counters["low"] / n_total), "value")
    if wcf["ln_clip_high"] is not None:
        log("zero-weight from y > %.2f: %.3f%%"
            % (wcf["ln_clip_high"], 100.0 * counters["high"] / n_total), "value")
    if use_ramp:
        log("telluric weight from the recon extension: ramp 0 at T = %.2f, 1 at T"
            " = %.2f" % (wcf["telluric_ramp_zero"], wcf["telluric_ramp_one"]))
        log("  mean telluric weight %.4f, %.3f%% of samples fully rejected"
            % (counters["telluric_sum"] / n_total,
               100.0 * counters["telluric_killed"] / n_total), "value")
    elif mode == "from_data":
        from . import telluric as _telluric
        weights = _telluric.apply_from_data(weights, grid, meta, config)
    elif mode == "file" and wcf["telluric_file"]:
        weights = apply_telluric(weights, grid, meta, config)

    _mask_thin_columns(weights, meta, float(wcf["min_good_fraction"]))
    log("total zero-weight fraction: %.3f%%"
        % (100.0 * np.mean(weights == 0)), "value")
    return weights


def _mask_thin_columns(weights, meta, min_good_fraction):
    """Kill grid columns that too few spectra constrain.

    Counted within each parity rather than over all rows. Half the domain is
    covered by the even orders only and half by the odd ones, so a global count
    would condemn every single-parity column no matter how well measured it is.
    """
    parity = np.asarray(meta["parity"]) if "parity" in meta.colnames else np.zeros(
        weights.shape[0], dtype=int)
    best = np.zeros(weights.shape[1])
    for value in np.unique(parity):
        rows = parity == value
        best = np.maximum(best, np.mean(weights[rows] > 0, axis=0))
    dead = best < min_good_fraction
    if np.any(dead):
        weights[:, dead] = 0.0
        where = " in either parity" if np.unique(parity).size > 1 else ""
        log("masked %d / %d grid columns below %.0f%% good spectra%s"
            % (dead.sum(), dead.size, 100 * min_good_fraction, where), "warn")


def apply_telluric(weights, grid, meta, config):
    """Down-weight pixels sitting under telluric absorption, from a text model.

    Expects a two-column text file (wavelength in nm, transmission in 0-1)
    sampled in the OBSERVER frame. That frame matters: tellurics are fixed in
    the Earth frame while the cube is registered to the stellar frame, so the
    model has to be shifted per spectrum by that spectrum's BERV before being
    applied. The weight is multiplied by transmission ** telluric_power, which
    for power = 2 is what a pure photon-noise argument gives.
    """
    path = config["weights"]["telluric_file"]
    power = float(config["weights"]["telluric_power"])
    if not os.path.exists(path):
        log("telluric file %s not found; skipping telluric weighting" % path, "warn")
        return weights

    table = np.loadtxt(path)
    wave_model, trans_model = table[:, 0], np.clip(table[:, 1], 0.0, 1.0)
    observer_frame = config["registration"]["frame"] == "observer"
    target = 0.0 if observer_frame else float(config["registration"]["target_berv"])

    for n in range(weights.shape[0]):
        berv = 0.0 if observer_frame else float(meta["berv"][n])
        wave_obs = grid * grids.doppler(target) / grids.doppler(berv)
        trans = np.interp(wave_obs, wave_model, trans_model, left=1.0, right=1.0)
        weights[n] *= np.clip(trans, 0.0, 1.0) ** power
    log("applied telluric weighting from %s (power %.1f)" % (path, power))
    return weights
