"""Lire un spectre de n'importe quel instrument, en passant par LBL.

« Tout ce qui passe dans le LBL peut être envoyé au PCA2D » (2026-10-14). LBL
sait déjà lire quinze combinaisons instrument/source: SPIRou et NIRPS en APERO,
CADC et ESO, HARPS et HARPS-N en ORIG, ESO et ESSP, ESPRESSO, CARMENES, SOPHIE,
CORALIE, MAROON-X, EXPRES, NEID, plus une classe générique. Réimplémenter cette
lecture ici reviendrait à la maintenir deux fois et à diverger; et puisque
pca2d a vocation à finir dans LBL, mieux vaut en emprunter dès maintenant le
chemin.

    read(path, instrument, data_source) -> le même dict que tfits.read_tfits

Ce que LBL donne nativement et ce qu'il faut en tirer:

    flux    inst.load_science_file(path), toujours 2D (ordres, pixels)
    wave    inst.get_wave_solution(path, data, header), en nm
    blaze   inst.load_blaze_from_science(...), normalisé par ordre
    meta    les mots-clés par leur nom LBL (KW_BERV, KW_MJDATE...), que chaque
            classe traduit pour son instrument

Ce que LBL ne donne pas, et ce qu'on en fait:

    recon   la transmission tellurique. APERO l'écrit dans le t.fits (Recon);
            HARPS et les autres ne l'ont pas, et LBL la calcule lui-même à
            partir de son modèle TAPAS (lbl/science/tellu_clean.py, le fichier
            tapas_lbl.fits qu'il télécharge). `tapas_transmission` ci-dessous
            l'évalue sur la grille du spectre, ce qui donne à pca2d le même
            objet que le Recon d'APERO.
    sky     l'émission du ciel. Aucun spectrographe visible ne la livre par
            pose; quality.max_sky_ratio est alors sans objet et le masque de
            ciel est laissé à l'instrument. None, et le reste du pipeline en
            tient compte.
    snr     le rapport signal sur bruit par ordre. Quand l'en-tête ne le porte
            pas, il est mesuré sur le flux lui-même (robuste, par ordre).

LES DEUX PARITÉS. pca2d ramène chaque pose à deux spectres, les ordres pairs et
les ordres impairs, parce que sur un échelle les ordres n et n+1 se recouvrent
alors que n et n+2 ne se touchent jamais: tout le spectre tient donc sans
redondance sur deux lignes. La règle est gardée pour tous les instruments
(décidé le 2026-10-14), mais elle est VÉRIFIÉE par `parity_is_safe`, qui mesure
le recouvrement réel et prévient quand l'hypothèse ne tient pas.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

#: ce que LBL appelle les instruments, et la source de données associée, tels
#: que select.InstDict les épelle. Ce qui n'est pas ici n'est pas refusé: la
#: paire est passée telle quelle à LBL, qui dira ce qu'il en pense.
KNOWN = {
    "SPIROU": ("APERO", "CADC"),
    "NIRPS_HA": ("APERO", "CADC", "ESO"),
    "NIRPS_HE": ("APERO", "CADC", "ESO"),
    "HARPS": ("ORIG", "ESO", "ESSP"),
    "HARPSN": ("ORIG", "ESO", "ESSP"),
    "ESPRESSO": ("None",),
    "CARMENES": ("None",),
    "SOPHIE": ("None",),
    "CORALIE": ("None",),
    "MAROONX": ("RED", "BLUE"),
    "EXPRES": ("ORIG", "ESSP"),
    "NEID": ("ESSP",),
}
#: GENERIC existe comme classe dans LBL mais pas dans sa table instrument/source:
#: il se configure à la main, donc pca2d ne l'annonce pas ici. Il reste
#: utilisable en passant son nom directement à `instrument`.


def instrument(name, data_source, config_file=None):
    """La classe LBL qui lit cet instrument, instanciée.

    `parse=False` est ce qui permet de l'instancier depuis du code plutôt que
    depuis une ligne de commande; c'est déjà ainsi que lbl.check_profile
    vérifie un profil avant une campagne.
    """
    from lbl.instruments import select

    kwargs = dict(instrument=str(name), data_source=str(data_source))
    if config_file:
        kwargs["config_file"] = os.path.abspath(config_file)
    args = select.parse_args(
        ["INSTRUMENT", "DATA_SOURCE", "DATA_TYPE", "INPUT_FILE"],
        kwargs, __name__, parse=False)
    return select.load_instrument(args, plogger=None)


def _lbl_root():
    """Le dossier du paquet LBL REELLEMENT installe, ou None.

    Trois chemins, du plus franc au plus obstine, parce que le dossier de
    donnees `lbl/` du depot peut masquer les deux premiers:

      1. l'import ordinaire, quand rien ne le masque;
      2. les entrees de sys.path qui portent un vrai lbl/__init__.py;
      3. la distribution installee, par importlib.metadata. C'est le seul qui
         survit a une installation EDITABLE (pip install -e), ou le paquet
         n'est sur aucune entree de sys.path mais atteint par un chercheur
         pose par un .pth: un paquet-espace-de-noms gagne contre ce
         chercheur, et seul le dossier declare par la distribution dit ou le
         code se trouve vraiment.
    """
    import importlib.metadata as metadata
    import importlib.util
    import sys

    spec = importlib.util.find_spec("lbl")
    if spec is not None and spec.origin:
        return os.path.dirname(spec.origin)
    for entry in sys.path:
        candidate = os.path.join(entry, "lbl", "__init__.py")
        if os.path.exists(candidate):
            return os.path.dirname(candidate)
    try:
        dist = metadata.distribution("lbl")
    except Exception:                                          # noqa: BLE001
        return None
    for item in dist.files or []:
        if item.parts[:2] == ("lbl", "__init__.py"):
            return os.path.dirname(str(dist.locate_file(item)))
    # une installation editable: le chercheur pose par le .pth sait ou est le
    # code, et il le dit dans sa table MAPPING
    for item in dist.files or []:
        text = str(item)
        if "__editable___" in text and text.endswith("_finder.py"):
            try:
                path = str(dist.locate_file(item))
                source = open(path, encoding="utf-8").read()
            except Exception:                                  # noqa: BLE001
                continue
            import re
            found = re.search(r"['\"]lbl['\"]\s*:\s*['\"]([^'\"]+)['\"]", source)
            if found and os.path.isdir(found.group(1)):
                return found.group(1)
    return None


def _lbl_science(name):
    """Un module de lbl.science, meme quand le dossier de donnees le masque.

    LE PIEGE, mesure le 2026-10-14: ce depot contient un dossier `lbl/` qui est
    une arborescence de DONNEES de LBL (calib, masks, models, science...). Des
    que la racine du depot est sur sys.path, `import lbl` trouve ce dossier
    avant le paquet installe et en fait un paquet-espace-de-noms: lbl.__file__
    vaut None et `from lbl.science import tellu_clean` echoue avec « unknown
    location ». Et c'est pire qu'un echec franc, parce que lbl.instruments,
    lui, n'est PAS masque (le dossier de donnees n'a pas ce sous-dossier):
    l'instrument se charge, la transmission non, et rien ne le dit.

    Charger le seul module voulu par son chemin ne suffit pas: il importe ses
    voisins par `from lbl.science import general`, qui retombe sur le dossier
    masque. On REPARE donc le paquet, en ajoutant le vrai dossier au __path__
    de lbl et de lbl.science, apres quoi l'import ordinaire trouve tout.
    """
    import importlib
    import sys

    full = "lbl.science.%s" % name
    if full in sys.modules:
        return sys.modules[full]
    try:
        root = _lbl_root()
        if root is not None:
            package = importlib.import_module("lbl")
            paths = list(getattr(package, "__path__", []))
            if root not in paths:
                package.__path__ = [root] + paths
                # lbl.science a pu etre resolu avant la reparation: il porte
                # alors le chemin du dossier de donnees et doit etre oublie
                for key in [k for k in sys.modules
                            if k == "lbl.science" or k.startswith("lbl.science.")]:
                    del sys.modules[key]
        return importlib.import_module(full)
    except Exception:                                          # noqa: BLE001
        return None


def tapas_transmission(inst, wave, model_dir, airmass=1.0):
    """La transmission tellurique sur `wave`, du modèle TAPAS de LBL.

    APERO écrit la transmission qu'il a divisée dans le t.fits (Recon), et
    pca2d s'en sert pour savoir où le spectre est aveugle. Les spectrographes
    visibles ne la livrent pas: LBL la modélise, et c'est ce modèle qu'on
    évalue ici, élevé à la masse d'air de la pose.

    Rend None quand le modèle n'est pas là, ce que l'appelant doit accepter:
    sans transmission, les coupures qui en dépendent sont simplement inactives.
    """
    tellu_clean = _lbl_science("tellu_clean")
    if tellu_clean is None:
        return None
    # LBL recopie le fichier TAPAS dans <DATA_DIR>/other/ avant de le lire, et
    # DATA_DIR est vide sur une classe instanciee hors d'une campagne: sans
    # lui, get_tapas_lbl s'arrete sur un realpath(None). Le dossier des
    # modeles est, par construction, <DATA_DIR>/models, donc son parent est la
    # reponse. On ne l'ecrase jamais quand une campagne l'a deja pose.
    try:
        if not inst.params.get("DATA_DIR"):
            inst.params.set("DATA_DIR",
                            os.path.dirname(os.path.abspath(model_dir)))
    except Exception:                                          # noqa: BLE001
        pass
    try:
        splines = tellu_clean.get_tapas_spl(inst, model_dir)
    except Exception:                                          # noqa: BLE001
        return None
    if splines is None:
        return None
    # get_tapas_spl rend les splines des deux composantes de TAPAS, l'eau et
    # le reste; la transmission totale est leur produit, chacune à la masse
    # d'air de la pose
    out = np.ones_like(np.asarray(wave, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spline in (splines if isinstance(splines, (list, tuple))
                       else [splines]):
            try:
                piece = np.asarray(spline(np.asarray(wave, dtype=float)),
                                   dtype=float)
            except Exception:                                  # noqa: BLE001
                return None
            out *= np.clip(piece, 0.0, 1.0) ** float(airmass)
    return out


def order_snr(flux, header_snr=None, n_orders=None):
    """Le rapport signal sur bruit par ordre.

    Celui de l'en-tête quand l'instrument l'écrit (APERO le fait, EXTSNxxx),
    mesuré sur le flux sinon: la médiane divisée par l'écart robuste des
    différences voisines, qui ne suppose rien sur le modèle de bruit.
    """
    if header_snr is not None and np.size(header_snr):
        return np.asarray(header_snr, dtype=float)
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    out = np.full(flux.shape[0], np.nan)
    for i, row in enumerate(flux):
        good = np.isfinite(row) & (row > 0)
        if good.sum() < 16:
            continue
        level = float(np.median(row[good]))
        step = np.diff(row[good])
        noise = 1.4826 * float(np.median(np.abs(step - np.median(step))))
        # la différence de deux voisins a sqrt(2) fois le bruit d'un seul
        out[i] = level / (noise / np.sqrt(2.0)) if noise > 0 else np.nan
    return out if n_orders is None else out[:n_orders]


def parity_is_safe(wave, tolerance=0.0):
    """(vrai si les ordres n et n+2 ne se recouvrent pas, le pire recouvrement).

    pca2d met les ordres pairs sur une ligne et les impairs sur l'autre, ce qui
    suppose que deux ordres de même parité ne mesurent jamais la même longueur
    d'onde. C'est vrai sur SPIRou et NIRPS; ailleurs il faut le vérifier plutôt
    que l'espérer, parce qu'un recouvrement ferait écrire deux mesures dans la
    même colonne de la grille.
    """
    wave = np.atleast_2d(np.asarray(wave, dtype=float))
    worst = 0.0
    for i in range(wave.shape[0] - 2):
        a, b = wave[i], wave[i + 2]
        if not (np.isfinite(a).any() and np.isfinite(b).any()):
            continue
        lo, hi = np.nanmax(a), np.nanmin(b)
        if np.isfinite(lo) and np.isfinite(hi):
            worst = max(worst, float(lo - hi))
    return worst <= float(tolerance), worst


def _doppler(velocity_kms):
    """Le rapport de longueurs d'onde pour une vitesse, RELATIVISTE.

    Un seul Doppler dans le depot, celui de grids: sqrt((1 + v/c) / (1 - v/c)),
    v > 0 si la source s'eloigne. Le premier ordre, 1 + v/c, se trompe de
    (v/c)^2 / 2, soit 1.5 m/s a 30 km/s de BERV, ce qui compte ici.
    """
    from .grids import doppler

    return float(doppler(float(velocity_kms)))


def berv_of(inst, header, from_header=None):
    """Le BERV a appliquer a ce spectre, en km/s, tel que LBL le decide.

    PAS CELUI DE L'EN-TETE. La question n'est pas « quel etait le mouvement
    barycentrique », que l'en-tete donne toujours, mais « reste-t-il a
    l'appliquer », qui depend du format. Un t.fits de SPIRou est dans le
    repere de l'observateur et il faut l'appliquer; un S2D de l'ESO porte
    WAVEDATA_VAC_BARY, une grille DEJA corrigee, et l'appliquer une seconde
    fois decale le spectre de tout le BERV.

    LBL a deja tranche, classe par classe: son get_berv lit l'en-tete pour
    SPIRou et rend zero pour HARPS, HARPS-N, ESPRESSO et SOPHIE. On lui pose
    donc la question plutot que d'y repondre, et le jour ou un format change
    de convention c'est LBL qui le saura.

    Mesure le 2026-10-14 sur tau Ceti: en prenant le BERV de l'en-tete, les
    poses au BERV le plus negatif sortaient a 1.3 et 4 km/s du groupe, et le
    rms des vitesses de LBL etait de 1311 m/s.
    """
    try:
        value = inst.get_berv(header)          # LBL rend des m/s
    except Exception:                                          # noqa: BLE001
        value = None
    if value is None:
        try:
            return float(from_header)
        except (TypeError, ValueError):
            return float("nan")
    return float(value) / 1000.0


def flux_extension(path, flux):
    """Le nom (ou le numero) de l'extension d'ou vient ce flux.

    RENDRE LE SPECTRE CORRIGE DANS LE FORMAT D'ORIGINE. Le fichier corrige est
    une copie de l'original dont on remplace le flux; encore faut-il savoir
    quelle extension remplacer, et elle n'a pas le meme nom partout: FluxAB
    sur un t.fits de SPIRou, SCIDATA sur un S2D de l'ESO. LBL ne la nomme pas:
    sa classe par defaut appelle fits.getdata(fichier) sans extension, ce qui
    rend la PREMIERE qui porte des donnees.

    On ne devine donc pas ce nom, on le RETROUVE: l'extension dont les donnees
    ont la forme du flux lu, la premiere en partant du debut, exactement comme
    astropy a choisi. Rend un nom quand il y en a un, sinon un numero, et None
    si rien ne correspond, auquel cas l'appelant refuse d'ecrire plutot que
    d'ecrire au mauvais endroit.
    """
    from astropy.io import fits

    shape = np.shape(flux)
    try:
        with fits.open(path, memmap=True) as hdulist:
            for index, hdu in enumerate(hdulist):
                data = getattr(hdu, "data", None)
                if data is None:
                    continue
                if np.atleast_2d(data).shape == shape:
                    return hdu.name or index
    except Exception:                                          # noqa: BLE001
        return None
    return None


#: the instruments already told that their blaze is taken as 1
_NO_BLAZE = set()


def _blaze(inst, path, flux, header, model_dir):
    """The blaze of a spectrum, as LBL reads it: corrected spectra need none,
    the others the file their header names, in DATA_DIR/calib where LBL
    keeps it (models/ as well, where an older campaign put it), else the
    default blaze of the setup (models/default-blaze, Instrument.blaze_path).
    Without one, the blaze is taken as 1, and said once per instrument and
    plainly, since LBL's own reader would log a red error for a file that is
    simply not there: PCA2D high-passes every spectrum, which takes the
    blaze's shape out; only the weights do without it."""
    if inst.params.get("BLAZE_CORRECTED"):
        return np.ones_like(flux)
    name = None
    try:
        name = header.get_hkey(inst.params["KW_BLAZE_FILE"])
    except Exception:                                           # noqa: BLE001
        pass
    folders = []
    if model_dir:
        folders.append(os.path.join(os.path.dirname(os.path.abspath(
            model_dir)), "calib"))
        folders.append(model_dir)
    for folder in folders:
        if name and os.path.exists(os.path.join(folder, os.path.basename(
                str(name)))):
            try:
                blaze, _ = inst.load_blaze_from_science(
                    path, flux, header, calib_directory=folder,
                    normalize=True)
                return np.atleast_2d(np.asarray(blaze, dtype=float))
            except Exception:                                   # noqa: BLE001
                break
    # the default blaze of the setup (models/default-blaze), which LBL's own
    #   reader takes when the night's is not in calib/; DATA_DIR is the
    #   models folder's parent for a class made outside a campaign
    if model_dir and not inst.params.get("DATA_DIR"):
        try:
            inst.params.set("DATA_DIR",
                            os.path.dirname(os.path.abspath(model_dir)))
        except Exception:                                       # noqa: BLE001
            pass
    default = None
    try:
        default = inst.default_blaze_file()
    except Exception:                                           # noqa: BLE001
        pass
    if default and os.path.exists(default) and folders:
        try:
            blaze, _ = inst.load_blaze_from_science(
                path, flux, header, calib_directory=folders[0],
                normalize=True)
            return np.atleast_2d(np.asarray(blaze, dtype=float))
        except Exception:                                       # noqa: BLE001
            pass
    instrument = str(inst.params.get("INSTRUMENT"))
    if instrument not in _NO_BLAZE:
        _NO_BLAZE.add(instrument)
        from .logger import log
        log("no blaze file for %s (%s, looked for in calib/, models/ and"
            " models/default-blaze/): the"
            " blaze is taken as 1. That is fine for PCA2D: its high pass takes"
            " the blaze's shape out of every spectrum, and only the weights do"
            " without it" % (instrument, os.path.basename(str(name))
                             if name else "none named"), "info")
    return np.ones_like(flux)


def read(path, instrument_name, data_source, config_file=None,
         model_dir=None, inst=None, to_observer=False):
    """Un spectre, dans la forme que pca2d attend, quel que soit l'instrument.

    Même dict que tfits.read_tfits: flux, wave, blaze, recon, sky, snr, meta.
    `inst` évite de réinstancier la classe LBL à chaque fichier d'une campagne.
    """
    from lbl.core import io as lbl_io                           # noqa: F401

    inst = inst or instrument(instrument_name, data_source, config_file)
    flux, header = inst.load_science_file(path)
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    science_ext = flux_extension(path, flux)
    to_observer = bool(to_observer)
    wave = np.atleast_2d(np.asarray(
        inst.get_wave_solution(path, flux, header), dtype=float))
    blaze = _blaze(inst, path, flux, header, model_dir)

    def key(name, default=None):
        """Un mot-clé par son nom LBL, que chaque classe traduit pour son
        instrument: KW_BERV est BERV sur APERO, HIERARCH ESO QC BERV sur ESO,
        HIERARCH TNG QC BERV sur HARPS-N et HIERARCH OHP DRS BERV sur SOPHIE.

        UNE ENTREE PEUT ETRE UNE LISTE. ESPRESSO donne quatre mots-clés pour
        la masse d'air, un par télescope unitaire du VLT, dont un seul est
        rempli selon l'UT qui a observé: `header[liste]` lèverait. On prend
        donc le premier qui est là (2026-10-14).
        """
        try:
            wanted = inst.params[name]
        except Exception:                                       # noqa: BLE001
            return default
        for candidate in (wanted if isinstance(wanted, (list, tuple))
                          else [wanted]):
            try:
                value = header[candidate]
            except Exception:                                   # noqa: BLE001
                continue
            if value is not None:
                return value
        return default

    def plain(name, default=None):
        """Un mot-clé par son nom FITS, sans passer par la table de LBL."""
        try:
            return header[name]
        except Exception:                                       # noqa: BLE001
            return default

    def number(*candidates):
        """Le premier des candidats qui est un nombre fini et non sentinelle.

        Un instrument qui n'a pas mesure quelque chose l'ecrit souvent par une
        valeur convenue plutot que par un vide: -1 pour un seeing que l'ESO
        n'a pas, -9999.9 pour un mat meteo en panne. La traiter comme un
        nombre reviendrait a mettre un seeing negatif dans un diagnostic, donc
        elle devient NaN, exactement comme dans tfits.read_tfits.
        """
        for value in candidates:
            try:
                out = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(out):
                continue
            if any(abs(out - s) < 1e-6 for s in SENTINELS):
                continue
            return out
        return np.nan

    airmass = key("KW_AIRMASS", 1.0)
    # LE TEMPS ET LA POSE NE VIENNENT PAS DE LA TABLE DE LBL. Mesuré le
    # 2026-10-14 sur un S2D HARPS: la classe HARPS fait pointer KW_MJDATE ET
    # KW_EXPTIME sur le même mot-clé, HIERARCH ESO QC BJD, parce que LBL n'a
    # besoin que d'un axe temporel et se moque de la durée de pose. pca2d s'en
    # sert autrement: mjdate est une date MODIFIÉE et exptime pondère. Les
    # vrais mots-clés sont là (MJD-OBS = 52859.31, EXPTIME = 60.0 s), donc on
    # les lit d'abord et la table de LBL ne sert plus que de dernier recours.
    mjd = number(plain("MJD-OBS"), key("KW_MJDATE"))
    # un en-tête qui ne donne qu'un BJD: on le ramène en MJD plutôt que de
    # laisser 2.45 millions de jours passer pour une date modifiée
    if np.isfinite(mjd) and mjd > 2.4e6:
        mjd -= 2400000.5
    # LE MEME CONTRAT QUE read_tfits, cle pour cle. cube.py lit meta["bjd"],
    # meta["berv"], meta["object"]... sans savoir d'ou vient le spectre: un
    # dict incomplet ne provoque pas une lecture degradee mais un KeyError au
    # milieu d'une campagne. Ce que l'en-tete ne porte pas vaut NaN, qui est
    # la reponse honnete, et jamais une valeur inventee.
    bjd = number(key("KW_BJD"), key("KW_MJDATE"))
    if np.isfinite(bjd) and bjd < 1.0e6:
        # un en-tete qui ne donne qu'un MJD: le BJD est a 2400000.5 pres, et
        # l'ecart barycentrique (au plus 8 minutes) n'a pas de sens ici
        bjd += 2400000.5
    # LE REPERE DANS LEQUEL ON REND LE SPECTRE.
    #
    # LA REGLE EST CELLE DE LBL, ET ELLE EST UNIFORME. Chaque classe de LBL
    # repond a une seule question par son get_berv: « que reste-t-il a
    # appliquer pour atteindre le repere de l'etoile ». Un t.fits de SPIRou
    # est dans le repere de l'observateur et la reponse est le BERV de
    # l'en-tete; un S2D de l'ESO porte une grille deja barycentrique et la
    # reponse est zero. LBL ne traite pas ces deux cas differemment ensuite:
    # il appelle get_berv une fois (science/general.py, « instrument specific
    # berv --> use instrument method ») et tout le reste est commun.
    #
    # pca2d donne a meta["berv"] EXACTEMENT ce sens-la, parce que c'est deja
    # celui qu'il lui donnait: twoframe decale le repere de l'etoile de
    # -pixel_shift(meta["berv"]). Passer par get_berv plutot que par l'en-tete
    # n'ajoute donc aucune regle, il en retire une: pca2d n'a plus a savoir
    # quel instrument livre quel repere, LBL le sait pour lui.
    #
    # CE QUE `to_observer` CHANGE, ET CE QU'IL NE CHANGE PAS. Verifie le
    # 2026-10-15 au pixel: avec ou sans, la raie stellaire tombe a la MEME
    # longueur d'onde (380.455648 nm sur la premiere pose de tau Ceti), donc
    # le repere de l'ETOILE est le meme et la physique est inchangee. Ce qui
    # change est le repere de l'OBSERVATEUR: sans l'option il est la grille
    # barycentrique, la meme pour toutes les poses, et le bloc observateur du
    # modele n'a alors rien a quoi s'accrocher; avec, il est le repere reel du
    # telescope, et les deux blocs redeviennent distinguables. C'est une
    # option de l'AJUSTEMENT, pas de la lecture.
    header_berv = number(key("KW_BERV"))
    berv = berv_of(inst, header, header_berv)
    if to_observer:
        if not np.isfinite(header_berv):
            raise ValueError(
                "%s: input.to_observer demande de defaire la correction"
                " barycentrique, mais l'en-tete ne porte pas de BERV"
                % os.path.basename(path))
        if berv not in (0.0,):
            # le fichier est deja dans le repere de l'observateur: il n'y a
            # rien a defaire, et le refaire decalerait tout
            pass
        else:
            # lambda_observateur = lambda_bary / doppler(BERV), relativiste
            wave = wave / _doppler(header_berv)
            berv = float(header_berv)
    # le debut et la fin de pose n'ont pas de nom chez LBL: on les cherche
    # sous les graphies connues, et a defaut la masse d'air qu'il donne
    air_start = number(plain("HIERARCH ESO TEL AIRM START"),
                       plain("HIERARCH TNG TEL AIRM START"),
                       plain("HIERARCH OHP TEL AIRM START"),
                       plain("AIRMASS"), airmass)
    air_end = number(plain("HIERARCH ESO TEL AIRM END"),
                     plain("HIERARCH TNG TEL AIRM END"),
                     plain("HIERARCH OHP TEL AIRM END"),
                     plain("AIRMASS"), airmass)
    meta = {
        "filename": os.path.basename(path),
        "bjd": float(bjd),
        "mjdmid": float(mjd),
        "mjdate": float(mjd),
        "berv": berv,
        "bervmax": float(number(key("KW_BERVMAX"),
                                plain("HIERARCH ESO QC BERVMAX"))),
        "exptime": float(number(plain("EXPTIME"), key("KW_EXPTIME"))),
        "airmass_start": float(air_start),
        "airmass_end": float(air_end),
        "airmass": float(airmass if airmass is not None else np.nan),
        "seeing": float(number(plain("HIERARCH ESO TEL AMBI FWHM START"),
                               plain("HIERARCH TNG METEO FWHM"))),
        "humidity": float(number(plain("HIERARCH ESO TEL AMBI RHUM"),
                                 plain("HIERARCH TNG METEO HUMIDITY"))),
        "sun_elevation": float("nan"),
        "drs_mode": str(plain("HIERARCH ESO INS MODE", "") or "").strip(),
        "dprtype": str(key("KW_DPRTYPE", "") or "").strip(),
        "object": str(key("KW_OBJNAME", "") or "").strip(),
        "instrument": str(instrument_name),
        "data_source": str(data_source),
    }
    meta["drsobjn"] = meta["object"]
    # LA TRANSMISSION N'EST JAMAIS None. order_quality et _order_sigma
    # l'indexent comme un tableau: un None les ferait planter, et la rendre
    # optionnelle partout en aval reviendrait a semer des `if` pour une
    # situation qui a une reponse honnete. Sans modele, on declare une
    # transmission de 1, c'est-a-dire « aucune absorption connue ici »: les
    # coupures qui en dependent ne retirent alors rien, au lieu de retirer
    # n'importe quoi. Le rapport d'une campagne dit laquelle des deux.
    recon = (tapas_transmission(inst, wave, model_dir, meta["airmass"])
             if model_dir else None)
    meta["telluric_model"] = "tapas" if recon is not None else "none"
    if recon is None:
        recon = np.ones_like(flux)
    meta["flux_extension"] = science_ext
    return dict(flux=flux, wave=wave, blaze=blaze, recon=recon, sky=None,
                snr=order_snr(flux), meta=meta)


#: ce qu'un en-tete ecrit quand la mesure n'a pas eu lieu, et qui doit devenir
#: NaN plutot que passer pour un nombre. Les trois premieres sont celles de
#: tfits.read_tfits (le mat meteo du CFHT); -1 est celle de l'ESO pour un
#: seeing absent, vue le 2026-10-14 sur un S2D HARPS.
SENTINELS = (-9999.0, -9999.9, -999.0, -1.0)


#: les instruments que pca2d lit avec son propre lecteur, parce que les t.fits
#: d'APERO portent des choses que LBL ne rend pas (le ciel dans OHLine, la
#: transmission dans Recon, le SNR par ordre dans l'en-tete) et que ce lecteur
#: les exploite. Tout le reste passe par LBL.
NATIVE = ("SPIROU", "NIRPS", "NIRPS_HA", "NIRPS_HE")


def reads_natively(config):
    """Faut-il lire ces spectres avec tfits, ou passer par LBL ?

    APERO est lu nativement (le t.fits porte le ciel et la transmission);
    HARPS, SOPHIE, ESPRESSO et les autres passent par LBL. `input.reader`
    tranche quand on veut forcer l'un ou l'autre.
    """
    asked = str((config.get("input") or {}).get("reader") or "auto").lower()
    if asked in ("tfits", "native"):
        return True
    if asked in ("lbl", "bridge"):
        return False
    instrument = str((config.get("input") or {}).get("instrument") or "").upper()
    source = str((config.get("lbl") or {}).get("data_source") or "").upper()
    # un t.fits d'APERO: notre lecteur en tire plus que LBL n'en donne
    return instrument in NATIVE and source in ("", "APERO", "CADC")


def star_frame(payload, config, path=None):
    """The star's frame of one exposure: its BERV composed with the star's
    velocity at that exposure, from a free fit of the velocities when there
    is one (registration.velocities, by file name: LBL's template), else the
    Keplerian velocity of the star's companions at its BJD
    (registration.orbits).

    `berv` is what every shift into the star's frame reads (the cube's rows,
    the template, the fit, the correction), so it becomes the velocity of that
    frame, (berv - vstar) / (1 - berv vstar / c2): the star at +vstar is
    where a star at rest would be under that BERV. The BERV itself is kept as
    `berv_bary`, for what is about the barycentre (the diagnostics of a bias
    with BERV), and the companions' velocity as `vstar` [km/s]. Without
    orbits nothing changes. Applied once: a payload that has berv_bary is
    left as it is.
    """
    registration = (config or {}).get("registration") or {}
    orbits = registration.get("orbits") or []
    velocities = registration.get("velocities") or {}
    meta = payload.get("meta")
    if (not orbits and not velocities) or meta is None or "berv_bary" in meta:
        return payload
    from .grids import C_KMS
    berv = float(meta.get("berv", np.nan))
    name = os.path.basename(str(path or meta.get("filename") or ""))
    if name in velocities:
        vstar = float(velocities[name])
    elif orbits:
        from lbl.science import planets
        bjd = float(meta.get("bjd", np.nan))
        vstar = float(planets.star_velocity([bjd], orbits)[0]) / 1000.0
    else:
        return payload
    meta["berv_bary"] = berv
    meta["vstar"] = vstar
    meta["berv"] = (berv - vstar) / (1.0 - berv * vstar / C_KMS ** 2)
    return payload


def read_any(path, config, inst=None):
    """Un spectre, par le lecteur que la configuration designe, dans le
    repere de l'etoile que la configuration designe (star_frame).

    C'est le seul point ou pca2d decide comment lire: cube, reconstruct et
    plotting passent par ici, donc ajouter un instrument ne touche qu'a
    `reads_natively`.
    """
    return star_frame(_read_any(path, config, inst), config, path)


def _read_any(path, config, inst=None):
    """read_any, without the star's frame"""
    if reads_natively(config):
        from . import tfits as sptf
        return sptf.read_tfits(path)
    block = config.get("lbl") or {}
    name = block.get("instrument") or (config.get("input") or {}).get("instrument")
    source = block.get("data_source") or "ESO"
    models = (config.get("output") or {}).get("model_directory")
    observer = bool((config.get("input") or {}).get("to_observer"))
    return read(path, name, source, model_dir=models, inst=inst,
                to_observer=observer)
