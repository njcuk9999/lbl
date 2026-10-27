"""The testers' guide to LBL's lbl-pca2d branch and its window, as one
bilingual page (English and French) in eartigau.github.io/lbl-gui/.

The reference parts are read from the code, so that the page says what the
window does: the help of every control and every PCA2D option, in both
languages (lbl.gui.i18n), and the defaults of the options for each
instrument (lbl.gui.lblinfo). The rest is written here.

    python tools/make_guide.py <eartigau.github.io folder>

The page is not linked from the site (not released yet): it carries a
warning and asks search engines not to index it.
"""
import datetime
import html
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lbl.gui import __version__, i18n, lblinfo        # noqa: E402
import guide_tutorial                                     # noqa: E402

THEMES = os.path.expanduser("~/GitHubProjects/themes_outils")
REPO = "https://github.com/njcuk9999/lbl/tree/lbl-pca2d/lbl/gui"
ISSUES = "https://github.com/njcuk9999/lbl/issues"
LBL_REPO = "https://github.com/njcuk9999/lbl"
#: the instruments whose PCA2D defaults are shown, and their names
DEFAULT_FOR = (("SPIROU", "SPIRou"), ("NIRPS_HE", "NIRPS"), ("HARPS", "HARPS"))


def esc(text):
    return html.escape(str(text), quote=False)


def para(text):
    """A help text of the window, its blank lines as paragraphs."""
    parts = [p.strip() for p in str(text).split("\n\n") if p.strip()]
    return "".join("<p>%s</p>" % esc(p).replace("\n", "<br>") for p in parts)


def both(en, fr, tag="div"):
    """The same content in both languages; the page shows one."""
    return '<%s lang="en">%s</%s><%s lang="fr">%s</%s>' % (tag, en, tag, tag,
                                                         fr, tag)


def inline(en, fr):
    return both(en, fr, "span")


def code(text):
    return '<pre class="codeblock">%s</pre>' % esc(text.strip("\n"))


def card(anchor, title_en, title_fr, body, span=3):
    return ('<section class="card span-%d" id="%s"><h2>%s</h2>%s</section>'
            % (span, anchor, inline(esc(title_en), esc(title_fr)), body))


def help_of(key):
    """A help of the window in both languages."""
    return both(para(i18n.EN[key]), para(i18n.FR[key]))


# ------------------------------------------------------------------ reference
def options_reference():
    """Every PCA2D option the window lays out: its label, key, defaults and
    help, by group, in both languages."""
    rows = {key: [] for key, _ in DEFAULT_FOR}
    for inst, _ in DEFAULT_FOR:
        rows[inst] = {r["key"]: r for r in lblinfo.pca2d_options(inst)}
    first = rows[DEFAULT_FOR[0][0]]
    order = ("model", "bases", "correction", "runs", "more")
    out = []
    for group in order:
        keys = [k for k, r in first.items() if r["group"] == group]
        if not keys:
            continue
        out.append("<h3>%s</h3>" % inline(esc(i18n.EN["group_" + group]),
                                          esc(i18n.FR["group_" + group])))
        for key in keys:
            text = i18n.option_text(key, "en", first[key].get("description"))
            text_fr = i18n.option_text(key, "fr", first[key].get("description"))
            values = [(name, rows[inst][key]["default"])
                      for inst, name in DEFAULT_FOR if key in rows[inst]]
            shown = {repr(v) for _, v in values}
            if len(shown) == 1:
                default = _value(values[0][1])
            else:
                default = ", ".join("%s %s" % (esc(name), _value(v))
                                    for name, v in values)
            help_en = para(text["help"] or text["lbl"])
            help_fr = para(text_fr["help"] or text_fr["lbl"])
            extra = ""
            if key == "joint":
                extra = both("<p><em>In the window: not offered for one"
                             " target, on by default for several.</em></p>",
                             "<p><em>Dans la fenêtre : pas proposée pour une"
                             " seule cible, activée par défaut pour plusieurs."
                             "</em></p>")
            out.append(
                '<div class="option"><div class="option-head">'
                '<span class="option-label">%s</span>'
                '<code>%s</code><span class="option-default">%s %s</span>'
                '</div>%s%s</div>'
                % (inline(esc(text["label"]), esc(text_fr["label"])),
                   esc(key), inline("default", "défaut"), default,
                   both(help_en, help_fr), extra))
    return "".join(out)


def _value(value):
    """A default as the page shows it (HTML, both languages for unset)."""
    if value is None:
        return inline("unset", "non réglé")
    return esc(repr(value) if isinstance(value, str) else str(value))


def instruments_table():
    rows = lblinfo.instruments()
    body = "".join("<tr><td><code>%s</code></td><td>%s</td></tr>"
                   % (esc(r["instrument"]),
                      " &middot; ".join("<code>%s</code>" % esc(s["source"])
                                        for s in r["sources"]))
                   for r in rows)
    return ('<div class="table-wrap"><table class="data"><thead>'
            '<tr><th>INSTRUMENT</th><th>DATA_SOURCE</th></tr></thead>'
            '<tbody>%s</tbody></table></div>' % body)


def switches_reference():
    out = []
    for row in lblinfo.lbl_switches():
        key = row["key"]
        out.append('<div class="option"><div class="option-head">'
                   '<span class="option-label">%s</span><code>%s</code>'
                   '<span class="option-default">%s %s</span></div>%s</div>'
                   % (inline(esc(i18n.EN["sw_" + key]),
                             esc(i18n.FR["sw_" + key])),
                      esc(key), inline("default", "défaut"),
                      esc("instrument" if row["default"] is None
                          else row["default"]),
                      help_of("help_sw_" + key)))
    return "".join(out)


# ---------------------------------------------------------------------- page
def page(results=None):
    today = datetime.date.today().isoformat()
    sections = []

    sections.append(card("short", "In short", "En bref", both(
        """
<p><strong>LBL</strong> (line-by-line, Artigau et al. 2022) measures radial
velocities line by line against a template of the star, and combines
thousands of lines into one velocity per exposure, resilient to outliers.
The <code>lbl-pca2d</code> branch adds <strong>PCA2D</strong>, a
precleaning step: it separates what moves with the star from what stays
fixed on the detector (telluric residuals, airglow, detector features),
divides the latter out of the spectra, and LBL then measures the corrected
spectra. The report shows the velocities before and after.</p>
<p><strong>lbl-gui</strong> is a window in your browser that writes an
ordinary LBL script (a <em>wrap</em>) from what you tick, runs it in the
environment LBL is installed in, and follows it: the steps, the log, the
reports. Nothing it does is hidden: every run keeps its wrap, which runs by
hand as it is.</p>
<p>One piece, one repository, one conda environment: <strong>LBL</strong>,
branch <code>lbl-pca2d</code> (github.com/njcuk9999/lbl), holds the
measurement, the precleaning and the window (<code>lbl/gui</code>, started
with <code>lbl_gui</code>), with its search of the public archives.</p>
""", """
<p><strong>LBL</strong> (line-by-line, Artigau et al. 2022) mesure les
vitesses radiales raie par raie contre un template de l'étoile, et combine
des milliers de raies en une vitesse par pose, de façon robuste aux valeurs
aberrantes. La branche <code>lbl-pca2d</code> y ajoute <strong>PCA2D</strong>,
un prénettoyage : il sépare ce qui bouge avec l'étoile de ce qui reste fixe
sur le détecteur (résidus telluriques, émission du ciel, défauts du
détecteur), divise ces derniers hors des spectres, puis LBL mesure les
spectres corrigés. Le rapport montre les vitesses avant et après.</p>
<p><strong>lbl-gui</strong> est une fenêtre dans votre navigateur qui
écrit un script LBL ordinaire (un <em>wrap</em>) à partir de ce que vous
cochez, le lance dans l'environnement où LBL est installé, et le suit : les
étapes, le journal, les rapports. Rien n'est caché : chaque run garde son
wrap, qui se relance tel quel à la main.</p>
<p>Un seul morceau, un seul dépôt, un seul environnement conda :
<strong>LBL</strong>, branche <code>lbl-pca2d</code>
(github.com/njcuk9999/lbl), contient la mesure, le prénettoyage et la
fenêtre (<code>lbl/gui</code>, lancée avec <code>lbl_gui</code>), avec sa
recherche dans les archives publiques.</p>
""") + '<figure class="lecture-figure">%s' % inline(
            '<img src="workflow_en.svg" alt="what a run does">',
            '<img src="workflow_fr.svg" alt="ce que fait un run">')
        + '<figcaption>%s</figcaption></figure>' % inline(
            "What a run does, for each target: LBL on the delivered spectra,"
            " PCA2D's four stages, LBL again on the corrected spectra, then"
            " one report.",
            "Ce que fait un run, pour chaque cible : LBL sur les spectres"
            " livrés, les quatre étapes de PCA2D, LBL de nouveau sur les"
            " spectres corrigés, puis un seul rapport.")))

    sections.append(card("before", "Before you start", "Avant de commencer",
                         both("""
<ul class="feature-list">
<li><strong>conda</strong> and <strong>git</strong>.</li>
<li><strong>Python 3.12</strong>: LBL accepts no other version.</li>
<li><strong>pdflatex</strong>, optional: with it the reports are typeset
documents; without it, the figures are bound into one PDF.</li>
<li><strong>Disk and memory.</strong> LBL writes its products beside the
spectra; PCA2D builds a cube of all the spectra of a target, which for
hundreds of exposures is several GB, and needs memory to fit it. On a
laptop, nightly stacking (the default for NIRPS) keeps it small.</li>
<li>macOS or Linux. The window is a web page served by your own machine:
any recent browser.</li>
</ul>""", """
<ul class="feature-list">
<li><strong>conda</strong> et <strong>git</strong>.</li>
<li><strong>Python 3.12</strong> : LBL n'accepte pas d'autre version.</li>
<li><strong>pdflatex</strong>, facultatif : avec lui, les rapports sont des
documents mis en page ; sans lui, les figures sont assemblées en un PDF.</li>
<li><strong>Disque et mémoire.</strong> LBL écrit ses produits à côté des
spectres ; PCA2D construit un cube de tous les spectres d'une cible, qui
pour des centaines de poses pèse plusieurs Go, et demande de la mémoire pour
l'ajuster. Sur un portable, l'empilement par nuit (le défaut pour NIRPS) le
garde petit.</li>
<li>macOS ou Linux. La fenêtre est une page web servie par votre propre
machine : n'importe quel navigateur récent.</li>
</ul>""")))

    install_en = """
<p><strong>1. An environment for LBL.</strong> Everything goes in one conda
environment: the window runs the wraps with its Python.</p>
""" + code("""
conda create -n lbl-pca2d python=3.12
conda activate lbl-pca2d
""") + """
<p><strong>2. LBL, branch <code>lbl-pca2d</code>.</strong> Installed in
editable mode (<code>-e</code>), so that a <code>git pull</code> is all an
update takes; <code>editable_mode=compat</code> puts the clone itself on
Python's path, so that no other folder named <code>lbl</code> can hide
it.</p>
""" + code("""
git clone -b lbl-pca2d https://github.com/njcuk9999/lbl.git lbl-pca2d
pip install -e lbl-pca2d --config-settings editable_mode=compat
""") + """
<p>Already have a clone of LBL? Switch it to the branch instead:</p>
""" + code("""
cd lbl
git fetch origin
git checkout lbl-pca2d
git pull
pip install -e . --config-settings editable_mode=compat
""") + """
<p>Check that Python finds this copy (the path printed is in your
clone):</p>
""" + code("""
python -c "import lbl; print(lbl.__file__)"
""") + """
<p><strong>3. The window comes with it</strong>, and so does its search of
the public archives (CADC, ESO, TNG, OHP): nothing else to install. Check
that the command is there:</p>
""" + code("""
lbl_gui --help
""") + """
<p><strong>Installed the first test version?</strong> It was a package of its
own (lbl-pca2d-gui) with a command <code>lbl-gui</code>, and koloa for the
archives. Remove it, so that no older window answers instead:</p>
""" + code("""
pip uninstall lbl-pca2d-gui
""")
    install_fr = """
<p><strong>1. Un environnement pour LBL.</strong> Tout va dans un seul
environnement conda : la fenêtre lance les wraps avec son Python.</p>
""" + code("""
conda create -n lbl-pca2d python=3.12
conda activate lbl-pca2d
""") + """
<p><strong>2. LBL, branche <code>lbl-pca2d</code>.</strong> Installé en mode
éditable (<code>-e</code>), pour qu'un <code>git pull</code> suffise à le
mettre à jour ; <code>editable_mode=compat</code> met le clone lui-même dans
le chemin de Python, pour qu'aucun autre dossier nommé <code>lbl</code> ne
puisse le cacher.</p>
""" + code("""
git clone -b lbl-pca2d https://github.com/njcuk9999/lbl.git lbl-pca2d
pip install -e lbl-pca2d --config-settings editable_mode=compat
""") + """
<p>Vous avez déjà un clone de LBL ? Passez-le plutôt sur la branche :</p>
""" + code("""
cd lbl
git fetch origin
git checkout lbl-pca2d
git pull
pip install -e . --config-settings editable_mode=compat
""") + """
<p>Vérifiez que Python trouve bien cette copie (le chemin affiché est dans
votre clone) :</p>
""" + code("""
python -c "import lbl; print(lbl.__file__)"
""") + """
<p><strong>3. La fenêtre vient avec</strong>, ainsi que sa recherche dans
les archives publiques (CADC, ESO, TNG, OHP) : rien d'autre à installer.
Vérifiez que la commande est là :</p>
""" + code("""
lbl_gui --help
""") + """
<p><strong>Vous aviez installé la première version de test ?</strong> C'était
un paquet à part (lbl-pca2d-gui) avec une commande <code>lbl-gui</code>, et
koloa pour les archives. Retirez-le, pour qu'aucune ancienne fenêtre ne
réponde à la place :</p>
""" + code("""
pip uninstall lbl-pca2d-gui
""")
    sections.append(card("install", "Installing", "Installation",
                         both(install_en, install_fr)))

    # THE TUTORIAL: one example from end to end, screenshot by screenshot
    sections.append(card("tutorial", "Step by step: a first run",
                         "Pas à pas : un premier run",
                         guide_tutorial.tutorial(code, results or {})))
    sections.append('<section class="card span-3 part" id="reference">'
                    '<h2>%s</h2><p class="hint">%s</p></section>' % (
                        inline("Reference", "Référence"),
                        inline("Every part of the window, every option, every"
                               " file a run writes.",
                               "Chaque partie de la fenêtre, chaque option,"
                               " chaque fichier qu'un run écrit.")))

    data_en = """
<p>LBL works in one folder, <code>DATA_DIR</code>. Each target is a
subfolder of <code>science/</code> that holds its spectra:</p>
""" + code("""
DATA_DIR/
  science/
    TOI-1234/        the spectra of one target
    GJ_1028/
  calib/             blaze files, for the instruments that need them
""") + """
<p>LBL creates everything else beside them on its first run
(<code>templates/</code>, <code>masks/</code>, <code>lblrv/</code>,
<code>lblrdb/</code>, <code>models/</code>, <code>reports/</code>...) and
downloads its models (the TAPAS atmosphere, the line masks). The window adds
<code>gui/runs/</code> and its logs, <code>gui/logs/</code>, and PCA2D
<code>pca2d/</code>.</p>
<p><strong>The models.</strong> <code>models/</code> holds what LBL
downloads and what is to be kept, one subfolder per kind:</p>
""" + code("""
DATA_DIR/models/
  phoenix/                    PHOENIX stellar models (for the masks)
  tellurics/                  the TAPAS atmosphere, tapas_lbl.fits
  resproj/                    RESPROJ tables: temperature gradients (DTEMP),
                              PCA2D star components (STRPCA)
  masks/                      default masks
  default-blaze/              a blaze per setup that needs one, used when a
                              spectrum's own is not in calib/
                              (blaze_NIRPS_HE_ESO.fits)
  template_SPIROU_APERO/      the vetted templates of an instrument and data
                              source: the depot's file and LBL's template
  pca2d_SPIROU_APERO/         the PCA2D observer bases you keep
""") + """
<p>Nothing removes <code>models/</code>: neither <code>lbl_reset</code>,
which empties <code>templates/</code>, nor the window's cleanup. A vetted
template is therefore kept there rather than with the templates LBL makes
from spectra. A <code>models/</code> from before (every file at its top) is
sorted by LBL itself at its next run.</p>
<p><strong>Which files.</strong> What the instrument's pipeline delivers,
for example APERO's or the CADC's <code>t.fits</code> for SPIRou and NIRPS,
ESO's telluric-corrected S2D for NIRPS, the DRS <code>e2ds</code> for HARPS
(with the blaze of the night in <code>calib/</code>). One folder holds one
instrument and one pipeline: the window reads which from the files
themselves, sets the instrument and the data source to match, and refuses a
run whose source does not fit the files (the CADC's <code>t.fits</code> read
as APERO's stop LBL on its first file).</p>
<p>The instruments LBL reads, and their data sources:</p>
""" + instruments_table()
    data_fr = """
<p>LBL travaille dans un dossier, <code>DATA_DIR</code>. Chaque cible est un
sous-dossier de <code>science/</code> qui contient ses spectres :</p>
""" + code("""
DATA_DIR/
  science/
    TOI-1234/        les spectres d'une cible
    GJ_1028/
  calib/             les fichiers de blaze, pour les instruments qui en ont besoin
""") + """
<p>LBL crée tout le reste à côté à son premier run (<code>templates/</code>,
<code>masks/</code>, <code>lblrv/</code>, <code>lblrdb/</code>,
<code>models/</code>, <code>reports/</code>...) et télécharge ses modèles
(l'atmosphère TAPAS, les masques de raies). La fenêtre ajoute
<code>gui/runs/</code> et ses journaux, <code>gui/logs/</code>, et PCA2D
<code>pca2d/</code>.</p>
<p><strong>Les modèles.</strong> <code>models/</code> contient ce que LBL
télécharge et ce qui doit être gardé, un sous-dossier par sorte :</p>
""" + code("""
DATA_DIR/models/
  phoenix/                    les modèles stellaires PHOENIX (pour les masques)
  tellurics/                  l'atmosphère TAPAS, tapas_lbl.fits
  resproj/                    les tables RESPROJ : gradients de température
                              (DTEMP), composantes stellaires PCA2D (STRPCA)
  masks/                      les masques par défaut
  default-blaze/              un blaze par configuration qui en demande un,
                              pris quand celui d'un spectre n'est pas dans
                              calib/ (blaze_NIRPS_HE_ESO.fits)
  template_SPIROU_APERO/      les templates validés d'un instrument et d'une
                              source : le fichier du dépôt et le template LBL
  pca2d_SPIROU_APERO/         les bases de l'observateur PCA2D que vous gardez
""") + """
<p>Rien n'efface <code>models/</code> : ni <code>lbl_reset</code>, qui vide
<code>templates/</code>, ni le ménage de la fenêtre. Un template validé y
est donc gardé plutôt qu'avec les templates que LBL fait à partir des
spectres. Un <code>models/</code> d'avant (tous les fichiers à sa racine)
est rangé par LBL lui-même à son run suivant.</p>
<p><strong>Quels fichiers.</strong> Ce que livre le pipeline de
l'instrument, par exemple les <code>t.fits</code> d'APERO ou de la CADC pour
SPIRou et NIRPS, les S2D corrigés des telluriques d'ESO pour NIRPS, les
<code>e2ds</code> du DRS pour HARPS (avec le blaze de la nuit dans
<code>calib/</code>). Un dossier contient un seul instrument et un seul
pipeline : la fenêtre les reconnaît d'après les fichiers eux-mêmes, règle
l'instrument et la source de données en conséquence, et refuse un run dont
la source ne correspond pas aux fichiers (les <code>t.fits</code> de la CADC
lus comme ceux d'APERO arrêtent LBL dès le premier fichier).</p>
<p>Les instruments que LBL sait lire, et leurs sources de données :</p>
""" + instruments_table()
    sections.append(card("data", "The data folder", "Le dossier de données",
                         both(data_en, data_fr)))

    launch_en = code("""
conda activate lbl-pca2d
lbl_gui --data-dir /path/to/DATA_DIR
""") + """
<p>The window opens in your browser at <code>http://127.0.0.1:8765</code>;
if that port is taken it takes the next free one and prints the address.</p>
<div class="table-wrap"><table class="data"><tbody>
<tr><td><code>--data-dir</code></td><td>the data folder; it can also be
changed in the window</td></tr>
<tr><td><code>--port 8800</code></td><td>another port</td></tr>
<tr><td><code>--no-browser</code></td><td>do not open the browser; open the
printed address yourself</td></tr>
</tbody></table></div>
<p><strong>On a remote machine</strong>, start it with
<code>--no-browser</code> and open a tunnel from your laptop, then go to
<code>http://127.0.0.1:8765</code> there:</p>
""" + code("""
ssh -L 8765:127.0.0.1:8765 you@machine
""") + """
<p>The window listens on the machine itself only (<code>127.0.0.1</code>)
and serves files from the data folders it is given only.
<strong>Ctrl+C</strong> in the terminal stops the window, not the runs: a
run in progress goes on, and the window finds it again when it is opened on
the same folder. What it remembers between sessions (the folder, the ticked
targets, the options) is in <code>~/.lbl_pca2d_gui.json</code>, what it
read of the headers in <code>~/.lbl_pca2d_gui/</code>, what the archive
search keeps in <code>~/.cache/lbl/gui/</code>, and its logs in the data
folder, <code>gui/logs/</code>: each run's and the archive search's
(<code>archives.log</code>).</p>
"""
    launch_fr = code("""
conda activate lbl-pca2d
lbl_gui --data-dir /chemin/vers/DATA_DIR
""") + """
<p>La fenêtre s'ouvre dans votre navigateur à l'adresse
<code>http://127.0.0.1:8765</code> ; si ce port est pris, elle prend le
suivant de libre et affiche l'adresse.</p>
<div class="table-wrap"><table class="data"><tbody>
<tr><td><code>--data-dir</code></td><td>le dossier de données ; on peut
aussi le changer dans la fenêtre</td></tr>
<tr><td><code>--port 8800</code></td><td>un autre port</td></tr>
<tr><td><code>--no-browser</code></td><td>n'ouvre pas le navigateur ;
ouvrez vous-même l'adresse affichée</td></tr>
</tbody></table></div>
<p><strong>Sur une machine distante</strong>, lancez-la avec
<code>--no-browser</code>, ouvrez un tunnel depuis votre portable, puis
allez à <code>http://127.0.0.1:8765</code> chez vous :</p>
""" + code("""
ssh -L 8765:127.0.0.1:8765 vous@machine
""") + """
<p>La fenêtre n'écoute que sur la machine elle-même
(<code>127.0.0.1</code>) et ne sert que les fichiers des dossiers de données
qu'on lui donne. <strong>Ctrl+C</strong> dans le terminal arrête la fenêtre,
pas les runs : un run en cours continue, et la fenêtre le retrouve quand on
la rouvre sur le même dossier. Ce qu'elle retient d'une session à l'autre
(le dossier, les cibles cochées, les options) est dans
<code>~/.lbl_pca2d_gui.json</code>, ce qu'elle a lu des en-têtes dans
<code>~/.lbl_pca2d_gui/</code>, ce que garde la recherche d'archive dans
<code>~/.cache/lbl/gui/</code>, et ses journaux dans le dossier de données,
<code>gui/logs/</code> : celui de chaque run et celui de la recherche
d'archive (<code>archives.log</code>).</p>
"""
    sections.append(card("launch", "Starting the window",
                         "Lancer la fenêtre", both(launch_en, launch_fr)))

    # the tabs, one by one: what the window shows, then its own help
    targets_en = """
<p><strong>My targets.</strong> Every subfolder of <code>science/</code>, read
from the headers only (one file per folder first, then the rest), so that a
folder of a thousand spectra is listed in seconds. Per target: the
instrument, the number of spectra and nights, the median SNR, the exposure
time, the BERV and its span, the dates and the Teff. Tick the targets to
run: one instrument at a time (the window refuses two), and several for a
joint PCA2D fit. The instrument filter hides the others; <em>Refresh</em>
reads the folder again.</p>
<p>The banner above describes <strong>the ticked targets only</strong>: their
spectra and dates, their nights and median SNR, their BERV coverage. The
plots under the table do the same: the BERV coverage (the more the BERV
moves, the better the star and the atmosphere separate), the star's
velocity in the observer's frame v_tot = v_sys - BERV (where the star's
lines fall against the telluric lines), the SNR against the BERV, and the
dates, where two bounds can restrict the PCA2D fit to a stretch of
time.</p>
"""
    targets_fr = """
<p><strong>Mes cibles.</strong> Chaque sous-dossier de <code>science/</code>,
lu dans les en-têtes seulement (un fichier par dossier d'abord, puis le
reste), pour qu'un dossier de mille spectres s'affiche en quelques secondes.
Par cible : l'instrument, le nombre de spectres et de nuits, le SNR médian,
le temps de pose, le BERV et son étendue, les dates et la Teff. Cochez les
cibles à traiter : un seul instrument à la fois (la fenêtre refuse d'en
mélanger deux), et plusieurs pour un fit PCA2D joint. Le filtre
d'instrument cache les autres ; <em>Rafraîchir</em> relit le dossier.</p>
<p>Le bandeau du haut décrit <strong>les cibles cochées seulement</strong> :
leurs spectres et leurs dates, leurs nuits et leur SNR médian, leur
couverture en BERV. Les graphiques sous la table font de même : la
couverture en BERV (plus le BERV bouge, mieux l'étoile et l'atmosphère se
séparent), la vitesse de l'étoile dans le référentiel de l'observateur
v_tot = v_sys - BERV (là où les raies de l'étoile tombent par rapport aux
raies telluriques), le SNR en fonction du BERV, et les dates, où deux
bornes peuvent limiter le fit PCA2D à une période.</p>
"""
    archive_en = """
<h3>Archive search</h3>
<p>The second sub-tab finds a star's public spectra
that LBL reads and fetches them into <code>science/</code>, four files at a
time, each folder with how far it is and its time left. It never moves or overwrites a file that is there; a folder fetched
from an archive remembers where it came from
(<code>.lbl_archive.json</code>), and the window sets the instrument, the
data source and the file pattern from it.</p>
"""
    archive_fr = """
<h3>Recherche d'archive</h3>
<p>Le second sous-onglet trouve les spectres publics
d'une étoile que LBL sait lire et les récupère dans <code>science/</code>,
quatre fichiers à la fois, chaque dossier avec où il en est et le temps qui
lui reste. Il ne déplace ni n'écrase jamais un fichier déjà là ;
un dossier récupéré d'une archive garde sa provenance
(<code>.lbl_archive.json</code>), d'où la fenêtre tire l'instrument, la
source de données et le motif de fichiers.</p>
"""
    sections.append(card(
        "tab-targets", "Tab: Targets", "Onglet : Cibles",
        both(targets_en, targets_fr) + help_of("help_data_dir")
        + help_of("help_instrument") + help_of("help_source")
        + help_of("help_vtot") + help_of("help_bounds")
        + both(archive_en, archive_fr) + help_of("help_archive")))

    lbl_en = """
<p><strong>The template of each target.</strong> Each ticked target is
measured against a template: its own by default, or another star's. For
targets between 2900 and 4300 K, <em>Match a template</em> picks LBL's
vetted template nearest in temperature (one per 100 K from 3000 to 4200 K,
for SPIRou and NIRPS). Every template offered says its temperature.</p>
<p><strong>Rewrite the files already on disk.</strong> One switch for every
LBL step: ticked, the template, the mask, the per-line velocities and the
rdb tables are all made again; unticked, what is on disk is kept and only
what is missing is made. Tick it when the spectra, the template or the mask
changed. The boxes below it set each step on its own.</p>
"""
    lbl_fr = """
<p><strong>Le template de chaque cible.</strong> Chaque cible cochée est
mesurée contre un template : le sien par défaut, ou celui d'une autre
étoile. Pour les cibles entre 2900 et 4300 K, <em>Apparier un template</em>
choisit le template validé de LBL le plus proche en température (un par
tranche de 100 K de 3000 à 4200 K, pour SPIRou et NIRPS). Chaque template
proposé indique sa température.</p>
<p><strong>Réécrire les fichiers déjà sur le disque.</strong> Un seul
interrupteur pour toutes les étapes LBL : cochée, le template, le masque,
les vitesses par raie et les tables rdb sont tous refaits ; décochée, ce qui
est sur le disque est gardé et seul ce qui manque est fait. À cocher quand
les spectres, le template ou le masque ont changé. Les cases dessous
règlent chaque étape séparément.</p>
"""
    sections.append(card(
        "tab-lbl", "Tab: LBL settings", "Onglet : Réglages LBL",
        both(lbl_en, lbl_fr) + help_of("help_comparison")
        + help_of("help_vetted") + help_of("help_teff")
        + "<h3>%s</h3>" % inline("The steps, one by one",
                                 "Les étapes, une par une")
        + help_of("help_overwrite_all") + switches_reference()
        + "<h3>%s</h3>" % inline("Temperature indicator",
                                 "Indicateur de température")
        + help_of("help_resproj")))

    tellu_en = """
<p>HARPS, HARPS-N, ESPRESSO, SOPHIE and the other optical spectrographs
deliver their spectra with the Earth's atmosphere still in them; NIRPS's and
SPIRou's come corrected by APERO. For the optical ones LBL cleans the
tellurics itself (<em>telluric cleaning</em>, on by default, in the LBL
settings), in <strong>two passes</strong>, and the run's list of steps shows
the four steps they take.</p>
<h3>What one pass does</h3>
<p>For each spectrum, a TAPAS model of the atmosphere's transmission, with
two exponents: one for water, one for the dry absorbers (O<sub>2</sub> and
the others). The exponents start from the airmass (dry) and 0.5 (water) and
are adjusted until the corrected spectrum no longer correlates with the
telluric lines (a cross-correlation against a mask of water lines and one of
dry lines, up to 20 iterations, to 0.001 in the exponents); for HARPS,
HARPS-N and ESPRESSO the water exponent is then fitted again by the
derivative method (the exponent that leaves the corrected spectrum
smoothest). The spectrum as delivered is divided by that transmission and
written to <code>science/&lt;object&gt;_tc/</code>; where the transmission
is below e<sup>-1</sup> (lines deeper than 63&nbsp;%), the pixels are set to
NaN rather than corrected.</p>
<h3>Why twice</h3>
<ol>
<li><strong>First pass, blind.</strong> The exponents are fitted on the
spectrum as it is, the star's own lines included. Where a stellar line falls
on a telluric line of the mask, it counts as atmosphere and biases the
exponents; and since the star moves against the atmosphere with the BERV
over the year, that bias changes from night to night.</li>
<li><strong>A template of the cleaned spectra</strong>
(<code>&lt;object&gt;_tc</code>). Put in the star's frame and combined,
the tellurics left by the first pass land at a different place on every
night and average out, as long as the nights span enough BERV: what
remains is the star's spectrum.</li>
<li><strong>Second pass, divided by that template.</strong> The spectra as
delivered (not those of the first pass) are fitted again, each divided by
the template shifted to its night's BERV: the star's lines are gone, and the
exponents are measured on the atmosphere alone. The template only enters the
fit: the file written is still the delivered spectrum divided by the
transmission, and it replaces the first pass's.</li>
<li><strong>The template made again</strong>, from the second pass's
spectra: the one the mask, the velocities and PCA2D are measured
against.</li>
</ol>
<p>LBL's own steps then run on <code>&lt;object&gt;_tc</code>: the mask, the
velocities, the rdb table and the report; with PCA2D, the precleaning works
on these cleaned spectra too. Each pass takes about a second per spectrum
(90 HARPS spectra of GJ 1214: 1 min 35 s).</p>
<h3>Checking it in the log</h3>
<p>Each spectrum writes one line with the exponents applied, the ones
written in its header (<code>TLPEH2O</code>, <code>TLPEOTR</code>), such as
<code>telluric exponents by the derivative method: H2O 0.1538, dry 1.0410
(airmass 1.0809), template=True</code>; <code>FALLBACK</code> (0.35) when
the derivative's cost had nothing to go on. The derivative method is the
only one, for every instrument, since 2026-10-27: the cross-correlation
loop that came before it often failed. The first pass says
<code>template=False</code>, the
second <code>template=True</code> (its banner says
<code>TELLUCLEAN_USE_TEMPLATE="True"</code>), with exponents that differ a
little from the first pass's. A second pass that says
<code>template=False</code> did not find the template and cleaned blind
again; LBL now says so as a warning. That was the case of every version of
LBL from 2026-02-24 to 2026-10-25, and until then the template was not made
again after the second pass either (the first pass's stayed on disk): a run
of an optical spectrograph made in that time is worth running again, with
<em>rewrite the files already on disk</em> ticked.</p>
"""
    tellu_fr = """
<p>HARPS, HARPS-N, ESPRESSO, SOPHIE et les autres spectrographes optiques
livrent leurs spectres avec l'atmosphère terrestre encore dedans ; ceux de
NIRPS et de SPIRou arrivent corrigés par APERO. Pour les optiques, LBL
nettoie lui-même les telluriques (<em>nettoyage tellurique</em>, actif par
défaut, dans les réglages LBL), en <strong>deux passages</strong>, et la
liste des étapes du run montre les quatre étapes qu'ils prennent.</p>
<h3>Ce que fait un passage</h3>
<p>Pour chaque spectre, un modèle TAPAS de la transmission de l'atmosphère,
avec deux exposants : un pour l'eau, un pour les absorbants secs
(O<sub>2</sub> et les autres). Les exposants partent de la masse d'air
(sec) et de 0,5 (eau), et sont ajustés jusqu'à ce que le spectre corrigé ne
corrèle plus avec les raies telluriques (une corrélation croisée avec un
masque de raies d'eau et un de raies sèches, au plus 20 itérations, à
0,001 près sur les exposants) ; pour HARPS, HARPS-N et ESPRESSO,
l'exposant de l'eau est ensuite réajusté par la méthode de la dérivée
(l'exposant qui laisse le spectre corrigé le plus lisse). Le spectre livré
est divisé par cette transmission et écrit dans
<code>science/&lt;objet&gt;_tc/</code> ; là où la transmission est sous
e<sup>-1</sup> (raies plus profondes que 63&nbsp;%), les pixels sont mis à
NaN plutôt que corrigés.</p>
<h3>Pourquoi deux fois</h3>
<ol>
<li><strong>Premier passage, à l'aveugle.</strong> Les exposants sont
ajustés sur le spectre tel quel, raies de l'étoile comprises. Là où une raie
stellaire tombe sur une raie tellurique du masque, elle compte comme de
l'atmosphère et biaise les exposants ; et comme l'étoile se déplace par
rapport à l'atmosphère avec le BERV au fil de l'année, ce biais change
d'une nuit à l'autre.</li>
<li><strong>Un template des spectres nettoyés</strong>
(<code>&lt;objet&gt;_tc</code>). Ramenées dans le référentiel de l'étoile
et combinées, les telluriques laissées par le premier passage tombent à un
endroit différent chaque nuit et se moyennent, pourvu que les nuits
couvrent assez de BERV : il reste le spectre de l'étoile.</li>
<li><strong>Second passage, divisé par ce template.</strong> Les spectres
livrés (pas ceux du premier passage) sont ajustés de nouveau, chacun divisé
par le template décalé au BERV de sa nuit : les raies de l'étoile ont
disparu, et les exposants sont mesurés sur l'atmosphère seule. Le template
n'entre que dans l'ajustement : le fichier écrit reste le spectre livré
divisé par la transmission, et il remplace celui du premier passage.</li>
<li><strong>Le template refait</strong>, à partir des spectres du second
passage : c'est contre lui que se mesurent le masque, les vitesses et
PCA2D.</li>
</ol>
<p>Les étapes de LBL tournent ensuite sur <code>&lt;objet&gt;_tc</code> :
le masque, les vitesses, la table rdb et le rapport ; avec PCA2D, le
pré-nettoyage travaille lui aussi sur ces spectres nettoyés. Chaque passage
prend environ une seconde par spectre (90 spectres HARPS de GJ 1214 :
1 min 35 s).</p>
<h3>Le vérifier dans le journal</h3>
<p>Chaque spectre écrit une ligne avec les exposants appliqués, ceux de
son en-tête (<code>TLPEH2O</code>, <code>TLPEOTR</code>), comme
<code>telluric exponents by the derivative method: H2O 0.1538, dry 1.0410
(airmass 1.0809), template=True</code> ; <code>FALLBACK</code> (0.35) quand
le coût de la dérivée n'avait rien pour trancher. La méthode de la dérivée
est la seule, pour tous les instruments, depuis le 2026-10-27 : la boucle
de corrélation croisée qui la précédait échouait souvent. Le premier passage
dit <code>template=False</code>, le
second <code>template=True</code> (sa bannière dit
<code>TELLUCLEAN_USE_TEMPLATE="True"</code>), avec des exposants un peu
différents de ceux du premier passage. Un second passage qui dit
<code>template=False</code> n'a pas trouvé le template et a nettoyé de
nouveau à l'aveugle ; LBL le dit maintenant par un avertissement. C'était
le cas de toutes les versions de LBL du 2026-02-24 au 2026-10-25, et
jusque-là le template n'était pas non plus refait après le second passage
(celui du premier restait sur le disque) : un run d'un spectrographe
optique fait dans cet intervalle mérite d'être relancé, <em>réécrire les
fichiers déjà sur le disque</em> coché.</p>
"""
    sections.append(card(
        "tellurics", "The telluric cleaning of optical spectra, in two passes",
        "Le nettoyage tellurique des spectres optiques, en deux passages",
        both(tellu_en, tellu_fr)))

    pca_en = """
<p>The precleaning is on by default and can be switched off (LBL alone).
Every option below is written into the wrap only when it differs from the
default, so a wrap says what was changed. The window shows each option's
default; <em>reset</em> (&#8634;) puts it back.</p>
<p><strong>In practice.</strong> Start with the defaults. Tick several
targets of one instrument observed on the same nights for a joint fit: one
observer basis for all, each star with its own spectrum. A basis saved from
a good joint fit (<em>save the observer basis as</em>) can be taken, frozen,
by a star with few spectra (<em>frozen observer basis</em>). Give each
variant a name, so that variants do not overwrite each other.</p>
"""
    pca_fr = """
<p>Le prénettoyage est actif par défaut et peut être désactivé (LBL seul).
Chaque option ci-dessous n'est écrite dans le wrap que si elle diffère du
défaut : un wrap dit donc ce qui a été changé. La fenêtre affiche le défaut
de chaque option ; <em>réinitialiser</em> (&#8634;) le remet.</p>
<p><strong>En pratique.</strong> Commencez avec les valeurs par défaut.
Cochez plusieurs cibles d'un même instrument observées les mêmes nuits pour
un fit joint : une seule base de l'observateur pour toutes, chaque étoile
avec son propre spectre. Une base sauvegardée d'un bon fit joint
(<em>sauver la base de l'observateur sous</em>) peut être reprise, figée,
par une étoile qui a peu de spectres (<em>base de l'observateur
figée</em>). Donnez un nom à chaque variante, pour que les variantes ne
s'écrasent pas.</p>
"""
    sections.append(card(
        "tab-pca2d", "Tab: PCA2D settings", "Onglet : Réglages PCA2D",
        both(pca_en, pca_fr) + help_of("help_pca2d_on")
        + help_of("help_skip_pca2d")
        + "<h3>%s</h3>" % inline("Every option", "Toutes les options")
        + both("<p class=\"hint\">Defaults for SPIRou, NIRPS and HARPS; one"
               " value when they agree.</p>",
               "<p class=\"hint\">Défauts pour SPIRou, NIRPS et HARPS ; une"
               " seule valeur quand ils sont identiques.</p>")
        + options_reference()))

    run_en = """
<p><strong>The wrap.</strong> The script that will run, built from the other
tabs, with every setting explained in a comment above it. You can edit it:
an edited wrap runs as you wrote it; <em>Rebuild from the settings</em>
starts it again from the tabs. <em>Save</em> writes it in the data folder;
<em>Load</em> reads a wrap back into the tabs (parsed, never run).</p>
<p><strong>Run.</strong> The big button starts the run, and changes colour
while it goes. One run at a time: a click during a run queues the next. The
<strong>steps</strong>, between the wrap and the log, tick off as the log
reaches them, and one that goes through files (the telluric cleaning, the
template, the velocities, the rdb table, PCA2D's cube and correction) fills
as it goes; click a step to jump to its place in the log. The
<strong>log</strong> keeps everything, with a scroll bar and two arrows to
its top and bottom; scrolled up, it stops following the run, and at the
bottom it follows again. <em>Stop</em> stops the run.</p>
"""
    run_fr = """
<p><strong>Le wrap.</strong> Le script qui va tourner, construit à partir
des autres onglets, chaque réglage expliqué par un commentaire au-dessus.
Vous pouvez le modifier : un wrap modifié tourne tel que vous l'avez écrit ;
<em>Reconstruire depuis les réglages</em> repart des onglets.
<em>Enregistrer</em> l'écrit dans le dossier de données ; <em>Charger</em>
relit un wrap dans les onglets (analysé, jamais exécuté).</p>
<p><strong>Lancer.</strong> Le gros bouton lance le run, et change de
couleur pendant qu'il tourne. Un run à la fois : un clic pendant un run met
le suivant en file. Les <strong>étapes</strong>, entre le wrap et le
journal, se cochent à mesure que le journal les atteint, et celle qui
parcourt des fichiers (le nettoyage tellurique, le template, les vitesses,
la table rdb, le cube et la correction de PCA2D) se remplit à mesure ;
cliquez sur une étape pour aller à sa place dans le journal. Le <strong>journal</strong>
garde tout, avec une barre de défilement et deux flèches vers son début et
sa fin ; remonté, il cesse de suivre le run, et en bas il le suit de
nouveau. <em>Arrêter</em> arrête le run.</p>
"""
    sections.append(card("tab-run", "Tab: Run", "Onglet : Lancer",
                         both(run_en, run_fr) + help_of("help_run_name")
                         + help_of("help_wrap") + help_of("help_load")
                         + help_of("help_steps")))

    others_en = """
<p><strong>Runs.</strong> The runs made from the window (name, state, start,
duration, the reports they wrote), kept up to date as runs start and end,
and every report of the data folder, made from the window or from a
terminal.</p>
<p><strong>Cleanup.</strong> What the data folder's products weigh, each in
one kind that says what deleting it costs: <em>scratch</em> (remade by the
next run that needs it, such as PCA2D's cubes), <em>rebuildable</em> (remade
by running again: superseded corrections, corrected spectra, per-exposure
velocities, templates, masks), <em>results</em> (rdb tables, reports,
observer bases: kept) and <em>spectra</em> (never touched). Only the first
two can be deleted, and only inside the data folder. Above each of them, a
button ticks the whole kind and one ticks each type in it (every
per-exposure velocity, every mask...), with how many and how much; a second
click unticks them, and what is ticked is said beside <em>Delete</em>.</p>
<p><strong>References.</strong> The paper that describes LBL, then a list,
not exhaustive, of the papers that used our codes: each one's title, its
authors and its page on ADS.</p>
"""
    others_fr = """
<p><strong>Runs.</strong> Les runs lancés depuis la fenêtre (nom, état,
début, durée, les rapports qu'ils ont écrits), tenus à jour à mesure que
les runs commencent et finissent, et tous les rapports du dossier de
données, faits depuis la fenêtre ou depuis un terminal.</p>
<p><strong>Ménage.</strong> Ce que pèsent les produits du dossier, chacun
rangé dans une catégorie qui dit ce que coûte sa suppression :
<em>jetable</em> (refait par le prochain run qui en a besoin, comme les
cubes de PCA2D), <em>refaisable</em> (refait en relançant : corrections
remplacées, spectres corrigés, vitesses par pose, templates, masques),
<em>résultats</em> (tables rdb, rapports, bases de l'observateur : gardés)
et <em>spectres</em> (jamais touchés). Seules les deux premières se
suppriment, et seulement dans le dossier de données. Au-dessus de chacune,
un bouton coche toute la catégorie et un autre chaque type qu'elle contient
(toutes les vitesses par pose, tous les masques...), avec leur nombre et
leur poids ; un second clic les décoche, et ce qui est coché est dit à côté
de <em>Supprimer</em>.</p>
<p><strong>Références.</strong> L'article qui décrit LBL, puis une liste
non exhaustive des articles ayant utilisé nos codes : le titre de chacun,
ses auteurs et sa page sur ADS.</p>
"""
    sections.append(card("tab-others", "Tabs: Runs, Cleanup, References",
                         "Onglets : Runs, Ménage, Références",
                         both(others_en, others_fr)))

    out_en = """
<p>What a run leaves, and where:</p>
<div class="table-wrap"><table class="data"><tbody>
<tr><td><code>lblrdb/lbl_&lt;target&gt;_&lt;template&gt;.rdb</code></td>
<td>the velocities, one row per exposure, with the indicators (d2v, d3v,
DTEMP...); <code>lbl2_</code> the same per night. This is what you
analyse.</td></tr>
<tr><td><code>reports/&lt;target&gt;_&lt;template&gt;.pdf</code></td><td>the
report of each target (see below)</td></tr>
<tr><td><code>science/&lt;target&gt;_pca2d/</code></td><td>the corrected
spectra, LBL's input for the corrected velocities (with the variant's name
when it has one)</td></tr>
<tr><td><code>pca2d/&lt;target&gt;/&lt;tag&gt;/</code></td><td>the PCA2D fit,
its figures and its configuration; <code>pca2d/joint/&lt;A+B&gt;/</code> for
a joint fit</td></tr>
<tr><td><code>pca2d/components/</code></td><td>the observer bases saved for
other stars to take frozen</td></tr>
<tr><td><code>models/pca2d_&lt;INSTRUMENT&gt;_&lt;SOURCE&gt;/</code></td><td>the
observer bases you keep (the <em>keep</em> button beside the basis, or put
the file there, or in any other subfolder of <code>models/</code>): read
first, never deleted by a cleanup; a fit of the same name updates the
basis there</td></tr>
<tr><td><code>gui/runs/&lt;run&gt;/</code></td><td><code>wrap.py</code> as it
ran and <code>run.json</code> (when, how it ended, the reports it
wrote)</td></tr>
<tr><td><code>gui/logs/</code></td><td>the window's logs: each run's whole
log (<code>&lt;run&gt;.log</code>) and the archive search's and fetch's
(<code>archives.log</code>)</td></tr>
<tr><td><code>wrap_&lt;name&gt;.py</code></td><td>the wraps you saved</td></tr>
</tbody></table></div>
<p><strong>The report.</strong> With PCA2D, its pages come first: after the
contents, <strong>the velocity series</strong>, every epoch coloured by
the star's velocity in the telluric frame, V<sub>tot</sub>, when SIMBAD has
its systemic velocity (darkest near zero, blue below and red above: the
epochs whose lines sit on the tellurics stand out, and those within 4 km/s
are outlined), by its BERV otherwise (a residual that follows the BERV is
not the star's, and shows as colours that line up); then the star, a summary of what the correction
changed (gain, loss, or changed where nothing should), the velocities
before and after, the run's settings, and the correction itself (the river
plots step by step, the components, their coefficients in time and their
periodograms). LBL's own report of the corrected target follows: the
velocity series again, a table of numbers, the velocity against the BERV
with the slope of a line through it, the periodograms and the activity
indicators.</p>
"""
    out_fr = """
<p>Ce qu'un run laisse, et où :</p>
<div class="table-wrap"><table class="data"><tbody>
<tr><td><code>lblrdb/lbl_&lt;cible&gt;_&lt;template&gt;.rdb</code></td>
<td>les vitesses, une ligne par pose, avec les indicateurs (d2v, d3v,
DTEMP...) ; <code>lbl2_</code> la même chose par nuit. C'est ce que vous
analysez.</td></tr>
<tr><td><code>reports/&lt;cible&gt;_&lt;template&gt;.pdf</code></td><td>le
rapport de chaque cible (voir plus bas)</td></tr>
<tr><td><code>science/&lt;cible&gt;_pca2d/</code></td><td>les spectres
corrigés, l'entrée de LBL pour les vitesses corrigées (avec le nom de la
variante quand elle en a un)</td></tr>
<tr><td><code>pca2d/&lt;cible&gt;/&lt;tag&gt;/</code></td><td>le fit PCA2D,
ses figures et sa configuration ; <code>pca2d/joint/&lt;A+B&gt;/</code>
pour un fit joint</td></tr>
<tr><td><code>pca2d/components/</code></td><td>les bases de l'observateur
sauvegardées, que d'autres étoiles peuvent reprendre figées</td></tr>
<tr><td><code>models/pca2d_&lt;INSTRUMENT&gt;_&lt;SOURCE&gt;/</code></td><td>les
bases de l'observateur que vous gardez (le bouton <em>garder</em> à côté de
la base, ou en déposant le fichier là ou dans tout autre sous-dossier de
<code>models/</code>) : lues en premier, jamais effacées par un ménage ; un
fit du même nom met la base à jour là</td></tr>
<tr><td><code>gui/runs/&lt;run&gt;/</code></td><td><code>wrap.py</code> tel
qu'il a tourné et <code>run.json</code> (quand, comment il s'est terminé,
les rapports écrits)</td></tr>
<tr><td><code>gui/logs/</code></td><td>les journaux de la fenêtre : le
journal complet de chaque run (<code>&lt;run&gt;.log</code>) et celui de la
recherche et des téléchargements d'archive (<code>archives.log</code>)</td></tr>
<tr><td><code>wrap_&lt;nom&gt;.py</code></td><td>les wraps que vous avez
enregistrés</td></tr>
</tbody></table></div>
<p><strong>Le rapport.</strong> Avec PCA2D, ses pages viennent en premier :
après la table des matières, <strong>la série temporelle des
vitesses</strong>, chaque époque colorée selon la vitesse de l'étoile dans
le référentiel des telluriques, V<sub>tot</sub>, quand SIMBAD connaît sa
vitesse systémique (plus foncé près de zéro, bleu en dessous et rouge
au-dessus : les époques dont les raies tombent sur les telluriques
ressortent, et celles à moins de 4 km/s sont cerclées), selon son BERV
sinon (un résidu qui suit le BERV ne vient pas de l'étoile, et se voit comme
des couleurs qui s'alignent) ; puis l'étoile, un résumé de ce que la correction a changé
(gain, perte, ou changement là où rien n'aurait dû bouger), les vitesses
avant et après, les réglages du run, et la correction elle-même (les
graphiques en rivière étape par étape, les composantes, leurs coefficients
dans le temps et leurs périodogrammes). Le rapport de LBL de la cible
corrigée suit : de nouveau la série temporelle, un tableau de chiffres, la
vitesse en fonction du BERV avec la pente d'une droite, les périodogrammes
et les indicateurs d'activité.</p>
"""
    sections.append(card("outputs", "What a run produces",
                         "Ce que produit un run", both(out_en, out_fr)))

    hand_en = """
<p>Every run keeps its wrap, and the window is not needed to run it
again:</p>
""" + code("""
conda activate lbl-pca2d
python DATA_DIR/gui/runs/<run>/wrap.py
""") + """
<p><strong>Updating.</strong> <code>git pull</code> in LBL's clone, then
start <code>lbl_gui</code> again. The install is editable: nothing to
reinstall unless the dependencies or the commands change (then
<code>pip install -e .</code> in the clone).</p>
"""
    hand_fr = """
<p>Chaque run garde son wrap, et la fenêtre n'est pas nécessaire pour le
relancer :</p>
""" + code("""
conda activate lbl-pca2d
python DATA_DIR/gui/runs/<run>/wrap.py
""") + """
<p><strong>Mettre à jour.</strong> <code>git pull</code> dans le clone de LBL,
puis relancez <code>lbl_gui</code>. L'installation est éditable : rien à
réinstaller, sauf si les dépendances ou les commandes changent (alors
<code>pip install -e .</code> dans le clone).</p>
"""
    sections.append(card("by-hand", "Without the window, and updating",
                         "Sans la fenêtre, et mises à jour",
                         both(hand_en, hand_fr)))

    trouble_en = """
<ul class="feature-list">
<li><strong><code>lbl_gui</code> stops at once, or
<code>python -c "import lbl; print(lbl.__file__)"</code> prints
<code>None</code>.</strong> A folder named <code>lbl</code> on Python's path
(an old clone in your home folder, when <code>PYTHONPATH</code> lists it)
hides the install. Reinstall with
<code>pip install -e &lt;your clone&gt; --config-settings
editable_mode=compat</code>, or take that folder's parent off
<code>PYTHONPATH</code>.</li>
<li><strong>Vetted templates are not on LBL's server yet.</strong> A run that
asks for one (<code>VETTED_&lt;star&gt;</code>) is refused by the window
while the file is missing. Measure each target against its own template in
the meantime.</li>
<li><strong>"Key WAVEORDN not found".</strong> The files were read with the
wrong data source (the CADC's <code>t.fits</code> as APERO's). The window
now reads the source from the files and refuses a mismatch; if you edited
the wrap by hand, check <code>DATA_SOURCE</code>.</li>
<li><strong>A report says "LBL has not measured this run yet", or that the
velocities predate the corrected spectra.</strong> LBL has not measured the
latest corrected spectra: run again with <em>rewrite the files already on
disk</em> ticked.</li>
<li><strong>Memory.</strong> A target with many hundreds of exposures makes
a large cube. Nightly stacking (on by default for NIRPS, automatic
elsewhere) coadds each night; run one heavy target at a time.</li>
<li><strong>SOPHIE's archive does not answer.</strong> The OHP's archive
server (atlas.obs-hp.fr) has not answered since 2026-10-24: a SOPHIE search
waits, then says so in the search's log. Stop the search to keep what the
other archives found.</li>
<li><strong>No LaTeX.</strong> Without <code>pdflatex</code> the reports are
the figures bound together, with nothing missing but the typesetting.</li>
<li><strong>The window shows nothing new after an update.</strong> Reload the
page without the cache (Cmd+Shift+R or Ctrl+Shift+R).</li>
</ul>"""
    trouble_fr = """
<ul class="feature-list">
<li><strong><code>lbl_gui</code> s'arrête aussitôt, ou
<code>python -c "import lbl; print(lbl.__file__)"</code> affiche
<code>None</code>.</strong> Un dossier nommé <code>lbl</code> dans le chemin
de Python (un ancien clone dans votre dossier personnel, quand
<code>PYTHONPATH</code> le contient) cache l'installation. Réinstallez avec
<code>pip install -e &lt;votre clone&gt; --config-settings
editable_mode=compat</code>, ou retirez le dossier parent de
<code>PYTHONPATH</code>.</li>
<li><strong>Les templates validés ne sont pas encore sur le serveur de
LBL.</strong> Un run qui en demande un (<code>VETTED_&lt;étoile&gt;</code>)
est refusé par la fenêtre tant que le fichier manque. Mesurez chaque cible
contre son propre template en attendant.</li>
<li><strong>« Key WAVEORDN not found ».</strong> Les fichiers ont été lus
avec la mauvaise source de données (les <code>t.fits</code> de la CADC comme
ceux d'APERO). La fenêtre lit maintenant la source dans les fichiers et
refuse une incohérence ; si vous avez modifié le wrap à la main, vérifiez
<code>DATA_SOURCE</code>.</li>
<li><strong>Un rapport dit « LBL has not measured this run yet », ou que les
vitesses précèdent les spectres corrigés.</strong> LBL n'a pas mesuré les
derniers spectres corrigés : relancez avec <em>réécrire les fichiers déjà
sur le disque</em> cochée.</li>
<li><strong>Mémoire.</strong> Une cible de plusieurs centaines de poses fait
un gros cube. L'empilement par nuit (actif par défaut pour NIRPS,
automatique ailleurs) additionne chaque nuit ; lancez une cible lourde à la
fois.</li>
<li><strong>L'archive de SOPHIE ne répond pas.</strong> Le serveur
d'archives de l'OHP (atlas.obs-hp.fr) ne répond plus depuis le 2026-10-24 :
une recherche SOPHIE attend, puis le dit dans le journal de la recherche.
Arrêtez la recherche pour garder ce que les autres archives ont trouvé.</li>
<li><strong>Pas de LaTeX.</strong> Sans <code>pdflatex</code>, les rapports
sont les figures assemblées, sans rien de moins que la mise en page.</li>
<li><strong>La fenêtre ne montre rien de nouveau après une mise à
jour.</strong> Rechargez la page sans le cache (Cmd+Maj+R ou
Ctrl+Maj+R).</li>
</ul>"""
    sections.append(card("trouble", "Known problems", "Problèmes connus",
                         both(trouble_en, trouble_fr)))

    sections.append(card("feedback", "Your feedback", "Vos retours", both(
        '<p>This is a test version: what breaks, what is unclear, what is'
        ' missing, all of it helps. Open an issue on <a href="%s">'
        'github.com/njcuk9999/lbl</a>, with the run\'s log'
        ' (<code>DATA_DIR/gui/logs/&lt;run&gt;.log</code>) and its'
        ' <code>wrap.py</code> when a run went wrong, or write to'
        ' Étienne.</p>' % ISSUES,
        '<p>C\'est une version de test : ce qui casse, ce qui n\'est pas'
        ' clair, ce qui manque, tout aide. Ouvrez un ticket sur <a'
        ' href="%s">github.com/njcuk9999/lbl</a>, avec le journal du run'
        ' (<code>DATA_DIR/gui/logs/&lt;run&gt;.log</code>) et son'
        ' <code>wrap.py</code> quand un run a mal tourné, ou écrivez à'
        ' Étienne.</p>' % ISSUES)))

    contents = "".join(
        '<li><a href="#%s">%s</a></li>' % (anchor, inline(esc(en), esc(fr)))
        for anchor, en, fr in (
            ("short", "In short", "En bref"),
            ("before", "Before you start", "Avant de commencer"),
            ("install", "Installing", "Installation"),
            ("tutorial", "Step by step: a first run",
             "Pas à pas : un premier run"),
            ("reference", "Reference", "Référence"),
            ("data", "The data folder", "Le dossier de données"),
            ("launch", "Starting the window", "Lancer la fenêtre"),
            ("tab-targets", "Tab: Targets", "Onglet : Cibles"),
            ("tab-lbl", "Tab: LBL settings", "Onglet : Réglages LBL"),
            ("tellurics", "The telluric cleaning, in two passes",
             "Le nettoyage tellurique, en deux passages"),
            ("tab-pca2d", "Tab: PCA2D settings", "Onglet : Réglages PCA2D"),
            ("tab-run", "Tab: Run", "Onglet : Lancer"),
            ("tab-others", "Tabs: Runs, Cleanup, References",
             "Onglets : Runs, Ménage, Références"),
            ("outputs", "What a run produces", "Ce que produit un run"),
            ("by-hand", "Without the window, and updating",
             "Sans la fenêtre, et mises à jour"),
            ("trouble", "Known problems", "Problèmes connus"),
            ("feedback", "Your feedback", "Vos retours")))

    return TEMPLATE % dict(
        sections="".join(sections), contents=contents, today=today,
        version=esc(__version__), repo=REPO, lbl_repo=LBL_REPO)


TEMPLATE = """<!doctype html>
<html lang="en" data-lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>lbl-gui</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="icon" href="favicon.ico?v=2" sizes="32x32">
<link rel="icon" type="image/svg+xml" href="favicon.svg?v=2">
<link rel="apple-touch-icon" href="apple-touch-icon.png?v=2">
<link rel="stylesheet" href="theme.css">
<link rel="stylesheet" href="guide.css">
<script>
  // the page's language: the one chosen before, else the browser's
  (function () {
    var lang = null;
    try { lang = localStorage.getItem('lbl-gui-guide-lang'); } catch (e) {}
    if (lang !== 'fr' && lang !== 'en') {
      lang = (navigator.language || 'en').toLowerCase().indexOf('fr') === 0 ? 'fr' : 'en';
    }
    document.documentElement.setAttribute('data-lang', lang);
    document.documentElement.setAttribute('lang', lang);
  })();
</script>
</head>
<body>
  <div class="bg-orb orb-a"></div>
  <div class="bg-orb orb-b"></div>

  <div class="site-banner">
    <div class="site-banner-inner">
      <div class="site-banner-text">
        <h2><img class="banner-icon" src="favicon.svg" alt="">lbl-gui</h2>
        <p><span lang="en">LBL with the PCA2D precleaning, and its window &middot; testers' guide</span><span lang="fr">LBL avec le prénettoyage PCA2D, et sa fenêtre &middot; guide des testeurs</span></p>
      </div>
      <button type="button" class="home-link lang-switch" id="lang" aria-label="language"><span lang="en">Français</span><span lang="fr">English</span></button>
      <a href="https://eartigau.github.io/" class="home-link">&larr; Home</a>
    </div>
  </div>

  <header class="hero">
    <p class="eyebrow"><span lang="en">Radial velocities &middot; line by line</span><span lang="fr">Vitesses radiales &middot; raie par raie</span></p>
    <h1>lbl-gui</h1>
    <p class="subtitle"><span lang="en">How to install LBL's lbl-pca2d branch and its window, and how to use them, step by step.</span><span lang="fr">Comment installer la branche lbl-pca2d de LBL et sa fenêtre, et comment s'en servir, étape par étape.</span></p>
  </header>

  <main class="layout">
    <section class="card span-3 warning" role="alert">
      <h2><span lang="en">Test version: not released yet</span><span lang="fr">Version de test : pas encore lancée</span></h2>
      <p lang="en">This page is for the testers of a version that is not released. Please do not share the link or cite it. The window, its options and its outputs can still change from one day to the next, and some of what it measures has not been validated yet: check the velocities against your usual ones before using them.</p>
      <p lang="fr">Cette page s'adresse aux testeurs d'une version qui n'est pas lancée. Merci de ne pas diffuser le lien ni le citer. La fenêtre, ses options et ses produits peuvent encore changer d'un jour à l'autre, et une partie de ce qu'elle mesure n'est pas encore validée : comparez les vitesses à celles que vous avez d'habitude avant de vous en servir.</p>
    </section>

    <nav class="card span-3 contents">
      <h2><span lang="en">Contents</span><span lang="fr">Sommaire</span></h2>
      <ol>%(contents)s</ol>
    </nav>

    %(sections)s
  </main>

  <footer class="footer">
    <p>
      <a href="%(repo)s" target="_blank">the window in LBL (lbl/gui)</a>
      &middot;
      <a href="%(lbl_repo)s/tree/lbl-pca2d" target="_blank">LBL, branch lbl-pca2d</a>
      &middot;
      <a href="https://eartigau.github.io/" target="_blank">More tools</a>
    </p>
    <p class="hint">LBL %(version)s &middot; <span lang="en">guide of</span><span lang="fr">guide du</span> %(today)s</p>
  </footer>

  <div id="lightbox" class="lightbox" hidden>
    <button id="lightboxClose" class="lightbox-close" aria-label="Close" type="button">&times;</button>
    <figure class="lightbox-figure">
      <img id="lightboxImg" src="" alt="" />
      <figcaption id="lightboxCaption"></figcaption>
    </figure>
  </div>
  <script src="lightbox.js"></script>
  <script>
    document.getElementById('lang').addEventListener('click', function () {
      var next = document.documentElement.getAttribute('data-lang') === 'fr' ? 'en' : 'fr';
      document.documentElement.setAttribute('data-lang', next);
      document.documentElement.setAttribute('lang', next);
      try { localStorage.setItem('lbl-gui-guide-lang', next); } catch (e) {}
    });
  </script>
</body>
</html>
"""

GUIDE_CSS = """/* the testers' guide: what theme.css does not have */
/* the window's tab icon, in the banner too */
.site-banner-text h2 { display: flex; align-items: center; gap: 10px; }
.site-banner-text h2 .banner-icon { width: 30px; height: 30px; border-radius: 7px; display: block; }
html[data-lang="fr"] [lang="en"], html[data-lang="en"] [lang="fr"] { display: none !important; }
.lang-switch { cursor: pointer; font: inherit; margin-right: 8px; }
.warning { border-color: rgba(255, 180, 84, 0.7); background: rgba(80, 45, 6, 0.55); }
.warning h2 { color: #ffb454; font-size: 1.3rem; }
.warning p { color: #ffe2bf; font-size: 1rem; }
.contents ol { columns: 2; column-gap: 32px; margin: 0; padding-left: 1.3em; }
.contents li { margin: 3px 0; break-inside: avoid; }
.card h3 { font-size: 0.98rem; margin: 18px 0 6px; color: var(--accent); }
.card p, .card li { line-height: 1.55; }
.card code { font-family: "IBM Plex Mono", monospace; font-size: 0.86em; color: #cfe4ff; }
.option { border-top: 1px solid var(--line); padding: 10px 0 4px; }
.option-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; }
.option-label { font-weight: 700; }
.option-default { color: var(--muted); font-family: "IBM Plex Mono", monospace; font-size: 0.82rem; }
.option p { color: var(--muted); margin: 6px 0; }
.table-wrap { overflow-x: auto; }
table.data { border-collapse: collapse; width: 100%%; font-size: 0.9rem; }
table.data td, table.data th { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
table.data th { color: var(--muted); font-weight: 500; }
.lecture-figure { margin: 14px 0 0; }
@media (max-width: 720px) { .contents ol { columns: 1; } }
/* the tutorial: numbered steps, badges as on the screenshots */
.lead { font-size: 1.05rem; }
.step { display: grid; grid-template-columns: 54px minmax(0, 1fr); gap: 16px; border-top: 1px solid var(--line); padding: 22px 0 8px; }
.step-num { width: 44px; height: 44px; border-radius: 50%%; background: var(--accent); color: #08111f; font: 700 1.3rem "Space Grotesk", sans-serif; display: flex; align-items: center; justify-content: center; }
.step h3 { margin: 6px 0 12px; font-size: 1.2rem; color: var(--text); }
.badge { display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; width: 26px; height: 26px; border-radius: 50%%; background: #ff5400; color: #fff; font-weight: 700; font-size: 0.9rem; border: 2px solid #fff; }
.marks { list-style: none; padding: 0; margin: 10px 0 14px; }
.marks li { display: flex; gap: 10px; align-items: flex-start; margin: 8px 0; }
.marks li span:last-child { padding-top: 2px; }
.shot.narrow img { max-width: 470px; display: block; }
.shot img { border-radius: 10px; border: 1px solid var(--line); width: 100%%; background: #fff; }
.callout { border-radius: 12px; padding: 10px 14px; margin: 12px 0; border: 1px solid var(--line); background: rgba(255, 255, 255, 0.03); }
.callout > strong { display: block; margin-bottom: 4px; font-size: 0.86rem; letter-spacing: 0.06em; text-transform: uppercase; }
.callout.see { border-color: rgba(108, 226, 173, 0.55); } .callout.see > strong { color: #6ce2ad; }
.callout.tip { border-color: rgba(98, 194, 255, 0.55); } .callout.tip > strong { color: var(--accent); }
.callout.why { border-color: rgba(178, 160, 255, 0.55); } .callout.why > strong { color: #b2a0ff; }
.callout.example { border-color: rgba(255, 180, 84, 0.6); } .callout.example > strong { color: #ffb454; }
.callout p { margin: 4px 0; }
.part { background: transparent; border: none; padding: 30px 16px 0; }
.part h2 { font-size: 1.6rem; }
@media (max-width: 720px) { .step { grid-template-columns: 1fr; } }
"""

WORKFLOW_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 250" font-family="Space Grotesk, sans-serif">
  <rect width="1120" height="250" fill="#0b1626" rx="14"/>
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#62c2ff"/></marker></defs>
  <g font-size="15" fill="#e8eef8" text-anchor="middle">
    <rect x="20" y="95" width="150" height="62" rx="10" fill="#14243c" stroke="#a8b4ca"/>
    <text x="95" y="122">science/&lt;{target}&gt;</text><text x="95" y="142" fill="#a8b4ca" font-size="13">{delivered}</text>
    <rect x="210" y="95" width="150" height="62" rx="10" fill="#14243c" stroke="#62c2ff"/>
    <text x="285" y="122">LBL</text><text x="285" y="142" fill="#a8b4ca" font-size="13">{before}</text>
    <rect x="400" y="30" width="420" height="190" rx="12" fill="#101e33" stroke="#6ce2ad" stroke-dasharray="5 4"/>
    <text x="610" y="55" fill="#6ce2ad">PCA2D</text>
    <rect x="420" y="95" width="80" height="62" rx="10" fill="#14243c" stroke="#6ce2ad"/><text x="460" y="131">cube</text>
    <rect x="520" y="95" width="80" height="62" rx="10" fill="#14243c" stroke="#6ce2ad"/><text x="560" y="131">fit</text>
    <rect x="620" y="95" width="80" height="62" rx="10" fill="#14243c" stroke="#6ce2ad"/><text x="660" y="131">figures</text>
    <rect x="720" y="95" width="80" height="62" rx="10" fill="#14243c" stroke="#6ce2ad"/><text x="760" y="131">correct</text>
    <text x="610" y="190" fill="#a8b4ca" font-size="13">{frames}</text>
    <rect x="850" y="60" width="250" height="56" rx="10" fill="#14243c" stroke="#62c2ff"/>
    <text x="975" y="85" font-size="14">{lbl_on} science/&lt;{target}&gt;_pca2d</text><text x="975" y="104" fill="#a8b4ca" font-size="13">{after}</text>
    <rect x="850" y="140" width="250" height="56" rx="10" fill="#14243c" stroke="#ffb454"/>
    <text x="975" y="165">{report}</text><text x="975" y="184" fill="#a8b4ca" font-size="13">reports/&lt;{target}&gt;_..pdf</text>
  </g>
  <g stroke="#62c2ff" stroke-width="2" fill="none" marker-end="url(#a)">
    <path d="M170 126 H208"/><path d="M360 126 H418"/><path d="M500 126 H518"/><path d="M600 126 H618"/><path d="M700 126 H718"/>
    <path d="M800 116 C825 100 832 88 848 88"/><path d="M975 116 V138"/>
  </g>
</svg>
"""


# ------------------------------------------------------- the example run
def _stamp(line):
    """The time of an LBL log line (2026-10-24 14:44:49.053|I|...)."""
    try:
        return datetime.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def machine():
    """The machine the example ran on, as its system says (macOS)."""
    import subprocess
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True).stdout)
        return "%s, %d GB" % (cpu, round(mem / 2 ** 30))
    except Exception:                                     # noqa: BLE001
        return ""


def _minutes(a, b):
    return max(1, int(round((b - a).total_seconds() / 60.0)))


def example_results(example, site_folder):
    """What the tutorial's example run produced, for its steps 9 and 10:
    how long each part took (its log), the first rows of its corrected
    velocities (lblrdb), and pages of its report drawn as images."""
    import glob
    import json
    import subprocess
    data = os.path.join(example, "data")
    runs = sorted(glob.glob(os.path.join(data, "gui", "runs", "*", "run.json")))
    if not runs:
        return {}
    record = json.load(open(runs[-1]))
    # the run's log, in the window's logs (or in its folder, from an older
    #   window)
    folder = os.path.dirname(runs[-1])
    log = os.path.join(data, "gui", "logs", os.path.basename(folder) + ".log")
    if not os.path.exists(log):
        log = os.path.join(folder, "run.log")
    lines = open(log).read().splitlines()
    marks = [(i, l) for i, l in enumerate(lines) if _stamp(l)]
    first = _stamp(marks[0][1]) if marks else None
    reports = [_stamp(l) for _, l in marks if "Report written" in l]
    cube = next((_stamp(l) for _, l in marks if "stage cube" in l), None)
    after = None
    if cube:
        seen_correct = False
        for _, l in marks:
            if "stage correct" in l:
                seen_correct = True
            elif seen_correct and "LBL Template" in l:
                after = _stamp(l)
                break
    out = {}
    if first and len(reports) >= 2 and cube and after:
        total = (record.get("ended") or 0) - (record.get("started") or 0)
        a, b, c = (_minutes(first, reports[0]), _minutes(cube, after),
                   _minutes(after, reports[-1]))
        mins = int(round(total / 60.0)) if total > 0 else a + b + c
        where = machine()
        out["durations_en"] = (
            "<p>On a laptop (%s), the %d spectra took about"
            " <strong>%d minutes</strong> in all: %d for the telluric cleaning"
            " and LBL on the delivered spectra, %d for PCA2D (cube, fit,"
            " figures, correction), %d for LBL on the corrected spectra and"
            " the report.</p>" % (where, guide_tutorial.EXAMPLE["spectra"],
                                  mins, a, b, c))
        out["durations_fr"] = (
            "<p>Sur un portable (%s), les %d spectres ont pris"
            " environ <strong>%d minutes</strong> en tout : %d pour le"
            " nettoyage tellurique et LBL sur les spectres livrés, %d pour"
            " PCA2D (cube, fit, figures, correction), %d pour LBL sur les"
            " spectres corrigés et le rapport.</p>"
            % (where.replace("GB", "Go"), guide_tutorial.EXAMPLE["spectra"],
               mins, a, b, c))
    # the corrected velocities: the newest lbl_ table of the corrected object
    tables = sorted(glob.glob(os.path.join(data, "lblrdb", "lbl_*pca2d*.rdb")),
                    key=os.path.getmtime)
    if tables:
        rows = [l.rstrip("\n").split("\t") for l in open(tables[-1])
                if l.strip()]
        head = rows[0]
        # a few columns, rounded as one reads them: the date to 1e-4 d,
        #   the velocity and its error to 1 cm/s, in the order of time
        formats = (("rjd", "%.4f"), ("vrad", "%.2f"), ("svrad", "%.2f"),
                   ("d2v", "%.0f"))
        keep = [(head.index(c), f) for c, f in formats if c in head]
        body = sorted(rows[2:], key=lambda row: float(row[head.index("rjd")]))
        lines = ["  ".join("%14s" % head[j] for j, _ in keep)]
        for row in body[:5]:
            lines.append("  ".join("%14s" % (f % float(row[j]))
                                   for j, f in keep))
        lines.append("  ".join("%14s" % "..." for _ in keep))
        shown = "\n".join(lines)
        name = os.path.basename(tables[-1])
        out["results_en"] = (
            "<p>The velocities are in <code>lblrdb/%s</code>, one row per"
            " exposure (<code>lbl2_</code>: per night). A few of its"
            " columns: the date (rjd, BJD - 2400000), the velocity and its"
            " error (vrad, svrad, in m/s, the systemic velocity included),"
            " and d2v (m&sup2;/s&sup2;), the line-width indicator:</p>%s"
            % (esc(name), code(shown)))
        out["results_fr"] = (
            "<p>Les vitesses sont dans <code>lblrdb/%s</code>, une ligne par"
            " pose (<code>lbl2_</code> : par nuit). Quelques-unes de ses"
            " colonnes : la date (rjd, BJD - 2400000), la vitesse et son"
            " erreur (vrad, svrad, en m/s, vitesse systémique comprise), et"
            " d2v (m&sup2;/s&sup2;), l'indicateur de largeur des raies :</p>%s"
            % (esc(name), code(shown)))
    out["report_pages"] = report_shots(data, site_folder)
    return out


#: the pages of the example's report shown in the tutorial: a line of their
#:    text that finds them, the image's name, the part of the page kept and
#:    what to point at, in pixels of the page drawn at 50 dpi (x, y, w, h)
REPORT_PAGES = (
    dict(find="1 The velocity series", name="report_series",
         crop=(15, 38, 395, 282),
         marks=[(1, (65, 108, 253, 108), "left"),
                (2, (320, 106, 26, 112), "right"),
                (3, (65, 243, 296, 63), "left")]),
    dict(find="3 Summary", name="report_summary",
         crop=(15, 225, 395, 322),
         marks=[(1, (36, 283, 300, 36), "left"),
                (2, (44, 452, 328, 92), "left")]),
)
#: the resolution the pages are drawn at, for the screen
REPORT_DPI = 110


def report_shots(data, site_folder):
    """Pages of the example's report as annotated images in shots/; the
    names of those made."""
    import glob
    import subprocess
    from annotate import annotate
    pdfs = sorted(glob.glob(os.path.join(data, "reports", "*pca2d*.pdf")),
                  key=os.path.getmtime)
    if not pdfs:
        return []
    pdf = pdfs[-1]
    pages = int(subprocess.run(["pdfinfo", pdf], capture_output=True,
                               text=True).stdout.split("Pages:")[1]
                .split()[0])
    target = os.path.join(site_folder, "shots")
    os.makedirs(target, exist_ok=True)
    scale = REPORT_DPI / 50.0
    made = []
    for spec in REPORT_PAGES:
        number = None
        for n in range(1, min(pages, 15) + 1):
            text = subprocess.run(["pdftotext", "-f", str(n), "-l", str(n),
                                   "-layout", pdf, "-"], capture_output=True,
                                  text=True).stdout
            text = " ".join(text.split())
            if spec["find"] in text and "Contents" not in text:
                number = n
                break
        if number is None:
            continue
        x0, y0, w, h = (int(round(v * scale)) for v in spec["crop"])
        base = os.path.join(target, spec["name"])
        subprocess.run(["pdftoppm", "-r", str(REPORT_DPI), "-png",
                        "-singlefile", "-f", str(number), "-l", str(number),
                        "-x", str(x0), "-y", str(y0), "-W", str(w), "-H",
                        str(h), pdf, base], check=True)
        boxes = [(k, dict(x=bx * scale, y=by * scale, width=bw * scale,
                          height=bh * scale), side)
                 for k, (bx, by, bw, bh), side in spec["marks"]]
        annotate(base + ".png", boxes, (x0, y0))
        made.append(spec["name"])
    return made


def main(site, shots=None, example=None):
    folder = os.path.join(os.path.abspath(os.path.expanduser(site)), "lbl-gui")
    os.makedirs(folder, exist_ok=True)
    if shots:
        # the tutorial's screenshots (tools/tutorial_shots.py)
        target = os.path.join(folder, "shots")
        os.makedirs(target, exist_ok=True)
        for name in sorted(os.listdir(shots)):
            if name.endswith(".webp"):
                shutil.copyfile(os.path.join(shots, name),
                                os.path.join(target, name))
    results = example_results(example, folder) if example else {}
    with open(os.path.join(folder, "index.html"), "w") as handle:
        handle.write(page(results))
    with open(os.path.join(folder, "guide.css"), "w") as handle:
        handle.write(GUIDE_CSS.replace("%%", "%"))
    words = dict(
        en=dict(target="target", delivered="delivered spectra",
                before="velocities before", lbl_on="LBL on",
                after="velocities after", report="report",
                frames="star frame + observer frame; the observer part"
                       " divided out"),
        fr=dict(target="cible", delivered="spectres livrés",
                before="vitesses avant", lbl_on="LBL sur",
                after="vitesses après", report="rapport",
                frames="référentiel de l'étoile + de l'observateur ; la"
                       " part de l'observateur divisée"))
    for lang, said in words.items():
        with open(os.path.join(folder, "workflow_%s.svg" % lang), "w") as handle:
            handle.write(WORKFLOW_SVG.format(**said))
    for name in ("theme.css", "lightbox.js"):
        shutil.copyfile(os.path.join(THEMES, name), os.path.join(folder, name))
    # the window's own icon
    static = os.path.join(HERE, "..", "static")
    # (the .ico for Safari's tabs, which take no SVG)
    for name in ("favicon.ico", "favicon.svg", "apple-touch-icon.png"):
        shutil.copyfile(os.path.join(static, name), os.path.join(folder, name))
    print("written: %s" % os.path.join(folder, "index.html"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
