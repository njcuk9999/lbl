"""The step-by-step part of the testers' guide: one worked example, Gliese 229
with HARPS, every step with its screenshot (tools/tutorial_shots.py), the
numbers on the screenshot explained, and what you should see. The numbers
quoted are those of the example run made for the screenshots.

Used by make_guide.py. A step is a list of pieces: an HTML string that holds
both languages or none (a screenshot, a command), or a pair (English,
French).
"""

#: the example, as the screenshots show it
EXAMPLE = dict(target="Gliese229", spectra=214, nights=207, snr=157,
               berv="-21.3 &rarr; 21.2", span=42.4, teff=3660,
               dates="2003-12-13 &rarr; 2024-10-23", folder_targets=3,
               folder_spectra=417)


def badge(n):
    return '<span class="badge">%d</span>' % n


def marks(en, fr):
    """The lists that explain a screenshot's numbers, both languages."""
    def one(items, lang):
        return '<ol class="marks" lang="%s">%s</ol>' % (lang, "".join(
            "<li>%s <span>%s</span></li>" % (badge(i + 1), text)
            for i, text in enumerate(items)))
    return one(en, "en") + one(fr, "fr")


def shot(name, en, fr, narrow=False):
    """A screenshot in the page's language, its caption below; `narrow`
    for a tall one, kept to the width it was taken at."""
    return ('<figure class="lecture-figure shot%s">'
            '<img lang="en" src="shots/%s_en.webp" alt="%s" loading="lazy">'
            '<img lang="fr" src="shots/%s_fr.webp" alt="%s" loading="lazy">'
            '<figcaption><span lang="en">%s</span><span lang="fr">%s</span>'
            '</figcaption></figure>'
            % (" narrow" if narrow else "", name, en, name, fr, en, fr))


def callout(kind, en, fr):
    """see: what you should see; tip: advice; why: the reason."""
    title = dict(see=("What you should see", "Ce que vous devez voir"),
                 tip=("Tip", "Conseil"), why=("Why", "Pourquoi"),
                 example=("In the example", "Dans l'exemple"))[kind]
    return ('<div class="callout %s"><strong><span lang="en">%s</span>'
            '<span lang="fr">%s</span></strong>'
            '<div lang="en">%s</div><div lang="fr">%s</div></div>'
            % (kind, title[0], title[1], en, fr))


def step(n, title, parts):
    body = []
    for piece in parts:
        if isinstance(piece, tuple):
            body.append('<div lang="en">%s</div><div lang="fr">%s</div>'
                        % piece)
        else:
            body.append(piece)
    return ('<div class="step" id="step-%d"><div class="step-num">%d</div>'
            '<div class="step-body"><h3><span lang="en">%s</span>'
            '<span lang="fr">%s</span></h3>%s</div></div>'
            % (n, n, title[0], title[1], "".join(body)))


def report_part(pages):
    """The report's pages in step 10, as the example drew them."""
    out = []
    if "report_series" in pages:
        out += [
            ("<p>Open the report. With PCA2D its first section, right after"
             " the contents, is the velocity series:</p>",
             "<p>Ouvrez le rapport. Avec PCA2D, sa première section, juste"
             " après la table des matières, est la série temporelle des"
             " vitesses :</p>"),
            '<figure class="lecture-figure shot"><img src="shots/'
            'report_series.webp" alt="velocity series" loading="lazy">'
            '<figcaption><span lang="en">The report, section 1.</span>'
            '<span lang="fr">Le rapport, section 1.</span></figcaption>'
            '</figure>',
            marks(["<strong>Every exposure's velocity</strong>, the median"
                   " taken out, coloured by its BERV.",
                   "<strong>The colour scale</strong> of the BERV (km/s). A"
                   " residual that follows the BERV is not the star's: it"
                   " shows as colours that line up (all the reds high, all"
                   " the blues low).",
                   "<strong>How the velocities are spread</strong>, with the"
                   " dashed lines at plus and minus one robust sigma (MAD)."],
                  ["<strong>La vitesse de chaque pose</strong>, la médiane"
                   " retirée, colorée selon son BERV.",
                   "<strong>L'échelle de couleur</strong> du BERV (km/s). Un"
                   " résidu qui suit le BERV ne vient pas de l'étoile : il se"
                   " voit comme des couleurs qui s'alignent (tous les rouges"
                   " en haut, tous les bleus en bas).",
                   "<strong>La distribution des vitesses</strong>, avec en"
                   " tirets plus et moins un sigma robuste (MAD)."]),
        ]
    if "report_summary" in pages:
        out += [
            ("<p>A few pages further, the summary says what the correction"
             " changed:</p>",
             "<p>Quelques pages plus loin, le résumé dit ce que la correction"
             " a changé :</p>"),
            '<figure class="lecture-figure shot"><img src="shots/'
            'report_summary.webp" alt="summary" loading="lazy">'
            '<figcaption><span lang="en">The report, section 3.</span>'
            '<span lang="fr">Le rapport, section 3.</span></figcaption>'
            '</figure>',
            marks(["<strong>In words</strong>: what got better (green), worse"
                   " (red), or changed where nothing should have.",
                   "<strong>In numbers</strong>: the delivered and the"
                   " corrected velocities side by side, on the same"
                   " exposures."],
                  ["<strong>En mots</strong> : ce qui s'est amélioré (vert),"
                   " dégradé (rouge), ou a changé là où rien n'aurait dû"
                   " bouger.",
                   "<strong>En chiffres</strong> : les vitesses livrées et"
                   " corrigées côte à côte, sur les mêmes poses."]),
            callout("example",
                    "<p>Gliese 229, 214 exposures: the robust sigma goes from"
                    " 5.84 to 4.57 m/s (1.28 times better), the rms from 9.21"
                    " to 8.73 m/s; the error bars barely move (0.48 to 0.50"
                    " m/s).</p>",
                    "<p>Gliese 229, 214 poses : le sigma robuste passe de 5,84"
                    " à 4,57 m/s (1,28 fois mieux), le rms de 9,21 à 8,73"
                    " m/s ; les barres d'erreur bougent à peine (0,48 à 0,50"
                    " m/s).</p>"),
        ]
    return out


def tutorial(code, results):
    """The tutorial's HTML. `code` makes a code block; `results`, what the
    example run produced (make_guide.example_results): its durations, the
    first rows of its table, the names of its files."""
    x = EXAMPLE
    r = results or {}
    out = []

    out.append(('<p class="lead"><span lang="en">The whole of a first run,'
                ' click by click, on one example: the %d public HARPS'
                ' spectra of <strong>Gliese 229</strong> (an M1 dwarf with'
                ' planets), from the data folder to the report. The orange'
                ' numbers on each screenshot are explained right under'
                ' it.</span><span lang="fr">Un premier run en entier, clic'
                ' par clic, sur un exemple : les %d spectres HARPS publics de'
                ' <strong>Gliese 229</strong> (une naine M1 qui a des'
                ' planètes), du dossier de données jusqu\'au rapport. Les'
                ' numéros orange de chaque capture sont expliqués juste'
                ' dessous.</span></p>') % (x["spectra"], x["spectra"]))

    out.append(step(1, ("Prepare the data folder",
                        "Préparer le dossier de données"), [
        ("<p>Make a folder for LBL, with a <code>science/</code> folder in"
         " it, and one folder per target holding its spectra. Here, the"
         " HARPS <code>e2ds</code> of three stars, and the blaze files of"
         " their nights in <code>calib/</code>:</p>",
         "<p>Créez un dossier pour LBL, avec dedans un dossier"
         " <code>science/</code>, et un dossier par cible qui contient ses"
         " spectres. Ici, les <code>e2ds</code> HARPS de trois étoiles, et"
         " les fichiers de blaze de leurs nuits dans <code>calib/</code>"
         " :</p>"),
        code("""
lbl-data/
  science/
    Gliese229/     HARPS.ADP.2014-...e2ds_A.fits   (214 files)
    HD189733/      ...                              (191 files)
    GL699/         ...                              (12 files)
  calib/           HARPS.2003-12-12T...blaze_A.fits
"""),
        callout("tip",
                "<p>No spectra yet? The window can fetch public ones for you:"
                " see <a href=\"#step-11\">step 11</a>. Keep one instrument"
                " per target folder.</p>",
                "<p>Pas encore de spectres ? La fenêtre peut récupérer des"
                " spectres publics pour vous : voir"
                " <a href=\"#step-11\">l'étape 11</a>. Gardez un seul"
                " instrument par dossier de cible.</p>"),
    ]))

    out.append(step(2, ("Start the window", "Lancer la fenêtre"), [
        ("<p>In a terminal, activate the environment and start the window on"
         " that folder:</p>",
         "<p>Dans un terminal, activez l'environnement et lancez la fenêtre"
         " sur ce dossier :</p>"),
        code("""
conda activate lbl-pca2d
lbl-gui --data-dir ~/lbl-data
"""),
        callout("see",
                "<p>One line in the terminal, which then waits (leave it"
                " open):</p>" + code("lbl-gui 0.1.0 at http://127.0.0.1:8765/"
                                     " (LBL from .../lbl-pca2d/lbl); Ctrl+C"
                                     " to stop")
                + "<p>and the window opens in your browser. If it does not,"
                " open <code>http://127.0.0.1:8765/</code> yourself.</p>",
                "<p>Une ligne dans le terminal, qui attend ensuite (laissez-le"
                " ouvert) :</p>" + code("lbl-gui 0.1.0 at"
                                        " http://127.0.0.1:8765/ (LBL from"
                                        " .../lbl-pca2d/lbl); Ctrl+C to stop")
                + "<p>et la fenêtre s'ouvre dans votre navigateur. Sinon,"
                " ouvrez vous-même <code>http://127.0.0.1:8765/</code>.</p>"),
    ]))

    out.append(step(3, ("Read the targets", "Lire les cibles"), [
        shot("01_start", "The window as it opens.",
             "La fenêtre à l'ouverture."),
        marks(["<strong>Data folder.</strong> The one given with"
               " <code>--data-dir</code>; paste another path here to change."
               " It is the folder that <em>holds</em> <code>science/</code>.",
               "<strong>Read the targets</strong>: reads the headers again"
               " (only new or changed files).",
               "<strong>The tabs</strong>, left to right, are the order of"
               " the work.",
               "<strong>The targets</strong>: one row per folder. A greyed"
               " row is still being read; wait until none is."],
              ["<strong>Dossier de données.</strong> Celui donné avec"
               " <code>--data-dir</code> ; collez un autre chemin ici pour en"
               " changer. C'est le dossier qui <em>contient</em>"
               " <code>science/</code>.",
               "<strong>Lire les cibles</strong> : relit les en-têtes (seuls"
               " les fichiers nouveaux ou modifiés).",
               "<strong>Les onglets</strong>, de gauche à droite, suivent"
               " l'ordre du travail.",
               "<strong>Les cibles</strong> : une ligne par dossier. Une"
               " ligne grisée est en cours de lecture ; attendez qu'il n'y en"
               " ait plus."]),
        callout("see",
                "<p>Under the field: <code>%d targets, %d spectra</code>.</p>"
                % (x["folder_targets"], x["folder_spectra"]),
                "<p>Sous le champ : <code>%d cibles, %d spectres</code>.</p>"
                % (x["folder_targets"], x["folder_spectra"])),
    ]))

    out.append(step(4, ("Tick the target", "Cocher la cible"), [
        shot("02_ticked", "Gliese 229 ticked.", "Gliese 229 cochée."),
        marks(["<strong>Tick</strong> the target to run. Several targets of"
               " one instrument can be ticked (a joint fit); two instruments"
               " cannot. The instrument and the data source are read from"
               " its files (you will check them in the next tab).",
               "<strong>The banner</strong> now describes the ticked"
               " targets only: their spectra and dates, their nights and"
               " median SNR, the range of their BERV."],
              ["<strong>Cochez</strong> la cible à traiter. On peut cocher"
               " plusieurs cibles d'un même instrument (un fit joint), pas"
               " deux instruments. L'instrument et la source de données sont"
               " lus dans ses fichiers (vous les vérifierez à l'onglet"
               " suivant).",
               "<strong>Le bandeau</strong> décrit maintenant les cibles"
               " cochées seulement : leurs spectres et leurs dates, leurs"
               " nuits et leur SNR médian, l'étendue de leur BERV."]),
        callout("example",
                "<p>1 target, %d spectra from %s, %d nights, median SNR %d,"
                " BERV from %s km/s: %.1f km/s of coverage.</p>"
                % (x["spectra"], x["dates"], x["nights"], x["snr"],
                   x["berv"], x["span"]),
                "<p>1 cible, %d spectres du %s, %d nuits, SNR médian %d, BERV"
                " de %s km/s : %.1f km/s de couverture.</p>"
                % (x["spectra"], x["dates"], x["nights"], x["snr"],
                   x["berv"], x["span"])),
        callout("why",
                "<p>PCA2D tells the star from the atmosphere because the star"
                " moves on the detector with the BERV while the atmosphere"
                " stays: the wider the BERV range, the better they"
                " separate.</p>",
                "<p>PCA2D distingue l'étoile de l'atmosphère parce que"
                " l'étoile bouge sur le détecteur avec le BERV alors que"
                " l'atmosphère reste fixe : plus l'étendue du BERV est grande,"
                " mieux elles se séparent.</p>"),
    ]))

    out.append(step(5, ("Look at the plots", "Regarder les graphiques"), [
        shot("03_plots", "The plots beside the table, for the ticked target.",
             "Les graphiques à côté de la table, pour la cible cochée.",
             narrow=True),
        marks(["<strong>BERV coverage</strong>: how many spectra per 3 km/s"
               " of BERV.",
               "<strong>The star's velocity in the observer's frame</strong>"
               " (v_tot = v_sys - BERV): where its lines fall against the"
               " telluric lines.",
               "<strong>SNR against BERV</strong>: one dot per spectrum.",
               "<strong>Date bounds</strong> (RJD): restrict the PCA2D fit"
               " to a stretch of time. Leave them empty in the example."],
              ["<strong>Couverture en BERV</strong> : combien de spectres par"
               " tranche de 3 km/s de BERV.",
               "<strong>Vitesse de l'étoile dans le référentiel de"
               " l'observateur</strong> (v_tot = v_sys - BERV) : là où ses"
               " raies tombent par rapport aux raies telluriques.",
               "<strong>SNR en fonction du BERV</strong> : un point par"
               " spectre.",
               "<strong>Bornes de dates</strong> (RJD) : limitent le fit"
               " PCA2D à une période. Laissez-les vides dans l'exemple."]),
    ]))

    out.append(step(6, ("LBL settings", "Réglages LBL"), [
        shot("04_lbl", "The LBL settings tab.", "L'onglet Réglages LBL."),
        marks(["<strong>Instrument and data source</strong>: detected from"
               " the ticked files, the green line says so (here HARPS ·"
               " ORIG). You can choose others in the two menus above it (a"
               " NIRPS mode, a source the files do not tell): the line then"
               " turns orange, says what the files say, and <em>Use the"
               " detection</em> brings it back.",
               "<strong>Python</strong> that runs the wraps: the one of the"
               " window by default; type another only to run in another"
               " environment.",
               "<strong>Template</strong>: what each exposure is measured"
               " against. Keep <em>its own</em> for a star with many"
               " spectra; for a star with few, choose another star's of a"
               " similar type.",
               "<strong>Teff</strong>: read from the headers (grey), else"
               " asked of SIMBAD by the target's name. What SIMBAD says is"
               " written under the field (the median of the Teff it holds),"
               " with <em>use it</em> when the headers say another. Type a"
               " value only to override it.",
               "<strong>Rewrite the files already on disk</strong>: leave it"
               " unticked the first time. Tick it when you run again after"
               " changing the spectra, the template or the mask.",
               "<strong>The steps</strong>: all ticked by default. For HARPS"
               " the <em>telluric cleaning</em> is on, in four steps that"
               " the run's list shows: a first pass blind, a template of"
               " its cleaned spectra, a second pass of the spectra as"
               " delivered divided by that template, and the template made"
               " again (<a href=\"#tellurics\">why twice</a>). For NIRPS and SPIRou"
               " it is off and locked (their pipeline corrects the"
               " tellurics)."],
              ["<strong>Instrument et source de données</strong> : détectés"
               " d'après les fichiers cochés, la ligne verte le dit (ici"
               " HARPS · ORIG). Vous pouvez en choisir d'autres dans les deux"
               " menus au-dessus (un mode de NIRPS, une source que les"
               " fichiers ne disent pas) : la ligne devient orange, dit ce"
               " que disent les fichiers, et <em>Revenir à la détection</em>"
               " la remet.",
               "<strong>Python</strong> qui lance les wraps : celui de la"
               " fenêtre par défaut ; n'en tapez un autre que pour tourner"
               " dans un autre environnement.",
               "<strong>Template</strong> : ce contre quoi chaque pose est"
               " mesurée. Gardez <em>le sien</em> pour une étoile qui a"
               " beaucoup de spectres ; pour une étoile qui en a peu,"
               " choisissez celui d'une autre étoile de type voisin.",
               "<strong>Teff</strong> : lue dans les en-têtes (en gris),"
               " sinon demandée à SIMBAD par le nom de la cible. Ce que dit"
               " SIMBAD est écrit sous le champ (la médiane des Teff qu'il"
               " a), avec <em>la prendre</em> quand les en-têtes en disent"
               " une autre. Tapez une valeur seulement pour la remplacer.",
               "<strong>Réécrire les fichiers déjà sur le disque</strong> :"
               " laissez-la décochée la première fois. Cochez-la quand vous"
               " relancez après avoir changé les spectres, le template ou le"
               " masque.",
               "<strong>Les étapes</strong> : toutes cochées par défaut. Pour"
               " HARPS, le <em>nettoyage tellurique</em> est actif, en quatre"
               " étapes que montre la liste du run : un premier passage à"
               " l'aveugle, un template de ses spectres nettoyés, un second"
               " passage des spectres livrés divisés par ce template, et le"
               " template refait (<a href=\"#tellurics\">pourquoi deux"
               " fois</a>)."
               " Pour NIRPS et SPIRou il est désactivé et bloqué (leur"
               " pipeline corrige déjà les telluriques)."]),
        callout("example",
                "<p>HARPS · ORIG detected, template <em>its own (%d K)</em>,"
                " Teff %d K from the headers, nothing else changed.</p>"
                % (x["teff"], x["teff"]),
                "<p>HARPS · ORIG détectés, template <em>le sien (%d K)</em>,"
                " Teff %d K lue dans les en-têtes, rien d'autre de changé.</p>"
                % (x["teff"], x["teff"])),
    ]))

    out.append(step(7, ("PCA2D settings", "Réglages PCA2D"), [
        shot("05_pca2d", "The PCA2D settings tab.",
             "L'onglet Réglages PCA2D."),
        marks(["<strong>PCA2D on</strong> (ticked by default). Unticked, the"
               " run is LBL alone, on the delivered spectra.",
               "<strong>A correction already on disk</strong>: redone only if"
               " the request changed (the default). Leave it.",
               "<strong>Observer components</strong> (n_earth): how many"
               " patterns fixed on the detector are fitted. 3 by default;"
               " leave it for a first run."],
              ["<strong>PCA2D actif</strong> (coché par défaut). Décoché, le"
               " run est LBL seul, sur les spectres livrés.",
               "<strong>Une correction déjà sur le disque</strong> : refaite"
               " seulement si la demande a changé (le défaut). Laissez-le.",
               "<strong>Composantes de l'observateur</strong> (n_earth) :"
               " combien de motifs fixes sur le détecteur sont ajustés. 3 par"
               " défaut ; gardez-le pour un premier run."]),
        ("<p>Every option has a (i) that says what it does, and its default"
         " under its name. The reference at the end of this page lists them"
         " all.</p>",
         "<p>Chaque option a un (i) qui dit ce qu'elle fait, et son défaut"
         " sous son nom. La référence à la fin de cette page les décrit"
         " toutes.</p>"),
        shot("06_pca2d_name", "A name for the variant, typed in.",
             "Un nom de variante, tapé."),
        marks(["<strong>Variant name</strong>: type a short name"
               " (here <code>demo</code>). It names the run and the corrected"
               " spectra, so that two variants never overwrite each other.",
               "<strong>&#8634;</strong> puts the option back to its"
               " default."],
              ["<strong>Nom de la variante</strong> : tapez un nom court (ici"
               " <code>demo</code>). Il nomme le run et les spectres"
               " corrigés, pour que deux variantes ne s'écrasent jamais.",
               "<strong>&#8634;</strong> remet l'option à son défaut."]),
        callout("see",
                "<p>A changed option is highlighted in blue, and it is the"
                " only kind of option written into the wrap.</p>",
                "<p>Une option changée est surlignée en bleu, et seules les"
                " options changées sont écrites dans le wrap.</p>"),
    ]))

    out.append(step(8, ("Check and run", "Vérifier et lancer"), [
        shot("07_run", "The Run tab before the run.",
             "L'onglet Lancer avant le run."),
        marks(["<strong>Run name</strong>: made from the target and the"
               " settings (<code>Gliese229_s1-o3_demo</code>: 1 star"
               " component, 3 observer components, variant demo). Type"
               " another if you like; <em>auto</em> brings it back.",
               "<strong>Run</strong>: starts the run.",
               "<strong>The wrap</strong>: the Python script that will run,"
               " every setting explained above it. You can read it, and edit"
               " it if you know what you are doing.",
               "<strong>Save</strong>: keeps it as"
               " <code>wrap_&lt;name&gt;.py</code> in the data folder, to run"
               " later without the window."],
              ["<strong>Nom du run</strong> : fait de la cible et des"
               " réglages (<code>Gliese229_s1-o3_demo</code> : 1 composante de"
               " l'étoile, 3 de l'observateur, variante demo). Tapez-en un"
               " autre si vous voulez ; <em>auto</em> le remet.",
               "<strong>Lancer</strong> : démarre le run.",
               "<strong>Le wrap</strong> : le script Python qui va tourner,"
               " chaque réglage expliqué au-dessus. Vous pouvez le lire, et"
               " le modifier si vous savez ce que vous faites.",
               "<strong>Enregistrer</strong> : le garde en"
               " <code>wrap_&lt;nom&gt;.py</code> dans le dossier de données,"
               " pour le relancer plus tard sans la fenêtre."]),
        callout("tip",
                "<p>If something is wrong (a missing template, two"
                " instruments ticked, a data source that does not fit the"
                " files) the window says so in red above the wrap and refuses"
                " to run. Fix what it says and click again.</p>",
                "<p>Si quelque chose ne va pas (un template manquant, deux"
                " instruments cochés, une source de données qui ne correspond"
                " pas aux fichiers), la fenêtre le dit en rouge au-dessus du"
                " wrap et refuse de lancer. Corrigez ce qu'elle dit et"
                " recliquez.</p>"),
    ]))

    durations_en = r.get("durations_en", "")
    durations_fr = r.get("durations_fr", "")
    out.append(step(9, ("Follow the run", "Suivre le run"), [
        shot("08b_button", "The button while the run goes.",
             "Le bouton pendant le run."),
        marks(["The button turns orange and says which step is running. A"
               " click now queues another run, after this one."],
              ["Le bouton devient orange et dit quelle étape tourne. Un clic"
               " maintenant met un autre run en file, après celui-ci."]),
        shot("08_running", "The steps and the log, during the run.",
             "Les étapes et le journal, pendant le run."),
        marks(["<strong>The steps</strong>, ticked as they end (a spinner on"
               " the one running). One line per object: the delivered"
               " spectra (here cleaned of tellurics: <code>_tc</code>), then"
               " the corrected ones (<code>_tc_pca2d_demo</code>). Click a"
               " step to jump to its place in the log.",
               "<strong>The log</strong>: everything LBL and PCA2D print.",
               "<strong>&uarr;</strong> to the start of the log (it then stops"
               " following the run).",
               "<strong>&darr;</strong> to the end (it follows again).",
               "<strong>Stop</strong> stops the run."],
              ["<strong>Les étapes</strong>, cochées à mesure qu'elles"
               " finissent (une roue sur celle qui tourne). Une ligne par"
               " objet : les spectres livrés (ici nettoyés des telluriques :"
               " <code>_tc</code>), puis les corrigés"
               " (<code>_tc_pca2d_demo</code>). Cliquez sur une étape pour"
               " aller à sa place dans le journal.",
               "<strong>Le journal</strong> : tout ce qu'impriment LBL et"
               " PCA2D.",
               "<strong>&uarr;</strong> vers le début du journal (il cesse"
               " alors de suivre le run).",
               "<strong>&darr;</strong> vers la fin (il le suit de nouveau).",
               "<strong>Arrêter</strong> arrête le run."]),
        callout("example", durations_en or "<p>-</p>",
                durations_fr or "<p>-</p>"),
        callout("tip",
                "<p>You can close the browser, and even the window (Ctrl+C):"
                " the run goes on. Start <code>lbl-gui</code> again on the"
                " same folder and it finds the run where it is.</p>",
                "<p>Vous pouvez fermer le navigateur, et même la fenêtre"
                " (Ctrl+C) : le run continue. Relancez <code>lbl-gui</code>"
                " sur le même dossier et elle retrouve le run où il en"
                " est.</p>"),
    ]))

    results_en = r.get("results_en", "")
    results_fr = r.get("results_fr", "")
    out.append(step(10, ("Read the results", "Lire les résultats"), [
        shot("10_runs", "The Runs tab after the run.",
             "L'onglet Runs après le run."),
        marks(["<strong>The run</strong>, its state, when it started and how"
               " long it took.",
               "<strong>The report</strong>: click to open the PDF."],
              ["<strong>Le run</strong>, son état, son début et sa durée.",
               "<strong>Le rapport</strong> : cliquez pour ouvrir le"
               " PDF."]),
        results_en and (results_en, results_fr) or "",
    ] + report_part(r.get("report_pages") or [])))

    out.append(step(11, ("Optional: fetch public spectra",
                         "Facultatif : récupérer des spectres publics"), [
        shot("09_archive", "The archive search, for GJ 1 with HARPS.",
             "La recherche d'archive, pour GJ 1 avec HARPS."),
        ("<p>In the <em>Targets</em> tab, sub-tab <em>Archive search</em>"
         " (koloa must be installed):</p>",
         "<p>Dans l'onglet <em>Cibles</em>, sous-onglet <em>Recherche"
         " d'archive</em> (koloa doit être installé) :</p>"),
        marks(["<strong>The star's name</strong>, as SIMBAD knows it (here"
               " <code>GJ 1</code>, which SIMBAD calls HD 225213).",
               "<strong>The instruments</strong> to ask (here HARPS only).",
               "<strong>Search the archives.</strong>",
               "<strong>What was found</strong>: per instrument, how many"
               " spectra, the LBL reader that will read them, the archive,"
               " how many are already here, how much to download, and the"
               " folder they will go to. Tick the rows to fetch.",
               "<strong>Fetch</strong>: downloads them into"
               " <code>science/</code>, four files at a time, a bar per"
               " folder with how far it is and its time left, without ever"
               " moving or overwriting a file. The target then appears in"
               " <em>My targets</em>."],
              ["<strong>Le nom de l'étoile</strong>, tel que SIMBAD le"
               " connaît (ici <code>GJ 1</code>, que SIMBAD appelle HD"
               " 225213).",
               "<strong>Les instruments</strong> à interroger (ici HARPS"
               " seulement).",
               "<strong>Chercher dans les archives.</strong>",
               "<strong>Ce qui a été trouvé</strong> : par instrument,"
               " combien de spectres, le lecteur LBL qui les lira, l'archive,"
               " combien sont déjà là, combien télécharger, et le dossier où"
               " ils iront. Cochez les lignes à récupérer.",
               "<strong>Récupérer</strong> : les télécharge dans"
               " <code>science/</code>, quatre fichiers à la fois, une barre"
               " par dossier avec où il en est et le temps qui lui reste,"
               " sans jamais déplacer ni écraser un fichier. La cible apparaît ensuite"
               " dans <em>Mes cibles</em>."]),
        callout("tip",
                "<p>While a search goes, <em>Stop the search</em> takes the"
                " place of <em>Search</em>: an archive that does not answer"
                " no longer holds the others up. Under the results, <em>What"
                " the search and the fetch are doing</em> says who was asked,"
                " what each archive answered and in how long, and why a file"
                " was not fetched.</p>",
                "<p>Pendant une recherche, <em>Arrêter la recherche</em> prend"
                " la place de <em>Chercher</em> : une archive qui ne répond"
                " pas ne retient plus les autres. Sous les résultats, <em>Ce"
                " que font la recherche et le téléchargement</em> dit qui a"
                " été interrogé, ce que chaque archive a répondu et en combien"
                " de temps, et pourquoi un fichier n'a pas été récupéré.</p>"),
    ]))

    out.append(step(12, ("Run again, differently", "Relancer, autrement"), [
        ("<ul class=\"feature-list\">"
         "<li><strong>A variant</strong>: change an option (for example"
         " n_earth to 5), give it another name (<code>demo5</code>), run."
         " The first one is kept beside it; the report says which is"
         " which.</li>"
         "<li><strong>More spectra arrived</strong>: put them in the target's"
         " folder, <em>Refresh</em>, tick <em>rewrite the files already on"
         " disk</em>, run.</li>"
         "<li><strong>From a terminal</strong>: every run kept its wrap,"
         " <code>python DATA_DIR/gui/runs/&lt;run&gt;/wrap.py</code>.</li>"
         "<li><strong>Disk space</strong>: the <em>Cleanup</em> tab deletes"
         " what can be made again (cubes, corrected spectra, per-exposure"
         " velocities), never the spectra nor the results.</li></ul>",
         "<ul class=\"feature-list\">"
         "<li><strong>Une variante</strong> : changez une option (par exemple"
         " n_earth à 5), donnez-lui un autre nom (<code>demo5</code>),"
         " lancez. La première est gardée à côté ; le rapport dit laquelle"
         " est laquelle.</li>"
         "<li><strong>De nouveaux spectres sont arrivés</strong> :"
         " mettez-les dans le dossier de la cible, <em>Rafraîchir</em>,"
         " cochez <em>réécrire les fichiers déjà sur le disque</em>,"
         " lancez.</li>"
         "<li><strong>Depuis un terminal</strong> : chaque run a gardé son"
         " wrap, <code>python DATA_DIR/gui/runs/&lt;run&gt;/wrap.py</code>."
         "</li>"
         "<li><strong>L'espace disque</strong> : l'onglet <em>Ménage</em>"
         " supprime ce qui peut être refait (cubes, spectres corrigés,"
         " vitesses par pose), jamais les spectres ni les résultats.</li>"
         "</ul>"),
    ]))
    return "".join(out)
