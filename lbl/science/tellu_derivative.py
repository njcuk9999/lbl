"""L'exposant H2O par la DERIVEE, la methode de telluric_fit.

Reprise de papier_helium/referee/tellu_derivative.py et de
telluric_fit/tellu_tools.py:2413 (optimize_exponents, appelee par
predict_abso.py), transposee du WINERED au visible.

LE PRINCIPE. On divise le spectre par la transmission modelisee et on retient
l'exposant qui rend le spectre corrige le plus LISSE, c'est-a-dire qui
minimise la dispersion ponderee du gradient pixel a pixel. Un exposant faux
laisse a chaque raie tellurique un residu en forme de DERIVEE, qui gonfle
cette dispersion; les raies stellaires y contribuent la meme chose quel que
soit l'exposant, donc le critere ne depend que des telluriques, meme la ou
elles sont minoritaires. C'est ce qui le rend stable la ou l'extrapolation
polynomiale de LBL s'effondre sur ses bornes.

LA FORMULATION EST CELLE DU CODE QUI MARCHE, et deux details y sont
essentiels (mesures le 2026-10-15, une premiere version qui les ignorait
tapait la borne sur 35% des poses):

    la PROFONDEUR OPTIQUE, tau = -ln(T), portee sur une grille log UNIFORME,
    puis exponentiee: tau se multiplie par l'exposant, ce qui est lineaire,
    la ou interpoler des transmissions deja exponentiees sur la grille
    irreguliere du spectre melange les deux operations;

    la CONVOLUTION a la resolution de l'instrument, APRES la puissance et
    AVANT la division. TAPAS est bien plus fin que le spectrographe (2.6
    km/s sur HARPS), et des raies trop etroites ne peuvent pas annuler des
    raies elargies.

CE QUI EST LIBRE, pour l'optique (consigne du 2026-10-15):

    H2O             libre PAR FICHIER: la colonne d'eau change trop vite pour
                    un modele par nuit. Et il y a TOUJOURS de l'eau: un
                    exposant sur la borne basse est un echec d'ajustement,
                    jamais une nuit seche.
    le reste        d'un seul bloc, pas par espece chimique: dans le visible
                    seul O2 compte vraiment parmi les secs, et ils suivent
                    tous la meme masse d'air. C'est aussi le decoupage que
                    TAPAS donne dans LBL (ABSO_WATER et ABSO_OTHERS).
    la vitesse      NON libre: le modele est evalue dans le repere du SOL, ou
                    l'atmosphere se trouve, et non sur la grille
                    barycentrique comme LBL le fait (docs/tapas_frame.md).
"""
import numpy as np

C = 299792.458
FWHM2SIG = 2.3548
#: les poids lisses de telluric_fit.tellu_tools.get_transmission_weights
WEIGHTS = dict(depth_max=0.6, depth_saturated=0.5, reject_saturated=0.8,
               transition_sigma=0.02)
OUTLIER_SIGMA = 6.0
#: le pas de la grille du modele, en km/s
STEP_KMS = 0.25

#: les bandes d'eau du domaine HARPS, mesurees sur TAPAS le 2026-10-15:
#: ailleurs l'eau ne depasse pas 1% de profondeur et l'ajustement se faisait
#: noyer par les raies stellaires
WATER_BANDS_HARPS = ((586.0, 600.0), (644.0, 664.0), (686.0, 696.0))

#: LES BANDES DU SEC: SEULEMENT LA OU IL CONTRAINT QUELQUE CHOSE.
#:
#: Le sec n'est pas l'eau et ne se mesure pas aux memes endroits. Releve sur
#: TAPAS par fenetres de 5 nm le 2026-10-16, les seules fenetres du domaine
#: HARPS ou il depasse 5% de profondeur:
#:
#:     575-585 nm   6%     marginal
#:     625-635 nm   70%    O2 gamma, et l'eau n'y fait que 9%
#:     685-690 nm   100%   bande B, saturee, et l'eau n'y fait que 2.6%
#:     690-695 nm   100%   ECARTEE: l'eau y monte a 77%, les deux especes
#:                         s'y confondent et l'ajustement ne peut plus les
#:                         separer
#:
#: Sous 450 nm il tombe a 0.2% de profondeur maximale, c'est-a-dire rien.
#: L'ajuster sur tout le domaine coute environ cinquante fois plus cher en
#: calcul pour une information qui ne vient que du rouge, et ajoute le bruit
#: de la moitie bleue a une mesure qu'elle ne contraint pas.
DRY_BANDS_HARPS = ((625.0, 635.0), (685.0, 690.0))

#: LA MARQUE DU REPLI. La valeur de repli (0.35) tombe exactement sur un point
#: du balayage, donc une pose qui converge vers 0.35 et une pose qui a echoue
#: rendraient le meme chiffre. On ajoute cet epsilon aux seules valeurs de
#: repli: trois millionnemes de l'exposant, c'est-a-dire rien pour la
#: physique, mais `is_fallback` les reconnait, et un journal aussi.
FALLBACK_MARK = 1e-7


def is_fallback(water, fallback=0.35):
    """Cet exposant vient-il du repli plutot que d'une mesure?"""
    return bool(abs(float(water) - (fallback + FALLBACK_MARK)) < FALLBACK_MARK / 10)


def band_mask(wave, bands):
    """Les pixels qui tombent dans une des bandes."""
    wave = np.asarray(wave, dtype=float)
    out = np.zeros(wave.shape, dtype=bool)
    for low, high in bands:
        out |= (wave > low) & (wave < high)
    return out


def water_mask(wave, bands=WATER_BANDS_HARPS):
    """Les pixels des bandes d'eau, ou l'exposant H2O s'ajuste."""
    wave = np.asarray(wave, dtype=float)
    out = np.zeros(wave.shape, dtype=bool)
    for lo, hi in bands:
        out |= (wave >= lo) & (wave <= hi)
    return out


def model_grid(spl_water, spl_dry, wave_min, wave_max, resolution,
               step_kms=STEP_KMS):
    """Les profondeurs optiques sur une grille log uniforme, et le noyau.

    tau plutot que T, parce que c'est tau qui se multiplie par l'exposant.
    """
    step = np.arctanh(step_kms / C)
    lnw = np.arange(np.log(wave_min), np.log(wave_max), step)
    wave = np.exp(lnw)
    tau_water = -np.log(np.clip(spl_water(wave), 1e-6, 1.0))
    tau_dry = -np.log(np.clip(spl_dry(wave), 1e-6, 1.0))
    sig_pix = np.arctanh(C / resolution / FWHM2SIG / C) / step
    half = int(5 * sig_pix)
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sig_pix) ** 2)
    return dict(wave=wave, tau_water=tau_water, tau_dry=tau_dry,
                kernel=kernel / kernel.sum())


def transmission(model, wave_ground, alpha_water, alpha_dry):
    """T sur `wave_ground`: la puissance d'abord, la convolution ensuite."""
    conv = np.convolve(np.exp(-alpha_water * model["tau_water"]
                              - alpha_dry * model["tau_dry"]),
                       model["kernel"], mode="same")
    return np.interp(wave_ground, model["wave"], conv)


def transmission_weights(trans, depth_max, depth_saturated, reject_saturated,
                         transition_sigma):
    """Les poids 0-1 de telluric_fit: une sigmoide sur la profondeur, et les
    raies saturees et leurs ailes mises a zero."""
    w_depth = 1.0 / (1.0 + np.exp(-(trans - depth_max) / transition_sigma))
    sat = trans < depth_saturated
    above = trans >= reject_saturated
    contam = np.zeros(len(trans), bool)
    for idx in (range(len(trans)), range(len(trans) - 1, -1, -1)):
        inside = False
        for i in idx:
            if sat[i]:
                inside = True
            elif inside and above[i]:
                inside = False
            contam[i] |= inside
    return w_depth * (~contam)


def weighted_nanstd(data, weights):
    """La dispersion ponderee de telluric_fit."""
    ok = np.isfinite(data) & np.isfinite(weights) & (weights > 0)
    d, w = data[ok], weights[ok]
    if d.size < 32:
        return np.inf
    mean = np.sum(w * d) / np.sum(w)
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    if denom <= 0:
        return np.inf
    return np.sqrt(np.sum(w * (d - mean) ** 2) / denom)


def objective(sp, trans):
    """La dispersion ponderee du gradient du spectre corrige."""
    grad = np.gradient(sp / np.clip(trans, 1e-3, None))
    w = transmission_weights(trans, **WEIGHTS)
    valid = w > 0.5
    if valid.sum() < 64:
        return np.inf
    p16, p84 = np.nanpercentile(grad[valid], [16, 84])
    sig = max((p84 - p16) / 2.0, 1e-12)
    dev = np.abs(grad - np.nanmedian(grad[valid])) / sig
    w = w / (1.0 + np.exp(np.clip(dev - OUTLIER_SIGMA, -500, 500)))
    return weighted_nanstd(grad, w)


def fit_exponents(wave_ground, sp, spl_water, spl_dry, airmass,
                  resolution=115000.0, free_dry=True, fallback=0.35,
                  flat_span=0.05):
    """(exposant H2O, exposant sec) de CE fichier, par la derivee.

    L'AJUSTEMENT SE FAIT EN LOGARITHME, et c'est ce qui empeche d'atteindre
    les bornes. Il y a TOUJOURS de l'eau dans l'atmosphere (des dizaines de
    milliers de spectres infrarouges le disent), donc un exposant nul est un
    echec d'ajustement et jamais une nuit seche. En ajustant ln(alpha),
    alpha reste strictement positif quoi que fasse le simplexe, et une
    variation relative compte autant en haut qu'en bas de la gamme, ce qui
    est la bonne metrique pour une colonne qui varie d'un facteur plusieurs.

    Mesure le 2026-10-15 sur 100 poses de HD 10700: en lineaire, une pose
    tombait exactement a zero et neuf restaient sous 0.05; en logarithme,
    aucune.

    `free_dry` est VRAI par defaut depuis le 2026-10-16. Le sec suit bien la
    masse d'air, mais pas avec le coefficient 1 que TAPAS suppose: mesure sur
    les 92 poses du mode HAM de HD 10700, ajuste librement sur les bandes
    seches, il vaut 0.961 fois la masse d'air avec un MAD de 0.026. Ces 4% de
    colonne en moins sont ce qu'une altitude de 2400 m donne, TAPAS etant
    calcule pour une colonne de reference qui n'est pas celle de La Silla.

    Le fixer revenait donc a imposer une erreur systematique de 4% sur toute
    l'absorption seche. Libre, l'ajustement est stable: le residu contre la
    droite ajustee a un MAD de 0.031, et pente et ordonnee se lisent
    separement (0.980 et -0.019 en moindres carres), ce qui est exactement la
    forme attendue d'un offset d'altitude.

    `fallback` et `flat_span` portent le garde-fou sur les poses sans signal,
    decrit au balayage ci-dessous. Il est ICI et non seulement dans
    fit_campaign parce que LBL appelle cette fonction une pose a la fois et
    n'a donc aucune mediane de campagne sous la main.
    """
    from scipy.optimize import minimize

    wave_ground = np.asarray(wave_ground, dtype=float)
    sp = np.asarray(sp, dtype=float)
    good = np.isfinite(sp) & (sp > 0) & np.isfinite(wave_ground)
    wave_ground, sp = wave_ground[good], sp[good]
    model = model_grid(spl_water, spl_dry, wave_ground.min() * 0.999,
                       wave_ground.max() * 1.001, resolution)

    def cost(p):
        water = float(np.exp(p[0]))
        dry = float(np.exp(p[1])) if free_dry else float(airmass)
        if not (1e-4 < water < 20) or not (0 < dry < 20):
            return np.inf
        return objective(sp, transmission(model, wave_ground, water, dry))

    # LE BALAYAGE D'ABORD, et il sert DEUX fois: il dit si la pose a de quoi
    # trancher, et il donne au simplexe un point de depart du bon cote.
    #
    # Mesure le 2026-10-15 sur deux poses de HD 10700 prises la meme nuit:
    #
    #                        alpha=0.005   alpha=2.0   variation   minimum
    #     pose qui marche        340.1       408.0       41%       interieur
    #     pose qui s'effondre     17.76       18.23      2.6%      monotone
    #
    # La seconde n'a pas de minimum du tout: son cout MONTE des la plus
    # petite valeur, donc le simplexe la pousse a zero. Comme il y a toujours
    # de l'eau, c'est un echec d'ajustement, et on rend `fallback` plutot
    # qu'une valeur inventee.
    #
    # LE BALAYAGE DESCEND A 0.002, et ce detail compte. Une version qui
    # s'arretait a 0.02 mettait au repli 19 poses sur 100: les 8 du mode EGGS
    # (un autre repere, voir plus bas), une a SNR 6.6, et ONZE poses
    # parfaitement saines. Ces onze, balayees jusqu'a 0.002, montrent un
    # minimum franc et INTERIEUR, de 0.2 a 1.0, avec 25 a 45% de variation:
    # le garde-fou les cassait au lieu de les mesurer, parce que leur
    # contraste de cout (1.6 a 2.2) est plus faible que celui des poses
    # faciles (2.7) et que le bord du balayage tombait du mauvais cote.
    scan = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6,
                     1.0, 1.6, 2.5])
    # le balayage ne bouge que l'eau; le sec reste a sa valeur de depart, ce
    # qui suffit pour localiser le creux en eau. Les DEUX parametres sont
    # passes quand le sec est libre, sinon cost() lit un p[1] absent (bug vu
    # le 2026-10-16: free_dry=True levait IndexError des le balayage).
    dry0 = np.log(max(float(airmass), 1e-3))
    costs = np.array([cost([np.log(a), dry0]) for a in scan])
    if not np.all(np.isfinite(costs)):
        return float(fallback) + FALLBACK_MARK, float(airmass)
    span = (costs.max() - costs.min()) / max(abs(costs.min()), 1e-30)
    jbest = int(np.argmin(costs))
    flat = span < flat_span
    on_edge = jbest in (0, len(scan) - 1)
    if flat or on_edge:
        # LE REPLI DOIT SE RECONNAITRE. `fallback` vaut 0.35, qui est AUSSI un
        # point du balayage: une pose qui converge vraiment vers 0.35 rend
        # alors le meme chiffre qu'une pose qui a echoue, et ni le journal ni
        # l'en-tete ne permettent de les separer. Vu le 2026-10-16 sur
        # HD 69830, ou j'ai compte 18% de repli alors que plusieurs de ces
        # poses convergeaient franchement vers 0.35 (variation 39 a 42%).
        #
        # On decale donc d'un epsilon invisible physiquement (0.35 contre
        # 0.3500001, soit trois millionnemes de l'exposant) mais qui rend le
        # repli reconnaissable a la lecture.
        return float(fallback) + FALLBACK_MARK, float(airmass)

    x0 = [np.log(scan[jbest]), np.log(max(float(airmass), 1e-3))]
    out = minimize(cost, x0=x0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-10, maxiter=300))
    water = float(np.exp(out.x[0]))
    dry = float(np.exp(out.x[1])) if free_dry else float(airmass)
    if not np.isfinite(water) or water < scan[0] or water > scan[-1]:
        return float(fallback) + FALLBACK_MARK, float(airmass)
    return water, dry


def fit_campaign(frames, spl_water, spl_dry, resolution=115000.0,
                 floor=0.05):
    """Les exposants H2O d'une campagne, avec un garde-fou sur le SNR.

    `frames` est une liste de (wave_ground, sp, airmass). Rend un tableau
    d'exposants, un par pose.

    POURQUOI UN GARDE-FOU. Il y a toujours de l'eau, donc un exposant qui
    s'effondre est un echec. Mesure le 2026-10-15 sur HD 10700: les poses qui
    derivent sont celles dont le SNR est faible, et leur courbe de cout est
    PLATE (17.76 a 17.94 entre alpha 0.005 et 0.8, soit 1% de variation, la
    ou une pose qui marche montre un minimum net a 0.3). Le critere n'a
    simplement pas de quoi trancher: les forcer ne ferait qu'inventer une
    valeur.

    Ces poses recoivent donc la MEDIANE de la campagne, ce qui est la
    meilleure estimation disponible pour elles, et le dit. `floor` est le
    seuil en dessous duquel on considere que l'ajustement n'a pas abouti:
    0.05 sur HARPS, bien au-dessous du 5e centile des poses qui convergent
    (0.15) et bien au-dessus de ce que donnent les echecs (moins de 0.01).
    """
    out = np.full(len(frames), np.nan)
    for i, (wave_ground, sp, airmass) in enumerate(frames):
        try:
            out[i] = fit_exponents(wave_ground, sp, spl_water, spl_dry,
                                   airmass, resolution)[0]
        except Exception:                                      # noqa: BLE001
            out[i] = np.nan
    usable = np.isfinite(out) & (out >= floor)
    if usable.any():
        out[~usable] = float(np.median(out[usable]))
    return out
