# Étoiles de base proposées pour les bases PCA2D figées

SPIRou et NIRPS (mode HE), trois bins de SNR, 4 étoiles par bin. Données :
toutes les t.fits d'APERO sur rali (`spirou_offline`, `nirps_he_offline`),
111 427 poses lues dans leurs en-têtes le 2026-11-11 (`survey_rali.py`),
classées par `select_baseline.py`.

## Critères

- **SNR** d'une pose : la médiane de ses `EXTSNxxx` (ce que montre lbl_gui) ;
  le bin d'une étoile : la médiane sur ses poses (20-40, 40-80, >80).
- **Couverture** : étendue du BERV > 30 km/s, sinon l'étoile est écartée.
- **Uniformité** : Gini de l'histogramme du BERV (cases de 2 km/s, du plus
  bas au plus haut BERV de l'étoile) : 0 uniforme, vers 1 concentré.
- **Classement** : score = N poses x (1 - Gini).
- Écartées : les étoiles chaudes (Teff >= 7000 K ou type O, B, A selon la base
  astrométrique d'APERO, les standards telluriques), et les objets de moins de
  20 t.fits.

## La liste (classement demandé)

| Instr. | SNR | Étoiles (N poses / nuits, étendue km/s, Gini) |
|---|---|---|
| NIRPS HE | 20-40 | WASP-127 (431/7, 36, 0.67) · WASP-107 (383/12, 52, 0.66) · TOI-782 (201/100, 58, 0.41) · TOI-5524 (198/100, 59, 0.44) |
| NIRPS HE | 40-80 | GJ 3090 (379/99, 37, 0.63) · TOI-667 (165/83, 37, 0.42) · TOI-4565 (166/65, 49, 0.44) · TOI-1078 (171/87, 48, 0.51) |
| NIRPS HE | >80 | Gl 699 (882/172, 53, 0.34) · Proxima (782/259, 43, 0.43) · eps Eri (729/73, 43, 0.51) · GJ 682 (560/143, 56, 0.49) |
| SPIRou | 20-40 | TOI-2120 (321/82, 38, 0.44) · TOI-736 (255/47, 56, 0.53) · TOI-4616 (243/61, 58, 0.52) · TOI-4479 (242/63, 46, 0.63) |
| SPIRou | 40-80 | GJ 905 (1968/476, 45, 0.38) · GJ 1289 (1333/315, 49, 0.39) · GJ 1286 (1263/314, 60, 0.36) · PM J21463+3813 (1213/299, 41, 0.34) |
| SPIRou | >80 | Gl 699 (2136/506, 53, 0.40) · Gl 846 (1851/455, 59, 0.42) · AU Mic (1859/422, 58, 0.46) · V1298 Tau (1258/314, 60, 0.25) |

Étoiles admissibles par bin : NIRPS 14, 26, 41 ; SPIRou 9, 76, 42.
Le détail de chacune (et pourquoi une étoile est écartée) :
`candidates_<INSTRUMENT>.csv` ; les histogrammes de BERV : `berv_*.pdf`.

## Suggestions

1. **Compter les nuits plutôt que les poses.** Un fond s'apprend sur des
   conditions différentes, et les lignes du fit sont souvent des nuits
   (`nightly_stack`). Le cas qui le montre : WASP-127 et WASP-107 tiennent
   leurs ~400 poses en 7 et 12 nuits (des transits). Classé sur
   nuits x (1 - Gini), NIRPS 20-40 devient **TOI-782, TOI-5524, TOI-4508,
   TOI-5155** (100, 100, 87, 47 nuits), des naines K7 à M (TOI-4508 sans
   Teff : son dossier APERO s'appelle `TOIM4508`, la base le connaît sous
   `TOI4508`). TOI-5524 est déjà dans la base NIRPS `2026a`.
2. **SPIRou >80 : remplacer AU Mic et V1298 Tau**, deux étoiles jeunes et très
   actives, dont la variabilité dans le référentiel de l'étoile n'est pas
   modélisée avec `n_star=1`. Remplaçantes calmes : **GJ 411** (393 nuits) et
   **Gl 880** (241 nuits) ou **Gl 15 A**.
3. **NIRPS >80 : eps Eri** est une K2 (pour une base de naines M) avec seulement
   73 nuits (10 poses par nuit). Par les nuits : **GJ 1** (M2V, 146 nuits)
   ou **GJ 3618** (M5.5V, 132 nuits, mais une étendue de 30,05 km/s, juste
   au-dessus du seuil).
4. **NIRPS 40-80** : par les nuits, **SMETHELLS 20** (130) et **GJ 3622** (114)
   passent en tête ; GJ 3622 est déjà dans la base NIRPS `2026a`.
5. **SPIRou 20-40** : seules 4 étoiles ont plus de 45 nuits ; la liste est
   celle de la base jointe déjà faite (TOI-2120, TOI-4616, TOI-4479) plus
   TOI-736 : cohérent.
6. Le Gini est pris sur l'étendue de chaque étoile : 36 km/s uniformes valent
   60 km/s uniformes. Pour récompenser la largeur, le prendre sur une plage
   fixe (-30 à +30 km/s).
7. Non vérifié : binaires (SB2), grands K planétaires, et NIRPS HA (non fait,
   même script). La Teff et le type viennent de la base d'APERO.

## Refaire

Sur rali, dans `~/proposed_basline_stars` (copie des scripts et de la base
astrométrique d'APERO ; rien n'est écrit à côté des données) :

    nohup ~/miniconda3/envs/lbl-pca2d/bin/python survey_rali.py > exposures.csv 2> survey.log &
    ~/miniconda3/envs/lbl-pca2d/bin/python select_baseline.py exposures.csv
