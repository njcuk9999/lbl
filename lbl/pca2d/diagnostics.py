"""Les tests de diagnostic de la nuit du 2026-10-17: ce qui decide POURQUOI
une correction coute sur une cible brillante, et non ce qui corrige.

Trois questions, chacune une option desactivee par defaut, donc sans effet
sur le nominal:

- twoframe.holdout (validation croisee): les nuits d'un rang donne sont
  tenues hors du fit, puis corrigees comme les autres par le reajustement des
  amplitudes. L'ecart avec une correction faite "dans l'echantillon" mesure
  ce que la base apprend du bruit des poses qu'elle corrige.
- correct.permute_b (test nul): chaque nuit recoit les amplitudes b d'une
  AUTRE nuit. Meme base, meme distribution d'amplitudes, aucun rapport avec la
  pose. Si les vitesses sortent aussi bonnes qu'avec la vraie correction,
  celle-ci n'apporte rien sur cette cible.
- correct.q_support (support tellurique): la base divisee n'est gardee que
  la ou la transmission tellurique du cube descend sous un seuil, sur
  n'importe quelle pose. Si le bruit injecte vient des colonnes propres,
  c'est la qu'il disparait.

Fonctions pures, testees dans tests/test_diagnostics.py.
"""

from __future__ import annotations

import numpy as np


def night_of(bjd):
    """La nuit d'une pose: floor(BJD - 0.5), la convention du cube."""
    return np.floor(np.asarray(bjd, dtype=float) - 0.5).astype(int)


def holdout_rows(bjd, which):
    """Les lignes a tenir hors du fit: les nuits de rang pair ou impair.

    Le rang est celui de la nuit parmi les nuits distinctes, par date
    croissante, de sorte que les deux moities couvrent toute la campagne et
    toute la plage de BERV au lieu d'en couper une saison. `which` vaut
    "even" ou "odd": les nuits TENUES A L'ECART, pas celles du fit. Toutes les
    lignes d'une nuit (les deux parites d'ordre, ou les poses d'une nuit non
    empilee) partagent son sort.
    """
    if which not in ("even", "odd"):
        raise ValueError("holdout is 'even' or 'odd', not %r" % (which,))
    nights = night_of(bjd)
    _, rank = np.unique(nights, return_inverse=True)
    return (rank % 2) == (0 if which == "even" else 1)


def night_derangement(nights, seed):
    """Pour chaque pose, l'indice d'une pose d'une AUTRE nuit.

    Les nuits distinctes sont permutees par l'algorithme de Sattolo, qui rend
    une permutation cyclique: aucune nuit ne garde la sienne. Chaque pose
    recoit ensuite les amplitudes d'une pose tiree au hasard dans la nuit qui
    lui est assignee. Deux poses d'une meme nuit ne peuvent donc pas
    s'echanger les leurs, ce qui serait presque une identite.
    """
    nights = np.asarray(nights)
    unique = np.unique(nights)
    if unique.size < 2:
        raise ValueError("a permutation between nights needs two nights")
    rng = np.random.default_rng(int(seed))
    order = list(range(unique.size))
    for i in range(unique.size - 1, 0, -1):      # Sattolo: j < i, strictement
        j = int(rng.integers(0, i))
        order[i], order[j] = order[j], order[i]
    target = dict(zip(unique.tolist(), unique[order].tolist()))
    members = {n: np.flatnonzero(nights == n) for n in unique.tolist()}
    return np.array([int(rng.choice(members[target[n]])) for n in nights.tolist()])


def support_columns(trans, threshold):
    """Les colonnes ou l'atmosphere absorbe: transmission sous `threshold`
    sur au moins une ligne du cube.

    Une colonne sans aucune transmission connue (NaN partout) n'est pas du
    support: on n'y sait rien de l'atmosphere, et la base n'y est pas divisee.
    """
    trans = np.asarray(trans)
    if trans.ndim != 2:
        raise ValueError("trans is (rows, columns), not %r" % (trans.shape,))
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("a transmission threshold is in (0, 1], not %r"
                         % (threshold,))
    with np.errstate(invalid="ignore"):
        lowest = np.full(trans.shape[1], np.inf)
        # par blocs de lignes: le cube est un memmap de plusieurs Go
        for start in range(0, trans.shape[0], 64):
            block = np.asarray(trans[start:start + 64], dtype=np.float32)
            block = np.where(np.isfinite(block), block, np.inf)
            lowest = np.minimum(lowest, block.min(axis=0))
    return lowest < float(threshold)
