"""The diagnostic tests from the night of 2026-10-17: what decides WHY a
correction costs on a bright target, and not what corrects.

Three questions, each an option disabled by default, hence with no effect
on the nominal run:

- twoframe.holdout (cross-validation): the nights of a given rank are
  held out of the fit, then corrected like the others by the refit of the
  amplitudes. The difference with a correction made "in sample" measures
  what the basis learns from the noise of the exposures it corrects.
- correct.permute_b (null test): each night receives the b amplitudes of
  ANOTHER night. Same basis, same distribution of amplitudes, no relation
  to the exposure. If the velocities come out as good as with the real
  correction, that correction brings nothing on this target.
- correct.q_support (telluric support): the divided basis is kept only
  where the telluric transmission of the cube drops below a threshold, on
  any exposure. If the injected noise comes from the clean columns, that
  is where it disappears.

Pure functions, tested in tests/test_diagnostics.py.
"""

from __future__ import annotations

import numpy as np


def night_of(bjd):
    """The night of an exposure: floor(BJD - 0.5), the cube's convention."""
    return np.floor(np.asarray(bjd, dtype=float) - 0.5).astype(int)


def holdout_rows(bjd, which):
    """The rows to hold out of the fit: the nights of even or odd rank.

    The rank is that of the night among the distinct nights, by increasing
    date, so that the two halves cover the whole campaign and the whole
    BERV range instead of cutting out one season. `which` is "even" or
    "odd": the nights HELD OUT, not those of the fit. All the rows of a
    night (the two order parities, or the exposures of an unstacked night)
    share its fate.
    """
    if which not in ("even", "odd"):
        raise ValueError("holdout is 'even' or 'odd', not %r" % (which,))
    nights = night_of(bjd)
    _, rank = np.unique(nights, return_inverse=True)
    return (rank % 2) == (0 if which == "even" else 1)


def night_derangement(nights, seed):
    """For each exposure, the index of an exposure from ANOTHER night.

    The distinct nights are permuted by Sattolo's algorithm, which returns
    a cyclic permutation: no night keeps its own. Each exposure then
    receives the amplitudes of an exposure drawn at random from the night
    assigned to it. Two exposures of the same night therefore cannot swap
    theirs, which would be almost an identity.
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
