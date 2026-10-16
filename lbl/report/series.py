"""La serie d'un run, lue de son .rdb, et ce qu'il faut pour la tracer.

CE MODULE EST LE POINT DE PARTAGE AVEC KOLOA, dans les deux sens: LBL s'en
sert pour son rapport, et koloa peut l'importer pour lire un .rdb de LBL sans
redire le format. Il ne depend que de numpy, pour qu'aucun des deux ne traine
l'autre.
"""
import os

import numpy as np

#: les colonnes que LBL ecrit toujours, et leur role
TIME = 'rjd'
VELOCITY = 'vrad'
ERROR = 'svrad'
#: les exposants telluriques, quand la correction a tourne
WATER = 'TLPEH2O'
DRY = 'TLPEOTR'
#: le nom du fichier d'origine, pour retrouver le spectre
FILENAME = 'FILENAME'


def read_rdb(path):
    """Un .rdb de LBL, en dictionnaire de colonnes.

    Les colonnes numeriques deviennent des tableaux de flottants, les autres
    des tableaux de chaines. Le format rdb met une ligne de tirets sous
    l'en-tete, que l'on saute.
    """
    with open(path, 'r') as handle:
        lines = [line.rstrip('\n') for line in handle if line.strip()]
    if len(lines) < 3:
        raise ValueError('%s holds no rows' % path)
    names = lines[0].split('\t')
    rows = [line.split('\t') for line in lines[2:]]
    out = dict()
    for index, name in enumerate(names):
        values = [row[index] if index < len(row) else '' for row in rows]
        try:
            out[name] = np.array(values, dtype=float)
        except ValueError:
            out[name] = np.array(values, dtype=str)
    return out


def has_telluric(table):
    """La correction tellurique a-t-elle tourne sur cette serie?"""
    return WATER in table and np.isfinite(np.asarray(
        table[WATER], dtype=float)).any()


def extremes(values, names=('best', 'median', 'worst')):
    """Les indices de la plus petite, de la mediane et de la plus grande.

    C'est ce que le rapport montre de la correction tellurique: une pose de
    chaque bout de la distribution et une du milieu, plutot qu'une pose au
    hasard qui ne dirait pas si le cas difficile passe.
    """
    values = np.asarray(values, dtype=float)
    good = np.where(np.isfinite(values))[0]
    if good.size == 0:
        return dict()
    order = good[np.argsort(values[good])]
    middle = order[order.size // 2]
    return {names[0]: int(order[0]), names[1]: int(middle),
            names[2]: int(order[-1])}


def spectrum_path(table, index, science_dir):
    """Le spectre derriere une ligne du .rdb, ou None.

    LE NOM DU .rdb EST CELUI DU FICHIER LBL, pas celui du spectre: LBL y colle
    l'objet, la comparaison et le suffixe _lbl, par exemple

        r.HARPS.2003-08-08T07:11:56.372_S2D_0-2_A_HD10700_TC_..._lbl.fits

    pour un spectre qui s'appelle r.HARPS.2003-08-08T07:11:56.372_S2D_A.fits.
    On retrouve donc le spectre par son HORODATAGE, qui est la seule partie
    sure du nom, plutot qu'en devinant quel suffixe retirer.
    """
    import glob

    if FILENAME not in table or not science_dir:
        return None
    name = str(np.asarray(table[FILENAME])[index])
    if not name or name == 'nan':
        return None
    base = os.path.basename(name)
    direct = os.path.join(science_dir, base)
    if os.path.exists(direct):
        return direct
    # l'horodatage: ce qui suit le nom de l'instrument, jusqu'au premier '_'
    stamp = None
    for piece in base.split('_'):
        if 'T' in piece and piece.count('-') >= 2:
            stamp = piece
            break
    if stamp is None:
        return None
    hits = sorted(glob.glob(os.path.join(science_dir, '*%s*' % stamp)))
    return hits[0] if hits else None
