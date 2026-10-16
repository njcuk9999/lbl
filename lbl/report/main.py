"""Assembler le rapport d'un run, une page par etape.

Appele a la toute fin, apres `compile`, et pilote par le parametre REPORT du
wrapper (vrai par defaut). Rien de ce qui se passe ici ne doit faire echouer
un run: le rapport est un compte rendu, pas une etape de calcul, donc chaque
page qui ne peut pas se dessiner est sautee et dite dans la derniere page.
"""
import os
import traceback

from lbl.core import logger

log = logger.Log(filename='')

#: le dossier du rapport, sous DATA_DIR
FOLDER = 'reports'


def make_report(data_dir, rdb_file, title, science_dir=None, tc_dir=None,
                known_periods=None):
    """Ecrire le rapport d'un run. Rend le chemin du PDF, ou None.

    :param data_dir: str, le DATA_DIR du run
    :param rdb_file: str, le .rdb que `compile` vient d'ecrire
    :param title: str, le nom du run, qui titre les pages
    :param science_dir: str, les spectres livres, pour la page tellurique
    :param tc_dir: str, les spectres nettoyes, pour la meme page
    :param known_periods: liste de periodes a marquer sur le periodogramme
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:                                  # noqa: BLE001
        log.warning('no report: matplotlib is not available ({0})'.format(exc))
        return None

    from lbl.report import figures, series

    if not os.path.exists(rdb_file):
        log.warning('no report: {0} is not there'.format(rdb_file))
        return None
    try:
        table = series.read_rdb(rdb_file)
    except Exception as exc:                                  # noqa: BLE001
        log.warning('no report: cannot read {0} ({1})'.format(rdb_file, exc))
        return None

    folder = os.path.join(data_dir, FOLDER)
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, '%s.pdf' % title)

    # CE QUE LE RAPPORT ESSAIE DE DESSINER, dans l'ordre. Une page qui echoue
    # est notee et le rapport continue: mieux vaut un rapport ampute qu'un run
    # qui s'arrete sur une figure.
    wanted = [('la sequence', lambda: figures.page_sequence(plt, table, title)),
              ('les periodogrammes',
               lambda: figures.page_periodogram(plt, table, title,
                                                known_periods)),
              ('le FIP', lambda: figures.page_fip(plt, table, title))]
    if series.has_telluric(table) and science_dir and tc_dir:
        wanted.append(('la correction tellurique, eau',
                       lambda: figures.page_telluric(plt, table, title,
                                                     science_dir, tc_dir,
                                                     'water')))
        wanted.append(('la correction tellurique, sec',
                       lambda: figures.page_telluric(plt, table, title,
                                                     science_dir, tc_dir,
                                                     'dry')))

    missing = []
    with PdfPages(out) as pdf:
        for name, draw in wanted:
            try:
                fig = draw()
            except Exception as exc:                          # noqa: BLE001
                missing.append('%s: %s' % (name, exc))
                log.warning('report page "{0}" failed: {1}'.format(name, exc))
                log.warning(traceback.format_exc(limit=2))
                continue
            if fig is None:
                missing.append('%s: rien a tracer, ou koloa absent' % name)
                continue
            pdf.savefig(fig)
            plt.close(fig)
        if missing:
            fig = plt.figure(figsize=(10, 4))
            fig.text(0.05, 0.9, 'Ce que ce rapport ne montre pas',
                     fontsize=12, va='top')
            fig.text(0.05, 0.78, '\n'.join('  - %s' % one for one in missing),
                     fontsize=9, va='top', family='monospace')
            pdf.savefig(fig)
            plt.close(fig)
    log.general('Report written: {0}'.format(out))
    return out
