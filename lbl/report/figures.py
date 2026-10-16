"""Les pages du rapport d'un run LBL.

Chaque fonction dessine une page et la rend; aucune n'ecrit de fichier, c'est
`main` qui les assemble. Les routines de periodogramme et de FIP viennent de
koloa quand il est la; sans lui ces pages-la sont sautees et les autres
restent.
"""
import numpy as np

from lbl.report import series as S


def _mad(values):
    """L'ecart-type robuste, 1.4826 fois l'ecart absolu median."""
    values = np.asarray(values, dtype=float)
    good = values[np.isfinite(values)]
    if good.size < 2:
        return float('nan')
    return float(1.4826 * np.median(np.abs(good - np.median(good))))


def page_sequence(plt, table, title):
    """La sequence des vitesses, avec ses barres d'erreur."""
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    centred = vrad - np.median(vrad)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    axes[0].errorbar(time, centred, yerr=svrad, fmt='o', ms=3, lw=0.8,
                     capsize=0, alpha=0.85)
    axes[0].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[0].set_ylabel('vitesse - mediane (m/s)')
    axes[0].set_title('%d poses, rms %.2f m/s, MAD %.2f, erreur mediane %.2f'
                      % (len(time), np.std(centred, ddof=1), _mad(centred),
                         np.median(svrad)), fontsize=10)
    # LES DEUX DISPERSIONS COTE A COTE, parce qu'elles ne mesurent pas la meme
    # chose: le MAD ne regarde que le coeur, le rms pese les extremes, et une
    # correction qui redresse les ailes en degradant le coeur les fait diverger
    axes[1].hist(centred, bins=max(10, int(np.sqrt(len(centred)))),
                 color='C0', alpha=0.75)
    for sign in (-1, 1):
        axes[1].axvline(sign * _mad(centred), color='k', lw=0.9, ls='--')
    axes[1].set_xlabel('vitesse - mediane (m/s)')
    axes[1].set_ylabel('poses')
    axes[1].set_title('les traits sont a plus ou moins un MAD', fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('%s: la sequence' % title)
    fig.tight_layout()
    return fig


def page_periodogram(plt, table, title, known=None):
    """Le periodogramme de la serie, et celui de la fenetre sous lui."""
    try:
        from koloa.periodogram import frequency_grid, gls
    except Exception:                                         # noqa: BLE001
        return None
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    if len(time) < 10:
        return None
    freq = frequency_grid(time)
    power = gls(time, vrad - np.median(vrad), svrad, freq)
    # LA FENETRE: le meme periodogramme d'une serie constante, qui ne porte
    # donc que l'echantillonnage. Un pic de la serie qui tombe sur un pic de
    # la fenetre est un alias, pas un signal.
    window = gls(time, np.ones_like(time), svrad, freq)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    period = 1.0 / freq
    axes[0].plot(period, power, lw=0.9)
    axes[0].set_ylabel('puissance')
    axes[0].set_title('la serie', fontsize=9)
    axes[1].plot(period, window, lw=0.9, color='0.4')
    axes[1].set_ylabel('puissance')
    axes[1].set_xlabel("periode (jours)")
    axes[1].set_title("la fenetre: un pic commun aux deux est un alias",
                      fontsize=9)
    for ax in axes:
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        for one in (known or []):
            ax.axvline(float(one), color='C3', lw=0.8, ls='--', alpha=0.7)
    fig.suptitle('%s: les periodogrammes' % title)
    fig.tight_layout()
    return fig


def page_fip(plt, table, title):
    """Le FIP, dans ses trois versions face aux aberrantes."""
    try:
        from koloa.data import RVData
        from koloa.fip import fip_comparison
    except Exception:                                         # noqa: BLE001
        return None
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    if len(time) < 20:
        return None
    try:
        data = RVData(time=time, rv=vrad, err=svrad, name=title)
        out = fip_comparison(data)
    except Exception:                                         # noqa: BLE001
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    drawn = 0
    for key, colour in (('gaussian', 'C0'), ('soft', 'C1'), ('hard', 'C2')):
        res = out.get(key)
        if res is None or not hasattr(res, 'freq'):
            continue
        fip = getattr(res, 'fip', None)
        if fip is None:
            continue
        ax.plot(1.0 / np.asarray(res.freq, dtype=float),
                np.asarray(fip, dtype=float), lw=1.0, color=colour,
                label=getattr(res, 'label', key))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('periode (jours)')
    ax.set_ylabel('FIP')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle('%s: le FIP, sans coupure et avec deux coupures\n'
                 "ce qui compte est l'ecart entre les trois: il dit ce que le"
                 ' traitement des aberrantes fait a la reponse' % title,
                 fontsize=10)
    fig.tight_layout()
    return fig


def page_telluric(plt, table, title, science_dir, tc_dir, which='water'):
    """La correction tellurique sur la meilleure, la mediane et la pire pose.

    `which` choisit la colonne qui classe les poses: l'eau ou le sec. Une pose
    de chaque bout et une du milieu, plutot qu'une au hasard qui ne dirait pas
    si le cas difficile passe.
    """
    from astropy.io import fits

    column = S.WATER if which == 'water' else S.DRY
    if column not in table:
        return None
    picks = S.extremes(np.asarray(table[column], dtype=float))
    if not picks:
        return None

    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.6 * len(picks)),
                             squeeze=False)
    drawn = 0
    for row, (tag, index) in enumerate(picks.items()):
        ax = axes[row, 0]
        raw = S.spectrum_path(table, index, science_dir)
        clean = S.spectrum_path(table, index, tc_dir)
        value = float(np.asarray(table[column], dtype=float)[index])
        if raw is None or clean is None:
            ax.text(0.5, 0.5, 'spectre absent', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        try:
            with fits.open(raw) as hdul:
                before = hdul[1].data.astype(float)
                wave = hdul[4].data.astype(float) / 10.0
            after = fits.getdata(clean, 1).astype(float)
        except Exception:                                     # noqa: BLE001
            continue
        band = ((wave > 688.0) & (wave < 690.5) & np.isfinite(before)
                & (before > 0) & np.isfinite(after))
        if band.sum() < 30:
            continue
        order = np.argsort(wave[band])
        w = wave[band][order]
        b = before[band][order]
        a = after[band][order]
        scale = np.nanpercentile(b, 95)
        ax.plot(w, b / scale, 'k-', lw=0.8, label='avant')
        ax.plot(w, a / scale, '-', lw=0.8, color='C3', label='apres')
        ax.axhline(1.0, color='0.6', lw=0.8, ls=':')
        ax.set_title('%s: %s = %.4f' % (tag, column, value), fontsize=9)
        ax.set_ylabel('flux normalise')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    axes[-1, 0].set_xlabel('longueur d onde (nm)')
    fig.suptitle('%s: la correction tellurique, colonne %s\n'
                 'la meilleure, la mediane et la pire pose'
                 % (title, 'd eau' if which == 'water' else 'seche'),
                 fontsize=10)
    fig.tight_layout()
    return fig
