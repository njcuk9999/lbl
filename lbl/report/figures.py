"""The pages of an LBL run report.

Each function draws one page and returns it; none writes a file, `main`
assembles them. The periodogram and FIP routines come from koloa when it is
installed; without it those pages are skipped and the rest still come out.
"""
import numpy as np

from lbl.report import series as S


from lbl.report.stats import robust_sigma as _mad


def page_sequence(plt, table, title):
    """The velocity series, with its error bars."""
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    centred = vrad - np.median(vrad)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    axes[0].errorbar(time, centred, yerr=svrad, fmt='o', ms=3, lw=0.8,
                     capsize=0, alpha=0.85)
    axes[0].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[0].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
    axes[0].set_ylabel('Radial velocity, median removed (m s$^{-1}$)')
    axes[0].set_title('%d epochs: rms %.2f, MAD %.2f, median error bar'
                      ' %.2f (all m s$^{-1}$)'
                      % (len(time), np.std(centred, ddof=1), _mad(centred),
                         np.median(svrad)), fontsize=10)
    # BOTH SCATTERS SIDE BY SIDE, because they do not measure the same thing:
    # the MAD only sees the core, the rms weighs the tails, and a correction
    # that straightens the tails while spoiling the core makes them disagree
    axes[1].hist(centred, bins=max(10, int(np.sqrt(len(centred)))),
                 color='C0', alpha=0.75)
    for sign in (-1, 1):
        axes[1].axvline(sign * _mad(centred), color='k', lw=0.9, ls='--')
    axes[1].set_xlabel('Radial velocity, median removed (m s$^{-1}$)')
    axes[1].set_ylabel('Number of epochs')
    axes[1].set_title('Distribution of the same velocities; dashed lines at'
                      ' plus and minus one MAD', fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('The velocity series, and how its scatter is distributed',
                 fontsize=10)
    fig.tight_layout()
    return fig


def page_periodogram(plt, table, title, known=None):
    """The periodogram of the series, and the spectral window below it."""
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
    # LA FENETRE SPECTRALE, et elle ne se calcule PAS avec gls: celle-ci est a
    # moyenne flottante, donc une serie constante lui donne exactement zero
    # (vu le 2026-10-16: la page sortait une ligne plate). La fenetre est la
    # transformee de l'echantillonnage lui-meme,
    #
    #     W(nu) = | somme_k w_k exp(-2i pi nu t_k) | / somme_k w_k,
    #
    # qui vaut 1 a frequence nulle et dit ou l'echantillonnage, a lui seul,
    # fabrique de la puissance. Un pic de la serie qui tombe sur un pic de la
    # fenetre est un alias, pas un signal.
    weight = 1.0 / np.asarray(svrad, dtype=float) ** 2
    weight = weight / np.sum(weight)
    phase = np.exp(-2j * np.pi * np.outer(np.asarray(freq, dtype=float),
                                          time - np.mean(time)))
    window = np.abs(phase @ weight)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    period = 1.0 / freq
    axes[0].plot(period, power, lw=0.9)
    axes[0].set_ylabel('GLS power (normalised, 0 to 1)')
    axes[0].set_title('the series', fontsize=9)
    axes[1].plot(period, window, lw=0.9, color='0.4')
    axes[1].set_ylabel('Window power (1 at zero frequency)')
    axes[1].set_xlabel('Period (days)')
    axes[1].set_title('the window: a peak shared with the series above is an'
                      ' alias, not a signal', fontsize=9)
    for ax in axes:
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        for one in (known or []):
            ax.axvline(float(one), color='C3', lw=0.8, ls='--', alpha=0.7)
    fig.suptitle('Periodogram of the velocities, and the spectral window'
                 ' of the sampling', fontsize=10)
    fig.tight_layout()
    return fig


def page_fip(plt, table, title):
    """The FIP, in its three ways of facing the outliers."""
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
    # L'ECHELLE DU FIP EST CELLE DE 1 - FIP, pas celle du FIP: une serie sans
    # detection reste collee a 1, et un log sur 0.6 a 1.0 ne montre rien. On
    # trace donc le FIP mais on gradue en 1 - FIP, qui est la probabilite
    # qu'il y AIT un signal et qui s'etale, elle, sur plusieurs decades.
    ax.set_yscale('function',
                  functions=(lambda y: -np.log10(np.clip(1.0 - y, 1e-12, 1.0)),
                             lambda z: 1.0 - 10.0 ** (-z)))
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('False inclusion probability'
                  ' (axis stretched in $1-\\mathrm{FIP}$)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle('False inclusion probability, with no clip and with two'
                 ' clips\nWhat matters is the gap between the three: it says'
                 ' what the handling of outliers does to the answer',
                 fontsize=10)
    fig.tight_layout()
    return fig


def page_telluric(plt, table, title, science_dir, tc_dir, which='water'):
    """The telluric correction on the best, median and worst epoch.

    `which` picks the column that ranks the epochs: water or dry. One epoch
    from each end and one from the middle, rather than one at random, which
    would not say whether the hard case comes out right.
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
            ax.text(0.5, 0.5, 'spectrum not found', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        try:
            with fits.open(raw) as hdul:
                before = hdul[1].data.astype(float)
                wave = hdul[4].data.astype(float) / 10.0
            after = fits.getdata(clean, 1).astype(float)
        except Exception:                                     # noqa: BLE001
            continue
        # THE SELECTION DOES NOT REQUIRE THE CORRECTED FLUX TO BE FINITE.
        # LBL sets to NaN every pixel whose modelled absorption is deeper
        # than exp(-1), about 63% absorbed, because nothing there can be
        # divided back out (tellu_clean, TELLUCLEAN_TRANSMISSION_THRESHOLD).
        # Dropping those pixels from the selection would join the points on
        # either side with a straight line, and the figure would show a
        # correction where in truth there is a hole. Keeping the NaNs makes
        # matplotlib break the curve, which is the honest picture.
        band = ((wave > 688.0) & (wave < 690.5) & np.isfinite(before)
                & (before > 0))
        if band.sum() < 30:
            continue
        order = np.argsort(wave[band])
        w = wave[band][order]
        b = before[band][order]
        a = after[band][order]
        scale = np.nanpercentile(b, 95)
        dead = int(np.sum(~np.isfinite(a)))
        ax.plot(w, b / scale, 'k-', lw=0.8, label='before')
        ax.plot(w, a / scale, '-', lw=0.8, color='C3',
                label='after (gaps: absorption too deep to correct)'
                      if dead else 'after')
        # and shade where the gaps are, so that a reader sees them as a
        # choice rather than as a plotting accident
        if dead:
            ax.fill_between(w, 0, 1, where=~np.isfinite(a), color='0.85',
                            transform=ax.get_xaxis_transform(), zorder=0)
        ax.axhline(1.0, color='0.6', lw=0.8, ls=':')
        ax.set_title('%s epoch: %s = %.4f%s'
                     % (tag, column, value,
                        ', %d pixels left uncorrected' % dead if dead else ''),
                     fontsize=9)
        ax.set_ylabel('Flux / continuum')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    axes[-1, 0].set_xlabel('Wavelength in the observatory rest frame (nm)')
    fig.suptitle('Telluric correction in the O$_2$ B band, ranked by %s'
                 ' column\nThe best, the median and the worst epoch of the'
                 ' campaign'
                 % ('water' if which == 'water' else 'dry'), fontsize=10)
    fig.tight_layout()
    return fig


def page_berv(plt, table, title):
    """Velocity against time, coloured by BERV, and against BERV itself.

    WHY THIS PAGE. A residual that follows the BERV is not stellar: the BERV
    is the Earth's own motion, so anything correlated with it comes from the
    atmosphere, from the wavelength solution, or from a frame that was not
    undone properly. Colouring the series by BERV shows it at a glance, and
    the panel below measures it.
    """
    berv = S.column(table, S.BERV)
    if berv is None or not np.isfinite(berv).any():
        return None
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    centred = vrad - np.median(vrad)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    scatter = axes[0].scatter(time, centred, c=berv, cmap='coolwarm', s=22,
                              zorder=3)
    axes[0].errorbar(time, centred, yerr=svrad, fmt='none', ecolor='0.6',
                     lw=0.7, zorder=2)
    axes[0].axhline(0.0, color='k', lw=0.8, ls=':')
    axes[0].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
    axes[0].set_ylabel('Radial velocity, median removed (m s$^{-1}$)')
    bar = fig.colorbar(scatter, ax=axes[0])
    bar.set_label('Barycentric Earth radial velocity (km s$^{-1}$)')
    axes[0].set_title('Velocity series, each epoch coloured by its BERV',
                      fontsize=9)

    good = np.isfinite(berv) & np.isfinite(centred)
    axes[1].errorbar(berv[good], centred[good], yerr=svrad[good], fmt='o',
                     ms=3, lw=0.7, capsize=0, alpha=0.8)
    axes[1].axhline(0.0, color='k', lw=0.8, ls=':')
    note = 'Too few epochs to fit a line'
    if good.sum() > 5:
        slope, offset = np.polyfit(berv[good], centred[good], 1)
        span = np.linspace(np.min(berv[good]), np.max(berv[good]), 10)
        axes[1].plot(span, slope * span + offset, '-', color='C3', lw=1.2)
        # the slope is the number that matters: in m/s per km/s it says how
        # much of the Earth's motion leaked into the measured velocity
        resid = centred[good] - (slope * berv[good] + offset)
        # THE SLOPE IS THE NUMBER TO READ: in m/s per km/s it says how much
        # of the Earth's own motion leaked into the measured velocity, and
        # the product with the BERV span says how many m/s that is in total.
        note = ('Linear fit: %.3f m s$^{-1}$ per km s$^{-1}$, i.e. %.2f'
                ' m s$^{-1}$ over the %.1f km s$^{-1}$ span.'
                ' Residual scatter about the line: %.2f m s$^{-1}$ (MAD)'
                % (slope, abs(slope) * np.ptp(berv[good]),
                   np.ptp(berv[good]), _mad(resid)))
    axes[1].set_xlabel('Barycentric Earth radial velocity (km s$^{-1}$)')
    axes[1].set_ylabel('Radial velocity, median removed (m s$^{-1}$)')
    axes[1].set_title(note, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('Does anything follow the BERV?\n'
                 'A residual that tracks the barycentric velocity is not'
                 ' stellar: it comes from the atmosphere, the wavelength'
                 ' solution, or a frame left undone', fontsize=10)
    fig.tight_layout()
    return fig
