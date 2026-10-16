"""The pages of an LBL run report.

Each function draws one page and returns it; none writes a file, `main`
assembles them. The periodogram and FIP routines come from koloa when it is
installed; without it those pages are skipped and the rest still come out.
"""
import numpy as np

from lbl.report import series as S


from lbl.report.stats import robust_sigma as _mad


#: THE PERIODS THE EYE SHOULD RECOGNISE on any periodogram. A peak that sits
#: on one of them is almost always the sampling and not the star: the day is
#: the observing cadence itself, the lunar month is when a target is
#: reachable, and the year is the seasonal window. They are drawn on every
#: page that has a period axis.
LANDMARKS = ((1.0, 'day'), (29.53, 'lunar month'), (365.25, 'year'))


def _landmarks(ax, periods=LANDMARKS, known=None):
    """Draw the day, month and year lines, and any known period."""
    import numpy as np

    low, high = ax.get_xlim()
    for value, name in periods:
        if not (low < value < high):
            continue
        ax.axvline(value, color='0.45', lw=0.9, ls='--', zorder=0)
        ax.annotate(name, xy=(value, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(2, -2), textcoords='offset points', rotation=90,
                    va='top', ha='left', fontsize=7, color='0.35')
    for one in (known or []):
        value = float(one)
        if not (low < value < high):
            continue
        ax.axvline(value, color='C3', lw=0.9, ls=':', alpha=0.8, zorder=0)
        ax.annotate('%.4g d' % value, xy=(value, 0.0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(2, 2), textcoords='offset points', rotation=90,
                    va='bottom', ha='left', fontsize=7, color='C3')


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
        _landmarks(ax, known=known)
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
    # A PROBABILITY IS READ ON ITS OWN SCALE: linear, 0 to 1, with the axis
    # pinned to those bounds. A FIP near 1 means the signal is almost surely
    # absent and a FIP near 0 that it is almost surely there, and the eye
    # should see where a curve sits between the two without having to decode
    # a stretched axis.
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('False inclusion probability')
    _landmarks(ax)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle('False inclusion probability, with no clip and with two'
                 ' clips\nWhat matters is the gap between the three: it says'
                 ' what the handling of outliers does to the answer',
                 fontsize=10)
    fig.tight_layout()
    return fig


def strongest_line(spl, wave_min, wave_max, step=0.002):
    """Where the deepest line of this species sits, in nm.

    Found from TAPAS rather than written down, so that the figure centres
    itself on whatever the instrument actually covers. Measured on the HARPS
    domain the 2026-10-16: water peaks at 647.76 nm (46% deep) and the dry
    absorbers at 687.41 nm, the saturated O2 B band.
    """
    grid = np.arange(wave_min, wave_max, step)
    depth = 1.0 - np.asarray(spl(grid), dtype=float)
    good = np.isfinite(depth)
    if not good.any():
        return None
    return float(grid[good][int(np.argmax(depth[good]))])


def page_telluric(plt, table, title, science_dir, tc_dir, which='water',
                  spl_water=None, spl_dry=None, half_width=1.2):
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
    # CENTRE ON THE STRONGEST LINE OF THE SPECIES THIS PAGE IS ABOUT: showing
    # the water column on a window dominated by O2, or the other way round,
    # would show whether the wrong species was corrected.
    spl = spl_water if which == 'water' else spl_dry
    centre = None
    if spl is not None:
        centre = strongest_line(spl, 380.0, 692.0)
    if centre is None:
        centre = 647.76 if which == 'water' else 687.41
    low, high = centre - half_width, centre + half_width
    colour = 'C0' if which == 'water' else 'C1'

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
        band = ((wave > low) & (wave < high) & np.isfinite(before)
                & (before > 0))
        if band.sum() < 30:
            continue
        order = np.argsort(wave[band])
        w = wave[band][order]
        b = before[band][order]
        a = after[band][order]
        scale = np.nanpercentile(b, 95)
        dead = int(np.sum(~np.isfinite(a)))
        ax.plot(w, b / scale, 'k-', lw=0.9, label='uncorrected')
        ax.plot(w, a / scale, '-', lw=0.9, color='tab:green',
                label='corrected (gaps: absorption too deep)'
                      if dead else 'corrected')
        # THE CORRECTION ITSELF, which is what the two curves differ by: blue
        # for water, orange for the dry absorbers, so that the species the
        # page is about is the one the eye follows.
        # THE TRANSMISSION ITSELF, which is before/after: 1 where nothing was
        # removed and dipping to the depth of each line. Plotting its inverse
        # would put the dips above the continuum and read as emission.
        with np.errstate(invalid='ignore', divide='ignore'):
            removed = np.where(np.isfinite(a) & (a != 0) & (b > 0),
                               b / a, np.nan)
        ax.plot(w, removed, '-', lw=1.1, color=colour, alpha=0.85,
                label='transmission divided out (%s)'
                      % ('water' if which == 'water' else 'dry'))
        # and shade where the gaps are, so that a reader sees them as a
        # choice rather than as a plotting accident
        # A THIN BAND ALONG THE BOTTOM, not a full-height wash: around a
        # strong line the gaps are many and scattered, and shading the whole
        # panel turned the figure into a solid block that hid the curves it
        # was meant to annotate.
        if dead:
            ax.fill_between(w, 0.0, 0.035, where=~np.isfinite(a),
                            color='0.45', transform=ax.get_xaxis_transform(),
                            step='mid', zorder=4)
        ax.axhline(1.0, color='0.6', lw=0.8, ls=':')
        # BOUND THE AXIS ON THE UNCORRECTED SPECTRUM. Dividing by a very low
        # transmission multiplies the noise, so just outside the pixels LBL
        # blanks the corrected flux swings over several units and, left to
        # itself, the axis squashes everything else into a line. The spikes
        # are real and worth seeing, but the scale belongs to the data.
        ax.set_ylim(-0.1, max(1.35, float(np.nanpercentile(b / scale, 99.5))
                              * 1.15))
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
    fig.suptitle('Telluric correction around the strongest %s line'
                 ' (%.2f nm), epochs ranked by %s column\n'
                 'The best, the median and the worst epoch of the campaign'
                 % ('water' if which == 'water' else 'dry', centre,
                    'water' if which == 'water' else 'dry'), fontsize=10)
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


def page_oafip(plt, table, title, known=None):
    """The outlier-aware FIP, beside the gaussian one.

    WHY A SECOND FIP. The gaussian FIP of the page before assumes every point
    is drawn from its error bar, so one bad night can either invent a signal
    or bury one, and the only defence is to clip, which is a decision made
    before the question is asked. The outlier-aware FIP instead MODELS the
    outliers: each point carries a latent indicator saying whether it belongs
    to the normal population or to a wider one, sampled along with everything
    else (koloa.fip.oafip, after Hara et al. 2022).

    Reading the two together is the point. Where they agree, the answer does
    not depend on how the outliers were handled. Where they differ, it does,
    and the outlier-aware one is the one to believe.
    """
    try:
        from koloa.data import RVData
        from koloa.fip import oafip
    except Exception:                                         # noqa: BLE001
        return None
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    if len(time) < 20:
        return None
    try:
        data = RVData(time=time, rv=vrad, err=svrad, name=title)
        # a short chain on purpose: this is a report, not a survey analysis,
        # and the question it answers is whether the outliers change the
        # answer, which a few hundred sweeps already settle
        res = oafip(data, kmax=1, nsweep=400, nburn=80, nchains=2,
                    progress=False, label='outlier-aware')
    except Exception:                                         # noqa: BLE001
        return None
    if not hasattr(res, 'freq') or getattr(res, 'fip', None) is None:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    period = 1.0 / np.asarray(res.freq, dtype=float)
    axes[0].plot(period, np.asarray(res.fip, dtype=float), lw=1.1,
                 color='C4', label='outlier-aware')
    axes[0].set_xscale('log')
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_yticks(np.linspace(0.0, 1.0, 11))
    axes[0].set_ylabel('False inclusion probability')
    axes[0].set_xlabel('Period (days)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    _landmarks(axes[0], known=known)

    # WHICH EPOCHS THE MODEL CALLS OUTLIERS, which is the other half of the
    # answer: a FIP that moved because three nights were down-weighted is a
    # different statement from one that moved because half the series was
    prob = getattr(res, 'outlier_prob', None)
    if prob is None:
        axes[1].text(0.5, 0.5, 'the sampler returned no outlier probabilities',
                     ha='center', va='center', transform=axes[1].transAxes)
    else:
        prob = np.asarray(prob, dtype=float)
        axes[1].plot(time, prob, 'o', ms=4, color='C4')
        axes[1].axhline(0.5, color='0.5', lw=0.9, ls='--')
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_ylabel('P(outlier)')
        axes[1].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
        axes[1].set_title('%d epochs above one half, out of %d'
                          % (int(np.sum(prob > 0.5)), prob.size), fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.suptitle('Outlier-aware false inclusion probability\\n'
                 'The outliers are modelled rather than clipped: compare with'
                 ' the gaussian page, and believe this one where they differ',
                 fontsize=10)
    fig.tight_layout()
    return fig


#: THE INDICATORS LBL MEASURES BESIDE THE VELOCITY, with the error column
#: that goes with each. DTEMP* is not listed here: there can be several of
#: them, one per temperature table, and they are found by their name.
INDICATORS = (('d2v', 'sd2v', 'Second derivative projection, d2v'),
              ('d3v', 'sd3v', 'Third derivative projection, d3v'),
              ('dW', 'sdW', 'Line width projection, dW'),
              ('CRX', 'sCRX', 'Chromatic index, CRX'),
              ('fwhm', 'sig_fwhm', 'FWHM'),
              ('contrast', 'sig_contrast', 'Contrast'))


def dtemp_columns(table):
    """Every DTEMP table in this .rdb, as (value, error, label).

    Found by name rather than written down: a run can project onto several
    temperature gradients at once (RESPROJ_TABLES), and each becomes its own
    DTEMP<T> column.
    """
    out = []
    for name in sorted(table):
        if not name.startswith('DTEMP'):
            continue
        error = 's%s' % name
        out.append((name, error if error in table else None,
                    'Temperature projection, %s' % name))
    return out


def page_indicators(plt, table, title):
    """Every indicator LBL measured, as a time series, one panel each.

    WHY ALL OF THEM ON ONE PAGE. A velocity that moves with d2v is a line
    shape changing, not a Doppler shift; one that moves with DTEMP is the
    star's effective temperature; one that moves with dW is the width alone.
    Seeing them under the same time axis is what tells a signal from an
    artefact, and that only works if none is left out.
    """
    time = np.asarray(table[S.TIME], dtype=float)
    wanted = []
    for name, error, label in list(dtemp_columns(table)) + list(INDICATORS):
        values = S.column(table, name)
        if values is None or not np.isfinite(values).any():
            continue
        wanted.append((values, S.column(table, error) if error else None,
                       label))
    if not wanted:
        return None

    height = max(6.0, 1.9 * len(wanted))
    fig, axes = plt.subplots(len(wanted), 1, figsize=(10, height),
                             sharex=True, squeeze=False)
    for row, (values, errors, label) in enumerate(wanted):
        ax = axes[row, 0]
        centred = values - np.nanmedian(values)
        if errors is not None and np.isfinite(errors).any():
            ax.errorbar(time, centred, yerr=errors, fmt='o', ms=3, lw=0.7,
                        capsize=0, alpha=0.85)
        else:
            ax.plot(time, centred, 'o', ms=3, alpha=0.85)
        ax.axhline(0.0, color='k', lw=0.8, ls=':')
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        # the scatter of each indicator, so the page is read and not only
        # looked at: an indicator flat at the level of its error bars says
        # something different from one that moves ten times as much
        scatter = _mad(centred)
        median_err = (float(np.nanmedian(errors))
                      if errors is not None and np.isfinite(errors).any()
                      else float('nan'))
        if np.isfinite(median_err) and median_err > 0:
            ax.set_title('MAD %.3g, median error %.3g, ratio %.1f'
                         % (scatter, median_err, scatter / median_err),
                         fontsize=8)
        else:
            ax.set_title('MAD %.3g' % scatter, fontsize=8)
    axes[-1, 0].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
    fig.suptitle('Every indicator measured beside the velocity, median'
                 ' removed\\nA velocity that follows one of these is a line'
                 ' shape changing, not a Doppler shift', fontsize=10)
    fig.tight_layout()
    return fig
