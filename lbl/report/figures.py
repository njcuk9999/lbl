"""The pages of an LBL run report.

Each function draws one page and returns it; none writes a file, `main`
assembles them. The FIP is computed here (lbl.report.fip); the periodogram
borrows koloa when it is installed, and is skipped without it.
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


#: |V_tot| [km/s] under which the star's lines sit on the telluric lines,
#:    within about one resolution element (lbl.pca2d.bervbias.CLOSE_KMS)
VTOT_CLOSE = 4.0


def vtot_cmap():
    """The colours of V_tot: blue below zero, red above, darkest at zero and
    paling away from it, the hue changing abruptly at zero. The epochs whose
    star lines sit on the tellurics are the ones that stand out, and which
    side of them each is reads at a glance; a diverging map pale at zero
    would hide exactly those."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    below = LinearSegmentedColormap.from_list('below', ['#d4e2f1', '#08306b'])
    above = LinearSegmentedColormap.from_list('above', ['#67000d', '#f6d3c9'])
    ramp = np.linspace(0.0, 1.0, 128)
    return ListedColormap(np.vstack([below(ramp), above(ramp)]), name='vtot')


def total_velocity(vrad, berv, systemic):
    """V_tot [km/s], the star's velocity in the telluric frame: its systemic
    velocity [km/s] plus the series less its median [m/s], less the BERV
    [km/s], composed relativistically (lbl.core.math.add_velocities)"""
    from lbl.core import math as mp
    star = systemic * 1000.0 + (vrad - np.nanmedian(vrad))
    return mp.add_velocities(star, -1000.0 * np.asarray(berv, float)) / 1000.0


def page_sequence(plt, table, title, systemic=None):
    """The velocity series, with its error bars, each epoch coloured by the
    star's velocity in the telluric frame, V_tot, when its systemic velocity
    is known (systemic, km/s: SIMBAD's, series.systemic_velocity), else by
    its BERV: the report's first figure (main._build puts it before the
    table). The figure's `coloured_by` says which ('V_tot' or 'BERV').

    WHY BY V_TOT. A residual from the telluric lines is largest where the
    star's lines sit on them, |V_tot| of a few km/s, and changes sign with
    V_tot: the epochs near zero, darkest, are the ones to look at, and
    their colours say on which side of the tellurics each one is. Without
    the systemic velocity, the BERV: a residual that follows it is not
    stellar (the Earth's own motion: the atmosphere, the wavelength
    solution, a frame left undone), and shows as a seasonal pattern whose
    colours line up. page_berv measures it.
    """
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    centred = vrad - np.median(vrad)
    berv = S.column(table, S.BERV)
    has_berv = berv is not None and np.isfinite(berv).any()
    vtot = (total_velocity(vrad, berv, systemic)
            if has_berv and systemic is not None else None)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    fig.coloured_by = None
    if vtot is not None:
        axes[0].errorbar(time, centred, yerr=svrad, fmt='none', ecolor='0.65',
                         lw=0.7, zorder=2)
        # symmetric about zero, so that zero is where the hue changes
        limit = max(10.0, 5.0 * np.ceil(np.nanmax(np.abs(vtot)) / 5.0))
        close = np.abs(vtot) < VTOT_CLOSE
        scatter = axes[0].scatter(time, centred, c=vtot, cmap=vtot_cmap(),
                                  vmin=-limit, vmax=limit, s=20,
                                  edgecolors='0.25', linewidths=0.3,
                                  zorder=3)
        # the epochs on the tellurics, outlined
        axes[0].scatter(time[close], centred[close], s=34, facecolors='none',
                        edgecolors='k', linewidths=0.9, zorder=4)
        bar = fig.colorbar(scatter, ax=axes[0], pad=0.01)
        for sign in (-1, 1):
            bar.ax.axhline(sign * VTOT_CLOSE, color='k', lw=0.8, ls='--')
        bar.set_label('$V_\\mathrm{tot}$ = $V_\\mathrm{sys}$ + vrad - BERV'
                      ' (km s$^{-1}$)')
        fig.coloured_by = 'V_tot'
    elif has_berv:
        axes[0].errorbar(time, centred, yerr=svrad, fmt='none', ecolor='0.65',
                         lw=0.7, zorder=2)
        scatter = axes[0].scatter(time, centred, c=berv, cmap='coolwarm',
                                  s=20, edgecolors='0.25', linewidths=0.3,
                                  zorder=3)
        bar = fig.colorbar(scatter, ax=axes[0], pad=0.01)
        bar.set_label('BERV (km s$^{-1}$)')
        fig.coloured_by = 'BERV'
    else:
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
    if fig.coloured_by == 'V_tot':
        coloured = ('each epoch coloured by $V_\\mathrm{tot}$, the star\'s'
                    ' velocity in the telluric frame,\n($V_\\mathrm{sys}$ ='
                    ' %.2f km s$^{-1}$ from SIMBAD; outlined: '
                    '$|V_\\mathrm{tot}|$ < %g km s$^{-1}$, the star\'s lines'
                    ' on the tellurics)' % (systemic, VTOT_CLOSE))
    elif fig.coloured_by == 'BERV':
        coloured = 'each epoch coloured by its BERV,\n'
    else:
        coloured = 'no BERV in the table,\n'
    fig.suptitle('The velocity series, %s and how its scatter is'
                 ' distributed' % coloured, fontsize=10)
    fig.tight_layout()
    return fig


def subtract_orbits(table, planets):
    """A copy of the table whose velocities have the known orbits taken out
    (PLANETS), for the periodogram and the FIP after them"""
    from lbl.science import planets as kepler

    out = dict(table)
    time = np.asarray(table[S.TIME], dtype=float) + 2400000.0
    out[S.VELOCITY] = (np.asarray(table[S.VELOCITY], dtype=float)
                       - kepler.star_velocity(time, planets['orbits']))
    return out


def page_phase(plt, table, title, planets):
    """The velocities folded on each orbit the template was made on
    (PLANETS), with its Keplerian. For each orbit, the others and one
    constant (their weighted mean residual) are taken out first."""
    if planets is None:
        return None
    from lbl.science import planets as kepler

    time = np.asarray(table[S.TIME], dtype=float) + 2400000.0
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    good = np.isfinite(time) & np.isfinite(vrad) & np.isfinite(svrad)
    time, vrad, svrad = time[good], vrad[good], svrad[good]
    orbits = planets['orbits']
    models = [kepler.orbit_velocity(time, orbit) for orbit in orbits]
    total = np.sum(models, axis=0)
    weights = 1.0 / np.maximum(svrad, 1e-3) ** 2
    offset = np.sum(weights * (vrad - total)) / np.sum(weights)
    residual = vrad - offset - total

    fig, axes = plt.subplots(len(orbits), 1, squeeze=False,
                             figsize=(10, 2.0 + 3.2 * len(orbits)))
    for ax, orbit, model in zip(axes[:, 0], orbits, models):
        # phase 0 at the conjunction (the transit) when the orbit gives it,
        #   at periastron otherwise
        at_conj = orbit.get('t0') is not None
        ref = orbit['t0'] if at_conj else kepler.periastron_time(orbit)
        period = float(orbit['period'])
        phase = np.mod((time - ref) / period + 0.5, 1.0) - 0.5
        folded = vrad - offset - (total - model)
        ax.errorbar(phase, folded, yerr=svrad, fmt='o', ms=3, lw=0.8,
                    capsize=0, alpha=0.8, label='LBL velocities')
        grid = np.linspace(-0.5, 0.5, 600)
        ax.plot(grid, kepler.orbit_velocity(ref + grid * period, orbit),
                color='C3', lw=1.2, label='Keplerian of the catalog')
        ax.axhline(0.0, color='k', lw=0.6, ls=':')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel('Phase (0 = %s)' % ('inferior conjunction' if at_conj
                                           else 'periastron'))
        ax.set_ylabel('Radial velocity (m s$^{-1}$)')
        ax.set_title('%s: P = %.6g d, K = %.2f m s$^{-1}$, e = %.3f;'
                     ' rms about the model %.2f m s$^{-1}$'
                     % (orbit.get('name', '?'), period, orbit['k'],
                        orbit['e'], np.std(residual, ddof=1)
                        if len(residual) > 1 else np.nan), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('%s: the velocities folded on the orbits the template was '
                 'made on (%s), other companions and a constant of %.1f '
                 'm s$^{-1}$ taken out' % (title, planets['catalog'], offset),
                 fontsize=10)
    fig.tight_layout()
    return fig


#: the jitter's search range [m/s], on a log scale
JITTER_RANGE = (1e-3, 1e3)
#: the acceleration's unit: m/s per year
YEAR = 365.25


def fit_orbits_trend(time, vrad, svrad, orbits):
    """The known orbits and an acceleration fitted together: a constant, a
    slope in time and each orbit's K (its shape, P, e, omega and the time of
    conjunction or periastron, held at the catalogue's), by weighted least
    squares, with a jitter added to the error bars and chosen by maximum
    likelihood. Returns dict(coef, err, cov, jitter, tmid, names, model,
    residual): coef = [constant, slope per day, K of each orbit]."""
    from scipy.optimize import minimize_scalar

    from lbl.science import planets as kepler

    tmid = float(np.mean(time))
    shapes = [kepler.orbit_velocity(time, dict(orbit, k=1.0))
              for orbit in orbits]
    design = np.column_stack([np.ones_like(time), time - tmid] + shapes)

    def solve(jitter):
        var = svrad ** 2 + jitter ** 2
        root = 1.0 / np.sqrt(var)
        normal = (design * (1.0 / var)[:, None]).T @ design
        cov = np.linalg.inv(normal)
        coef = cov @ ((design * (1.0 / var)[:, None]).T @ vrad)
        resid = vrad - design @ coef
        nll = 0.5 * float(np.sum(resid ** 2 / var + np.log(var)))
        return coef, cov, resid, nll, root

    found = minimize_scalar(lambda lj: solve(np.exp(lj))[3],
                            bounds=np.log(JITTER_RANGE), method='bounded')
    jitter = float(np.exp(found.x))
    # the bounded search can miss a minimum at its lower edge (no jitter)
    if solve(JITTER_RANGE[0])[3] <= solve(jitter)[3]:
        jitter = JITTER_RANGE[0]
    coef, cov, resid, _, _ = solve(jitter)
    return dict(coef=coef, err=np.sqrt(np.diag(cov)), cov=cov,
                jitter=jitter, tmid=tmid,
                names=[o.get('name', '?') for o in orbits],
                model=design @ coef, residual=resid, shapes=shapes)


def page_phase_trend(plt, table, title, planets):
    """The known orbits and an acceleration fitted together
    (fit_orbits_trend): the series with the total model, its residuals, and
    each orbit folded with the rest of the model taken out, its fitted K
    beside the catalogue's."""
    if planets is None or not planets.get('orbits'):
        return None
    from lbl.science import planets as kepler

    time = np.asarray(table[S.TIME], dtype=float) + 2400000.0
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    good = (np.isfinite(time) & np.isfinite(vrad) & np.isfinite(svrad)
            & (svrad > 0))
    orbits = planets['orbits']
    if good.sum() < len(orbits) + 4:
        return None
    time, vrad, svrad = time[good], vrad[good], svrad[good]
    fit = fit_orbits_trend(time, vrad, svrad, orbits)
    coef, err = fit['coef'], fit['err']
    slope, slope_err = coef[1] * YEAR, err[1] * YEAR
    total_err = np.sqrt(svrad ** 2 + fit['jitter'] ** 2)
    rjd = time - 2400000.0

    fig, axes = plt.subplots(2 + len(orbits), 1, squeeze=False,
                             figsize=(10, 5.0 + 3.0 * len(orbits)),
                             gridspec_kw=dict(height_ratios=[2.2, 1.0]
                                              + [2.0] * len(orbits)))
    axes = axes[:, 0]
    # THE SERIES AND THE TOTAL MODEL, on a fine grid in time, and the
    #   acceleration alone
    grid = np.linspace(time.min(), time.max(), 20000)
    shapes = [kepler.orbit_velocity(grid, dict(o, k=1.0)) for o in orbits]
    trend = coef[0] + coef[1] * (grid - fit['tmid'])
    total = trend + sum(k * f for k, f in zip(coef[2:], shapes))
    axes[0].errorbar(rjd, vrad, yerr=total_err, fmt='o', ms=3, lw=0.7,
                     capsize=0, alpha=0.8, label='LBL velocities (errors'
                     ' with the jitter)')
    # a short period over years fills a band: drawn thin, with the model's
    #   value at each epoch, which is what the points are compared with
    axes[0].plot(grid - 2400000.0, total, color='C3', lw=0.4, alpha=0.35,
                 label='total model: orbits + acceleration + constant')
    axes[0].plot(rjd, fit['model'], 's', ms=4, mfc='none', mec='C3',
                 mew=0.9, label='total model at the epochs')
    axes[0].plot(grid - 2400000.0, trend, color='k', lw=1.2, ls='--',
                 label='acceleration %+.3f $\\pm$ %.3f m s$^{-1}$ yr$^{-1}$'
                       % (slope, slope_err))
    axes[0].set_ylabel('Radial velocity (m s$^{-1}$)')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].set_title('jitter %.2f m s$^{-1}$, rms of the residuals %.2f'
                      ' m s$^{-1}$, %d epochs' % (
                          fit['jitter'], float(np.std(fit['residual'],
                                                      ddof=len(coef))),
                          time.size), fontsize=9)
    axes[1].errorbar(rjd, fit['residual'], yerr=total_err, fmt='o', ms=3,
                     lw=0.7, capsize=0, alpha=0.8, color='0.3')
    axes[1].axhline(0.0, color='k', lw=0.6, ls=':')
    axes[1].set_ylabel('Residual (m s$^{-1}$)')
    axes[1].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
    # EACH ORBIT FOLDED, the constant, the acceleration and the other
    #   orbits taken out: the fitted K drawn, the catalogue's dashed
    for i, (ax, orbit) in enumerate(zip(axes[2:], orbits)):
        at_conj = orbit.get('t0') is not None
        ref = orbit['t0'] if at_conj else kepler.periastron_time(orbit)
        period = float(orbit['period'])
        phase = np.mod((time - ref) / period + 0.5, 1.0) - 0.5
        own = coef[2 + i] * fit['shapes'][i]
        folded = fit['residual'] + own
        ax.errorbar(phase, folded, yerr=total_err, fmt='o', ms=3, lw=0.7,
                    capsize=0, alpha=0.8, label='LBL velocities, the rest of'
                    ' the model taken out')
        pgrid = np.linspace(-0.5, 0.5, 600)
        unit = kepler.orbit_velocity(ref + pgrid * period, dict(orbit, k=1.0))
        ax.plot(pgrid, coef[2 + i] * unit, color='C3', lw=1.4,
                label='fitted K = %.2f $\\pm$ %.2f m s$^{-1}$'
                      % (coef[2 + i], err[2 + i]))
        ax.plot(pgrid, float(orbit['k']) * unit, color='0.4', lw=1.0,
                ls='--', label='catalogue K = %.2f m s$^{-1}$'
                               % float(orbit['k']))
        ax.axhline(0.0, color='k', lw=0.6, ls=':')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel('Phase (0 = %s)' % ('inferior conjunction' if at_conj
                                           else 'periastron'))
        ax.set_ylabel('Radial velocity (m s$^{-1}$)')
        ax.set_title('%s: P = %.6g d, e = %.3f (held); K and the acceleration'
                     ' fitted together' % (orbit.get('name', '?'), period,
                                           float(orbit.get('e') or 0.0)),
                     fontsize=9)
        ax.legend(fontsize=8, loc='best')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('%s: the known orbits (%s) and an acceleration fitted'
                 ' together, with a free jitter' % (title, planets['catalog']),
                 fontsize=10)
    fig.tight_layout()
    return fig


#: the rdb column of the number of iterations compute took (KW_NITERATIONS)
ITERATIONS = 'ITE_RV'


def page_iterations(plt, table, title):
    """How many iterations compute took to converge, epoch by epoch"""
    iterations = S.column(table, ITERATIONS)
    if iterations is None:
        return None
    time = np.asarray(table[S.TIME], dtype=float)
    good = np.isfinite(iterations) & np.isfinite(time)
    time, iterations = time[good], iterations[good].astype(int)
    if len(iterations) == 0:
        return None
    median = np.median(iterations)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    axes[0].plot(time, iterations, 'o', ms=3, alpha=0.8)
    axes[0].axhline(median, color='k', lw=0.8, ls='--',
                    label='median %g' % median)
    axes[0].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
    axes[0].set_ylabel('Iterations to converge')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].set_title('%d epochs: %d to %d iterations, median %g, mean %.2f'
                      % (len(iterations), iterations.min(), iterations.max(),
                         median, np.mean(iterations)), fontsize=10)
    from matplotlib.ticker import MaxNLocator
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    bins = np.arange(iterations.min() - 0.5, iterations.max() + 1.5)
    counts, _, _ = axes[1].hist(iterations, bins=bins, color='C0', alpha=0.75,
                                rwidth=0.85)
    # the count on each bar: a few epochs beside hundreds draw no bar at all
    for value, count in zip(bins[:-1] + 0.5, counts):
        if count > 0:
            axes[1].annotate('%d' % count, xy=(value, count),
                             xytext=(0, 2), textcoords='offset points',
                             ha='center', va='bottom', fontsize=8)
    axes[1].set_ylim(0, 1.15 * max(counts))
    axes[1].set_xlabel('Iterations to converge')
    axes[1].set_ylabel('Number of epochs')
    axes[1].set_xticks(np.arange(iterations.min(), iterations.max() + 1))
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('How fast compute converged: iterations per epoch', fontsize=10)
    fig.tight_layout()
    return fig


def page_periodogram(plt, table, title, known=None):
    """The periodogram of the series, the spectral window below it, and the
    false inclusion probability under both: outlier-aware and alias-aware
    (lbl.report.fip.alias_aware), the aliases of its deepest dip marked."""
    from lbl.report import fip as _fip
    from lbl.report.periodogram import frequency_grid, gls
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
    # THE FIP ON THE SAME PERIODS, on a grid of its own: uniform in
    #   frequency, which its prior and its intervals need
    period = 1.0 / freq
    fip_freq = _fip.frequency_grid(time, pmin=float(np.min(period)),
                                   pmax=float(np.max(period)))
    robust = _fip.alias_aware(time, vrad, svrad, freq=fip_freq)
    gauss = _fip.alias_aware(time, vrad, svrad, freq=fip_freq,
                             outlier_aware=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 0.8, 1.1]))
    axes[0].plot(period, power, lw=0.9)
    axes[0].set_ylabel('GLS power (normalised, 0 to 1)')
    axes[0].set_title('the series', fontsize=9)
    axes[1].plot(period, window, lw=0.9, color='0.4')
    axes[1].set_ylabel('Window power (1 at zero frequency)')
    axes[1].set_title('the window: a peak shared with the series above is an'
                      ' alias, not a signal', fontsize=9)
    curves = []
    if gauss['freq'].size:
        curves.append(_log_fip(gauss['fip']))
        axes[2].plot(1.0 / gauss['freq'], curves[-1], lw=0.8, color='C0',
                     alpha=0.55, label='gaussian')
    if robust['freq'].size:
        curves.append(_log_fip(robust['fip']))
        axes[2].plot(1.0 / robust['freq'], curves[-1], lw=1.2, color='C4',
                     label='outlier-aware (%d epochs above one half)'
                           % int(np.sum(robust['prob'] > 0.5)))
    _fip_axis(axes[2], curves=curves)
    axes[2].set_title('false inclusion probability over intervals 1/T wide,'
                      ' every period competing for one posterior (Hara et'
                      ' al. 2022): a peak and its alias share it', fontsize=9)
    axes[2].legend(fontsize=7, loc='lower right')
    for ax in axes:
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        _landmarks(ax, known=known)
    # THE DEEPEST DIP AND ITS ALIASES, with the day and the year, on every
    #   panel: where one of them is the stronger peak of the series, the
    #   period is in doubt, and the FIP says by how much
    low, high = float(np.min(period)), float(np.max(period))
    if robust['freq'].size and robust['best_fip'] < 0.5:
        best = robust['best_period']
        drawn, close = [], []
        for alias, name in _fip.alias_periods(best, low, high):
            k = int(np.argmin(np.abs(robust['freq'] - 1.0 / alias)))
            # an alias within 3% of the peak is the same peak on this axis:
            #   said in the peak's label rather than drawn on top of it
            if abs(np.log(alias / best)) < 0.03:
                close.append('%s alias %.5g d: FIP %.2g'
                             % (name, alias, robust['fip'][k]))
            else:
                drawn.append((alias, name, k))
        axes[2].annotate('\n'.join(['%.5g d, FIP %.2g'
                                     % (best, robust['best_fip'])] + close),
                         xy=(best, _log_fip(robust['best_fip'])),
                         xytext=(4, -10), textcoords='offset points',
                         fontsize=7, color='C4', va='top')
        for alias, name, k in drawn:
            for ax in axes:
                ax.axvline(alias, color='C1', lw=0.8, ls=':', zorder=0)
            axes[2].annotate('%s alias, FIP %.2g' % (name, robust['fip'][k]),
                             xy=(alias, 0.0), xycoords=('data', 'data'),
                             xytext=(2, -2), textcoords='offset points',
                             rotation=90, va='top', ha='left', fontsize=6.5,
                             color='C1')
    axes[2].set_xlabel('Period (days)')
    fig.suptitle('Periodogram of the velocities, the spectral window of the'
                 ' sampling, and the false inclusion probability', fontsize=10)
    fig.tight_layout()
    return fig


#: the floor of the log scale. A FIP below this is "certain" as far as any
#: of this machinery can tell, and plotting it lower only stretches noise.
LOG_FIP_FLOOR = 1e-12
#: where the log axis stops when nothing goes lower, in log10 units
DEFAULT_FIP_BOTTOM = -2.0


def _log_fip(values):
    """log10 of a FIP, floored so that an exact zero still plots."""
    return np.log10(np.clip(np.asarray(values, dtype=float),
                            LOG_FIP_FLOOR, 1.0))


def _fip_axis(ax, curves=()):
    """The shared axis of the two FIP pages.

    LOG10 RATHER THAN THE PROBABILITY ITSELF: a FIP of 0.5 and one of 1e-8
    are both "detected" on a linear axis, yet they differ by eight orders of
    magnitude of confidence, and that is the whole content of the number.

    THE AXIS STOPS AT -2 UNLESS SOMETHING GOES LOWER. A series with no
    detection sits between -0.3 and 0, and an axis running down to the
    floor of the scale would squash all of it into the top centimetre. Two
    decades is enough to see the shape; when a real dip goes deeper, the
    axis follows it and the dip is what the page is about.
    """
    bottom = DEFAULT_FIP_BOTTOM
    for values in curves:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            bottom = min(bottom, float(np.min(values)))
    ax.set_xlabel('Period (days)')
    ax.set_ylabel(r'$\log_{10}$ (false inclusion probability)')
    ax.set_ylim(bottom - 0.1 * abs(bottom), 0.1)
    ax.axhline(np.log10(0.5), color='0.6', lw=0.8, ls='-.')
    ax.annotate('FIP = 0.5', xy=(0.004, np.log10(0.5)),
                xycoords=('axes fraction', 'data'),
                xytext=(0, 3), textcoords='offset points', fontsize=7,
                color='0.4')


def page_fip(plt, table, title):
    """The FIP, in its three ways of facing the outliers."""
    from lbl.report import fip as _fip
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    if len(time) < 20:
        return None
    freq, values = _fip.fip(time, vrad, svrad)
    if freq.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(1.0 / freq, _log_fip(values), lw=1.0, color='C0',
            label='gaussian, one signal')
    ax.set_xscale('log')
    # A PROBABILITY IS READ ON ITS OWN SCALE: linear, 0 to 1, with the axis
    # pinned to those bounds. A FIP near 1 means the signal is almost surely
    # absent and a FIP near 0 that it is almost surely there, and the eye
    # should see where a curve sits between the two without having to decode
    # a stretched axis.
    _fip_axis(ax, curves=[_log_fip(values)])
    _landmarks(ax)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle('False inclusion probability, every point taken at its error'
                 ' bar\nCompare with the outlier-aware page: where the two'
                 ' differ, the answer depends on how outliers are treated',
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


def _best_order(wave, low, high):
    """Which echelle order owns a window, or None if the grid is not 2D.

    The order kept is the one whose middle the window sits closest to,
    measured in units of that order's own half-width so that orders of
    different widths compare: 0 is dead centre, 1 is an edge.
    """
    wave = np.asarray(wave, dtype=float)
    if wave.ndim != 2:
        return None
    centre = 0.5 * (low + high)
    best, score = None, np.inf
    for index in range(wave.shape[0]):
        row = wave[index]
        if not np.isfinite(row).any():
            continue
        first, last = np.nanmin(row), np.nanmax(row)
        if first > high or last < low:
            continue
        half = 0.5 * (last - first)
        if half <= 0:
            continue
        offset = abs(centre - 0.5 * (first + last)) / half
        if offset < score:
            best, score = index, offset
    return best


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
        # ONE ORDER, THE ONE WHOSE CENTRE THE WINDOW SITS CLOSEST TO.
        # Consecutive echelle orders share their ends, so a window inside an
        # overlap is covered twice: once near the middle of an order, where
        # the blaze peaks and the extraction is at its best, and once at the
        # very edge of its neighbour, where the throughput has fallen off and
        # the same wavelengths are far noisier. Drawing both stacks a
        # measurement on a poor copy of itself. Measured on HARPS the
        # 2026-10-16: the water window at 647.76 nm falls in the overlap of
        # orders 64 and 65.
        chosen = _best_order(wave, low, high)
        band = ((wave > low) & (wave < high) & np.isfinite(before)
                & (before > 0))
        if chosen is not None:
            keep_order = np.zeros(wave.shape, dtype=bool)
            keep_order[chosen] = True
            band &= keep_order
        if band.sum() < 30:
            continue
        order = np.argsort(wave[band])
        w = wave[band][order]
        b = before[band][order]
        a = after[band][order]
        scale = np.nanpercentile(b, 95)
        dead = int(np.sum(~np.isfinite(a)))
        # HALF OPACITY ON THE THREE CURVES: they lie on top of one another
        # almost everywhere, and at full opacity the last one drawn simply
        # hides the others. At one half the eye sees where they agree (the
        # colours blend) and where they part (one colour shows through).
        ax.plot(w, b / scale, 'k-', lw=0.9, alpha=0.5, label='uncorrected')
        ax.plot(w, a / scale, '-', lw=0.9, color='tab:green', alpha=0.5,
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
        ax.plot(w, removed, '-', lw=1.1, color=colour, alpha=0.5,
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
        # THE FLOOR FOLLOWS THE DATA. A window whose deepest line only
        # reaches 0.8 of the continuum wastes four fifths of the panel on
        # empty space if the axis always starts at zero, and the lines one
        # came to look at are squashed into the top. The floor is therefore
        # set a little below the deepest point actually drawn, and only
        # pinned to zero when the spectrum really does go there.
        shown = np.concatenate([(b / scale)[np.isfinite(b)],
                                (a / scale)[np.isfinite(a)],
                                removed[np.isfinite(removed)]])
        bottom = float(np.nanpercentile(shown, 0.5)) if shown.size else 0.0
        top = float(np.nanpercentile(shown, 99.5)) if shown.size else 1.2
        pad = 0.08 * max(top - bottom, 0.05)
        ax.set_ylim(max(-0.05, bottom - pad) if bottom > 0.1 else -0.05,
                    max(1.1, top + pad))
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


#: the zero point measured on the tellurics, for instruments without a
#:    simultaneous reference (lbl.instruments.espadons): the RV correction
#:    they gave and its error [m/s]
TELLDV, TELLDVE = 'TELLDV', 'TELLDVE'


def page_telluric_zero_point(plt, table, title):
    """The zero point each exposure got from its tellurics (TELLDV), against
    time and against the BERV, and the velocities against it.

    WHY THIS PAGE. Without a simultaneous reference the wavelengths of an
    exposure are put where its telluric lines sit at rest, and that velocity
    is applied to the whole spectrum. Its scatter is how far the instrument
    drifts; what it follows (the BERV, the season) is what the measurement
    of the tellurics may have picked up from the star; and the velocities
    must not follow it once it is applied: a slope of -1 there means it was
    not applied, a slope of +1 that it was applied twice.
    """
    zero = S.column(table, TELLDV)
    if zero is None or not np.isfinite(zero).any() or not np.any(zero != 0):
        return None
    error = S.column(table, TELLDVE, np.zeros_like(zero))
    time = np.asarray(table[S.TIME], dtype=float)
    berv = S.column(table, S.BERV, np.full_like(zero, np.nan))
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    centred = vrad - np.median(vrad)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5))
    axes[0].errorbar(time, zero, yerr=error, fmt='o', ms=3, lw=0.7,
                     capsize=0)
    axes[0].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
    axes[0].set_ylabel('TELLDV (m s$^{-1}$)')
    axes[0].set_title('The RV correction each exposure got from its'
                      ' tellurics: rms %.1f m s$^{-1}$, median error %.1f'
                      ' m s$^{-1}$' % (np.nanstd(zero), np.nanmedian(error)),
                      fontsize=9)
    good = np.isfinite(berv) & np.isfinite(zero)
    axes[1].errorbar(berv[good], zero[good], yerr=error[good], fmt='o',
                     ms=3, lw=0.7, capsize=0)
    note = 'Too few epochs to fit a line'
    if good.sum() > 5:
        slope, _ = np.polyfit(berv[good], zero[good], 1)
        note = ('TELLDV against the BERV: %.2f m s$^{-1}$ per km s$^{-1}$,'
                ' r = %.2f (the star seen through the tellurics would'
                ' follow it)' % (slope, np.corrcoef(berv[good],
                                                    zero[good])[0, 1]))
    axes[1].set_xlabel('Barycentric Earth radial velocity (km s$^{-1}$)')
    axes[1].set_ylabel('TELLDV (m s$^{-1}$)')
    axes[1].set_title(note, fontsize=9)
    good = np.isfinite(zero) & np.isfinite(centred)
    axes[2].errorbar(zero[good], centred[good], yerr=svrad[good], fmt='o',
                     ms=3, lw=0.7, capsize=0)
    axes[2].axhline(0.0, color='k', lw=0.8, ls=':')
    note = 'Too few epochs to fit a line'
    if good.sum() > 5:
        slope, _ = np.polyfit(zero[good], centred[good], 1)
        note = ('The velocities against it: slope %.3f, r = %.2f (0 once'
                ' applied; -1 had it not been)'
                % (slope, np.corrcoef(zero[good], centred[good])[0, 1]))
    axes[2].set_xlabel('TELLDV (m s$^{-1}$)')
    axes[2].set_ylabel('Radial velocity, median removed (m s$^{-1}$)')
    axes[2].set_title(note, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle('%s: the zero point from the tellurics\n'
                 'no simultaneous reference: each exposure is moved so that'
                 ' its telluric lines sit at rest' % title, fontsize=10)
    fig.tight_layout()
    return fig


def page_berv(plt, table, title):
    """Velocity against BERV, and the slope of a line through it.

    WHY THIS PAGE. A residual that follows the BERV is not stellar: the BERV
    is the Earth's own motion, so anything correlated with it comes from the
    atmosphere, from the wavelength solution, or from a frame that was not
    undone properly. The series coloured by BERV (page_sequence, the
    report's first figure, when the star's systemic velocity is not known)
    shows it at a glance; this page measures it.
    """
    berv = S.column(table, S.BERV)
    if berv is None or not np.isfinite(berv).any():
        return None
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    centred = vrad - np.median(vrad)

    # one panel: the series coloured by BERV is the report's first figure
    fig, axes = plt.subplots(1, 1, figsize=(10, 4.6))
    axes = [None, axes]

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
    axes[1].grid(alpha=0.3)
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
    else (lbl.report.fip.oafip).

    Reading the two together is the point. Where they agree, the answer does
    not depend on how the outliers were handled. Where they differ, it does,
    and the outlier-aware one is the one to believe.
    """
    from lbl.report import fip as _fip
    time = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    if len(time) < 20:
        return None
    freq, values, prob = _fip.oafip(time, vrad, svrad)
    if freq.size == 0:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    # the gaussian curve under it, so the two are read on one axis rather
    # than by flipping pages: where they part, the answer is the outliers'
    plain_freq, plain = _fip.fip(time, vrad, svrad, freq=freq)
    if plain.size:
        axes[0].plot(1.0 / plain_freq, _log_fip(plain), lw=0.9, color='C0',
                     alpha=0.6, label='gaussian')
    axes[0].plot(1.0 / freq, _log_fip(values), lw=1.2, color='C4',
                 label='outlier-aware')
    axes[0].set_xscale('log')
    _fip_axis(axes[0], curves=[_log_fip(values)]
              + ([_log_fip(plain)] if plain.size else []))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    _landmarks(axes[0], known=known)

    # WHICH EPOCHS THE MODEL CALLS OUTLIERS, which is the other half of the
    # answer: a FIP that moved because three nights were down-weighted is a
    # different statement from one that moved because half the series was
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
                 'The outliers are modelled rather than clipped: where the'
                 ' two curves part, the answer depends on them',
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


#: activity lines drawn on one page, in two columns
ACTIVITY_PER_PAGE = 8
#: a spectrum this many sigmas from the median of a line is red, a sigma
#:    being its error bar and the line's own variability together
#:    (activity_spread)
ACTIVITY_NSIG = 4.0


def activity_lines(table, labels=None):
    """The activity lines of the rdb measured in at least one spectrum
    (lbl_compile, lbl.science.activity_lines), in the rdb's order, as (key,
    name, label).

    The columns of a line are <column>_<key>, the key being its name with
    the space as an underscore: dEW_He_1.08333 and sdEW_He_1.08333 for
    He 1.08333.
    """
    labels = labels or dict()
    out = []
    for column in table:
        if not column.startswith('dEW_'):
            continue
        if 'sdEW_' + column[4:] not in table:
            continue
        if not np.isfinite(np.asarray(table[column], dtype=float)).any():
            continue
        key = column[4:]
        name = key.replace('_', ' ', 1)
        out.append((key, name, labels.get(name, name)))
    return out


def activity_stats(table, key):
    """The measures of one activity line: (good, weighted mean, reduced
    chi2 about it), the last two NaN with fewer than two measures."""
    ew = np.asarray(table['dEW_' + key], dtype=float)
    sig = np.asarray(table['sdEW_' + key], dtype=float)
    good = np.isfinite(ew) & np.isfinite(sig) & (sig > 0)
    if np.sum(good) < 2:
        return good, float('nan'), float('nan')
    weights = 1 / sig[good] ** 2
    mean = float(np.sum(weights * ew[good]) / np.sum(weights))
    chi2 = float(np.sum(((ew[good] - mean) / sig[good]) ** 2)
                 / (np.sum(good) - 1))
    return good, mean, chi2


def activity_spread(ew, sig):
    """The variability of a line beyond its error bars: the robust scatter
    of its measures (1.4826 MAD about their median) less their median error
    bar in quadrature, 0 when the error bars explain it. A star whose H alpha
    follows its rotation (GJ 229, rms 15 mA for errors of 1 mA) is then not
    red at every epoch: what stands out is what stands out from that."""
    ew, sig = np.asarray(ew, dtype=float), np.asarray(sig, dtype=float)
    if ew.size < 3:
        return 0.0
    scatter = 1.4826 * np.median(np.abs(ew - np.median(ew)))
    return float(np.sqrt(max(scatter ** 2 - np.median(sig) ** 2, 0.0)))


def page_activity(plt, table, lines, title):
    """The differential equivalent width of each activity line, in time.

    WHAT IS PLOTTED. For each line of lbl/resources/activity_lines.yaml that
    the spectra cover, what the line absorbs more than in the template (in
    mA, lbl.science.activity_lines): 0 is the template, a line filling in
    with chromospheric emission goes negative. The spectra further than
    ACTIVITY_NSIG sigmas from the median are red, a sigma being the error
    bar and the line's own variability together (activity_spread): a flare
    lights He I and the Paschen lines together, an artefact usually one line
    only. Against the error bars alone, a line that varies with the star's
    rotation was red at every epoch.

    :param table: dict of columns, the rdb
    :param lines: (key, name, label) of the lines on this page
    :param title: str, the name of the run
    """
    if not lines:
        return None
    time = np.asarray(table[S.TIME], dtype=float)
    ncol = 2 if len(lines) > 1 else 1
    nrow = int(np.ceil(len(lines) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 1.2 + 2.4 * nrow),
                             sharex=True, squeeze=False)
    for index, (key, name, label) in enumerate(lines):
        ax = axes[index // ncol, index % ncol]
        ew = np.asarray(table['dEW_' + key], dtype=float)
        sig = np.asarray(table['sdEW_' + key], dtype=float)
        good, mean, chi2 = activity_stats(table, key)
        spread = activity_spread(ew[good], sig[good]) if good.any() else 0.0
        centre = np.median(ew[good]) if good.any() else mean
        with np.errstate(invalid='ignore', divide='ignore'):
            far = good & (np.abs(ew - centre) / np.sqrt(sig ** 2 + spread ** 2)
                          > ACTIVITY_NSIG)
        near = good & ~far
        ax.errorbar(time[near], ew[near], yerr=sig[near], fmt='o', ms=3,
                    lw=0.7, capsize=0, alpha=0.85, color='C0')
        if far.any():
            ax.errorbar(time[far], ew[far], yerr=sig[far], fmt='o', ms=3.5,
                        lw=0.7, capsize=0, color='C3')
        ax.axhline(0.0, color='k', lw=0.8, ls=':')
        ax.set_title('%s (%s): %d spectra\nrms %.2g, median error %.2g,'
                     ' variability %.2g m$\\AA$, $\\chi^2_r$ %.2g, %d beyond'
                     ' %g$\\sigma$'
                     % (label, name, np.sum(good),
                        np.std(ew[good]) if good.any() else np.nan,
                        np.median(sig[good]) if good.any() else np.nan,
                        spread, chi2, np.sum(far), ACTIVITY_NSIG),
                     fontsize=8)
        ax.set_ylabel('$\\Delta$EW (m$\\AA$)', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for index in range(len(lines), nrow * ncol):
        axes[index // ncol, index % ncol].set_visible(False)
    for col in range(ncol):
        # the lowest panel drawn in each column carries the time axis
        last = max(i for i in range(len(lines)) if i % ncol == col) \
            if any(i % ncol == col for i in range(len(lines))) else None
        if last is not None:
            axes[last // ncol, col].set_xlabel(
                'Reduced Julian date (RJD = BJD - 2400000, days)', fontsize=8)
            axes[last // ncol, col].xaxis.set_tick_params(labelbottom=True)
    fig.suptitle('Activity lines: differential equivalent width against the'
                 ' template, %s\n0 is the template, negative is a line'
                 ' filling in (emission); red: beyond %g sigmas of the median,'
                 ' error bar and the line\'s own variability together'
                 % (title, ACTIVITY_NSIG), fontsize=10)
    fig.tight_layout()
    return fig


def subtract_fit(table, result):
    """A copy of the table whose velocities have the fitted candidates and
    the known orbits taken out (lbl.report.kepfit), for the periodogram and
    the FIP after them"""
    from lbl.report import kepfit

    out = dict(table)
    time = np.asarray(table[S.TIME], dtype=float) + 2400000.0
    out[S.VELOCITY] = (np.asarray(table[S.VELOCITY], dtype=float)
                       - kepfit.model(result, time, offset=False))
    return out


def _candidate_label(cand, k, sk):
    """'TOI-904 b (CP): P = 10.877 d, K = 1.2 +- 0.4 m/s (3.0 sigma)'"""
    status = cand.get('disposition')
    return ('%s%s: P = %.6g d, K = %.2f $\\pm$ %.2f m s$^{-1}$ (%.1f$\\sigma$)'
            % (cand.get('name', '?'), ' (%s)' % status if status else '',
               float(cand['period']), k, sk, abs(k) / sk if sk > 0 else 0))


def page_candidate_fit(plt, table, title, result):
    """The velocities in time with the joint Keplerian fit of the transiting
    planets and candidates (lbl.report.kepfit), and the residuals below."""
    from lbl.report import kepfit

    rjd = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    good = np.isfinite(rjd) & np.isfinite(vrad) & np.isfinite(svrad)
    rjd, vrad, svrad = rjd[good], vrad[good], svrad[good]
    offset = result['offset']
    resid = vrad - kepfit.model(result, rjd + 2400000.0)
    # the model finely enough for the shortest period, across the span
    step = min(float(c['period']) for c in result['candidates']) / 40.0
    npoint = int(min(200000, (rjd.max() - rjd.min()) / step + 2))
    grid = np.linspace(rjd.min(), rjd.max(), max(npoint, 2))
    curve = kepfit.model(result, grid + 2400000.0) - offset

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    axes[0].plot(grid, curve, color='C3', lw=0.6, alpha=0.8,
                 label='joint Keplerian fit', zorder=3)
    axes[0].errorbar(rjd, vrad - offset, yerr=svrad, fmt='o', ms=3, lw=0.7,
                     capsize=0, alpha=0.75, label='LBL velocities', zorder=2)
    axes[0].axhline(0.0, color='k', lw=0.6, ls=':')
    axes[0].set_ylabel('Radial velocity, offset removed (m s$^{-1}$)')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].set_title('\n'.join(_candidate_label(c, k, sk) for c, k, sk in
                                zip(result['candidates'], result['k'],
                                    result['sk'])), fontsize=9)
    axes[1].errorbar(rjd, resid, yerr=np.sqrt(svrad ** 2 +
                                              result['jitter'] ** 2),
                     fmt='o', ms=3, lw=0.7, capsize=0, alpha=0.75)
    axes[1].axhline(0.0, color='k', lw=0.6, ls=':')
    axes[1].set_ylabel('Residual (m s$^{-1}$)')
    axes[1].set_xlabel('Reduced Julian date (RJD = BJD - 2400000, days)')
    axes[1].set_title('rms %.2f m s$^{-1}$ before the candidates, %.2f after;'
                      ' jitter %.2f m s$^{-1}$ (added to the error bars)'
                      % (result['rms_before'], result['rms_after'],
                         result['jitter']), fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    known = (', %d known orbit(s) held fixed' % len(result['known'])
             if result['known'] else '')
    # the reduced chi2, when the fit gives it (lbl.report.kepfit)
    chi2 = ''
    if result.get('chi2_after') is not None:
        chi2 = ('\nreduced $\\chi^2$ on the error bars alone: %.2f before'
                ' the candidates, %.2f after (%d velocities, %d degrees of'
                ' freedom); %.2f with the jitter'
                % (result['chi2_before'], result['chi2_after'],
                   result['nfit'], result['dof'], result['chi2_jitter']))
    fig.suptitle('%s: joint fit of the transiting planets and candidates,'
                 ' circular, P and T0 of the catalog fixed%s%s'
                 % (title, known, chi2), fontsize=10)
    fig.tight_layout()
    return fig


def page_candidate_phase(plt, table, title, result, nbins=12):
    """The velocities folded on each transiting planet or candidate (phase 0
    at the transit), the offset, the known orbits and the other candidates
    taken out, with the fitted Keplerian and the velocities binned in
    phase."""
    from lbl.report import kepfit

    rjd = np.asarray(table[S.TIME], dtype=float)
    vrad = np.asarray(table[S.VELOCITY], dtype=float)
    svrad = np.asarray(table[S.ERROR], dtype=float)
    good = np.isfinite(rjd) & np.isfinite(vrad) & np.isfinite(svrad)
    bjd = rjd[good] + 2400000.0
    vrad, svrad = vrad[good], svrad[good]
    error = np.sqrt(svrad ** 2 + result['jitter'] ** 2)
    ncand = len(result['candidates'])
    fig, axes = plt.subplots(ncand, 1, squeeze=False,
                             figsize=(10, 2.0 + 3.0 * ncand))
    for index, cand in enumerate(result['candidates']):
        ax = axes[index, 0]
        period, t0 = float(cand['period']), float(cand['t0'])
        phase = np.mod((bjd - t0) / period + 0.5, 1.0) - 0.5
        folded = vrad - kepfit.model(result, bjd, skip=index)
        ax.errorbar(phase, folded, yerr=error, fmt='o', ms=2.5, lw=0.6,
                    capsize=0, alpha=0.35, color='C0',
                    label='LBL velocities (error bars with the jitter)')
        # the velocities binned in phase, weighted
        edges = np.linspace(-0.5, 0.5, nbins + 1)
        which = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
        for ibin in range(nbins):
            sel = which == ibin
            if np.sum(sel) < 2:
                continue
            weights = 1.0 / error[sel] ** 2
            mean = np.sum(weights * folded[sel]) / np.sum(weights)
            ax.errorbar(0.5 * (edges[ibin] + edges[ibin + 1]), mean,
                        yerr=1.0 / np.sqrt(np.sum(weights)), fmt='s', ms=5,
                        color='k', capsize=2, lw=1.0, zorder=5,
                        label='binned' if ibin == 0 else None)
        grid = np.linspace(-0.5, 0.5, 400)
        ax.plot(grid, result['k'][index] * kepfit.basis(t0 + grid * period,
                                                        cand),
                color='C3', lw=1.4, label='fitted Keplerian', zorder=6)
        ax.axhline(0.0, color='k', lw=0.6, ls=':')
        ax.axvline(0.0, color='0.5', lw=0.6, ls='--')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel('Phase (0 = transit, T0 = %.5f BJD)' % t0)
        ax.set_ylabel('Radial velocity (m s$^{-1}$)')
        ax.set_title(_candidate_label(cand, result['k'][index],
                                      result['sk'][index]), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')
    fig.suptitle('%s: the velocities folded on each candidate, the offset,'
                 ' the known orbits and the other candidates taken out'
                 % title, fontsize=10)
    fig.tight_layout()
    return fig
