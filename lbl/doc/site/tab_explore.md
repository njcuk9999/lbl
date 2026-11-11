# Data Explorer

The velocity tables of the data folder, column against column: a sequence
of one star, several sequences overplotted (delivered against corrected,
two templates, two instruments), their periodograms with the false inclusion
probability, their histograms, and the template they were measured against
over the atmosphere's transmission. Above the plot, a card per star shown:
what is known of it and its published planets.

{{help explore | The tables and their flavours}}

## The star

The card sums up the star the way the
[star-o-meter](https://eartigau.github.io/starometer/) does, asked by the
window's server (`lbl/gui/starinfo.py`), which reaches archives a browser
cannot (ESA's Gaia archive and the NASA Exoplanet Archive refuse requests
from web pages).

| Part | From | Where else when it does not answer |
|---|---|---|
| names, types, position, parallax, proper motion, RV, V G J H Ks, measured Teff, [Fe/H], log g, v sin i | SIMBAD at the CDS | its mirrors at the CfA and at IDIA |
| BP, RP, astrometry, RUWE, GSP-Phot and FLAME | Gaia DR3 at ESA | GAVO's copy (no GSP-Phot, no FLAME) |
| W1 to W4 | AllWISE at GAVO | IRSA |
| planets, host parameters | NASA Exoplanet Archive (`pscomppars`) | LBL's catalog (`lbl/resources/planets`: period, K) |

A service that fails is left alone for 5 minutes. The star is asked by its
folder's name less LBL's suffixes (`_pca2d`...), then by the names of the
table's headers, as SIMBAD writes them. The answer is kept in
`DATA_DIR/gui/star_info.json` for 30 days; **Ask again** asks them again.

The relations, within the ranges they were calibrated on (a value outside
gives nothing, and the card says which and why):

| Star | Quantity | Relation | Holds for |
|---|---|---|---|
| M dwarf | [Fe/H] | Duque-Arribas et al. (2023), Eq. 4: W1-W2, BP-RP, M_G; 0.089 dex | -0.10 to 0.24 in W1-W2, 1.85 to 3.10 in G-J |
| M dwarf | Teff | Mann et al. (2015): BP-RP with J-H (49 K), with [Fe/H] (45 K), alone (52 K) | BP-RP 1.5 to 4.5, Teff 2700 to 4100 K |
| M dwarf | R | Mann et al. (2015): M_Ks (2.89 %); times 1 + 0.04458 [Fe/H] (2.70 %) | M_Ks 4.6 to 9.8 |
| M dwarf | M | Mann et al. (2015): M_Ks (1.8 %); Delfosse et al. (2000): M_K (10 %) | M_Ks 4.6 to 9.8; M_K 4.5 to 9.5 |
| AFGK dwarf | Teff, [M/H], R, M, log g, L | Gaia DR3 GSP-Phot and FLAME; else Pecaut & Mamajek (2013) dwarf sequence at BP-RP and M_G (within 1 mag of it) or at the spectral type | BP-RP -0.05 to 1.95, M_G 0.5 to 7.5 for Gaia's |

An M dwarf is one of M_G 7.5 to 16.5 or of type M; giants, subgiants,
white dwarfs and O and B stars get no derived parameters. Mann et al.
(2015) is read as its erratum (Mann et al. 2016, ApJ 819, 87) prints its
tables.

**Lines on the tellurics**: the days of the year when the star's velocity
seen from the Earth's centre, RV minus the BERV (composed relativistically),
is within 5 km/s, its lines then sitting on the telluric lines. The BERV is
the Earth's barycentric velocity on the star's direction (astropy's
ephemeris), without the Earth's rotation (under 0.5 km/s).

{{help explore_star | The star}}

{{help explore_nightly | Per night}}

{{help explore_fip | False inclusion probability}}

{{help template_explorer | The template}}
