# LBL's vetted templates

High-SNR templates of slowly rotating M dwarfs, one per 100 K of Teff from
3000 to 4200 K, for NIRPS (HE) and SPIRou: a star with few spectra or a
narrow BERV range is measured against one of them rather than against its
own noisy template.

A wrap names one as `OBJECT_COMPARISON = 'VETTED_<object>'`; nothing else
changes. The first time LBL looks for that template it downloads
`vetted_<object>_<instrument>.fits` from `MODEL_REPO_URL` (where
`tapas_lbl.fits` is) into `DATA_DIR/models` and writes it as one of its own
templates, with its Savitzky-Golay columns (`lbl/core/vetted.py`). The list
LBL reads is `lbl/resources/vetted_templates.yaml`.

## How they were chosen (2026-10-23)

Every APERO `Template_s1dv_*` of the UdeM LBL trees (maestria:
`/cosmos99/{nirps,spirou}/lbl-data/*/templates`, 1405 templates) was read for
its number of spectra (`FTTTEMPN`) and Teff (`PP_TEFF`). In each 100 K bin,
the star with the most spectra was taken, passing over:

1. young accreting stars (SIMBAD types TT\*, Or\*, Y\*O, Y\*?, Ae\*);
2. fast rotators, from **published rotation periods**: `v_eq = 2 pi R / P`
   above 2 km/s (an upper bound on v sin i). Periods from Shan+ 2024 (A&A
   684, A9, best periods) first, then MEarth (Newton+ 2016, 2018), Diez
   Alonso+ 2019 and Lambier+ 2025, matched by position; R from the same
   catalog, or from Teff (Mann+ 2015);
3. without a period: a median high-resolution v sin i above 4 km/s (APOGEE's
   and the old compilations' left out: they put GJ 411 at 50 to 58 km/s);
4. without a period: lines measurably broadened, an H-band autocorrelation
   more than 1 km/s wider than that of slow rotators of the same instrument
   at the same Teff (`linewidth.py`).

| bin [K] | NIRPS_HE | SPIRou |
|---|---|---|
| 3000 | Teegarden's Star (38) | GJ 905 (1581) |
| 3100 | GJ 166 C (64) | GJ 1151 (613) |
| 3200 | TOI-210 (214) | GJ 1289 (1301) |
| 3300 | GJ 699 (468) | GJ 699 (1733) |
| 3400 | L 98-59 (301) | GJ 4063 (1134) |
| 3500 | TOI-756 (189) | GJ 411 (1403) |
| 3600 | GJ 3090 (351) | GJ 15 A (1484) |
| 3700 | GJ 317 (44) | GJ 880 (959) |
| 3800 | TOI-270 (239) | GJ 846 (1533) |
| 3900 | TOI-442 (18) | GJ 338 B (232) |
| 4000 | GJ 9425 (69) | TOI-1759 (215) |
| 4100 | WASP-43 (62) | none: only young stars |

The number of spectra in each template is in parentheses. What was passed
over, and why, is in `selection.json` (`dropped`).

## Remaking them

```
python vet_simbad.py; python vsini_rows.py; python prot_match.py  # SIMBAD, VizieR
python linewidth.py                                               # the templates fetched
python choose.py                                                  # selection.json
python make_vetted.py <APERO templates> selection.json <depot>    # the depot files
```

The depot files then go next to `tapas_lbl.fits` in the model repository.
