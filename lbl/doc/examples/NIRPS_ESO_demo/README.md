# NIRPS ESO demo

Example data set to try the lbl on NIRPS ESO data

---

# Contents

1. [Download the example data set](#1-download-the-example-data-set)
2. [Setup a data directory](#2-setup-data-dir)
3. [Run the LBL](#3-run-lbl)
4. [Compare results](#4-compare-results)

---

# 1. Download the example data set

An example data of Proxima Cen. observed with NIRPS can be downloaded [here](https://udemontreal-my.sharepoint.com/:u:/g/personal/charles_cadieux_1_umontreal_ca/EYBhKsqvYixHjCELgz4XsSEBRWgXztcTP_ZDwX4kzz0Hgg?e=Q6jJoQ) (`PROXIMA_NIRPS_HE_ESO.tar`)

This data set was reduced with the standard ESO pipeline developed and maintained at the Geneva Observatory. This pipeline was adapted from the ESPRESSO data reduction software (DRS, detailed user manual available [here](https://www.eso.org/sci/software/pipelines/espresso/espresso-pipe-recipes.html)). A post-processing telluric correction (Allart et al. 2022) is also applied to the extracted 2D spectrum produced by the ESPRESSO DRS.

The example data set contains 15 spectra of Proxima Centauri observed during a NIRPS commissioning run (January 2023). It also includes 5 blaze calibration files produced by the ESPRESSO DRS and generated from daily calibration sequences. Finally, an LBL wrapper script (`NIRPS_ESO_wrap.py`) is provided.

[back to top](#contents)

---

# 2. Setup a data directory

- Extract the `PROXIMA_NIRPS_HE_ESO.tar` file into your `{DATA_DIR}` directory
- Change the `rparams['DATA_DIR']` parameter inside the `NIRPS_ESO_wrap.py` script to point to your `{DATA_DIR}` (absolute path)

[back to top](#contents)

---

# 3. Run the LBL

```bash
cd {DATA_DIR}
python NIRPS_ESO_wrap.py
```

The `NIRPS_ESO_wrap.py` wrapper will run the `lbl_template`, `lbl_mask`, `lbl_compute`, and `lbl_compile` recipes.

# 4. Compare results

The produced LBL radial velocities (`lbl_PROXIMA_PROXIMA.rdb`) will be created in `{DATA_DIR}/lblrdb`. This `.rdb` file is also available in ./lbl/doc/examples/NIRPS\_ESO\_demo/lblrdb for comparison and for testing your installation.

[back to top](#contents)
