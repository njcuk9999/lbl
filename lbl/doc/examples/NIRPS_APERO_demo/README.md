# NIRPS APERO demo

Example data set to try the lbl on NIRPS APERO data

---

# Contents

1. [Download the example data set](#1-download-the-example-data-set)
2. [Setup a data directory](#2-setup-data-dir)
3. [Run the LBL](#3-run-lbl)
4. [Compare results](#4-compare-results)

---

# 1. Download the example data set

An example data of Proxima Cen. observed with NIRPS can be downloaded [here](https://udemontreal-my.sharepoint.com/:u:/g/personal/charles_cadieux_1_umontreal_ca/EVL-zT9NO9xPsKhN6msNe5UBdh39t9co1jTTWccCXNY6SA?e=NsyqnZ) (`PROXIMA_NIRPS_HE_APERO.tar`)

This data set was reduced with the APERO data reduction software ([Cook et al. 2022](doi.org/10.3847/1538-3881/ab8237)). It contains 30 spectra of Proxima Centauri observed during NIRPS commissioning runs (January and March 2023). It also includes the associated blaze calibration files produced by APERO and generated from daily calibration sequences. Finally, an LBL wrapper script (`NIRPS_APERO_wrap.py`) is provided.

[back to top](#contents)

---

# 2. Setup a data directory

- Extract the `PROXIMA_NIRPS_HE_APERO.tar` file into your `{DATA_DIR}` directory
- Change the `rparams['DATA_DIR']` parameter inside the `NIRPS_APERO_wrap.py` script to point to your `{DATA_DIR}` (absolute path)

[back to top](#contents)

---

# 3. Run the LBL

```bash
cd {DATA_DIR}
python NIRPS_wrap.py
```

The `NIRPS_APERO_wrap.py` wrapper will run the `lbl_template`, `lbl_mask`, `lbl_compute`, and `lbl_compile` recipes.

# 4. Compare results

The produced LBL radial velocities (`lbl_PROXIMA_PROXIMA.rdb`) will be created in `{DATA_DIR}/lblrdb`. This `.rdb` file is also available in ./lbl/doc/examples/NIRPS\_APERO\_demo/lblrdb for comparison and for testing your installation.

[back to top](#contents)
