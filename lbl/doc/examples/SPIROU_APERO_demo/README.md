# SPIRou APERO demo

Example data set to try lbl on SPIRou APERO data

---

# Contents

1. [Download the example data set](#1-download-the-example-data-set)
2. [Setup a data directory](#2-setup-data-dir)
3. [Run LBL](#3-run-lbl)
4. [Compare results](#4-compare-results)

---

# 1. Download the example data set

An example data set of GL699 observed with SPIRou can be downloaded [here](https://udemontreal-my.sharepoint.com/:u:/g/personal/charles_cadieux_1_umontreal_ca/EVL-zT9NO9xPsKhN6msNe5UBdh39t9co1jTTWccCXNY6SA?e=NsyqnZ) (`GL699_SPIROU_APERO.tar`)

This data set was reduced with the APERO data reduction software ([Cook et al. 2022](doi.org/10.3847/1538-3881/ab8237)) version 0.7.282. It contains 201 spectra of GL699 observed with SPIRou as part of the SPICE CFHT large program (password protected). It also includes the associated blaze calibration files produced by APERO and generated from daily calibration sequences. Finally, an LBL wrapper script (`example_wrap_spirou_apero.py`) is provided.

[back to top](#contents)

---

# 2. Setup a data directory

- Extract the `GL699_SPIROU_APERO.tar` file into your `{DATA_DIR}` directory
- Change the `rparams['DATA_DIR']` parameter inside the `example_wrap_spirou_apero.py` script to point to your `{DATA_DIR}` (absolute path)

[back to top](#contents)

---

# 3. Run the LBL

```bash
cd {DATA_DIR}
python example_wrap_spirou_apero.py
```

The `example_wrap_spirou_apero.py` wrapper will run the `lbl_template`, `lbl_mask`, `lbl_compute`, and `lbl_compile` recipes.

# 4. Compare results

The produced LBL radial velocities (`lbl_GL699_GL699.rdb`) will be created in `{DATA_DIR}/lblrdb`. This `.rdb` file is also available in ./lbl/doc/examples/SPIROU\_APERO\_demo/lblrdb for comparison and for testing your installation.

[back to top](#contents)
