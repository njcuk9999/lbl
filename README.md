# lbl
Line by line code for radial velocity

---

# Contents

1. [Installation](#1-installation)
    - [Step 1: Download from github](#step-1-download-the-github-repository)
    - [Step 2: Choose your branch](#step-2-choose-your-branch)
    - [Step 3: Install python 3.8 and python requirements](#step-3-install-python-38-and-required-modules)
    - [Step 4: Add to PYTHONPATH](#step-4-add-to-the-path-and-pythonpath-environment)
2. [Using LBL compute](#2-using-lbl-compute)
3. [Using LBL compile](#3-using-lbl-compile)
4. [The config file](#4-the-configuration-file)
5. [Things that the LBL code is NOT meant to do](#5-things-that-the-lbl-code-is-not-meant-to-do)

---

# 1. Installation
### Step 1: Download the github repository
```bash
>> git clone git@github.com:njcuk9999/lbl.git
```

Note from now on we refer to this directory as `{LBL_ROOT}`

[back to top](#contents)

---

## Step 2: Choose your branch
#### Main
The main branch should be the most stable version but may not be the most
up-to-date version.
```bash
>> git checkout main
```

#### Developer
The developer branch should be generally be a stable and update-to-date, but
may contain experimental functionality.
```bash
>> git checkout developer
```

#### Working
This is the working branch it may or may not be stable and will probably contain
experimental functionality currently in development.
```bash
>> git checkout working
```

[back to top](#contents)

---

## Step 3: Install python 3.8 and required modules
Install python 3.8 (either with venv, manually or with conda).

#### With conda:
With conda create a new environment:
```bash
conda create --name lbl-env python=3.8
```
Then activate the environment
```bash
conda activate lbl-env
```

#### Installing modules (venv, manually or conda):
Then install packages with `pip`
```bash
cd {LBL_ROOT}
pip install -r requirements.txt
```

[back to top](#contents)

---

## Step 4: Add to the PATH and PYTHONPATH environment

I.e. in `~/.bashrc` or `~/.bash_profile` or `~/.profile` or a sh script you 
can source

For bash:
```shell
export PYTHONPATH={LBL_ROOT}:$PYTHONPATH
export PATH={LBL_ROOT}:$PATH
export PATH={LBL_ROOT}/lbl/recipes/:$PATH
export PATH={LBL_ROOT}/lbl/resources/:$PATH
```


Note remember to source your profile after making these changes.

[back to top](#contents)

---

# 2. Using LBL Compute

[back to top](#contents)

Simply use the following (once configuration file set up correctly)
```bash
lbl_compute.py --config=config.yaml
```

All valid command line arguments can be found using:
```bash
lbl_compute.py --help
```

---

# 3. Using LBL Compile

[back to top](#contents)

Simply use the following (once configuration file set up correctly)

```bash
lbl_compile.py --config=config.yaml
```

All valid command line arguments can be found using:
```bash
lbl_compile.py --help
```

---

# 4. The configuration file

Copy the `config.yaml` file from the {LBL_ROOT} directory to some other location
on the computer.

This file contains by default only some of the parameters one can edit.
These can then be used in `lbl_compute` and `lbl_compile.

You can create directories that are the same are your `DATA_DIR` by using
```bash
lbl_admin.py --create_dirs --config=config.yaml
```
where config.yaml includes the path to the config.yaml file you copied (do not
use the one in the github directory).

You can then use lbl_compute or lbl_compile

```bash
lbl_compute.py --config=config.yaml
```

```bash
lbl_compile.py --config=config.yaml
```

Note an example `full_config.yaml` contains all current allowed parameters
this is only as an example it is not best to override all parameters.


## 4.1 List of parameters

|                               KEY | DEFAULT_VALUE |           SPIROU_VALUE |                                                                                                                         DESCRIPTION |
| --------------------------------- |   ----------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
|                       CONFIG_FILE |          None |                   None |                                                                                       Config file for user settings (absolute path) |
|                          DATA_DIR |          None |                   None |                                                                                                 Main data directory (absolute path) |
|                       MASK_SUBDIR |         masks |                  masks |                                                                                     mask sub directory (relative to data directory) |
|                   TEMPLATE_SUBDIR |     templates |              templates |                                                                                 template sub directory (relative to data directory) |
|                      CALIB_SUBDIR |         calib |                  calib |                                                                                    calib sub directory (relative to data directory) |
|                    SCIENCE_SUBDIR |       science |                science |                                                                                  science sub directory (relative to data directory) |
|                      LBLRV_SUBDIR |         lblrv |                  lblrv |                                                                                   LBL RV sub directory (relative to data directory) |
|                  LBLREFTAB_SUBDIR |   lblreftable |            lblreftable |                                                                            LBL ref table sub directory (relative to data directory) |
|                     LBLRDB_SUBDIR |        lblrdb |                 lblrdb |                                                                                  LBL RDB sub directory (relative to data directory) |
|                        INSTRUMENT |          None |                 SPIROU |                                                                                                               The instrument to use |
|                         SKIP_DONE |         False |                  False |                                                                                                          Whether to skip done files |
|                    OBJECT_SCIENCE |          None |                   None |                                                                                            The object name for the compute function |
|                   OBJECT_TEMPLATE |          None |                   None |                                                             The object name to use for the template (If None set to OBJECT_SCIENCE) |
|                        BLAZE_FILE |          None |                   None |                                                                          Blaze file to use (must be present in the CALIB directory) |
|                     TEMPLATE_FILE |          None |                   None |       Template file to use (if not defined will try to find template for OBJECT_TEMPLATE) must be present in theTEMPLATES directory |
|                        INPUT_FILE |          None |        *e2dsff*AB.fits |                                                                             The input file expression to use (i.e. *e2dsff*AB.fits) |
|                     REF_TABLE_FMT |           csv |                    csv |                                                                                                         Ref table format (i.e. csv) |
|                          HP_WIDTH |          None |                    223 |                                                                                                          The High pass width [km/s] |
|                     SNR_THRESHOLD |          None |                     10 |                                                                                                           The SNR cut off threshold |
|                   USE_NOISE_MODEL |         False |                  False |                                                                    Switch whether to use noise model or not for the RMS calculation |
|                  ROUGH_CCF_MIN_RV |     -300000.0 |              -300000.0 |                                                                                               The rough CCF rv minimum limit in m/s |
|                  ROUGH_CCF_MAX_RV |      300000.0 |               300000.0 |                                                                                               The rough CCF rv maximum limit in m/s | The rough CCF rv step in m/s |
|            ROUGH_CCF_EWIDTH_GUESS |          2000 |                   2000 |                                                                                           The rough CCF ewidth guess for fit in m/s |
|           COMPUTE_RV_N_ITERATIONS |            10 |                     10 |                                                                        The number of iterations to do to converge during compute RV |
|         COMPUTE_MODEL_PLOT_ORDERS |          None |                   [35] |                                            The plot order for the compute rv model plotthis can be an integer of a list of integers |
|        COMPUTE_LINE_MIN_PIX_WIDTH |             5 |                      5 |                                                                           The minimum line width (in pixels) to consider line valid |
|           COMPUTE_LINE_NSIG_THRES |             8 |                      8 |                                                                           The threshold in sigma on nsig (dv / dvrms) to keep valid |
| COMPUTE_RV_BULK_ERROR_CONVERGENCE |           0.2 |                    0.2 |                                               fraction of the bulk error the rv mean must be above for compute rv to have converged |
|       COMPUTE_RV_MAX_N_GOOD_ITERS |             8 |                      8 |                                                                        The maximum number of iterations deemed to lead to a good RV |
|                        RDB_SUFFIX |               |                        |                                                                                                    The suffix to give the rdb files |
|                   COMPIL_WAVE_MIN |          None |                    900 |                                                                                The compil minimum wavelength allowed for lines [nm] |
|                   COMPIL_WAVE_MAX |          None |                   2500 |                                                                                The compil maximum wavelength allowed for lines [nm] |
|            COMPIL_MAX_PIXEL_WIDTH |          None |                     50 |                                                                                  The maximum pixel width allowed for lines [pixels] |
|              COMPILE_BINNED_BAND1 |          None |                      H |                                                                         The first band (from get_binned_parameters) to plot (band1) |
|              COMPILE_BINNED_BAND2 |          None |                      J |                                The second band (from get_binned_parameters) to plot (band2) this is used for colour (band2 - band3) |
|              COMPILE_BINNED_BAND3 |          None |                      H |                                 The third band (from get_binned_parameters) to plot (band3) this is used for colour (band2 - band3) |
|                       FP_REF_LIST |          None |              ['FP_FP'] | define the FP reference string that defines that an FP observation was a reference (calibration) file - should be a list of strings |
|                       FP_STD_LIST |          None | ['OBJ_FP', 'POLAR_FP'] |          # define the FP standard string that defines that an FP observation was NOT a reference file - should be a list of strings |
|                              PLOT |         False |                  False |                                                                                        Whether to do plots for the compute function |
|                  PLOT_COMPUTE_CCF |         False |                  False |                                                                                                  Whether to do the compute ccf plot |
|                PLOT_COMPUTE_LINES |         False |                  False |                                                                                                 Whether to do the compute line plot |
|                 PLOT_COMPIL_CUMUL |         False |                  False |                                                                                            Whether to do the compil cumulative plot |
|                PLOT_COMPIL_BINNED |         False |                  False |                                                                                                Whether to do the compil binned plot |
|                 COMMAND_LINE_ARGS |          None |                     [] |                                                                                              storage of command line arguments used |
|                     KW_WAVECOEFFS |          None |            WAVE{0:04d} |                                                                                                        Wave coefficients header key |
|                       KW_WAVEORDN |          None |               WAVEORDN |                                                                                                       wave num orders key in header |
|                       KW_WAVEDEGN |          None |               WAVEDEGN |                                                                                                           wave degree key in header |
|                   KW_MID_EXP_TIME |          None |                 MJDMID |                                                                                                            mid exposure time in MJD |
|                            KW_SNR |          None |               EXTSN035 |                                                                                                                   snr key in header |
|                           KW_BERV |          None |                   BERV |                                                                                                  the barycentric correction keyword |
|                     KW_BLAZE_FILE |          None |               CDBBLAZE |                                                                                                          The Blaze calibration file |
|                    KW_NITERATIONS |        ITE_RV |                 ITE_RV |                                                                                                            the number of iterations |
|                  KW_SYSTEMIC_VELO |      SYSTVELO |               SYSTVELO |                                                                                                        the systemic velocity in m/s |
|                      KW_RMS_RATIO |      RMSRATIO |               RMSRATIO |                                                                                                       the rms to photon noise ratio |
|                         KW_CCF_EW |        CCF_EW |                 CCF_EW |                                                                                                              the e-width of LBL CCF |
|                       KW_HP_WIDTH |      HP_WIDTH |               HP_WIDTH |                                                                                                      the high-pass LBL width [km/s] |
|                        KW_VERSION |      LBL_VERS |               LBL_VERS |                                                                                                                     the LBL version |
|                          KW_VDATE |      LBLVDATE |               LBLVDATE |                                                                                                                     the LBL version |
|                          KW_PDATE |      LBLPDATE |               LBLPDATE |                                                                                                              the LBL processed date |
|                     KW_INSTRUMENT |      LBLINSTR |               LBLINSTR |                                                                                                              the LBL processed date |
|                         KW_MJDATE |          None |                 MJDATE |                                                                                                   the start time of the observation |
|                        KW_EXPTIME |          None |                EXPTIME |                                                                                                the exposure time of the observation |
|                        KW_AIRMASS |          None |                AIRMASS |                                                                                                      the airmass of the observation |
|                       KW_FILENAME |          None |               FILENAME |                                                                                                     the filename of the observation |
|                           KW_DATE |          None |               DATE-OBS |                                                                                                   the human date of the observation |
|                        KW_TAU_H2O |          None |                TAU_H2O |                                                                                                      the tau_h20 of the observation |
|                     KW_TAU_OTHERS |          None |               TAU_OTHE |                                                                                                    the tau_other of the observation |
|                        KW_DPRTYPE |          None |                DPRTYPE |                                                                                                      the DPRTYPE of the observation |
|                       KW_WAVETIME |          None |               WAVETIME |                                                                                     the observation time (mjd) of the wave solution |
|                       KW_WAVEFILE |          None |               WAVEFILE |                                                                                                   the filename of the wave solution |
|                       KW_TLPDVH2O |          None |               TLPDVH2O |                                                                                   the telluric TELLUCLEAN velocity of water absorbers |
|                       KW_TLPDVOTR |          None |               TLPDVOTR |                                                                                   the telluric TELLUCLEAN velocity of other absorbers |
|                        KW_CDBWAVE |          None |                CDBWAVE |                                                                                                              the wave solution used |
|                        KW_OBJNAME |          None |                OBJNAME |                                                                                                            the original object name |
|                         KW_RHOMB1 |          None |               SBRHB1_P |                                                                                                     the rhomb 1 predefined position |
|                         KW_RHOMB2 |          None |               SBRHB2_P |                                                                                                     the rhomb 2 predefined position |
|                         KW_CDEN_P |          None |               SBCDEN_P |                                                                                                         the calib-reference density |
|                        KW_SNRGOAL |          None |                SNRGOAL |                                                                                                    the SNR goal per pixel per frame |
|                        KW_EXT_SNR |          None |               EXTSN035 |                                                                                                             the SNR in chosen order |
|                            KW_BJD |          None |                    BJD |                                                                                                         The barycentric julian date |
|                       KW_SHAPE_DX |          None |               SHAPE_DX |                                                                                                             The shape code dx value |
|                       KW_SHAPE_DY |          None |               SHAPE_DY |                                                                                                             The shape code dy value |
|                        KW_SHAPE_A |          None |                SHAPE_A |                                                                                                              The shape code A value |
|                        KW_SHAPE_B |          None |                SHAPE_B |                                                                                                              The shape code B value |
|                        KW_SHAPE_C |          None |                SHAPE_C |                                                                                                              The shape code C value |
|                        KW_SHAPE_D |          None |                SHAPE_D |                                                                                                              The shape code D value |
|                        KW_REF_KEY |          None |                DPRTYPE |          define the reference header key (must also be in rdb table) to distinguish FP calibration files from FP simultaneous files |

[back to top](#contents)

---

# 5. Things that the LBL code is NOT meant to do

The purpose of the LBL library of codes is to optimally determine stellar velocities from a set of input extracted science frames. We fully understand that a number of data processing steps are required *prior* to the LBL computation and that the science analysis to derive keplerian orbits will require many more tools. We do not intend to cover the following items with the LBL, and the user is exected to perform these tasks prior/after the LBL analysis to obtain scientifically meaningful results:

- Extraction of the science data.
- Telluric absorption correction.
- OH line subtraction.
- Proper parsing of the objects in sub-directory; if you put files from different objects in a folder and call it as if they were from the same target, the code will not work.
- Proper matching of science, mask and template targets. You can use a G star as a template for a late-M and the code will run... but the results will be useless!
- Scientific analysis of the RV time series, keplerian fits, GP filtering.
- Fancy plotting; the LBL code returns big csv tables and these can be used to generate many different plots.


[back to top](#contents)

---
