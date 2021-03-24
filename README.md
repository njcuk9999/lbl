# lbl
Line by line code for radial velocity

---

# Contents

1. [Installation](#1-installation)
    - [Step 1: Download from github](#step-1-download-the-github-repository)
    - [Step 2: Choose your branch](#step-2-choose-your-branch)
    - [Step 3: Install python 3.8 and python requirements](#step-3-install-python-38-and-required-modules)
    - [Step 4: Add to PYTHONPATH](#step-4-add-to-the-pythonpath-environment)
2. [Using LBL compute](#2-using-lbl-compute)
3. [Using LBL compile](#3-using-lbl-compil)
4. [The config file](#4-the-configuration-file)

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
cd {LBL_ROOT}/lbl
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
export PATH={LBL_ROOT}:{LBL_ROOT}/lbl/recipes/:$PATH
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


[back to top](#contents)

---

# 45. Things that the LBL code is *NOT* meant to do

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
