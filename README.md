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