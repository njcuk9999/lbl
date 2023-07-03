# Adding an Instrument

## Step 1: Setup the Instrument

Create a new branch from the `working` branch using git.

If you do not have lbl github repo locally download it using the following:

```bash
git clone git@github.com:njcuk9999/lbl.git
```

Change into the lbl directory and create a new branch

```bash
git branch -b {instrument_name} working
```

where `{instrument_name}` is the name of the instrument you are going to add.
It should be all upper case.

This is the branch you will add your changes to please push changes and 
create a PR request when you are done.


## Step 2: Add the Instrument to `base.py`

Find the following lines in `lbl/core/base.py`

```python
# currently supported instruments
INSTRUMENTS = ['SPIROU', 'HARPS', 'ESPRESSO', 'CARMENES', 'NIRPS_HA',
               'NIRPS_HE', 'HARPSN', 'MAROONX']
```

Add your instrument (no spaces or punctuation other than underscores)


# Step 3: Add your instrument to `select.py`

First add an import similar to the following:

```python 
from lbl.instruments import {instrument_name}
```

where `{instrument name}` is the lower case version of the one added to
`base.py` in step 2.

Then add to the `InstrumentType` and `InstrumentList`

Note that we expect it in the form `{module}.{class}` where `{module}` is the
lower case instrument name and Instrument is the class name (CamelCase).
Note you will either be adding one class or a few classes (for different modes).

Next you must add to the `InstDict` (directly below `InstrumentList`) 

```python
InstDict['HARPS'] = dict()
InstDict['HARPS']['None'] = harps.Harps
InstDict['MAROONX'] = dict()
InstDict['MAROONX']['RED'] = maroonx.MaroonXRed
InstDict['MAROONX']['BLUE'] = maroonx.MaroonXBlue
```

Above you see two examples, one with a mode (MAROONX has RED and BLUE) and one
without a mode (the mode is set to `"None"`)


# Step 4: Add your instrument python code

Add your instrument python code to `lbl/instruments/` directory.

Follow the format of the other instruments.

Note you should not need to change any other code for basic instruments.

Note you can override any parmaeter (from `lbl/core/parameters.py` or any 
function in the `lbl/instruments/default.py`) any additional parameters or 
functions required should be requested via github and not manually added.


# Step 5: Push your changes

Once complete and tested please do a Pull Request on github.
It is good practise to make your branch up-to-date with the `working` branch 
before doing this (with a merge or rebase).

If you would like your instrument supported after this point we ask for a 
minimum working data set so we can run tests before releasing new versions.

This data set can remain private but we prefer public data sets to provide
users with a working example of instruments.




