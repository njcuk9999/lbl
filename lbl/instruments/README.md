# Instrument directory

- This contains instrument python scripts (and `select.py`)

- Each instrument contains a class i.e. for SPIRou:

```python
from lbl.instruments import default


class Spirou(default.Instrument):
    def __init__(self, params):
        # call to super function
        # noinspection PyTypeChecker
        super().__init__("SPIROU")
        # set parameters for instrument
        self.params = params
        # constructor
        pass

```

These functions should be overloaded to provide instrument specific functionality

Note as well as adding the `{instrument}.py` file you must add a line to 
`select.py`:


```

from lbl.instruments import {INSTRUMENT}

# ...
# and in load_instrument():

# select SPIROU
if instrument.upper() == 'SPIROU':
    inst = spirou.Spirou(params)
# select HARPS
elif instrument.upper() == 'HARPS':
    inst = harps.Harps(params)
    
# select {INSTRUMENT}
elif instrument.upper() == '{INSTRUMENT}':
    inst = instrument.Instrument(params)
    
# else instrument is invalid
else:
    emsg = 'Instrument name "{0}" invalid'
    eargs = [instrument]
    raise base_classes.LblException(emsg.format(*eargs))

```


## Param Override method

This method is used to override default parameters (these in turn can be overridden
from command line / yaml / main function inputs), but if these are not overriden
by command line / yaml / main function and the default value (found in 
`core.parameters.py`) is not correct for the instrument it should be overriden here.

## Spectrum read function

TODO

## Template read function

TODO

## Blaze read function

TODO

## Wave read function

TODO