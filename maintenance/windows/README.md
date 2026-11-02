# PCA2D's river-plot windows, placed in the orders

`order_windows.py` checks and places the windows of PCA2D's figures, the
instrument's `PCA2D_WINDOWS` parameter (set in its class's
`param_override`, `lbl/instruments/<instrument>.py`): deep in the tellurics
(LBL's TAPAS model) and in the middle of an order's blaze (each
instrument's wavelength and blaze, read as PCA2D reads a real file). A
window's blaze is the lowest blaze inside it over the peak of the order
that holds it best; a window is proposed only at 0.8 or more. The blaze
file is looked for in calib/ beside the models folder given; a file read
with a flat blaze (CADC t.fits) cannot place anything, and says so.

```
python order_windows.py <DATA_DIR>/models \
    SPIROU:CADC:<a t.fits> NIRPS_HE:CADC:<a t.fits> \
    HARPS:ORIG:<an e2ds_A.fits> HARPSN:ESO:<an S2D_A.fits>:HARPS \
    ESPRESSO:None:<an S2D_A.fits>
```

For each configured window it prints its blaze and telluric content and,
when a better one is found, the proposed `centre:width` (nm): that list is
what goes into the instrument's `PCA2D_WINDOWS`. `:HARPS` at the end starts
from HARPS's windows (for an instrument that has none of its own). The
windows on a stellar line (Na D, Mg b) stay on it and are only moved a
little, unless the line sits on the blaze's flank: they then go to the
middle of the blaze, and the tool says so. The windows of every
instrument are best checked by the blaze of a real file of that
instrument; a data source whose files carry no blaze (CADC t.fits) needs
its blaze file in calib/.
