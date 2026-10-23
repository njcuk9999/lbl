# PCA2D's river-plot windows, placed in the orders

`order_windows.py` places the windows of `output.windows` (nominal.yaml):
deep in the tellurics (LBL's TAPAS model) and as close to the centre of an
order as can be (each instrument's own wavelength solution, through LBL's
class on a real file). A window's centrality is 1 at the centre of the
order that holds it best, 0 at its edge, -1 across two orders.

```
python order_windows.py <DATA_DIR>/models \
    SPIROU:CADC:<a t.fits> NIRPS_HE:CADC:<a t.fits> \
    HARPS:ORIG:<an e2ds_A.fits> HARPSN:ESO:<an S2D_A.fits>:HARPS \
    ESPRESSO:None:<an S2D_A.fits>
```

`:HARPS` at the end starts from HARPS's windows (for an instrument that has
none of its own). The windows on a stellar line (Na D, Mg b) stay on it and
are only moved a little. Run on 2026-10-23: see the comments of each list in
nominal.yaml. SOPHIE is still to place (its archive did not answer).
