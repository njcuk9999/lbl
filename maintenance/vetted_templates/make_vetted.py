"""
The depot files of LBL's vetted templates, from the APERO s1dv templates the
selection chose (choose.py, selection.json).

One file per star and instrument, vetted_<OBJECT>_<instrument>.fits, next to
tapas_lbl.fits in LBL's model repository (MODEL_REPO_URL): the template's
wavelength, flux, errors and n_valid (TEMPLATE), the spectra it was made of
(SCI_TABLE), and in the header why it was chosen. LBL turns it into one of
its own templates when a wrap names VETTED_<OBJECT> (lbl.core.vetted).

    python make_vetted.py <folder of the APERO templates> <selection.json> <out folder>

The APERO templates are named <INSTRUMENT>_<APERO file name> in the folder.
"""
import datetime
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table

#: the APERO header keys kept, with the vetting keys added
KEEP = ("DRSOBJN", "PP_TEFF", "PP_TEFFS", "FTTTEMPN", "VERSION", "DRSMODE")
#: the columns of the spectra table kept (an SNR column is added if any)
SCI_COLUMNS = ("Filename", "BERV", "BJD", "USED")


def depot_name(objname: str, instrument: str) -> str:
    return 'vetted_{0}_{1}.fits'.format(objname, instrument.lower())


def make(apero_path: str, entry: dict, out_path: str):
    with fits.open(apero_path) as hdul:
        head0 = hdul[0].header
        spec = hdul[1].data
        sci = hdul[2].data if len(hdul) > 2 else None
    header = fits.Header()
    for key in KEEP:
        if key in head0:
            header[key] = (head0[key], head0.comments[key])
    header['OBJECT'] = (entry['object'], 'the star, as APERO names it')
    header['INSTRUME'] = (entry['instrument'], 'LBL instrument')
    header['VETTED'] = (True, 'an LBL vetted template')
    header['VETTEFF'] = (float(entry['teff']), '[K] Teff, APERO PP_TEFF')
    header['VETNFIL'] = (int(entry['nfiles']), 'spectra in the template')
    header['VETSIMB'] = (str(entry['simbad'] or ''), 'SIMBAD main identifier')
    header['VETPROT'] = (entry['prot'] if entry['prot'] else 'none',
                         '[d] published rotation period')
    header['VETPREF'] = (str(entry['prot_ref'] or ''), 'its catalog')
    header['VETVEQ'] = (round(entry['veq'], 2) if entry['veq'] else 'none',
                        '[km/s] 2 pi R / P_rot')
    header['VETACF'] = (round(entry['fwhm_acf'], 2) if entry['fwhm_acf']
                        else 'none', '[km/s] H-band autocorrelation FWHM')
    header['VETSRC'] = ('{0}/{1}'.format(entry['tree'], entry['file']),
                        'APERO tree and template')
    header['VETDATE'] = (datetime.date.today().isoformat(), 'made on')
    table = Table()
    table['wavelength'] = np.asarray(spec['wavelength'], dtype=np.float64)
    for col in ('flux', 'eflux', 'rms'):
        table[col] = np.asarray(spec[col], dtype=np.float32)
    table['n_valid'] = np.asarray(spec['n_valid'], dtype=np.int32)
    hdus = [fits.PrimaryHDU(header=header),
            fits.BinTableHDU(table, name='TEMPLATE')]
    if sci is not None:
        cols = [c for c in SCI_COLUMNS if c in sci.columns.names]
        cols += [c for c in sci.columns.names if c.startswith('SNR')][:1]
        hdus.append(fits.BinTableHDU(Table([sci[c] for c in cols],
                                           names=cols), name='SCI_TABLE'))
    fits.HDUList(hdus).writeto(out_path, overwrite=True)


def main(apero_dir: str, selection: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    chosen = json.load(open(selection))['chosen']
    for entry in chosen:
        apero = os.path.join(apero_dir, '{0}_{1}'.format(entry['instrument'],
                                                         entry['file']))
        out = os.path.join(out_dir, depot_name(entry['object'],
                                               entry['instrument']))
        make(apero, entry, out)
        print(out, '%.1f MB' % (os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main(*sys.argv[1:4])
