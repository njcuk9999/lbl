"""
Everything public about a star, from its SIMBAD name: the velocities of
DACE and of CARMENES, and the TESS light curves, kept in one folder.

    from .gather import gather, load
    manifest = gather('GJ 436', 'archives')   # archives/GJ_436/...
    star = load('archives/GJ_436')            # star['rv'], an RVData

The folder of a star (the name as given, spaces as underscores):

    target.json          SIMBAD: main name, every identifier, position, TIC,
                         and the periods of variability it lists (ROT for a
                         rotation), with the rotation period of CARMENES
    manifest.json        what each archive gave: files, instruments, number
                         of points, time span, or why nothing
    rv/all_rv.csv        every velocity, one instrument per instrument era,
                         with the indicators of each (nan for the others):
                         RVData.from_csv(path, inst='inst')
    rv/dace/raw_<name>.csv   DACE's answer as it came (koloa.dace.fetch)
    rv/dace/<INST>.csv   one instrument era of DACE (HARPS03, HARPS15,
                         ESPRESSO19, NIRPS...): the velocities that pass the
                         pipeline's quality control, with the indicators
    rv/carmenes/raw.csv  the CARMENES DR1 rows of the star (GAVO's TAP
                         service, Ribas et al. 2023), every scalar column
    rv/carmenes/CARMENES.csv   the velocities corrected for the nightly zero
                         points (SERVAL's a_rv), with the indicators
    rv/published/        the velocities published on VizieR (koloa.published:
                         the surveys, Keck HIRES, the APF, the Lick
                         Hamilton, HARPS by SERVAL; the tables of the star's
                         papers), a CSV per source and published.json
    phot/tess/fits/      the TESS light curves of MAST, one per sector (the
                         best of SPOC 2-minute, TESS-SPOC, QLP)
    phot/tess/s<sector>_<pipeline>.csv   each sector: rjd, flux, sflux [ppt]
    phot/tess.csv        every sector together, with its sector and pipeline
    phot/kepler.csv, phot/k2.csv, phot/corot.csv   the same from Kepler, K2
                         and CoRoT when the star lay in their fields (and
                         phot/<mission>/<stretch>.csv, each quarter,
                         campaign or run)

Times are BJD - 2400000 (rjd), velocities m/s. A file that is there is read
back rather than asked for again, unless refresh=True.

DACE: with no API key, the public data only. A key is looked for in the
environment variable DACE_API_KEY, then in ~/.dacerc (koloa.dace.find_key);
with one, DACE adds what its account may see, data that may not be public,
and the manifest says which key was used (where it came from, never the
key). Some networks filter DACE: it is then reported as unreachable and the
rest goes on. CARMENES DR1 (the
GTO of 2016 to 2020) covers about 360 M dwarfs of the north; its velocities
(a_rv) are per exposure, corrected for the nightly zero points, and leave
out the exposures of nights the survey did not keep (transit sequences,
mostly). GAVO's metadata give them in km/s, but they are m/s (errors of
about 1 m/s).

Created on 2026-10-01

@author: artigau

FROM KOLOA (github.com/eartigau/koloa, 584b4d2, module gather): only the part
LBL's window uses to find and fetch public spectra, kept as it was there.
"""
import re


# =============================================================================
# Define functions
# =============================================================================
def folder_name(target: str) -> str:
    """the folder of a star: its name as given, spaces as underscores"""
    return re.sub(r'[^A-Za-z0-9+\-.]+', '_', target.strip()).strip('_')
