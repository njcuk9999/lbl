"""
The spectra DACE (University of Geneva) serves, for the archive search:
CORALIE, every era reduced again by espdr (DACE's DRS-3.3.12, 2026), its
S2D_A files blaze corrected on a barycentric grid (LBL's CORALIE DACE).
Asked anonymously: public spectra only.

The spectroscopy service as the dace-query client asks it
(Spectroscopy.browse_products and download): the products of a star
listed by its catalogue name (download/browse/latest, the latest
reduction), then each file asked for by the id of its exposure
(download, a key) and fetched (download/<key>, one file as it is).

After koloa.dace (github.com/eartigau/koloa), which asks the same.
"""
import io
import json
import os
import re
import tarfile
import urllib.request
from typing import Any, Dict, List, Sequence, Tuple

from .paths import USER_AGENT

#: the spectroscopy service of DACE (the one dace-query uses)
SPECTROSCOPY_API = 'https://spectroscopy-webapp-pg.obsuksprd2.unige.ch'


def _post(endpoint: str, body: Dict[str, Any], timeout: float) -> Any:
    """a question to the spectroscopy service, its answer as JSON"""
    request = urllib.request.Request(
        f'{SPECTROSCOPY_API}/{endpoint}', data=json.dumps(body).encode(),
        headers={'Accept': 'application/json', 'User-Agent': USER_AGENT,
                 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _columns(payload: Any) -> Dict[str, list]:
    """DACE's answer as columns (its values are run-length encoded:
    values[i] repeats occurrences[i] times)"""
    columns: Dict[str, list] = {}
    if not isinstance(payload, dict):
        return columns
    for par in payload.get('parameters') or []:
        name = par.get('variableName')
        if not name:
            continue
        values = next((par[key] for key in ('doubleValues', 'floatValues',
                                             'intValues', 'stringValues',
                                             'boolValues')
                       if par.get(key) is not None), [])
        occ = par.get('occurrences')
        if occ:
            values = [val for val, num in zip(values, occ) for _ in range(num)]
        columns.setdefault(name, []).extend(values)
    return columns


def names(ident: Dict[str, Any]) -> List[str]:
    """the names DACE may know a star by: its catalogue names without
    spaces (HD23079, GJ551, HIP...), its SIMBAD name the last"""
    out = []
    for name in [ident.get('main') or ''] + list(ident.get('aliases') or []):
        bare = re.sub(r'^(NAME|\*\*|\*|V\*)\s+', '', name.strip())
        if bare.split(' ')[0] in ('HD', 'GJ', 'Gl', 'HIP', 'TOI', 'LHS',
                                  'Wolf', 'Ross', 'WASP', 'K2'):
            out.append(bare.replace(' ', ''))
    out.append(re.sub(r'^(NAME|\*\*|\*|V\*)\s+', '',
                      str(ident.get('main') or '').strip()).replace(' ', ''))
    return list(dict.fromkeys(one for one in out if one))


def spectra(star_names: Sequence[str], instrument: str = 'CORALIE',
            file_type: str = 'S2D_A', timeout: float = 60.0
            ) -> Tuple[str, List[Dict[str, Any]]]:
    """
    The public spectra of a star on DACE from an instrument, in their
    latest reduction

    :param star_names: the names DACE may know the star by (names()), the
                       first it has spectra for is used
    :param instrument: what the name of the instrument's eras starts with
    :param file_type: the product of each spectrum

    :return: the name used ('' when none has any) and, by date, for each
             spectrum: id (its file on DACE), spectrum (DACE's id of the
             exposure), era (CORALIE98...), name (of its file)
    """
    for name in star_names:
        cols = _columns(_post('download/browse/latest', {
            'fileType': [file_type], 'drsIds': [],
            'filters': {'target_catname': {'equal': [name]},
                        'instrument_name': {'contains': [instrument]}}},
            timeout))
        found = {}
        for path, spec, era in zip(cols.get('file_rootname', []),
                                   cols.get('spectrum_id', []),
                                   cols.get('instrument_name', [])):
            found[path] = dict(id=path, spectrum=str(spec), era=str(era),
                               name=os.path.basename(path))
        if found:
            # the night of each file is its folder: by date
            return name, sorted(found.values(), key=lambda one: one['id'])
    return '', []


def download(spectrum: str, file_type: str, path: str,
             timeout: float = 300.0) -> None:
    """a file of a spectrum (public), written to a path whole or not at
    all"""
    key = _post('download', {'fileType': [file_type], 'drsIds': [],
                             'filters': {'spectrum_id': {'equal': [
                                 str(spectrum)]}}}, timeout).get('key')
    if not key:
        raise RuntimeError(f'DACE has no {file_type} of spectrum {spectrum}')
    request = urllib.request.Request(
        f'{SPECTROSCOPY_API}/download/{key}?compressed=false',
        headers={'Accept': 'application/octet-stream',
                 'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    if not data.startswith(b'SIMPLE'):
        # several files come as a tar: the one asked for, out of it
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                member = next(one for one in tar.getmembers() if one.isfile()
                              and one.name.endswith(f'_{file_type}.fits'))
                data = tar.extractfile(member).read()
        except (tarfile.TarError, StopIteration) as err:
            raise RuntimeError(f'DACE sent no FITS file for spectrum '
                               f'{spectrum}') from err
    with open(path + '.part', 'wb') as handle:
        handle.write(data)
    os.replace(path + '.part', path)
