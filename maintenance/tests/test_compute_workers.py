"""The worker processes of lbl_compute (COMPUTE_NPROC > 1) import the lbl
package of the main process, even when started from a folder that holds
another lbl package (python -c puts the current folder first)."""
import os
import subprocess
import sys

from lbl.recipes import lbl_compute


def test_a_worker_imports_the_lbl_of_the_main_process(tmp_path):
    # another lbl package in the folder the worker is started from
    fake = tmp_path / 'lbl' / 'recipes'
    fake.mkdir(parents=True)
    (tmp_path / 'lbl' / '__init__.py').write_text('')
    (fake / '__init__.py').write_text('')
    (fake / 'lbl_compute.py').write_text('')
    lbl_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(lbl_compute.__file__))))
    command = lbl_compute.WORKER_COMMAND.replace(
        'lbl_compute.worker_main(sys.argv[1])', 'print(lbl_compute.__file__)')
    out = subprocess.run([sys.executable, '-c', command, 'task', lbl_path],
                         cwd=tmp_path, capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == os.path.abspath(lbl_compute.__file__)
