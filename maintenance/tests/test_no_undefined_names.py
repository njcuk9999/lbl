"""Every global name a function of LBL reads exists. A name left behind by a
merge (a loop's old body after its numba kernel) stops a run only when it is
reached, so nothing short of a real run shows it otherwise."""
import builtins
import importlib
import inspect
import os
import symtable

import pytest

import lbl

#: not modules of the package: folders of scripts run by hand
LEFT_OUT = (os.path.join('gui', 'tools'),)


def _modules():
    """Every module of the package lbl, by its dotted name: the folders with
    an __init__.py (the doc's example scripts are not in one)."""
    root = os.path.dirname(os.path.abspath(lbl.__file__))
    names = []
    for here, subs, files in os.walk(root):
        relative = os.path.relpath(here, root)
        subs[:] = [s for s in subs
                   if os.path.exists(os.path.join(here, s, '__init__.py'))
                   and os.path.normpath(os.path.join(relative, s))
                   not in LEFT_OUT]
        for name in files:
            if name.endswith('.py'):
                parts = os.path.normpath(os.path.join(
                    'lbl', relative, name[:-3])).split(os.sep)
                names.append('.'.join(parts[:-1] if parts[-1] == '__init__'
                                      else parts))
    return sorted(names)


def _missing(table, namespace):
    missing = []
    for symbol in table.get_symbols():
        if table.get_type() == 'function' and symbol.is_referenced() \
                and symbol.is_global() and symbol.get_name() not in namespace:
            missing.append('%s: %s' % (table.get_name(), symbol.get_name()))
    for child in table.get_children():
        missing += _missing(child, namespace)
    return missing


def test_the_sweep_finds_the_package():
    names = _modules()
    assert len(names) > 100
    assert 'lbl.pca2d.reconstruct' in names
    assert 'lbl.report.fip' in names


@pytest.mark.parametrize('name', _modules())
def test_no_undefined_names(name):
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        pytest.skip('not importable here: %s' % exc)
    table = symtable.symtable(inspect.getsource(module), name, 'exec')
    # and what the module assigns anywhere at its top level, its
    #   `if __name__ == "__main__":` block included, which a script's
    #   functions read when it runs as one
    assigned = {s.get_name() for s in table.get_symbols() if s.is_assigned()
                or s.is_imported()}
    namespace = set(vars(module)) | set(vars(builtins)) | assigned
    assert _missing(table, namespace) == []
