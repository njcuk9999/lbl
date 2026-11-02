"""Every global name a function of LBL's measuring code reads exists. A name
left behind by a merge (a loop's old body after its numba kernel) stops a run
only when it is reached, so nothing short of a real run shows it otherwise."""
import builtins
import importlib
import inspect
import symtable

import pytest


def _missing(table, namespace):
    missing = []
    for symbol in table.get_symbols():
        if table.get_type() == 'function' and symbol.is_referenced() \
                and symbol.is_global() and symbol.get_name() not in namespace:
            missing.append('%s: %s' % (table.get_name(), symbol.get_name()))
    for child in table.get_children():
        missing += _missing(child, namespace)
    return missing


@pytest.mark.parametrize('name', ['lbl.science.general',
                                  'lbl.recipes.lbl_compute',
                                  'lbl.instruments.expres'])
def test_no_undefined_names(name):
    module = importlib.import_module(name)
    table = symtable.symtable(inspect.getsource(module), name, 'exec')
    namespace = set(vars(module)) | set(vars(builtins))
    assert _missing(table, namespace) == []
