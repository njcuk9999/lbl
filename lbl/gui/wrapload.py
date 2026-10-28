"""Reading an LBL wrap back into the window: its run parameters, without
running a line of it.

A wrap is Python, and the window never executes it: the file is parsed
(ast), and only what is written as a literal is taken: `rparams['KEY'] =
<literal>`, `rparams['PCA2D'] = dict(key=<literal>, ...)` or `{...}`, and
names given a literal at the top of the file (STARS = [...]) wherever they
are used. What is computed instead is listed, not guessed.

Then the run parameters become the window's state: the targets, their
templates and temperatures, LBL's steps, the PCA2D options. What the window
does not hold (INPUT_FILE, a mask file, a date range of LBL's own...) is
kept as it is and written back into every wrap the window makes from it, so
that a wrap loaded and saved again loses nothing.
"""
from __future__ import annotations

import ast
import glob
import os

from . import lblinfo

#: the run parameters the window holds itself (the rest is kept as extra)
MANAGED = {"INSTRUMENT", "DATA_SOURCE", "DATA_DIR", "DATA_TYPES",
           "OBJECT_SCIENCE", "OBJECT_COMPARISON", "OBJECT_TEFF", "PCA2D",
           "RESPROJ_TABLES", lblinfo.SKIP_PCA2D}


class _Reader(ast.NodeVisitor):
    def __init__(self):
        self.names = {}
        self.rparams = {}
        self.skipped = []

    def value(self, node):
        """A literal, a name given one, or dict(key=literal, ...)."""
        if isinstance(node, ast.Name) and node.id in self.names:
            return self.names[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "dict" and not node.args):
            return {kw.arg: self.value(kw.value) for kw in node.keywords}
        if isinstance(node, ast.Dict):
            return {self.value(k): self.value(v)
                    for k, v in zip(node.keys, node.values)}
        if isinstance(node, (ast.List, ast.Tuple)):
            items = [self.value(e) for e in node.elts]
            return items if isinstance(node, ast.List) else tuple(items)
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            # ['SCIENCE'] * len(STARS), ['SCIENCE'] * 3
            left = self.value(node.left)
            right = node.right
            if (isinstance(right, ast.Call) and isinstance(right.func, ast.Name)
                    and right.func.id == "len" and len(right.args) == 1):
                return left * len(self.value(right.args[0]))
            return left * self.value(right)
        return ast.literal_eval(node)

    def visit_Assign(self, node):                         # noqa: N802
        for target in node.targets:
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "rparams"):
                key = target.slice
                key = key.value if isinstance(key, ast.Constant) else None
                if key is None:
                    continue
                try:
                    self.rparams[key] = self.value(node.value)
                except (ValueError, TypeError, SyntaxError, KeyError):
                    self.skipped.append(key)
            elif isinstance(target, ast.Name):
                try:
                    self.names[target.id] = self.value(node.value)
                except (ValueError, TypeError, SyntaxError, KeyError):
                    pass
        self.generic_visit(node)


def read(path: str) -> dict:
    """A wrap's run parameters: dict(rparams, skipped), `skipped` the keys
    whose value is computed rather than written."""
    with open(path) as handle:
        return read_text(handle.read(), path)


def read_text(text: str, filename: str = "<wrap>") -> dict:
    """read, on a wrap's text."""
    tree = ast.parse(text, filename=filename)
    reader = _Reader()
    reader.visit(tree)
    return dict(rparams=reader.rparams, skipped=reader.skipped)


def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def to_state(rparams: dict, instrument_hint: str = "") -> dict:
    """The window's state from a wrap's run parameters: dict(state,
    notes)."""
    notes = []
    targets = [str(t) for t in _as_list(rparams.get("OBJECT_SCIENCE"))]
    comparisons = _as_list(rparams.get("OBJECT_COMPARISON"))
    teffs = _as_list(rparams.get("OBJECT_TEFF"))
    types = _as_list(rparams.get("DATA_TYPES"))
    if any(str(t).upper() != "SCIENCE" for t in types):
        notes.append("data types other than SCIENCE (%s): the window builds"
                     " SCIENCE wraps" % ", ".join(sorted(set(map(str, types)))))
    state = dict(
        data_dir=str(rparams.get("DATA_DIR") or ""),
        instrument=str(rparams.get("INSTRUMENT") or instrument_hint or ""),
        source=str(rparams.get("DATA_SOURCE") or "None"),
        ticked=targets,
        comparison={t: str(c) for t, c in zip(targets, comparisons)
                    if c and str(c) != t},
        teff={t: v for t, v in zip(targets, teffs)
              if isinstance(v, (int, float))},
        lbl={}, pca2d_on="PCA2D" in rparams, pca2d={}, pca2d_extra={},
        skip_pca2d=rparams.get(lblinfo.SKIP_PCA2D),
        resproj=bool(rparams.get("RESPROJ_TABLES")), extra={})
    switches = {row["key"] for row in lblinfo.lbl_switches()}
    for key, value in rparams.items():
        if key in switches:
            state["lbl"][key] = bool(value)
        elif key not in MANAGED:
            state["extra"][key] = value
    if state["extra"]:
        notes.append("kept as they are, the window not holding them: %s"
                     % ", ".join(sorted(state["extra"])))
    pca2d = rparams.get("PCA2D")
    if pca2d is True:
        pca2d = {}
    if isinstance(pca2d, dict):
        known = {row["key"] for row in lblinfo.pca2d_options(
            state["instrument"])} if state["instrument"] else set()
        for key, value in pca2d.items():
            if key in ("min_rjd", "max_rjd"):
                state[key] = value
            elif key in known:
                state["pca2d"][key] = value
            else:
                # section.key, a section's dict, or a short name the window
                #   does not lay out: kept and written back as it is
                state["pca2d_extra"][key] = value
        if state["pca2d_extra"]:
            notes.append("PCA2D keys kept as they are: %s"
                         % ", ".join(sorted(state["pca2d_extra"])))
    return dict(state=state, notes=notes)


def wraps_in(data_dir: str) -> list:
    """The wraps of a data folder: its own (*.py that call lbl_wrap) and
    those of the window's runs, newest first."""
    out = []
    candidates = glob.glob(os.path.join(data_dir, "*.py"))
    candidates += glob.glob(os.path.join(data_dir, "gui", "runs", "*",
                                         "wrap.py"))
    for path in candidates:
        try:
            with open(path) as handle:
                text = handle.read(20000)
        except OSError:
            continue
        if "lbl_wrap" not in text:
            continue
        out.append(dict(path=path, mtime=os.path.getmtime(path),
                        name=os.path.relpath(path, data_dir)))
    out.sort(key=lambda w: w["mtime"], reverse=True)
    return out
