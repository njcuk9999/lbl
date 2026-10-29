"""What to call a run, from the command it is.

Two things have to be read off a folder name in an output root months later:
WHAT was reduced, and whether it was reduced the same way as the one beside
it. The targets give the first, and no list of settings short enough to be a
folder name gives the second, so the command itself is hashed: same command,
same six characters; one flag different anywhere, a different six. Unique in
the only sense that matters here, which is that a clash is improbable rather
than impossible.

The command, and not a handful of settings: the roots a run reads and writes
and every LBL setting change what it produces as surely as the component
counts do, and hashing ten keys left two different runs under one name. What
is taken out of it, so that the name means the same on both sides and a run
resumed is the same run:

    the program's own name   `pca2d-preclean` in the window, nothing on the
                             command line: the flags are what matter
    the name itself          or it would hash the name it is computing
    the targets' order       the same set in another order is the same
                             reduction
    how much is run this time   --stages, --dry-run, --rebuild-cube,
                             --clean-cache: a run that died in its LBL stage
                             and is resumed with `--stages lbl` is the same
                             run, and must land in the folder it left
                             (2026-09-18, TOI2120 earth3 on rali)
    --lbl-before             whether the DELIVERED spectra are measured. It
                             cannot change what the corrected ones give, and
                             every scenario after a target's first passes
                             false, which would otherwise split runs that
                             belong together

The window hashes the command it shows (`gui.build_command`); the pipeline
hashes its own `sys.argv`. Both give the same six characters for the same
run.
"""

from __future__ import annotations

import hashlib
import re

#: the flags whose value is a list of targets
TARGETS = ("--objects",)
#: the flag left out of the hash
NAME = "--name"
#: flags left out of the hash, with their value when they take one: they say
#: how much of a run happens this time, not what the run is
IGNORED_WITH_VALUE = ("--stages", "--lbl-before")
IGNORED_ALONE = ("--dry-run", "--rebuild-cube", "--clean-cache")


def normalised(argv):
    """The command line a name is hashed from, as a list."""
    argv = list(argv or [])
    # whatever comes before the first flag is how this program was called:
    # `pca2d-preclean` from the window, nothing from the command line itself
    while argv and not argv[0].startswith("-"):
        argv.pop(0)
    out = []
    skip = False
    for i, word in enumerate(argv):
        if skip:
            skip = False
            continue
        flag = word.split("=", 1)[0]
        if flag == NAME or flag in IGNORED_WITH_VALUE:
            skip = "=" not in word           # its value goes with it
            continue
        if flag in IGNORED_ALONE:
            continue
        if flag in TARGETS:
            names = word.split("=", 1)[1] if "=" in word else None
            if names is None and i + 1 < len(argv):
                names, skip = argv[i + 1], True
            out += [flag, ",".join(sorted((names or "").split(",")))]
            continue
        out.append(word)
    return out


def targets_head(names, most=3):
    """The part of a name that says what was reduced."""
    names = sorted(str(n).strip() for n in (names or []) if str(n).strip())
    if not names:
        head = "run"
    elif len(names) <= most:
        head = "+".join(names)
    else:
        # three names is already a long folder name; past that, say how many
        head = "%s+%d" % ("+".join(names[:2]), len(names) - 2)
    return re.sub(r"[^0-9A-Za-z._+-]", "_", head)[:40].strip("_+") or "run"


def run_hash(argv, digits=6):
    """The six characters that say which run a command is."""
    payload = " ".join(normalised(argv))
    return hashlib.blake2b(payload.encode("utf-8"),
                           digest_size=8).hexdigest()[:digits]


def report_name(object_name, tag, run_hash=None):
    """The stem of a run's compilation PDF: <object>_<tag>[_<hash>].

    The hash is what tells two runs of the same targets and the same counts
    apart once their PDFs are side by side in a folder, or attached to a
    message (asked for on 2026-09-18). A run from before the hash existed
    has none, and keeps the shorter name.
    """
    stem = "%s_%s" % (object_name, tag)
    return "%s_%s" % (stem, run_hash) if run_hash else stem


def label_with_hash(label, stamp):
    """`label` with the command's hash after it, unless it carries it
    already: a name typed twice at different settings is two runs."""
    label = str(label)
    return label if label.endswith("_" + str(stamp)) \
        else "%s_%s" % (label, stamp)


def run_name(argv, names=None, digits=6):
    """'<targets>_<hash of the command>' for the command `argv`.

    `names` are the targets when the caller knows them; otherwise they are
    read from the command's own --object and --objects.
    """
    argv = list(argv or [])
    if names is None:
        names = []
        for i, word in enumerate(argv):
            flag, _, value = word.partition("=")
            if flag in TARGETS + ("--object",):
                value = value or (argv[i + 1] if i + 1 < len(argv) else "")
                names += [n for n in value.split(",") if n]
    return "%s_%s" % (targets_head(names), run_hash(argv, digits))
