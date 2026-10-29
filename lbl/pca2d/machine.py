"""What this machine can actually do, as opposed to what a config asks for.

Everything here reads the hardware, so nothing here may ever reach a cube's
cache key: `output.max_memory_gb` is a CONFIG number precisely so that the same
key means the same cube on every computer (cube.should_stack). What the machine
is for is the guards: refusing a run that cannot fit, and sizing the blocks a
run is cut into.
"""
from __future__ import annotations

import os
import subprocess

#: Half, and not more. The other half is the operating system, the file cache
#: the cube is read through, LBL if it is running beside this, and the window
#: that started it. A fit allowed all of memory is a fit that gets killed with
#: no message (exit -9) by the operating system, after however long it had
#: already run.
DEFAULT_FRACTION = 0.5


def total_ram_bytes():
    """The machine's physical memory, or None if it cannot be read.

    psutil where it is installed (every system, Windows included), and without
    it os.sysconf where it knows the page count, then sysctl (macOS, BSD),
    then /proc/meminfo (Linux).
    None rather than a guess: a guard built on an invented number is worse than
    no guard, because it refuses runs that would have worked. The guards say
    so when they get None (twoframe.check_memory, spill_dir).
    """
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except (ImportError, OSError, RuntimeError):
        pass
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and size > 0:
                return int(pages) * int(size)
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def budget_bytes(fraction=DEFAULT_FRACTION):
    """What a run may use of this machine, or None if the memory is unknown."""
    total = total_ram_bytes()
    if not total:
        return None
    return int(total * float(fraction))


def describe(fraction=DEFAULT_FRACTION):
    """(total GB, budget GB) or (None, None), for saying it in a log line."""
    total = total_ram_bytes()
    if not total:
        return None, None
    return total / 1e9, total * float(fraction) / 1e9
