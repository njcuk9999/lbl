"""What lets the window run on Windows as on macOS and Linux: the launcher a
run is started with (no shell), its process group and Stop, the process
checks, the cleanup's view of the runs going, the memory, the links, and the
page's paths.

What can run here runs for real (the launcher, Stop on this system, the
lsof fallback, the bars in the log). The Windows branches cannot: they are
run with os.name, Popen, os.kill, taskkill and the Windows API stood in for,
which checks what is asked of Windows, not what Windows does with it.
"""
import json
import ntpath
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import types

import pytest

from lbl.gui import cleanup, runner


def _wait_for(check, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


def _launch(tmp_path, body, name="wrap.py"):
    """A wrap run through the launcher, to its end: (output, exit file's
    text or None, return code)."""
    folder = tmp_path / "run"
    folder.mkdir(exist_ok=True)
    wrap = folder / name
    wrap.write_text(body, encoding="utf-8")
    exit_file = folder / "exit"
    with open(folder / "run.raw", "ab") as raw:
        proc = runner.launch(sys.executable, str(wrap), str(exit_file),
                             str(tmp_path), dict(os.environ,
                                                 PYTHONIOENCODING="utf-8"),
                             raw)
    code = proc.wait(timeout=60)
    text = (folder / "run.raw").read_text(encoding="utf-8")
    written = exit_file.read_text() if exit_file.exists() else None
    return text, written, code


# -- the launcher, for real ---------------------------------------------------

def test_the_launcher_runs_a_wrap_as_python_would(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    # a module beside the wrap: its folder is first on sys.path
    (folder / "beside.py").write_text("WORD = 'found'\n")
    out, written, code = _launch(tmp_path, (
        "import os, sys\n"
        "import beside\n"
        "print('name', __name__)\n"
        "print('argv', os.path.basename(sys.argv[0]), len(sys.argv))\n"
        "print('beside', beside.WORD)\n"
        "print('cwd', os.getcwd())\n"
        "print('caf\u00e9 \u00b1 1')\n"))
    assert code == 0 and written == "0"
    assert "name __main__" in out and "argv wrap.py 1" in out
    assert "beside found" in out
    assert "cwd %s" % os.path.realpath(tmp_path) in out \
        or "cwd %s" % tmp_path in out
    assert "caf\u00e9 \u00b1 1" in out


def test_the_launcher_passes_on_an_exit_code(tmp_path):
    out, written, code = _launch(tmp_path, "print('before')\n"
                                           "raise SystemExit(3)\n")
    assert "before" in out and written == "3" and code == 3
    out, written, code = _launch(tmp_path, "import sys\n"
                                           "sys.exit('a message')\n")
    assert "a message" in out and written == "1" and code == 1


def test_the_launcher_reports_an_exception_from_the_wrap_on(tmp_path):
    out, written, code = _launch(tmp_path, "print('first')\n"
                                           "def f():\n"
                                           "    return 1 / 0\n"
                                           "f()\n")
    assert written == "1" and code == 1
    assert "first" in out and "ZeroDivisionError" in out
    # the traceback starts at the wrap, as `python wrap.py` would print it,
    #   not in the launcher or runpy
    assert "Traceback (most recent call last)" in out
    assert "wrap.py\", line 4" in out and "runpy" not in out


def test_a_wrap_using_spawned_workers_runs_through_the_launcher(tmp_path):
    """Windows starts every worker by spawn, which imports the wrap again
    (as __mp_main__): that has to find it, here as there."""
    out, written, code = _launch(tmp_path, (
        "import multiprocessing as mp\n"
        "def square(x):\n"
        "    return x * x\n"
        "if __name__ == '__main__':\n"
        "    with mp.get_context('spawn').Pool(2) as pool:\n"
        "        print('squares', pool.map(square, [1, 2, 3]))\n"))
    assert code == 0 and written == "0", out
    assert "squares [1, 4, 9]" in out


def test_stop_ends_a_run_and_what_it_started(tmp_path):
    """macOS, Linux: SIGTERM to the run's group reaches its children."""
    if os.name == "nt":                                   # pragma: no cover
        pytest.skip("the POSIX stop")
    folder = tmp_path / "run"
    folder.mkdir()
    wrap = folder / "wrap.py"
    wrap.write_text("import subprocess, sys, time\n"
                    "child = subprocess.Popen([sys.executable, '-c',"
                    " 'import time; time.sleep(60)'])\n"
                    "print('child', child.pid, flush=True)\n"
                    "time.sleep(60)\n")
    raw_path = folder / "run.raw"
    with open(raw_path, "ab") as raw:
        proc = runner.launch(sys.executable, str(wrap),
                             str(folder / "exit"), str(tmp_path),
                             dict(os.environ), raw)
    try:
        assert _wait_for(lambda: "child" in raw_path.read_text())
        child = int(re.search(r"child (\d+)", raw_path.read_text()).group(1))
        assert runner._alive(proc.pid) and runner._alive(child)
        assert runner.stop_group(proc.pid) == "SIGTERM"
        assert proc.wait(timeout=10) == -signal.SIGTERM
        assert _wait_for(lambda: not runner._alive(child), 10)
        # killed before the launcher could write it: a stopped run has none
        assert not (folder / "exit").exists()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


# -- the window's runs, for real ----------------------------------------------

def _visible(events):
    """The log's lines as the page shows them (app.js: a 'replace' rewrites
    the last line)."""
    lines = []
    for event in events:
        if event["op"] == "replace" and lines:
            lines[-1] = event["text"]
        else:
            lines.append(event["text"])
    return lines


def test_a_run_gets_bars_in_utf8_and_a_bar_is_one_line(tmp_path):
    queue = runner.Runner()
    folder = tmp_path / "gui" / "runs" / "bars"
    folder.mkdir(parents=True)
    wrap = folder / "wrap.py"
    wrap.write_text(
        "import os, sys, time\n"
        "print('bars', os.environ.get('LBL_PROGRESS_BARS'),"
        " sys.stdout.encoding, flush=True)\n"
        "print('\u00e9t\u00e9 \u00b1 \u2588', flush=True)\n"
        "sys.stderr.write('\\rbar 1/3'); sys.stderr.flush()\n"
        "time.sleep(1.5)\n"
        "sys.stderr.write('\\rbar 2/3'); sys.stderr.flush()\n"
        "time.sleep(0.5)\n"
        "sys.stderr.write('\\r' + ' ' * 8 + '\\r'); sys.stderr.flush()\n"
        "print('done', flush=True)\n", encoding="utf-8")
    job = queue.submit("bars", str(folder), str(wrap), sys.executable,
                       str(tmp_path), {})
    # the draw a bar is on shows while the run is quiet, before the
    #   carriage return that ends it
    assert _wait_for(lambda: any(e["text"] == "bar 1/3"
                                 for e in list(job.events)), 10)
    assert "bar 2/3" not in (folder / "run.raw").read_text(encoding="utf-8")
    assert _wait_for(lambda: job.state in ("done", "failed"), 30)
    assert job.state == "done" and job.exit == 0
    texts = [e["text"] for e in job.events]
    assert "bars 1 utf-8" in texts
    assert "\u00e9t\u00e9 \u00b1 \u2588" in texts
    # each draw once, and the bar's line taken by the line after it
    bar = [(e["op"], e["text"]) for e in job.events if "bar " in e["text"]
           or e["text"] == "done"]
    assert bar == [("add", "bar 1/3"), ("replace", "bar 2/3"),
                   ("replace", "done")]
    shown = _visible(job.events)
    assert not any(line.startswith("bar ") for line in shown)
    assert "done" in shown
    # the log file: lines only, no bar, in UTF-8
    log = (tmp_path / "gui" / "logs" / "bars.log").read_text(encoding="utf-8")
    assert "\u00e9t\u00e9 \u00b1 \u2588" in log and "bar " not in log


def test_a_draw_already_shown_is_not_added_again(tmp_path):
    job = runner.Job("j", "j", str(tmp_path), "w.py", sys.executable,
                     str(tmp_path), {})
    job.add("frame 1", True)          # shown while the run was quiet
    job.add("frame 1", True)          # the same, ended by its \r
    job.add("frame 2", True)
    job.add("next line", False)
    assert [(e["op"], e["text"]) for e in job.events] == [
        ("add", "frame 1"), ("replace", "frame 2"), ("replace", "next line")]


def test_progress_bars_follow_the_variable_the_window_sets(monkeypatch):
    from lbl.pca2d import progress
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    monkeypatch.delenv("LBL_PROGRESS_BARS", raising=False)
    assert not progress._drawn()
    monkeypatch.setenv("LBL_PROGRESS_BARS", "1")
    assert progress._drawn()


def test_reports_in_a_superseded_folder_are_not_a_runs(tmp_path):
    kept = tmp_path / "pca2d" / "STAR" / "tag"
    old = tmp_path / "pca2d" / "STAR" / "superseded"
    kept.mkdir(parents=True)
    old.mkdir(parents=True)
    (kept / "a.pdf").write_bytes(b"%PDF")
    (old / "b.pdf").write_bytes(b"%PDF")
    found = runner.reports_since(str(tmp_path), 0)
    assert [os.path.basename(p) for p in found] == ["a.pdf"]


# -- the Windows branches, stood in for ---------------------------------------

@pytest.fixture
def windows(monkeypatch):
    """os.name 'nt', and the Windows-only constants this system lacks."""
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200,
                        raising=False)
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 1, raising=False)
    monkeypatch.setattr(os, "name", "nt")
    return monkeypatch


def test_windows_starts_a_run_in_a_new_process_group(windows, tmp_path):
    asked = {}

    def popen(args, **kwargs):
        asked.update(args=args, **kwargs)
        return types.SimpleNamespace(pid=4242)
    windows.setattr(subprocess, "Popen", popen)
    proc = runner.launch("C:\\env\\python.exe", "D:\\data\\wrap.py",
                         "D:\\data\\exit", "D:\\data", {}, None)
    windows.setattr(os, "name", "posix")
    assert proc.pid == 4242
    assert asked["args"] == ["C:\\env\\python.exe", "-u", "-c",
                             runner.LAUNCHER, "D:\\data\\wrap.py",
                             "D:\\data\\exit"]
    assert asked["creationflags"] == 0x200
    assert "start_new_session" not in asked
    assert asked["stderr"] == subprocess.STDOUT
    # no shell anywhere, and nothing a Windows command line has to escape
    assert "/bin/sh" not in asked["args"]
    assert '"' not in runner.LAUNCHER and "\\" not in runner.LAUNCHER


def test_windows_stop_sends_ctrl_break_then_taskkill(windows):
    kills, runs = [], []
    windows.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    windows.setattr(subprocess, "run",
                    lambda args, **kw: runs.append(args))
    assert runner.stop_group(4242) == "Ctrl+Break"
    assert kills == [(4242, signal.CTRL_BREAK_EVENT)] and runs == []
    assert runner.stop_group(4242, hard=True) == "taskkill /T /F"
    assert runs == [["taskkill", "/T", "/F", "/PID", "4242"]]
    assert len(kills) == 1                       # the hard way sends none


def test_windows_stop_without_a_shared_console_kills_at_once(windows):
    runs = []

    def refused(pid, sig):
        raise OSError(87, "no console shared with this process")
    windows.setattr(os, "kill", refused)
    windows.setattr(subprocess, "run", lambda args, **kw: runs.append(args))
    assert runner.stop_group(4242) == "taskkill /T /F"
    assert runs == [["taskkill", "/T", "/F", "/PID", "4242"]]


def test_windows_stop_of_a_run_escalates_after_the_grace(windows, tmp_path):
    kills, runs = [], []
    windows.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    windows.setattr(subprocess, "run", lambda args, **kw: runs.append(args))
    windows.setattr(runner, "GRACE", 0.05)
    windows.setattr(runner, "_alive", lambda pid: True)
    queue = runner.Runner()
    job = runner.Job("w", "w", str(tmp_path), "wrap.py", "python",
                     str(tmp_path), {})
    job.state, job.pid = "running", 4242
    with queue.lock:
        queue.jobs[job.id] = job
        queue.order.append(job.id)
    assert queue.stop(job.id)
    assert _wait_for(lambda: runs, 5)
    assert kills == [(4242, signal.CTRL_BREAK_EVENT)]
    assert runs == [["taskkill", "/T", "/F", "/PID", "4242"]]
    notes = [e["text"] for e in job.events]
    assert any("Ctrl+Break to the run" in n for n in notes)
    assert _wait_for(lambda: any("taskkill /T /F" in e["text"]
                                 for e in job.events), 5)
    job.state = "stopped"


def test_windows_never_looks_at_a_process_with_os_kill(windows):
    def forbidden(*args):
        raise AssertionError("os.kill ends a process on Windows")
    windows.setattr(os, "kill", forbidden)
    # psutil stood in for: the one here is the POSIX one, which uses os.kill
    windows.setitem(sys.modules, "psutil", types.SimpleNamespace(
        pid_exists=lambda pid: pid == 4242))
    assert runner._alive(4242) and runner._alive("4242")
    assert not runner._alive(4243)
    assert not runner._alive(None) and not runner._alive("x")


def test_windows_without_psutil_asks_the_windows_api(windows):
    import ctypes
    windows.setitem(sys.modules, "psutil", None)
    windows.setattr(os, "kill", lambda *a: (_ for _ in ()).throw(
        AssertionError("os.kill ends a process on Windows")))
    state = dict(code=259, handle=7, closed=[])

    def get_exit_code(handle, ref):
        ref._obj.value = state["code"]
        return 1
    kernel32 = types.SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: state["handle"],
        GetExitCodeProcess=get_exit_code,
        CloseHandle=lambda handle: state["closed"].append(handle))
    windows.setattr(ctypes, "windll", types.SimpleNamespace(
        kernel32=kernel32), raising=False)
    assert runner._alive(4242)                   # STILL_ACTIVE
    state["code"] = 0
    assert not runner._alive(4242)               # ended
    state["handle"] = 0
    assert not runner._alive(4242)               # no such process
    assert state["closed"] == [7, 7]


# -- the cleanup's view of the runs going -------------------------------------

def test_without_psutil_lsof_still_finds_a_run(tmp_path, monkeypatch):
    if shutil.which("lsof") is None:                      # pragma: no cover
        pytest.skip("no lsof here")
    monkeypatch.setitem(sys.modules, "psutil", None)
    proc = subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(30)"], cwd=str(tmp_path))
    try:
        assert _wait_for(lambda: any(p["pid"] == proc.pid
                                     for p in cleanup.busy(str(tmp_path))),
                         10)
    finally:
        proc.kill()
        proc.wait()


def test_no_way_to_tell_is_said_and_nothing_is_deleted(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)

    def no_lsof(*args, **kwargs):
        raise FileNotFoundError("lsof")
    monkeypatch.setattr(subprocess, "run", no_lsof)
    found = cleanup.busy(str(tmp_path))
    assert found == [dict(pid=None, command="", unknown=True)]
    cache = tmp_path / "pca2d" / "cache"
    cache.mkdir(parents=True)
    out = cleanup.delete(str(tmp_path), [str(cache)])
    assert out[0]["done"] is False and "cannot be told" in out[0]["why"]
    assert cache.exists()


def test_psutil_finds_a_run_and_its_command_line(tmp_path):
    proc = subprocess.Popen([sys.executable, "-u", "-c",
                             "import time; time.sleep(30)", "wrap.py"],
                            cwd=str(tmp_path))
    try:
        assert _wait_for(lambda: any(
            p["pid"] == proc.pid and "-c" in p["command"]
            and "wrap.py" in p["command"]
            for p in cleanup.busy(str(tmp_path))), 10)
    finally:
        proc.kill()
        proc.wait()


# -- the memory, the links -----------------------------------------------------

def test_the_memory_is_read_with_or_without_psutil(monkeypatch):
    from lbl.pca2d import machine
    with_psutil = machine.total_ram_bytes()
    assert with_psutil and with_psutil > 0
    monkeypatch.setitem(sys.modules, "psutil", None)
    if os.name != "nt":
        assert machine.total_ram_bytes() == with_psutil


def test_a_refused_link_stops_the_run_and_puts_the_folder_back(tmp_path,
                                                              monkeypatch):
    from lbl.pca2d import storage
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "outputs" / "STAR" / "tag"
    local.mkdir(parents=True)
    (local / "a.fits").write_bytes(b"spectrum")
    config = dict(output=dict(fits_directory=str(tmp_path / "disk")))

    def refused(*args, **kwargs):
        raise OSError(1314, "A required privilege is not held by the client")
    monkeypatch.setattr(os, "symlink", refused)
    said = []
    monkeypatch.setattr(storage, "log", lambda text, level: said.append(
        (level, text)))
    with pytest.raises(SystemExit) as stop:
        storage.link_dir(config, str(local))
    assert stop.value.code == 2
    # the folder is where it was, as it was
    assert local.is_dir() and not local.is_symlink()
    assert (local / "a.fits").read_bytes() == b"spectrum"
    assert not (tmp_path / "outputs" / "STAR" / "tag.__moving__").exists()
    assert any(level == "error" and "cannot be made a link" in text
               for level, text in said)
    # a folder that does not exist yet: the same clear stop
    with pytest.raises(SystemExit):
        storage.link_dir(config, str(tmp_path / "outputs" / "OTHER" / "t"))


def test_a_path_on_another_windows_drive_stays_under_the_fits_directory(
        monkeypatch):
    from lbl.pca2d import storage
    fake = types.SimpleNamespace(path=ntpath, sep="\\", altsep="/",
                                 pardir="..")
    monkeypatch.setattr(storage, "os", fake)
    monkeypatch.setattr(ntpath, "abspath", lambda p: ntpath.normpath(
        p if ntpath.isabs(p) else ntpath.join("C:\\work", p)))

    def relpath(path):
        # from C:\work: no relative path to another drive or a share
        path = ntpath.normpath(path)
        if not path.lower().startswith("c:\\work\\"):
            raise ValueError("path is on another mount than C:")
        return path[len("C:\\work\\"):]
    monkeypatch.setattr(ntpath, "relpath", relpath)
    config = dict(output=dict(fits_directory="E:\\fits"))
    assert storage.external(config, "D:\\data\\out\\STAR") \
        == "E:\\fits\\D\\data\\out\\STAR"
    assert storage.external(config, "\\\\server\\share\\out") \
        == "E:\\fits\\server_share\\out"
    assert storage.external(config, "C:\\work\\outputs\\STAR") \
        == "E:\\fits\\outputs\\STAR"


def test_windows_link_mode_says_why_it_copies(monkeypatch, tmp_path):
    from lbl.pca2d import lbl as pca2d_lbl
    monkeypatch.setattr(pca2d_lbl, "can_link", lambda folder: False)
    monkeypatch.setattr(os, "name", "nt")
    mode, why = pca2d_lbl.link_mode(str(tmp_path))
    monkeypatch.setattr(os, "name", "posix")
    assert mode == "copy" and "developer mode" in why


# -- the wrap and the page -----------------------------------------------------

def test_a_wrap_for_a_windows_interpreter_is_valid_python():
    import ast
    from lbl.gui import wrapgen, wrapload
    rparams = dict(INSTRUMENT="NIRPS_HE", DATA_SOURCE="APERO",
                   DATA_DIR="C:\\Users\\me\\data", OBJECT_SCIENCE=["GJ1214"])
    text = wrapgen.render(rparams, "x",
                          "C:\\Users\\me\\miniconda3\\envs\\lbl\\python.exe")
    ast.parse(text)
    assert wrapload.read_text(text)["rparams"]["DATA_DIR"] \
        == "C:\\Users\\me\\data"


def test_a_wrap_is_read_as_utf8_whatever_the_system_says(tmp_path):
    from lbl.gui import wrapload
    wrap = tmp_path / "wrap_x.py"
    wrap.write_text("# r\u00e9duction\n"
                    "if __name__ == '__main__':\n"
                    "    rparams = dict()\n"
                    "    rparams['OBJECT_SCIENCE'] = ['\u00c9TOILE']\n",
                    encoding="utf-8")
    assert wrapload.read(str(wrap))["rparams"]["OBJECT_SCIENCE"] \
        == ["\u00c9TOILE"]


def test_the_page_cuts_server_paths_on_both_separators():
    node = shutil.which("node")
    if node is None:                                      # pragma: no cover
        pytest.skip("no node here")
    here = os.path.dirname(os.path.abspath(__file__))
    source = open(os.path.join(here, "..", "..", "lbl", "gui", "static",
                               "app.js"), encoding="utf-8").read()
    start = source.index("const flatPath")
    end = source.index("\n}\n", source.index("function relative(")) + 3
    script = "let S = {};\n" + source[start:end] + """
const out = [];
S.data_dir = 'C:\\\\data\\\\lbl';
out.push(baseName('C:\\\\data\\\\lbl\\\\reports\\\\r.pdf'));
out.push(baseName('/data/lbl/reports/r.pdf'));
out.push(relative('C:\\\\data\\\\lbl\\\\pca2d\\\\cache'));
out.push(relative('C:\\\\other\\\\x'));
out.push(childPath(S.data_dir, 'science'));
S.data_dir = '/data/lbl/';
out.push(relative('/data/lbl/pca2d/cache'));
out.push(childPath(S.data_dir, 'science'));
console.log(JSON.stringify(out));
"""
    result = subprocess.run([node, "-e", script], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "r.pdf", "r.pdf", "pca2d\\cache", "C:\\other\\x",
        "C:\\data\\lbl\\science", "pca2d/cache", "/data/lbl/science"]
