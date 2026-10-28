"""The run engine: cutting a run's output, its levels, the queue, Stop, and
finding again the runs a closed window left going."""
import json
import os
import sys
import time

from lbl.gui import runner


def test_cut_output_keeps_bars_apart_from_lines():
    segments, rest = runner.cut_output("a\nbar 1\rbar 2\rdone\nhalf")
    assert segments == [("a", False), ("bar 1", True), ("bar 2", True),
                        ("done", False)]
    assert rest == "half"
    segments, rest = runner.cut_output("x\r\ny\n")
    assert segments == [("x", False), ("y", False)] and rest == ""


def test_levels_from_colours_and_markers():
    assert runner.level_of("\x1b[92;1mstage cube\x1b[0;0m") == "info"
    assert runner.level_of("\x1b[94;1mvalue\x1b[0;0m") == "value"
    assert runner.level_of("\x1b[93;1mhm\x1b[0;0m") == "warn"
    assert runner.level_of("\x1b[91mbad\x1b[0m") == "error"
    assert runner.level_of("2026-10-23 12:00:00.000|W| plain") == "warn"
    assert runner.level_of("nothing") == "plain"


def test_a_bar_replaces_itself_and_the_stage_is_followed(tmp_path):
    job = runner.Job("j", "j", str(tmp_path), "w.py", sys.executable,
                     str(tmp_path), {})
    job.add("", True)                         # the lone \r a bar starts with
    job.add(" 1/3", True)
    job.add(" 2/3", True)
    job.add(" 3/3", False)
    job.add("2026-10-23 12:00:00.000|I| \tLBL Compute", False)
    job.add("x|G| stage fit", False)
    job.add("tellu-cleaning E2DS for file 7 of 39", False)
    ops = [(e["op"], e["text"]) for e in job.events]
    assert ops[:3] == [("add", " 1/3"), ("replace", " 2/3"),
                       ("replace", " 3/3")]
    assert job.stage == "PCA2D fit"
    assert job.counter == "7 / 39"


def _wrap(tmp_path, body):
    path = tmp_path / "wrap.py"
    path.write_text(body)
    return str(path)


def _wait(job, states=("done", "failed", "stopped"), timeout=30):
    end = time.time() + timeout
    while job.state not in states and time.time() < end:
        time.sleep(0.1)
    return job.state


def test_a_run_its_log_and_its_exit_code(tmp_path):
    queue = runner.Runner()
    folder = tmp_path / "gui" / "runs" / "x"
    folder.mkdir(parents=True)
    wrap = _wrap(folder, "print('\\x1b[92mhello\\x1b[0m')\nraise SystemExit(3)\n")
    job = queue.submit("x", str(folder), wrap, sys.executable,
                       str(tmp_path), {})
    assert _wait(job) == "failed" and job.exit == 3
    # the log in the window's logs, not in the run's folder
    assert "hello" in (tmp_path / "gui" / "logs" / "x.log").read_text()
    assert not (folder / "run.log").exists()
    record = json.loads((folder / "run.json").read_text())
    assert record["state"] == "failed" and record["exit"] == 3


def test_runs_go_one_at_a_time_and_stop_stops(tmp_path):
    queue = runner.Runner()
    first, second = (tmp_path / "gui" / "runs" / "a",
                     tmp_path / "gui" / "runs" / "b")
    first.mkdir(parents=True)
    second.mkdir()
    slow = _wrap(first, "import time\nprint('start', flush=True)\n"
                        "time.sleep(60)\n")
    quick = _wrap(second, "print('quick')\n")
    one = queue.submit("one", str(first), slow, sys.executable,
                       str(tmp_path), {})
    two = queue.submit("two", str(second), quick, sys.executable,
                       str(tmp_path), {})
    assert _wait(one, ("running",)) == "running"
    time.sleep(0.5)
    assert two.state == "queued"
    assert queue.stop(one.id)
    assert _wait(one) == "stopped"
    assert _wait(two) == "done"


def test_a_closed_window_finds_its_runs_again(tmp_path):
    runs = tmp_path / "gui" / "runs"
    gone = runs / "a"
    gone.mkdir(parents=True)
    (gone / "run.raw").write_text("\x1b[92mfinished meanwhile\x1b[0m\n")
    (gone / "exit").write_text("0\n")
    (gone / "run.json").write_text(json.dumps(dict(
        id="a", name="a", state="running", folder=str(gone),
        wrap=str(gone / "wrap.py"), pid=999999999, created=1.0,
        started=1.0)))
    waiting = runs / "b"
    waiting.mkdir()
    _wrap(waiting, "print('later')\n")
    (waiting / "run.json").write_text(json.dumps(dict(
        id="b", name="b", state="queued", folder=str(waiting),
        wrap=str(waiting / "wrap.py"), python=sys.executable, created=2.0)))
    queue = runner.Runner()
    queue.adopt(str(tmp_path), os.path.join("gui", "runs"))
    first, second = queue.get("a"), queue.get("b")
    assert _wait(first) == "done"
    assert any("finished meanwhile" in e["text"] for e in first.events)
    assert _wait(second) == "done"
    assert "later" in (tmp_path / "gui" / "logs" / "b.log").read_text()


def test_an_older_run_log_is_moved_into_the_logs(tmp_path):
    from lbl.gui import history
    folder = tmp_path / "gui" / "runs" / "old"
    folder.mkdir(parents=True)
    (folder / "run.log").write_text("from an older window\n")
    (folder / "run.json").write_text(json.dumps(dict(id="old", name="old",
                                                     state="done")))
    record = history.window_runs(str(tmp_path))[0]
    assert record["log"] == str(tmp_path / "gui" / "logs" / "old.log")
    assert (tmp_path / "gui" / "logs" / "old.log").read_text().startswith(
        "from an older window")


def _line(text, level="I"):
    return "\x1b[94;1m2026-10-23 12:33:37.953|%s| %s\x1b[0;0m" % (level, text)


def test_the_telluric_cleaning_shows_its_two_passes(tmp_path):
    from lbl.gui import wrapgen
    rparams = dict(OBJECT_SCIENCE=["GJ1214"], RUN_LBL_TELLUCLEAN=True,
                   RUN_LBL_TEMPLATE=True, RUN_LBL_MASK=True,
                   RUN_LBL_COMPUTE=True, RUN_LBL_COMPILE=True, REPORT=True)
    steps = wrapgen.steps(rparams)
    assert [(s["kind"], s["obj"], s.get("label")) for s in steps[:5]] == [
        ("telluclean", "GJ1214", "step_telluclean_1"),
        ("template", "GJ1214_tc", "step_template_tc"),
        ("telluclean", "GJ1214", "step_telluclean_2"),
        ("template", "GJ1214_tc", "step_template_tc2"),
        ("mask", "GJ1214_tc", None)]
    # the log of lbl_wrap: the blind pass, its template, the second pass,
    #   the template made again, then LBL's own template (found on disk)
    job = runner.Job("x", "x", str(tmp_path), "wrap.py", "python",
                     str(tmp_path), dict(steps=steps))
    for text in ("\tLBL Tellu Clean", '\tOBJECT_SCIENCE="GJ1214"',
                 "\tLBL Template", '\tOBJECT_SCIENCE="GJ1214_tc"',
                 "\tLBL Tellu Clean", '\tOBJECT_SCIENCE="GJ1214"'):
        job.add(_line(text), False)
    assert [s["state"] for s in job.steps[:4]] == [
        "done", "done", "running", "pending"]
    for text in ("\tLBL Template", '\tOBJECT_SCIENCE="GJ1214_tc"',
                 "\tLBL Template", '\tOBJECT_SCIENCE="GJ1214_tc"'):
        job.add(_line(text), False)
    assert [s["state"] for s in job.steps[:5]] == [
        "done", "done", "done", "running", "pending"]
    for text in ("\tLBL Mask", '\tOBJECT_SCIENCE="GJ1214_tc"'):
        job.add(_line(text), False)
    assert [s["state"] for s in job.steps[:5]] == ["done"] * 4 + ["running"]


def test_a_counted_step_says_how_far_it_is(tmp_path):
    from lbl.gui import wrapgen
    rparams = dict(OBJECT_SCIENCE=["GJ1214"], RUN_LBL_TELLUCLEAN=False,
                   RUN_LBL_TEMPLATE=True, RUN_LBL_MASK=True,
                   RUN_LBL_COMPUTE=True, RUN_LBL_COMPILE=True, REPORT=True)
    job = runner.Job("x", "x", str(tmp_path), "wrap.py", "python",
                     str(tmp_path), dict(steps=wrapgen.steps(rparams)))
    for text in ("\tLBL Compute", '\tOBJECT_SCIENCE="GJ1214"',
                 "Processing file 30 / 90   (60 left)"):
        job.add(_line(text), False)
    going = [s for s in job.steps if s["state"] == "running"]
    assert going[0]["kind"] == "compute"
    assert abs(going[0]["progress"] - 1 / 3) < 1e-9
    # the next step starts empty, the one before is done without a bar
    for text in ("\tLBL Compil", '\tOBJECT_SCIENCE="GJ1214"'):
        job.add(_line(text), False)
    compute = next(s for s in job.steps if s["kind"] == "compute")
    assert compute["state"] == "done" and "progress" not in compute
    job.add("100%|##########| 63/63 [00:02<00:00, 25.08it/s]", True)
    going = [s for s in job.steps if s["state"] == "running"]
    assert going[0]["kind"] == "compile" and going[0]["progress"] == 1.0


def test_the_steps_tick_off_as_the_log_goes(tmp_path):
    from lbl.gui import wrapgen
    rparams = dict(OBJECT_SCIENCE=["TOI904"], RUN_LBL_TELLUCLEAN=False,
                   RUN_LBL_TEMPLATE=True, RUN_LBL_MASK=True,
                   RUN_LBL_COMPUTE=True, RUN_LBL_COMPILE=True, REPORT=True,
                   PCA2D=dict(n_star=1, name="ext"))
    steps = wrapgen.steps(rparams)
    kinds = [(s["kind"], s["obj"]) for s in steps]
    assert kinds[:5] == [("template", "TOI904"), ("mask", "TOI904"),
                         ("compute", "TOI904"), ("compile", "TOI904"),
                         ("report", "TOI904")]
    assert kinds[5:9] == [("pca2d_cube", "TOI904"), ("pca2d_fit", "TOI904"),
                          ("pca2d_figures", "TOI904"),
                          ("pca2d_correct", "TOI904")]
    assert kinds[-1] == ("report", "TOI904_pca2d_ext")
    job = runner.Job("x", "x", str(tmp_path), "wrap.py", "python",
                     str(tmp_path), dict(steps=steps))
    for text in ("\tLBL Template", '\tOBJECT_SCIENCE="TOI904"',
                 "\tLBL Template", '\tOBJECT_SCIENCE="TOI904"',
                 "\tLBL Mask", '\tOBJECT_SCIENCE="TOI904"'):
        job.add(_line(text), False)
    states = [s["state"] for s in job.steps]
    assert states[:3] == ["done", "running", "pending"]
    # where the log reached each step (a click shows that line): the first
    #   time, not the template's second pass
    assert [s.get("at") for s in job.steps[:3]] == [1, 5, None]
    for text in ("\tLBL Compute", '\tOBJECT_SCIENCE="TOI904"',
                 "\tLBL Compil", '\tOBJECT_SCIENCE="TOI904"',
                 "Report written: /a/TOI904_GL514.pdf", "stage cube",
                 "stage fit"):
        job.add(_line(text, "G" if "stage" in text else "I"), False)
    states = [s["state"] for s in job.steps]
    assert states[:7] == ["done"] * 6 + ["running"]
    assert job.stage == "PCA2D fit"
    # it fails in the fit: that step is the failed one
    job.state = "failed"
    job.finish_steps()
    assert job.steps[6]["state"] == "failed"
    assert all(s["state"] == "pending" for s in job.steps[7:])
