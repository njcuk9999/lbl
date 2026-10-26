"""The window's server: a page, its JSON, and the live log, on 127.0.0.1.

    lbl_gui [--data-dir DIR] [--port 8765] [--no-browser]

Standard library only (http.server, one thread per request): the window
installs with nothing LBL does not already need. It listens on the local
machine alone, and serves files from the data folders it has been given
only (reports, logs, wraps).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, cleanup, history, i18n, lblinfo, runner
from . import archives, simbad, targets
from . import wrapgen, wrapload

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
#: what the window remembers between sessions
STATE_FILE = os.path.expanduser("~/.lbl_pca2d_gui.json")
#: what the window serves from a data folder
SERVED = (".pdf", ".log", ".py", ".txt", ".json", ".yaml", ".csv", ".rdb",
          ".png", ".svg")


class Context:
    """What the requests share."""

    def __init__(self, data_dir: str = ""):
        self.scanner = targets.Scanner()
        self.archives = archives.Archives()
        self.runner = runner.Runner()
        self.lock = threading.Lock()
        self.state = self.load_state()
        if data_dir:
            self.state["data_dir"] = os.path.abspath(
                os.path.expanduser(data_dir))
        self.roots = set()
        if self.state.get("data_dir"):
            self.roots.add(os.path.abspath(self.state["data_dir"]))

    @staticmethod
    def load_state() -> dict:
        try:
            with open(STATE_FILE) as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def save_state(self, state: dict) -> None:
        with self.lock:
            self.state = state
            if state.get("data_dir"):
                self.roots.add(os.path.abspath(state["data_dir"]))
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(state, handle, indent=1)
            os.replace(tmp, STATE_FILE)

    def allow(self, data_dir: str) -> str:
        """A data folder the page names: its files may be served, and the
        runs it holds that this window did not start are found (once)."""
        path = os.path.abspath(os.path.expanduser(data_dir or ""))
        with self.lock:
            self.roots.add(path)
        if os.path.isdir(path):
            self.runner.adopt(path, history.GUI_RUNS)
        return path

    def servable(self, path: str) -> bool:
        real = os.path.realpath(path)
        plain = os.path.abspath(path)
        if not os.path.isfile(real) or not plain.lower().endswith(SERVED):
            return False
        with self.lock:
            roots = list(self.roots)
        return any(plain.startswith(root + os.sep) for root in roots)


def suggest_name(state: dict, rparams: dict) -> str:
    """A run's name from what it is: its targets, its components, its
    frozen basis."""
    names = list(rparams.get("OBJECT_SCIENCE") or [])
    stem = "+".join(names[:3]) + ("+%d" % (len(names) - 3)
                                  if len(names) > 3 else "")
    pca2d = rparams.get("PCA2D")
    if pca2d is None:
        return (stem or "run") + "_lbl"
    stem += "_s%s-o%s" % (pca2d.get("n_star"), pca2d.get("n_earth"))
    if pca2d.get("input_name"):
        stem += "_" + str(pca2d["input_name"])
    # the variant's own name, which its corrected spectra carry too
    if pca2d.get("name"):
        stem += "_" + str(pca2d["name"])
    return stem


class Handler(BaseHTTPRequestHandler):
    server_version = "lbl_gui/" + __version__
    ctx: Context = None
    #: a HEAD request: the headers of a file, not its body
    head = False

    # -- plumbing ------------------------------------------------------------
    def log_message(self, fmt, *args):            # quiet: the page is the log
        return

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, mime=None):
        mime = mime or mimetypes.guess_type(path)[0] or "text/plain"
        if path.endswith((".log", ".py", ".rdb", ".yaml")):
            mime = "text/plain; charset=utf-8"
        elif (mime.startswith("text/") or "javascript" in mime) \
                and "charset" not in mime:
            mime += "; charset=utf-8"
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not self.head:
            self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode())
        except ValueError:
            return {}

    # -- GET -----------------------------------------------------------------
    def do_GET(self):                                     # noqa: N802
        url = urllib.parse.urlparse(self.path)
        query = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        route = url.path
        ctx = self.ctx
        try:
            if route in ("/", "/index.html"):
                return self._file(os.path.join(STATIC, "index.html"),
                                  "text/html; charset=utf-8")
            if route == "/favicon.ico":
                # asked by browsers whatever the page says, and the only
                #   icon Safari's tabs take: a true .ico (16 to 64 px)
                return self._file(os.path.join(STATIC, "favicon.ico"),
                                  "image/x-icon")
            if route.startswith("/static/"):
                name = os.path.basename(route)
                path = os.path.join(STATIC, name)
                if os.path.isfile(path):
                    return self._file(path)
                return self.send_error(HTTPStatus.NOT_FOUND)
            if route == "/api/strings":
                return self._json(i18n.strings(query.get("lang", "en")))
            if route == "/api/state":
                return self._json(dict(state=ctx.state,
                                       python=sys.executable,
                                       version=__version__,
                                       lbl=lblinfo.lbl_home()))
            if route == "/api/instruments":
                return self._json(lblinfo.instruments())
            if route == "/api/options":
                lang = query.get("lang", "en")
                rows = lblinfo.pca2d_options(query.get("instrument", ""))
                for row in rows:
                    row.update(i18n.option_text(row["key"], lang,
                                                row["description"]))
                return self._json(dict(options=rows,
                                       switches=lblinfo.lbl_switches()))
            if route == "/api/bases":
                data_dir = ctx.allow(query.get("data_dir", ""))
                return self._json(lblinfo.bases(
                    data_dir, query.get("instrument", ""),
                    query.get("source", "")))
            if route == "/api/archive":
                return self._json(ctx.archives.status())
            if route == "/api/vetted":
                return self._json(lblinfo.vetted(query.get("instrument", "")))
            if route == "/api/templates":
                data_dir = ctx.allow(query.get("data_dir", ""))
                return self._json(lblinfo.templates(
                    data_dir, query.get("instrument", "")))
            if route == "/api/targets":
                data_dir = ctx.allow(query.get("data_dir", ""))
                science = os.path.join(data_dir, "science")
                # no folder of targets: said, not an empty table
                if not os.path.isdir(science):
                    return self._json(dict(targets=[], science="missing",
                                           folder=science,
                                           scan=ctx.scanner.status()))
                names = targets.target_names(data_dir)
                if not names:
                    return self._json(dict(targets=[], science="empty",
                                           folder=science,
                                           scan=ctx.scanner.status()))
                status = ctx.scanner.status()
                if status["data_dir"] == data_dir:
                    index = ctx.scanner.snapshot()
                else:
                    index = targets.load_index(data_dir)
                # spectra the index has not read: read them, one file per
                #   folder first (targets.Scanner), without being asked
                if not (status["running"] and status["data_dir"] == data_dir) \
                        and targets.incomplete(index, data_dir, names):
                    ctx.scanner.start(data_dir)
                    status = ctx.scanner.status()
                rows = []
                for name in names:
                    records = (index.get("targets") or {}).get(name, {})
                    row = targets.summary(records)
                    on_disk = len(targets.files_of(data_dir, name))
                    # files: on disk; read: in the index so far
                    row.update(name=name, read=row["files"], files=on_disk,
                               on_disk=on_disk)
                    # a folder fetched from an archive: its LBL instrument
                    #   and source, and the star's Teff when the headers
                    #   have none
                    origin = archives.provenance(os.path.join(science, name))
                    if origin:
                        row["archive"] = dict(lbl=origin.get("lbl"),
                                              source=origin.get("source"))
                        if row.get("teff") is None and origin.get("teff"):
                            row["teff"] = float(origin["teff"])
                    rows.append(row)
                return self._json(dict(targets=rows, scan=status,
                                       science="ok", folder=science))
            if route == "/api/points":
                data_dir = ctx.allow(query.get("data_dir", ""))
                status = ctx.scanner.status()
                index = (ctx.scanner.snapshot()
                         if status["data_dir"] == data_dir
                         else targets.load_index(data_dir))
                names = [n for n in query.get("names", "").split(",") if n]
                return self._json({name: targets.points(
                    (index.get("targets") or {}).get(name, {}))
                    for name in names})
            if route == "/api/queue":
                if query.get("data_dir"):
                    ctx.allow(query["data_dir"])
                return self._json(ctx.runner.listing())
            if route == "/api/events":
                job = ctx.runner.get(query.get("id", ""))
                if job is None:
                    return self._json(dict(error="no such run"),
                                      HTTPStatus.NOT_FOUND)
                events, upto, finished = job.events_from(
                    int(query.get("from", 0)))
                return self._json(dict(events=events, next=upto,
                                       finished=finished,
                                       job=job.summary()))
            if route == "/api/stream":
                return self._stream(query.get("id", ""),
                                    int(query.get("from", 0)))
            if route == "/api/runs":
                data_dir = ctx.allow(query.get("data_dir", ""))
                return self._json(dict(
                    window=history.window_runs(data_dir),
                    reports=history.reports(data_dir)))
            if route == "/api/wraps":
                data_dir = ctx.allow(query.get("data_dir", ""))
                return self._json(wrapload.wraps_in(data_dir))
            if route == "/api/cleanup":
                data_dir = ctx.allow(query.get("data_dir", ""))
                return self._json(dict(items=cleanup.survey(data_dir),
                                       busy=cleanup.busy(data_dir)))
            if route == "/file":
                path = query.get("path", "")
                if not ctx.servable(path):
                    return self.send_error(HTTPStatus.FORBIDDEN)
                return self._file(path)
            return self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:                          # noqa: BLE001
            return self._json(dict(error="%s: %s" % (type(exc).__name__,
                                                     exc)),
                              HTTPStatus.INTERNAL_SERVER_ERROR)

    def _stream(self, job_id: str, start: int):
        """The live log of a run, as server-sent events: every event from
        `start` on, then the new ones as they come, until the run ends."""
        job = self.ctx.runner.get(job_id)
        if job is None:
            return self.send_error(HTTPStatus.NOT_FOUND)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        position = start
        try:
            while True:
                events, position, finished = job.events_from(position,
                                                              wait=15.0)
                if events:
                    payload = json.dumps(dict(events=events, next=position,
                                              job=job.summary()))
                    self.wfile.write(("data: %s\n\n" % payload).encode())
                elif finished:
                    payload = json.dumps(dict(events=[], next=position,
                                              job=job.summary(), end=True))
                    self.wfile.write(("event: end\ndata: %s\n\n"
                                      % payload).encode())
                    self.wfile.flush()
                    return None
                else:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return None

    # -- POST ----------------------------------------------------------------
    def do_HEAD(self):                                    # noqa: N802
        """The headers of the page or of a static file (a browser asks them
        of an icon before taking it); the API answers GET only."""
        route = urllib.parse.urlparse(self.path).path
        if route not in ("/", "/index.html", "/favicon.ico") \
                and not route.startswith("/static/"):
            return self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
        self.head = True
        return self.do_GET()

    def do_POST(self):                                    # noqa: N802
        route = urllib.parse.urlparse(self.path).path
        body = self._body()
        ctx = self.ctx
        try:
            if route == "/api/teff":
                data_dir = ctx.allow(body.get("data_dir", ""))
                return self._json(simbad.teffs(data_dir,
                                               body.get("targets") or [],
                                               bool(body.get("refresh"))))
            if route == "/api/state":
                ctx.save_state(body)
                return self._json(dict(ok=True))
            if route == "/api/scan":
                data_dir = ctx.allow(body.get("data_dir", ""))
                if not os.path.isdir(os.path.join(data_dir, "science")):
                    return self._json(dict(error="%s has no science/ folder"
                                           % data_dir), HTTPStatus.BAD_REQUEST)
                ctx.scanner.start(data_dir)
                return self._json(ctx.scanner.status())
            if route == "/api/build":
                built = wrapgen.build(body)
                name = body.get("name") or suggest_name(body,
                                                        built["rparams"])
                python = body.get("python") or sys.executable
                built.update(name=name, python=python,
                             wrap=wrapgen.render(built["rparams"], name,
                                                 python),
                             steps=wrapgen.steps(built["rparams"]))
                return self._json(built)
            if route == "/api/queue":
                return self._queue(body)
            if route == "/api/archive/search":
                data_dir = ctx.allow(body.get("data_dir", "")) \
                    if body.get("data_dir") else ""
                return self._json(ctx.archives.start_search(
                    body.get("target", ""), body.get("instruments"),
                    data_dir))
            if route == "/api/archive/fetch":
                data_dir = ctx.allow(body.get("data_dir", ""))
                return self._json(ctx.archives.start_fetch(
                    data_dir, body.get("ids") or []))
            if route == "/api/archive/search/stop":
                return self._json(ctx.archives.stop_search())
            if route == "/api/archive/stop":
                ctx.archives.stop_fetch()
                return self._json(ctx.archives.status())
            if route == "/api/stop":
                return self._json(dict(ok=ctx.runner.stop(body.get("id",
                                                                   ""))))
            if route == "/api/save_wrap":
                data_dir = ctx.allow(body.get("data_dir", ""))
                name = "".join(c if c.isalnum() or c in "_-+" else "_"
                               for c in body.get("name", "run"))
                path = os.path.join(data_dir, "wrap_%s.py" % name)
                if os.path.exists(path) and not body.get("overwrite"):
                    return self._json(dict(error="%s exists" % path,
                                           exists=True), HTTPStatus.CONFLICT)
                # the window's own wrap is written commented and current,
                #   a wrap edited by hand as it is (wrapgen.window_text)
                text = body.get("wrap", "")
                if body.get("state"):
                    built = wrapgen.build(body["state"])
                    text = wrapgen.window_text(
                        text, built["rparams"], body.get("name", ""),
                        body["state"].get("python") or sys.executable)
                with open(path, "w") as handle:
                    handle.write(text)
                os.chmod(path, 0o755)
                return self._json(dict(path=path))
            if route == "/api/load_wrap":
                path = os.path.abspath(os.path.expanduser(body.get("path",
                                                                   "")))
                if not path.endswith(".py") or not os.path.isfile(path):
                    return self._json(dict(error="%s is not a wrap (.py)"
                                           % path), HTTPStatus.BAD_REQUEST)
                parsed = wrapload.read(path)
                loaded = wrapload.to_state(parsed["rparams"])
                if loaded["state"].get("data_dir"):
                    ctx.allow(loaded["state"]["data_dir"])
                name = os.path.splitext(os.path.basename(path))[0]
                if name == "wrap":
                    name = os.path.basename(os.path.dirname(path))
                elif name.startswith("wrap_"):
                    name = name[len("wrap_"):]
                loaded["state"]["name"] = name
                return self._json(dict(state=loaded["state"],
                                       notes=loaded["notes"],
                                       skipped=parsed["skipped"], path=path))
            if route == "/api/delete":
                data_dir = ctx.allow(body.get("data_dir", ""))
                return self._json(cleanup.delete(data_dir,
                                                 body.get("paths") or []))
            return self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:                          # noqa: BLE001
            return self._json(dict(error="%s: %s" % (type(exc).__name__,
                                                     exc)),
                              HTTPStatus.INTERNAL_SERVER_ERROR)

    def _queue(self, body: dict):
        """Write the run's folder and its wrap, then queue it. The wrap the
        page sends is run as it is (it may have been edited); the state is
        checked first all the same, so that a run with errors is refused."""
        state = body.get("state") or {}
        built = wrapgen.build(state)
        if built["errors"]:
            return self._json(dict(errors=built["errors"]),
                              HTTPStatus.BAD_REQUEST)
        data_dir = self.ctx.allow(state.get("data_dir", ""))
        name = body.get("name") or suggest_name(state, built["rparams"])
        python = state.get("python") or sys.executable
        text = wrapgen.window_text(body.get("wrap"), built["rparams"], name,
                                   python)
        job_id = runner.new_id(name)
        folder = os.path.join(data_dir, history.GUI_RUNS, job_id)
        os.makedirs(folder, exist_ok=True)
        wrap = os.path.join(folder, "wrap.py")
        with open(wrap, "w") as handle:
            handle.write(text)
        os.chmod(wrap, 0o755)
        job = self.ctx.runner.submit(
            name, folder, wrap, python, data_dir,
            dict(targets=built["rparams"].get("OBJECT_SCIENCE"),
                 instrument=built["rparams"].get("INSTRUMENT"),
                 pca2d=built["rparams"].get("PCA2D"),
                 steps=wrapgen.steps(built["rparams"]),
                 edited=text != wrapgen.render(built["rparams"], name,
                                               python)),
            job_id=job_id)
        return self._json(job.summary())


def free_port(preferred: int) -> int:
    import socket
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="A browser window to build, run and follow LBL wraps"
                    " with the PCA2D precleaning")
    parser.add_argument("--data-dir", default="",
                        help="the data folder to open (DATA_DIR)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the page")
    args = parser.parse_args(argv)
    Handler.ctx = Context(args.data_dir)
    port = free_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    url = "http://127.0.0.1:%d/" % server.server_address[1]
    print("lbl_gui %s at %s (LBL from %s); Ctrl+C to stop"
          % (__version__, url, lblinfo.lbl_home()), flush=True)
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        running = Handler.ctx.runner.running()
        if running is not None:
            print("a run is going (%s): it goes on, its log in %s"
                  % (running.name, running.folder), flush=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
