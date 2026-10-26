"""The screenshots of the step-by-step guide (eartigau.github.io/lbl-gui),
taken in a running window and annotated: a frame around what to fill or
click, a numbered badge, an arrow from one to the other. The numbers are
the ones the guide's text explains, in both languages.

    python tools/tutorial_shots.py URL OUT LANG PHASE

URL a window (started on the guide's demo folder, with a HOME of its own so
that it does not touch the user's), OUT the folder the images go to, LANG
fr or en, PHASE one of
    setup     the window before a run: targets, settings, the wrap
    launch    the same, and the run started (once, in one language)
    running   the run going: its steps and its log
    archive   the archive search, searched and not fetched
    done      the runs and their report

Needs playwright (with the Google Chrome installed on the machine) and
Pillow, in an environment of their own.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotate import WIDTH, annotate                       # noqa: E402


class Shots:
    def __init__(self, page, out, lang):
        self.page, self.out, self.lang = page, out, lang

    def box(self, selector, nth=0):
        loc = self.page.locator(selector)
        if loc.count() <= nth:
            print("  no element for %s" % selector)
            return None
        loc.nth(nth).scroll_into_view_if_needed()
        return loc.nth(nth).bounding_box()

    def doc_box(self, selector, nth=0):
        """An element's box in the coordinates of the whole page."""
        box = self.box(selector, nth)
        if box is None:
            return None
        return dict(box, y=box["y"] + self.page.evaluate("() => window.scrollY"))

    def region(self, name, top_selector, marks, bottom_selector=None,
               margin=60, height=None, column=None):
        """A screenshot from the top of one element to the bottom of another
        (the same by default), the page's width, annotated; every box in
        the coordinates of the whole page."""
        page = self.page
        time.sleep(0.4)
        t = self.doc_box(top_selector)
        b = self.doc_box(bottom_selector) if bottom_selector else t
        boxes = []
        for number, selector, side in marks:
            bx = self.doc_box(*selector) if isinstance(selector, tuple) \
                else self.doc_box(selector)
            boxes.append((number, bx, side))
        y0 = max(0.0, t["y"] - margin)
        y1 = (y0 + height) if height else b["y"] + b["height"] + margin
        # `column`: only the page's column that holds this element (the
        #   plots sit in a narrow one), with room for the badges
        x0, width = 0.0, WIDTH
        if column:
            c = self.doc_box(column)
            x0 = max(0.0, c["x"] - 70)
            width = min(WIDTH - x0, c["width"] + 90)
        clip = dict(x=x0, y=y0, width=width, height=y1 - y0)
        path = os.path.join(self.out, "%s_%s.png" % (name, self.lang))
        page.screenshot(path=path, clip=clip, full_page=True)
        done = annotate(path, boxes, (x0, y0))
        print("  %s" % done)


def set_lang(page, lang):
    for _ in range(2):
        now = page.evaluate("() => document.documentElement.lang")
        if now == lang:
            return
        page.click("#lang")
        time.sleep(1.0)


def open_window(p, url, lang):
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport=dict(width=WIDTH, height=1000))
    page.goto(url)
    # the window opens on the tab it was left on: the targets first
    page.wait_for_selector('#tabs button[data-tab="targets"]', timeout=60000)
    time.sleep(1)
    page.click('#tabs button[data-tab="targets"]')
    time.sleep(0.5)
    page.click('#target_subtabs button[data-sub="local"]')
    page.wait_for_selector("#targets_table tr", timeout=60000)
    # every header read: no row still greyed out as being read
    for _ in range(120):
        if page.locator("#targets_table tr.reading").count() == 0:
            break
        time.sleep(1)
    time.sleep(2)
    set_lang(page, lang)
    # the banner sticks to the top of the window as the page scrolls: in a
    #   picture of the middle of the page it would cover what is shown
    page.add_style_tag(content=".site-banner { position: static !important; }")
    time.sleep(1)
    return browser, page


def tab(page, name):
    page.click('#tabs button[data-tab="%s"]' % name)
    time.sleep(1.2)


def ensure_ticked(page, target="Gliese229"):
    """The window opens with nothing ticked: the guide's target ticked."""
    tab(page, "targets")
    box = page.locator(target_row(target))
    if box.count() and not box.first.is_checked():
        box.first.check()
        time.sleep(3)


def target_row(target):
    return ('#targets_table tr:has(td:text-is("%s")) input[type=checkbox]'
            % target)


def setup(page, out, lang, target="Gliese229", launch=False):
    s = Shots(page, out, lang)
    tab(page, "targets")
    # untick everything, so that the first picture is the window as it opens
    page.click("#tick_none")
    time.sleep(1)
    s.region("01_start", "#setup",
             [(1, "#data_dir", "below"), (2, "#scan", "below"),
              (3, "#tabs", "left"), (4, "#targets_table", "left")],
             bottom_selector="#targets_wrap", margin=30)
    page.check(target_row(target))
    time.sleep(4)
    s.region("02_ticked", "#setup",
             [(1, target_row(target), "left"), (2, "#tiles", "left")],
             bottom_selector="#targets_wrap", margin=30)
    s.region("03_plots", "#plot_berv",
             [(1, "#plot_berv", "left"), (2, "#plot_vtot", "left"),
              (3, "#plot_snr", "left"), (4, "#min_rjd", "below")],
             bottom_selector="#max_rjd", margin=40, column="#plot_berv")
    # LBL settings, once SIMBAD has answered for the Teff
    tab(page, "lbl")
    for _ in range(60):
        if not page.locator('.teff-note:text-is("SIMBAD...")').count():
            break
        time.sleep(0.5)
    s.region("04_lbl", "#pane-lbl",
             [(1, "#lbl_setup_status", "left"), (2, "#python", "above"),
              (3, "#per_target select", "above"),
              (4, "#per_target input", "above"),
              (5, "#overwrite_all", "left"), (6, "#switches", "left")],
             bottom_selector="#resproj", margin=40)
    pca2d_shots(page, out, lang)
    run_shots(page, out, lang, launch)


def pca2d_shots(page, out, lang):
    """The PCA2D tab, with a name for the variant typed in, as an example."""
    s = Shots(page, out, lang)
    ensure_ticked(page)
    tab(page, "pca2d")
    name = page.locator('#options .option:has(.key:text-matches("^name ")) input')
    if name.count():
        name.first.fill("demo")
        name.first.dispatch_event("change")
        time.sleep(1)
    s.region("05_pca2d", "#pane-pca2d",
             [(1, "#pca2d_on", "above"), (2, "#skip_pca2d", "above"),
              (3, '#options .option:has(.key:text-matches("^n_earth ")) input',
               "left")],
             margin=90, height=760)
    s.region("06_pca2d_name", '#options .option:has(.key:text-matches("^name "))',
             [(1, '#options .option:has(.key:text-matches("^name ")) input', "left"),
              (2, '#options .option:has(.key:text-matches("^name ")) .reset', "right")],
             margin=120, height=300)


def run_shots(page, out, lang, launch=False):
    """The wrap and the button, and the run started when asked."""
    s = Shots(page, out, lang)
    tab(page, "run")
    time.sleep(2)
    s.region("07_run", "#pane-run",
             [(1, "#run_name", "below"), (2, "#queue", "left"),
              (3, "#wrap_text", "left"), (4, "#save_wrap", "right")],
             bottom_selector="#save_wrap", margin=30)
    if launch:
        page.click("#queue")
        time.sleep(12)


def running(page, out, lang):
    s = Shots(page, out, lang)
    ensure_ticked(page)
    tab(page, "run")
    time.sleep(3)
    s.region("08_running", "#steps",
             [(1, "#steps", "left"), (2, "#log", "left"),
              (3, "#log_top", "left"), (4, "#log_bottom", "left"),
              (5, "#stop", "above")],
             bottom_selector="#log", margin=60)
    # the button while a run goes
    s.region("08b_button", "#queue",
             [(1, "#queue", "left")], margin=30)


def archive(page, out, lang, star="GJ 1"):
    s = Shots(page, out, lang)
    tab(page, "targets")
    page.click('#target_subtabs button[data-sub="archive"]')
    time.sleep(1)
    page.fill("#archive_target", star)
    # one instrument, HARPS, for a short search
    for box in page.locator("#archive_instruments input[type=checkbox]").all():
        label = box.evaluate("e => e.closest('label').textContent")
        want = "HARPS" in label and "HARPS-N" not in label and "HARPSN" not in label
        if box.is_checked() != want:
            box.click()
    page.click("#archive_search")
    page.wait_for_selector("#archive_table tr td", timeout=240000)
    time.sleep(3)
    s.region("09_archive", "#sub-archive",
             [(1, "#archive_target", "left"),
              (2, "#archive_instruments", "below"),
              (3, "#archive_search", "below"),
              (4, "#archive_table", "left"), (5, "#archive_fetch", "left")],
             margin=30, height=400)
    page.click('#target_subtabs button[data-sub="local"]')


def done(page, out, lang):
    s = Shots(page, out, lang)
    tab(page, "runs")
    time.sleep(2)
    s.region("10_runs", "#pane-runs",
             [(1, ("#runs_window tbody tr", 0), "left"),
              (2, ("#runs_reports tbody tr a", 0), "left")],
             margin=30)


def main(url, out, lang, phase):
    os.makedirs(out, exist_ok=True)
    with sync_playwright() as p:
        browser, page = open_window(p, url, lang)
        try:
            if phase in ("setup", "launch"):
                setup(page, out, lang, launch=phase == "launch")
            elif phase == "running":
                running(page, out, lang)
            elif phase == "plots":
                s = Shots(page, out, lang)
                ensure_ticked(page)
                time.sleep(3)
                s.region("03_plots", "#plot_berv",
                         [(1, "#plot_berv", "left"), (2, "#plot_vtot", "left"),
                          (3, "#plot_snr", "left"), (4, "#min_rjd", "below")],
                         bottom_selector="#max_rjd", margin=40,
                         column="#plot_berv")
            elif phase == "pca2d":
                pca2d_shots(page, out, lang)
            elif phase == "archive":
                archive(page, out, lang)
            elif phase == "done":
                done(page, out, lang)
        finally:
            browser.close()


if __name__ == "__main__":
    main(*sys.argv[1:5])
