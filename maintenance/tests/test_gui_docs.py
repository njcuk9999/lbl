"""The Documentation tab (lbl/gui/docs.py, lbl/doc/site): every page there,
every help text of the window in a page, every tab with its page, every link
between pages leading somewhere, and the Markdown it reads rendered right.

The documentation is updated with every change of the window or of what LBL
does (CLAUDE.md): this test fails when a page, a help text or a tab is
left out."""
import os
import re

from lbl.gui import docs, i18n

STATIC = os.path.join(os.path.dirname(docs.__file__), "static")


def _pages():
    return [p for s in docs.contents() for p in s["pages"]]


def test_every_page_of_the_contents_is_there_with_a_title():
    pages = _pages()
    assert len(pages) >= 15
    ids = [p["id"] for p in pages]
    assert len(ids) == len(set(ids))
    for one in pages:
        assert os.path.isfile(one["path"]), one["file"]
        assert docs._title(one["path"]), one["file"] + " has no '# ' title"


def test_every_help_text_of_the_window_is_in_a_page():
    said = set()
    for one in _pages():
        with open(one["path"], encoding="utf-8") as handle:
            said |= set(re.findall(r"^\{\{\s*help\s+(\w+)", handle.read(),
                                   re.M))
    helps = {k[5:] for k in i18n.EN if k.startswith("help_")}
    assert helps - said == set(), "help texts in no page of the docs"
    assert said - helps == set(), "pages ask for help texts that are gone"


def test_every_tab_of_the_window_has_its_page():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as h:
        tabs = re.findall(r'data-tab="(\w+)"', h.read())
    assert tabs.index("docs") == tabs.index("cleanup") + 1
    assert tabs.index("refs") == tabs.index("docs") + 1
    welcome = docs.page("welcome")["html"]
    for tab in tabs:
        assert "<td>%s</td>" % i18n.EN["tab_" + tab] in welcome, tab


def test_every_page_renders_and_its_links_lead_somewhere():
    ids = {p["id"] for p in _pages()}
    for one in _pages():
        got = docs.page(one["id"])
        assert "error" not in got
        body = got["html"]
        assert "Unknown directive" not in body and "No help text" not in body
        for target in re.findall(r'data-doc="([^"]+)"', body):
            assert target in ids, (one["id"], target)
        # a link to a Markdown file that is no page, left as it was
        for href in re.findall(r'href="([^"]+)"', body):
            if not href.startswith(("http", "#")):
                assert not href.endswith(".md"), (one["id"], href)
        for _, anchor, title in got["toc"]:
            assert 'id="%s"' % anchor in body and title


def test_links_with_an_anchor_find_it():
    ids = {p["id"]: docs.page(p["id"])["html"] for p in _pages()}
    for one, body in ids.items():
        for page, anchor in re.findall(
                r'data-doc="([^"]+)" data-anchor="([^"]+)"', body):
            assert 'id="%s"' % anchor in ids[page], (one, page, anchor)


def test_the_markdown_it_reads():
    text = """# Title

A *word*, a **bold** one, `code_with_underscores`, KIC_12060710, a
[link](https://example.org) and [a page](naming.md#three-ways).

- one
- two
  - two and a half
- three

1. first
2. second

| a | b |
|---|---|
| `x \\| y` | 2 |

> a note

```
<raw> & code
```
"""
    pages = _pages()
    body, toc = docs.render(text, docs._links(pages, os.path.join(
        docs.SITE, "welcome.md")))
    assert '<h1 id="title">Title</h1>' in body
    assert "<em>word</em>" in body and "<strong>bold</strong>" in body
    assert "<code>code_with_underscores</code>" in body
    assert "KIC_12060710" in body
    assert 'href="https://example.org" target="_blank"' in body
    assert 'data-doc="naming" data-anchor="three-ways"' in body
    assert "<ul><li>one</li><li>two<ul><li>two and a half</li></ul></li>" \
        "<li>three</li></ul>" in body
    assert "<ol><li>first</li><li>second</li></ol>" in body
    assert "<td><code>x | y</code></td>" in body
    assert "<blockquote><p>a note</p></blockquote>" in body
    assert "&lt;raw&gt; &amp; code" in body
    assert toc == []


def test_what_follows_the_code():
    page = docs.page("data-root")["html"]
    # the window's own help text, and the instrument folders it knows
    assert "Sorts the files of Downloads/" in page
    assert "<code>NIRPS_HE_APERO/</code>" in page
    options = docs.page("pca2d-options")["html"]
    assert "<code>n_star</code>" in options and "<code>n_earth</code>" in options


def test_the_search():
    found = docs.search("naming apero")
    assert found and found[0]["page"] == "naming"
    assert docs.search("") == []
    assert docs.search("zzzz-nothing-here") == []
