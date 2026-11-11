"""The window's Documentation tab: a small website, read from Markdown.

The pages are Markdown files: those of lbl/doc/site/ (the window, the data,
the steps) and the documents of lbl/doc/ itself (PCA2D.md, TEMPLATE.md...),
in the order and sections lbl/doc/site/SITE.yaml gives. A page's title is
its first '# ' heading. They are read when asked (a page edited shows at
the next click, no restart).

What the code already says is not written twice: a line of its own

    {{help data_dir | Data root}}     the window's (i) text of that field
    {{options}}                       PCA2D's options: key, default, what
    {{folders}}                       the instrument folders a root holds
    {{cleanup}}                       the kinds of files the Cleanup weighs
    {{version}}                       the versions of LBL and the window

is replaced by it when the page is read, so these parts follow the code.
maintenance/tests/test_gui_docs.py checks that every page of SITE.yaml is
there, that every help text of the window is in a page, and that every
link between pages leads somewhere.

Markdown as GitHub writes it, the part these pages use: headings, lists
(nested by indentation), fenced code, tables, block quotes, rules, inline
code, bold, italics, links and images. A link to another page is its file
name (naming.md, ../PCA2D.md, naming.md#renaming); a file of the repository
that is no page goes to GitHub.
"""
from __future__ import annotations

import html
import os
import re

from . import i18n

#: the folder of the pages, and the documents of LBL beside it
DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "doc")
SITE = os.path.join(DOC, "site")
#: the table of contents
CONTENTS = os.path.join(SITE, "SITE.yaml")
#: where a file of the repository that is no page is read
GITHUB = "https://github.com/njcuk9999/lbl/blob/lbl-pca2d/lbl/doc/"
#: a directive, on a line of its own
DIRECTIVE = re.compile(r"^\{\{\s*(\w+)\s*(.*?)\s*\}\}\s*$")

_CACHE = {}


# -----------------------------------------------------------------------------
# The site: its sections and pages
# -----------------------------------------------------------------------------
def contents() -> list:
    """The sections of the site, each with its pages (id, file, title)."""
    import yaml
    with open(CONTENTS, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or []
    out = []
    for section in raw:
        pages = []
        for one in section.get("pages") or []:
            path = os.path.normpath(os.path.join(SITE, one["file"]))
            pages.append(dict(id=one["id"], file=one["file"], path=path,
                              title=_title(path) or one["id"]))
        out.append(dict(title=section["section"], pages=pages))
    return out


def _pages() -> list:
    return [page for section in contents() for page in section["pages"]]


def _title(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        return ""
    return ""


def nav() -> dict:
    """What the tab lays out: the sections and their pages' titles."""
    return dict(sections=[dict(title=s["title"], pages=[
        dict(id=p["id"], title=p["title"]) for p in s["pages"]])
        for s in contents()], first=_pages()[0]["id"] if _pages() else None)


def page(page_id: str) -> dict:
    """A page: its title, its HTML, its table of contents (the ## and ###
    headings), the pages before and after it, its file."""
    pages = _pages()
    ids = [p["id"] for p in pages]
    if page_id not in ids:
        return dict(error="no page %s" % page_id)
    k = ids.index(page_id)
    one = pages[k]
    text = _source(one["path"])
    body, toc = render(expand(text), _links(pages, one["path"]))
    near = [dict(id=p["id"], title=p["title"]) if p else None for p in (
        pages[k - 1] if k else None,
        pages[k + 1] if k + 1 < len(pages) else None)]
    return dict(id=page_id, title=one["title"], html=body, toc=toc,
                prev=near[0], next=near[1],
                file=os.path.relpath(one["path"], os.path.dirname(DOC)))


def _source(path: str) -> str:
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return "# Missing page\n\n`%s` is not there." % path
    kept = _CACHE.get(path)
    if kept and kept[0] == stamp:
        return kept[1]
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    _CACHE[path] = (stamp, text)
    return text


def _links(pages, here) -> dict:
    """Each page by its file, as a link from `here` names it."""
    out = {}
    for p in pages:
        out[os.path.normpath(p["path"])] = p["id"]
    return dict(files=out, folder=os.path.dirname(here))


# -----------------------------------------------------------------------------
# What the code says: the directives
# -----------------------------------------------------------------------------
def expand(text: str) -> str:
    """The directives of a page replaced by what they stand for (Markdown,
    rendered with the page)."""
    out = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        found = None if fenced else DIRECTIVE.match(line.strip())
        out.append(_directive(found.group(1), found.group(2)) if found
                   else line)
    return "\n".join(out)


def _directive(name: str, args: str) -> str:
    if name == "help":
        key, _, title = args.partition("|")
        key = key.strip()
        text = i18n.EN.get("help_" + key)
        if text is None:
            return "> No help text `%s` in the window." % key
        return "### %s\n\n%s\n" % (title.strip() or key, text)
    if name == "options":
        return _options()
    if name == "folders":
        return _folders()
    if name == "cleanup":
        from .cleanup import WHAT
        return "\n".join(["| kind | what it is |", "|---|---|"] + [
            "| `%s` | %s |" % (code, _cell(what)) for code, what in
            WHAT.items()]) + "\n"
    if name == "version":
        from . import __version__
        from .lblinfo import lbl_home
        return ("The window `lbl_gui` %s, LBL in `%s`."
                % (__version__, lbl_home()))
    return "> Unknown directive `%s`." % name


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _options() -> str:
    from .lblinfo import pca2d_options
    rows = ["| option | default | what it does |", "|---|---|---|"]
    for row in pca2d_options(""):
        label = i18n.OPTIONS.get(row["key"])
        name = "`%s`" % row["key"]
        if label:
            name += " (%s)" % label[0][0]
        rows.append("| %s | `%s` | %s |" % (
            _cell(name), _cell(row["default"]), _cell(row["description"])))
    return "\n".join(rows) + "\n"


def _folders() -> str:
    from .dispatch import readers
    rows = ["| folder | instrument | data source |", "|---|---|---|"]
    for name, (inst, source) in sorted(readers().items()):
        rows.append("| `%s/` | `%s` | `%s` |" % (name, inst, source))
    return "\n".join(rows) + "\n"


# -----------------------------------------------------------------------------
# Markdown
# -----------------------------------------------------------------------------
LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)\s*([\w+-]*)\s*$")


def slug(text: str, taken: set) -> str:
    """A heading's anchor: its words, lower case, joined by -; once."""
    base = re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", text).lower()
                  ).strip("-") or "section"
    out, n = base, 1
    while out in taken:
        n += 1
        out = "%s-%d" % (base, n)
    taken.add(out)
    return out


def render(text: str, links=None) -> tuple:
    """(HTML, table of contents [(level, anchor, title)]) of Markdown."""
    state = dict(taken=set(), toc=[], links=links or {})
    return _blocks(text.split("\n"), state), state["toc"]


def _blocks(lines, state) -> str:
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        fence = FENCE.match(line)
        if fence:
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(
                    fence.group(1)):
                j += 1
            code = "\n".join(lines[i + 1:j])
            lang = fence.group(2)
            out.append('<pre><code%s>%s</code></pre>' % (
                ' class="lang-%s"' % lang if lang else "",
                html.escape(code, quote=False)))
            i = j + 1
            continue
        head = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if head:
            level = len(head.group(1))
            inner = _inline(head.group(2), state)
            anchor = slug(head.group(2), state["taken"])
            if level in (2, 3):
                state["toc"].append((level, anchor, re.sub(
                    r"<[^>]+>", "", inner)))
            out.append('<h%d id="%s">%s</h%d>' % (level, anchor, inner,
                                                  level))
            i += 1
            continue
        if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", line):
            out.append("<hr>")
            i += 1
            continue
        if line.lstrip().startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(_table(lines[i], lines[i + 2:j], state))
            i = j
            continue
        if line.lstrip().startswith(">"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith(">"):
                j += 1
            inner = [re.sub(r"^\s*>\s?", "", one) for one in lines[i:j]]
            out.append("<blockquote>%s</blockquote>" % _blocks(inner,
                                                                state))
            i = j
            continue
        if LIST.match(line):
            kind = LIST.match(line).group(2)[0].isdigit()
            base = len(LIST.match(line).group(1).expandtabs(4))

            def same(text):
                # an item of this list: as deep, of the same kind
                found = LIST.match(text)
                return found and (len(found.group(1).expandtabs(4)) > base + 1
                                  or found.group(2)[0].isdigit() == kind)

            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip():
                    # a blank line ends the list unless the next line is
                    #   indented or another item of it
                    if j + 1 < len(lines) and (
                            same(lines[j + 1])
                            or lines[j + 1].startswith((" ", "\t"))):
                        j += 1
                        continue
                    break
                if LIST.match(nxt) and not same(nxt):
                    break
                if LIST.match(nxt) or nxt.startswith((" ", "\t")):
                    j += 1
                    continue
                # a line of the item's paragraph (lazy continuation)
                j += 1
            out.append(_list(lines[i:j], state))
            i = j
            continue
        j = i
        while j < len(lines) and lines[j].strip() and not (
                FENCE.match(lines[j]) or LIST.match(lines[j])
                or lines[j].lstrip().startswith(("#", ">", "|"))):
            j += 1
        if j == i:
            j = i + 1
        out.append("<p>%s</p>" % _inline(" ".join(
            one.strip() for one in lines[i:j]), state))
        i = j
    return "\n".join(out)


def _list(lines, state) -> str:
    """A list, nested by the indentation of its items."""
    first = LIST.match(lines[0])
    base = len(first.group(1).expandtabs(4))
    ordered = first.group(2)[0].isdigit()
    items = []
    for line in lines:
        found = LIST.match(line)
        indent = len(line) - len(line.lstrip()) if line.strip() else None
        if found and len(found.group(1).expandtabs(4)) <= base + 1:
            items.append([found.group(3)])
        elif items:
            items[-1].append(line[base + 2:] if indent and indent >= base + 2
                             else line.strip())
    body = []
    for item in items:
        sub = [one for one in item[1:]]
        # the item's own text, then what is indented under it (a list, a
        #   paragraph)
        lead = [item[0]]
        while sub and sub[0].strip() and not LIST.match(sub[0]) \
                and not FENCE.match(sub[0]):
            lead.append(sub.pop(0).strip())
        inner = _inline(" ".join(lead), state)
        rest = _blocks(sub, state) if any(s.strip() for s in sub) else ""
        body.append("<li>%s%s</li>" % (inner, rest))
    tag = "ol" if ordered else "ul"
    return "<%s>%s</%s>" % (tag, "".join(body), tag)


def _row(line) -> list:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells, cur, code = [], "", False
    k = 0
    while k < len(text):
        c = text[k]
        if c == "\\" and k + 1 < len(text) and text[k + 1] == "|":
            cur += "|"
            k += 2
            continue
        if c == "`":
            code = not code
        if c == "|" and not code:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += c
        k += 1
    cells.append(cur.strip())
    return cells


def _table(head, rows, state) -> str:
    out = ["<div class=\"doc-table\"><table><thead><tr>"]
    out += ["<th>%s</th>" % _inline(c, state) for c in _row(head)]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>%s</tr>" % "".join(
            "<td>%s</td>" % _inline(c, state) for c in _row(row)))
    out.append("</tbody></table></div>")
    return "".join(out)


def _href(url: str, state) -> tuple:
    """(href, attributes) of a link: a page of the site, a file of the
    repository on GitHub, or the web."""
    if re.match(r"^[a-z]+:", url) or url.startswith("//"):
        return url, ' target="_blank" rel="noopener"'
    if url.startswith("#"):
        return url, ' data-anchor="%s"' % html.escape(url[1:])
    path, _, anchor = url.partition("#")
    links = state["links"]
    full = os.path.normpath(os.path.join(links.get("folder", SITE), path))
    page_id = links.get("files", {}).get(full)
    if page_id:
        return ("#doc/%s%s" % (page_id, "/" + anchor if anchor else ""),
                ' data-doc="%s" data-anchor="%s"' % (page_id,
                                                     html.escape(anchor)))
    rel = os.path.relpath(full, DOC)
    if not rel.startswith(".."):
        return GITHUB + rel, ' target="_blank" rel="noopener"'
    return url, ""


def _inline(text: str, state) -> str:
    """The inline Markdown of a line: code, images, links, bold, italics;
    the rest escaped."""
    kept = []

    def keep(fragment):
        kept.append(fragment)
        return "\x00%d\x00" % (len(kept) - 1)

    text = re.sub(r"`([^`]+)`", lambda m: keep(
        "<code>%s</code>" % html.escape(m.group(1), quote=False)), text)

    def image(m):
        href, _ = _href(m.group(2), state)
        return keep('<img src="%s" alt="%s">' % (
            html.escape(href), html.escape(m.group(1))))

    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", image, text)

    def link(m):
        href, extra = _href(m.group(2), state)
        return keep('<a href="%s"%s>' % (html.escape(href), extra)) \
            + m.group(1) + keep("</a>")

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    text = re.sub(r"<(https?://[^>\s]+)>", lambda m: keep(
        '<a href="%s" target="_blank" rel="noopener">%s</a>' % (
            html.escape(m.group(1)), html.escape(m.group(1)))), text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])",
                  r"<em>\1</em>", text)
    text = re.sub(r"(?<![A-Za-z0-9_])_(?!\s)(.+?)(?<!\s)_(?![A-Za-z0-9_])",
                  r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------
def _plain(fragment: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ",
                                                     fragment))).strip()


def search(query: str, most: int = 30) -> list:
    """The sections of the pages that hold every word of `query`, the best
    first: a word in a page's title or a section's heading counts more.
    Each: page, its title, the section's heading and anchor, a snippet."""
    words = [w for w in re.findall(r"[\w./+-]+", (query or "").lower())
             if len(w) > 1]
    if not words:
        return []
    found = []
    for one in _pages():
        got = page(one["id"])
        if "html" not in got:
            continue
        parts = re.split(r'(<h[1-4] id="[^"]+">.*?</h[1-4]>)', got["html"])
        heading, anchor = got["title"], ""
        for part in parts:
            m = re.match(r'<h[1-4] id="([^"]+)">(.*?)</h[1-4]>', part)
            if m:
                anchor, heading = m.group(1), _plain(m.group(2))
                continue
            text = _plain(part)
            low = (heading + " " + text).lower()
            if not all(w in low for w in words):
                continue
            score = sum(low.count(w) for w in words) \
                + 5 * sum(w in heading.lower() for w in words) \
                + 3 * sum(w in got["title"].lower() for w in words)
            at = min((text.lower().find(w) for w in words
                      if w in text.lower()), default=0)
            start = max(0, at - 60)
            snippet = ("..." if start else "") + text[start:start + 200] \
                + ("..." if start + 200 < len(text) else "")
            found.append(dict(page=one["id"], title=got["title"],
                              heading=heading, anchor=anchor,
                              snippet=snippet, score=score))
    found.sort(key=lambda f: -f["score"])
    # one hit a section heading of a page (the best)
    out, seen = [], set()
    for one in found:
        key = (one["page"], one["heading"])
        if key not in seen:
            seen.add(key)
            out.append(one)
    return out[:most]
