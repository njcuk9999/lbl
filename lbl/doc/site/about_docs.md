# About this documentation

These pages are the window's Documentation tab. They are Markdown files of
the repository:

- `lbl/doc/site/*.md`: the window, the data, the steps;
- `lbl/doc/*.md`: LBL's own documents (PCA2D, the template, the known
  limitations, adding an instrument), shown as they are;
- `lbl/doc/site/SITE.yaml`: the sections and the order of the pages.

A page edited shows at the next click, without restarting the window.

## What follows the code by itself

What the code already says is not written twice. A line of its own:

| line | becomes |
|---|---|
| `{{help naming \| Naming of new stars}}` | the window's (i) text of that field, under that title |
| `{{options}}` | PCA2D's options, their defaults, what they do |
| `{{folders}}` | the instrument folders a root can hold |
| `{{cleanup}}` | the kinds of files the Cleanup weighs |
| `{{version}}` | the window's version and the LBL it runs |

So the help of every field is the same in the window and here, and changes
with it.

## Kept up to date

**Every change of the window or of what LBL does updates these pages in the
same commit**: a new tab, field, option, folder, file or behaviour, a
default changed, a step renamed. `maintenance/tests/test_gui_docs.py`
checks that every page of the contents is there, that every help text of
the window is in a page, that every tab of the window has its page, and that
every link between pages leads somewhere.

## Writing a page

Markdown as GitHub writes it: headings (`##` and `###` make the page's
table of contents), lists, fenced code, tables, block quotes, inline code,
bold, italics, links and images. A link to another page is its file
(`naming.md`, `../PCA2D.md`, `naming.md#renaming-by-hand`); a file of the
repository that is no page opens on GitHub. Terse: fragments, every fact
and number kept, no filler.
