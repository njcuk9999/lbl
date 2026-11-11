# Working in this repository

## The documentation is always kept up to date

The window `lbl_gui` has a **Documentation** tab: a website made of the
Markdown pages of `lbl/doc/site/` and of LBL's documents in `lbl/doc/`
(`PCA2D.md`, `TEMPLATE.md`, `KnownLimitations.md`, `AddingAnInstrument.md`),
in the sections and order of `lbl/doc/site/SITE.yaml` (served by
`lbl/gui/docs.py`).

**Every change of the window or of what LBL does updates that documentation
in the same commit**: a new or renamed tab, field, button, option, folder,
file, step or recipe; a default changed; a behaviour changed (how files are
dispatched, how stars are named, what a run writes, what the cleanup
deletes...); a new instrument or data source. Edit the page that describes
it (or add one, and list it in `SITE.yaml`); a change that no page
describes yet gets a page. Do not consider a change done while the
documentation still describes the old behaviour.

What the code already says is pulled in rather than copied, so that it
cannot drift: `{{help <key> | <title>}}` (the window's (i) text, from
`lbl/gui/i18n.py`), `{{options}}` (PCA2D's options), `{{folders}}` (the
instrument folders), `{{cleanup}}`, `{{version}}`. A new (i) help text must
be placed in a page with `{{help ...}}`.

`maintenance/tests/test_gui_docs.py` fails when a page of `SITE.yaml` is
missing, when a help text of the window is in no page, when a tab of the
window has no line in the welcome page, or when a link between pages leads
nowhere. Run it with the GUI tests:

```
python -m pytest maintenance/tests -k gui
```

Pages are in English (LBL's documentation is), terse: every fact and
number kept, no filler; no em dashes.
