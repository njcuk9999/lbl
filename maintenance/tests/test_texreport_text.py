"""Text that goes into pca2d's LaTeX report never holds a paragraph break.

A blank line inside a macro argument or a table cell stops pdflatex
("Paragraph ended before \\@textcolor was complete"). SIMBAD's error, when
it cannot be reached, is several lines with a blank one among them, and it
went into \\watch{...} in the star's table: the report of the EXPRES HD 10700
run (2026-11-03) was not made.

    PYTHONPATH=<this checkout> python -m pytest \
        maintenance/tests/test_texreport_text.py
"""
from lbl.pca2d import texreport

SIMBAD_DOWN = ("SIMBAD: DALServiceError: Unable to access the capabilities"
               " endpoint at:\n- https://simbad.cds.unistra.fr/simbad/sim-tap/"
               "capabilities: Connection failed\n\nThis could be due to")


def test_text_has_no_line_break():
    out = texreport.tex("a\n\nb\r\nc\td")
    assert "\n" not in out and "\r" not in out
    assert out.split() == ["a", "b", "c", "d"]


def test_the_star_table_survives_simbad_not_answering():
    table = texreport.star_table("HD10700", {"ok": False,
                                             "error": SIMBAD_DOWN,
                                             "header": {"OBJECT": "10700"}},
                                 ([], {"ok": True}))
    row = [line for line in table.splitlines() if "not reached" in line]
    assert len(row) == 1
    assert "This could be due to" in row[0]
    assert "\n\n" not in table
