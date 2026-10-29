"""lbl/gui/static/references.json, what the References tab shows: the paper
that describes LBL and the papers whose velocities LBL measured
(lbl_references.yaml, read paper by paper, with the sentence that
says so), each as its title, its first authors and its link to ADS. The
authors come from CrossRef (by DOI, found with OpenAlex), else from the
arXiv record.

    python maintenance/gui/lbl_references.py <folder with meta/papers.json,
        meta/citing_biblio.json, meta/lbl_biblio.json>
"""
import datetime
import html
import json
import os
import re
import sys
import time
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "..", "lbl", "gui", "static")
UA = {"User-Agent": "lbl-gui-references/1.0 (+https://github.com/njcuk9999/lbl)"}


def get_json(url):
    time.sleep(0.5)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def crossref(doi):
    try:
        return get_json("https://api.crossref.org/works/" + doi)["message"]
    except Exception:                                      # noqa: BLE001
        return None


#: the authors named, the others as et al.
AUTHORS_SHOWN = 3


def authors_short(families):
    families = [f for f in families if f]
    if not families:
        return "?"
    if len(families) > AUTHORS_SHOWN:
        return "%s et al." % ", ".join(families[:AUTHORS_SHOWN])
    if len(families) == 1:
        return families[0]
    return "%s & %s" % (", ".join(families[:-1]), families[-1])


def ads(arxiv, doi):
    """A paper's ADS page: ADS finds a paper by its arXiv id or its DOI in
    place of its bibcode (the bibcode needs the ADS API and a token)"""
    return "https://ui.adsabs.harvard.edu/abs/%s/abstract" % (
        "arXiv:" + arxiv if arxiv else doi)


def openalex_of(arxiv, title, works):
    """The OpenAlex record of a paper: by its arXiv id, else its title"""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())[:70]
    for w in works:
        for loc in w.get("locations") or []:
            url = (loc.get("landing_page_url") or "") + (loc.get("pdf_url") or "")
            if arxiv and arxiv in url:
                return w
    for w in works:
        if title and norm(w.get("title")) == norm(title):
            return w
    return None


def licence_name(url):
    """'CC BY 4.0', 'CC0 1.0' of a Creative Commons licence's URL"""
    parts = url.rstrip("/").split("/")
    if "zero" in parts:
        return "CC0 " + parts[-1]
    return "CC %s %s" % (parts[-2].upper(), parts[-1])


def describe(arxiv, title, work, year):
    """A paper's entry: authors, year, title, DOI, arXiv, its ADS page"""
    doi = (work or {}).get("doi") or ""
    doi = doi.replace("https://doi.org/", "") or None
    # an arXiv DOI is not the journal's
    if doi and doi.lower().startswith("10.48550"):
        doi = None
    out = dict(arxiv=arxiv, doi=doi, year=year, title=title)
    meta = crossref(doi) if doi else None
    if meta:
        out["authors"] = authors_short([a.get("family") for a in meta.get("author", [])])
        # a series' papers carry the series as title, their own as subtitle
        #   ("The GAPS Programme at TNG" alone)
        full = (meta.get("title") or [title])[0]
        if meta.get("subtitle"):
            full = "%s. %s" % (full.rstrip(". "), meta["subtitle"][0])
        out["title"] = html.unescape(re.sub(r"<[^>]+>", "", full))
        issued = (meta.get("issued") or {}).get("date-parts") or [[year]]
        out["year"] = issued[0][0] or year
    else:
        names = [(a.get("author") or {}).get("display_name") for a in (work or {}).get("authorships", [])]
        out["authors"] = authors_short([n.split()[-1] if n else n for n in names])
    out["ads"] = ads(arxiv, doi)
    return out


def main(folder):
    meta_dir = os.path.join(folder, "meta")
    papers = json.load(open(os.path.join(meta_dir, "papers.json")))
    by_arxiv = {r.get("arxiv"): r for r in papers.values() if r.get("arxiv")}
    works = json.load(open(os.path.join(meta_dir, "citing_biblio.json")))["results"]
    lbl_work = json.load(open(os.path.join(meta_dir, "lbl_biblio.json")))
    chosen = yaml.safe_load(open(os.path.join(HERE, "lbl_references.yaml")))

    method_arxiv = str(chosen["method"])
    method = describe(method_arxiv, lbl_work["title"], lbl_work,
                      lbl_work["publication_year"])

    out, seen = [], set()
    for row in chosen["listed"]:
        # by arXiv id, or by DOI for a paper with no source on arXiv (read
        #   in its abstract)
        arxiv = str(row["arxiv"]) if row.get("arxiv") else None
        doi = str(row["doi"]) if row.get("doi") else None
        if (arxiv or doi) in seen:
            continue
        seen.add(arxiv or doi)
        rec = by_arxiv.get(arxiv) or {}
        if arxiv:
            work = openalex_of(arxiv, rec.get("title"), works)
        else:
            work = next((w for w in works if (w.get("doi") or "").lower()
                         .endswith(doi.lower())), None) or dict(
                             doi="https://doi.org/" + doi)
        entry = describe(arxiv, rec.get("title") or (work or {}).get("title"),
                         work, rec.get("year") or (work or {}).get("publication_year"))
        out.append(entry)
        print("%s %s" % (entry["year"], entry["authors"]))
    out.sort(key=lambda e: (-int(e["year"] or 0), e["authors"]))

    credits = []
    path = os.path.join(STATIC, "backdrop_credits.json")
    if os.path.exists(path):
        listed = {e["arxiv"]: e for e in out + [method]}
        for c in json.load(open(path)):
            e = listed.get(c["arxiv"])
            # one line per paper, however many of its figures are used
            if e and all(k["ads"] != e["ads"] for k in credits):
                credits.append(dict(authors=e["authors"], year=e["year"],
                                    title=e["title"], ads=e["ads"],
                                    licence=licence_name(c["licence"])))
    result = dict(updated=datetime.date.today().isoformat(), method=method,
                  papers=out, credits=credits)
    with open(os.path.join(STATIC, "references.json"), "w") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
    print("%d papers listed, %d background credits" % (len(out), len(credits)))


if __name__ == "__main__":
    main(sys.argv[1])
