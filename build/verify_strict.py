"""Compare the rendered text of source and output character by character.

The word-level check in verify.py cannot see whitespace that the rewrite added
or removed inside a word, which is exactly the kind of damage a pretty-printer
does. This walks the text as the browser would see it instead.
"""

from __future__ import annotations

import difflib
import os
import re
import sys
import warnings

from bs4 import BeautifulSoup

from build import CHAPTERS, V3, source

warnings.filterwarnings("ignore")

NAV_WORDS = {"previous", "next", "index"}


def rendered(html: str, is_output: bool) -> str:
    soup = BeautifulSoup(html, "html5lib")
    for t in soup.find_all(["script", "style", "title"]):
        t.decompose()
    if is_output:
        for t in soup.select("nav, header.topbar"):
            t.decompose()
    else:
        for p in soup.find_all("p"):
            if {a.get_text(strip=True).lower() for a in p.find_all("a")} & NAV_WORDS:
                p.decompose()
    h1 = soup.find("h1")
    if h1:
        h1.decompose()

    # Block elements imply a break; inline ones must not add space.
    for t in soup.find_all(["li", "ul", "ol", "div", "p", "br", "h2", "h3", "figure", "tr", "pre"]):
        t.insert_before(" ")
        t.insert_after(" ")

    text = soup.get_text("")
    text = text.replace("\xa0", " ").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    """Compare rendered text, applying the intended corrections to the source first.

    Anything that still differs is damage from the conversion rather than a fix.
    """
    from fixes import FIXES

    expected: dict[str, list[tuple[str, str]]] = {}
    for stem, before, after, _count, _note in FIXES:
        expected.setdefault(stem, []).append((before, after))

    bad = 0
    for stem, _ in CHAPTERS:
        src = rendered(open(source(stem), encoding="utf-8", errors="replace").read().lstrip("﻿"), False)
        out = rendered(open(os.path.join(V3, stem + ".html"), encoding="utf-8").read(), True)

        for before, after in expected.get(stem, []):
            b = re.sub(r"\s+", " ", BeautifulSoup(before, "html5lib").get_text("")).strip()
            a = re.sub(r"\s+", " ", BeautifulSoup(after, "html5lib").get_text("")).strip()
            src = src.replace(b, a)

        if src == out:
            print(f"ok   {stem:52s} {len(src):>7,d} chars")
            continue

        bad += 1
        print(f"DIFF {stem:52s} {len(src):>7,d} -> {len(out):,d} chars")
        sm = difflib.SequenceMatcher(None, src, out, autojunk=False)
        shown = 0
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal" or shown >= 12:
                continue
            shown += 1
            print(f"       {op:7s} src={src[max(0,i1-30):i2+30]!r}")
            print(f"       {'':7s} out={out[max(0,j1-30):j2+30]!r}")
    print("\nchapters differing:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
