"""Check that the V3 rewrite preserved every word and every image of the source."""

from __future__ import annotations

import difflib
import os
import re
import sys
import warnings

from bs4 import BeautifulSoup

from build import CHAPTERS, V3, source

warnings.filterwarnings("ignore")

# Text the rewrite deliberately drops (the old hand-rolled nav) or adds (the TOC).
NAV_WORDS = {"previous", "next", "index"}


def words(html: str, drop_toc: bool = False) -> list[str]:
    soup = BeautifulSoup(html, "html5lib")
    for t in soup.find_all(["script", "style", "title"]):
        t.decompose()
    if drop_toc:
        for t in soup.select("nav, header.topbar"):
            t.decompose()
        h1 = soup.find("h1")
        if h1:
            h1.decompose()
    else:
        h1 = soup.find("h1")
        if h1:
            h1.decompose()
        for p in soup.find_all("p"):
            if {a.get_text(strip=True).lower() for a in p.find_all("a")} & NAV_WORDS:
                p.decompose()

    text = soup.get_text(" ")
    text = text.replace(" ", " ").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def images(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html5lib")
    return [i.get("src", "") for i in soup.find_all("img")]


def main() -> int:
    bad = 0
    for stem, _ in CHAPTERS:
        src = open(source(stem), encoding="utf-8", errors="replace").read().lstrip("﻿")
        out = open(os.path.join(V3, stem + ".html"), encoding="utf-8").read()

        a, b = words(src), words(out, drop_toc=True)
        ia, ib = images(src), images(out)

        ok_text = a == b
        ok_img = ia == ib
        status = "ok " if (ok_text and ok_img) else "FAIL"
        if not (ok_text and ok_img):
            bad += 1
        print(f"{status} {stem:52s} words {len(a):>6,d} -> {len(b):>6,d}   imgs {len(ia):>3d} -> {len(ib):>3d}")

        if not ok_text:
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal":
                    continue
                print(f"     {op:8s} src[{i1}:{i2}]={a[i1:i2][:14]}  out[{j1}:{j2}]={b[j1:j2][:14]}")
        if not ok_img:
            miss = [x for x in ia if x not in ib]
            extra = [x for x in ib if x not in ia]
            if miss:
                print("     dropped images:", miss[:8])
            if extra:
                print("     added images:", extra[:8])
    print("\nchapters with differences:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
