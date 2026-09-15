"""Compare two sets of rendered pages after normalising whitespace.

Two HTML files count as the same page when their ``<title>`` and ``<main>``
parse to the same tree once whitespace that a browser ignores has been
normalised: runs of whitespace collapse to one space, whitespace next to a
block boundary is dropped, attributes are sorted, and text inside ``<pre>``
is kept exactly. This is the test that the Markdown sources reproduce the
hand-built HTML, and later the regression test that an edit to the sources
changed only what it meant to.

Usage::

    python build/check.py DIR_A DIR_B          # compare every page in both
    python build/check.py DIR_A DIR_B 06 index  # only the pages named
"""

from __future__ import annotations

import os
import re
import sys

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

BLOCK_TAGS = {
    "main",
    "section",
    "nav",
    "header",
    "ul",
    "ol",
    "li",
    "div",
    "figure",
    "figcaption",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "caption",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "dl",
    "dt",
    "dd",
}
# Inside these, whitespace between elements carries no meaning at all.
STRUCTURAL = {
    "math",
    "mrow",
    "msub",
    "msup",
    "msubsup",
    "mfrac",
    "mtable",
    "mtr",
    "mtd",
    "munder",
    "mover",
    "munderover",
    "msqrt",
    "mroot",
    "mstyle",
    "mpadded",
    "mphantom",
    "menclose",
    "mfenced",
    "semantics",
    "svg",
    "g",
    "defs",
    "table",
    "thead",
    "tbody",
    "tr",
    "ul",
    "ol",
}
WS_RE = re.compile(r"[ \t\r\n\f]+")


def canonical(node: Tag, in_pre: bool = False) -> str:
    """Serialise an element in a whitespace-normalised form.

    Parameters
    ----------
    node : Tag
        Element to serialise.
    in_pre : bool
        Whether an enclosing ``<pre>`` makes whitespace significant.

    Returns
    -------
    str
        A string that is equal for two elements a browser would render alike.
    """
    name = node.name
    raw_classes = node.get("class")
    classes: list[str] = (
        raw_classes
        if isinstance(raw_classes, list)
        else ([raw_classes] if raw_classes else [])
    )
    attrs = {
        k: " ".join(v) if isinstance(v, list) else str(v) for k, v in node.attrs.items()
    }
    if name in ("nav", "div") and "toc" in classes:
        # Seven chapters had a <div class="toc"> without the aria-label; the
        # build emits the accessible form for all of them.
        name = "toc"
        attrs.pop("aria-label", None)
    attr_text = "".join(f' {k}="{v}"' for k, v in sorted(attrs.items()))
    in_pre = in_pre or name == "pre"

    # Merge adjacent text nodes, then normalise.
    pieces: list[str | Tag] = []
    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            if pieces and isinstance(pieces[-1], str):
                pieces[-1] += str(child)
            else:
                pieces.append(str(child))
        elif isinstance(child, Tag):
            pieces.append(child)

    out = []
    block_parent = name in BLOCK_TAGS or name in STRUCTURAL
    for i, piece in enumerate(pieces):
        if isinstance(piece, Tag):
            out.append(canonical(piece, in_pre))
            continue
        text = piece
        if not in_pre:
            text = WS_RE.sub(" ", text)
            prev = pieces[i - 1] if i > 0 else None
            nxt = pieces[i + 1] if i + 1 < len(pieces) else None
            prev_block = (i == 0 and block_parent) or (
                isinstance(prev, Tag) and prev.name in BLOCK_TAGS | STRUCTURAL
            )
            next_block = (i == len(pieces) - 1 and block_parent) or (
                isinstance(nxt, Tag) and nxt.name in BLOCK_TAGS | STRUCTURAL
            )
            if name in STRUCTURAL and not text.strip():
                text = ""
            if prev_block:
                text = text.lstrip(" ")
            if next_block:
                text = text.rstrip(" ")
            if name in STRUCTURAL:
                text = text.strip()
        if text:
            out.append(text.replace("&", "&amp;").replace("<", "&lt;"))
    return f"<{name}{attr_text}>{''.join(out)}</{name}>"


def page_key(path: str) -> str:
    """The canonical form of one page.

    Parameters
    ----------
    path : str
        HTML file.

    Returns
    -------
    str
        Canonical title plus canonical ``<main>``.
    """
    with open(path, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")
    title = WS_RE.sub(" ", soup.title.get_text()).strip() if soup.title else ""
    main = soup.find("main")
    return f"<title>{title}</title>\n" + (
        canonical(main) if isinstance(main, Tag) else ""
    )


def diff_context(a: str, b: str, width: int = 160) -> str:
    """Show where two canonical strings first differ.

    Parameters
    ----------
    a, b : str
        Canonical forms.
    width : int
        Characters of context on each side.

    Returns
    -------
    str
        A two-line excerpt around the first difference.
    """
    i = next(
        (k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b))
    )
    lo = max(0, i - width)
    return f"  A: …{a[lo:i]}▶{a[i : i + width]}…\n  B: …{b[lo:i]}▶{b[i : i + width]}…"


def main(argv: list[str]) -> int:
    """Compare pages between two directories.

    Parameters
    ----------
    argv : list[str]
        ``DIR_A DIR_B [stem ...]``.

    Returns
    -------
    int
        0 when every page matches, 1 otherwise.
    """
    if len(argv) < 2:
        print(__doc__)
        return 2
    dir_a, dir_b, wanted = argv[0], argv[1], argv[2:]
    names = sorted(n for n in os.listdir(dir_a) if n.endswith(".html"))
    if wanted:
        names = [n for n in names if any(n.startswith(w) for w in wanted)]
    bad = 0
    for name in names:
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        if not os.path.exists(pb):
            print(f"{name:52s} missing in {dir_b}")
            bad += 1
            continue
        ka, kb = page_key(pa), page_key(pb)
        if ka == kb:
            print(f"{name:52s} same")
        else:
            bad += 1
            print(f"{name:52s} DIFFERENT")
            print(diff_context(ka, kb))
    print(f"{len(names) - bad} of {len(names)} pages identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
