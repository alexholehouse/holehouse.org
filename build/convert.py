"""Convert the Evernote-exported ML course notes into clean, semantic HTML.

The V1 pages were produced by Evernote's HTML exporter and carry the usual
damage: unclosed ``</font>`` tags, ``<basefont>``, pixel font sizes standing in
for headings, sub-lists smuggled inside invisible ``display: inline`` list
items, and long runs of WebKit paste junk. The prose itself is fine, so this
script rewrites the structure and leaves the words alone.

The transformation is deliberately mechanical: every piece of text in the source
survives into the output. Content fixes (typos, notation errors) are applied
separately from ``fixes.py`` so that they stay reviewable and can be listed in
the changelog.
"""

from __future__ import annotations

import html
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

warnings.filterwarnings("ignore")

# Block-level elements, used to decide when an inline run has ended and when a
# <br> is redundant spacing rather than a real line break.
BLOCK = {
    "ul", "ol", "li", "div", "p", "table", "blockquote", "img",
    "figure", "h1", "h2", "h3", "h4", "hr", "br", "section",
}

# Anything that already forms a block of its own and so ends an inline run.
BLOCK_LEVEL = {
    "ul", "ol", "li", "figure", "table", "tr", "td", "th", "div", "p", "pre",
    "h1", "h2", "h3", "h4", "section", "nav", "header", "blockquote",
}

# Tags carrying no meaning once their styling is gone.
UNWRAP = {"font", "span", "basefont", "center", "blockquote"}

# Colours the notes use to pick out terms and warnings, mapped onto classes.
COLOUR_CLASS = {
    "1c3387": "term",     # dark blue - key term being defined
    "e30000": "hl-red",
    "ff0000": "hl-red",
    "ad0000": "hl-red",
    "7b003d": "hl-red",
    "328712": "hl-green",
    "41007d": "hl-purple",
    "e500ff": "hl-purple",
    "0000ff": "term",
    "1c3387;": "term",
}

# Colours that are simply body text and should not produce a class.
COLOUR_IGNORE = {"010101", "222222", "000000", "333333", "1a1a1a", "010102"}

HEADING_PX = 27


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Turn heading text into a stable, URL-safe id."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text) or "section"


def style_of(tag: Tag) -> str:
    return (tag.get("style") or "").lower()


def font_px(tag: Tag) -> int | None:
    """Return the font-size in px declared on a tag, if any."""
    m = re.search(r"font-size:\s*(\d+)px", style_of(tag))
    return int(m.group(1)) if m else None


def colour_of(tag: Tag) -> str | None:
    """Normalise a tag's colour, from either the attribute or inline style."""
    raw = tag.get("color")
    if not raw:
        m = re.search(r"(?<!-)\bcolor:\s*([^;]+)", style_of(tag))
        raw = m.group(1) if m else None
    if not raw:
        return None
    raw = raw.strip().lower()
    m = re.match(r"rgb\(\s*(\d+)\D+(\d+)\D+(\d+)", raw)
    if m:
        return "".join(f"{int(v):02x}" for v in m.groups())
    return raw.lstrip("#").strip()


# Deliberately excludes the non-breaking space: the notes use <i>&nbsp;</i> as a
# spacer between a symbol and the text after it, and dropping those runs words
# together ("weights of" -> "weightsof").
ASCII_SPACE = " \t\r\n\f\v​"


def is_blank(node) -> bool:
    """True for whitespace text and for tags holding nothing visible."""
    if isinstance(node, NavigableString):
        return not str(node).strip(ASCII_SPACE)
    if not isinstance(node, Tag):
        return True
    if node.name in ("img", "br", "hr", "figure", "table"):
        return False
    return not node.get_text().strip(ASCII_SPACE) and not node.find(["img", "br", "table"])


def prev_element(node):
    """Previous sibling that is not whitespace."""
    sib = node.previous_sibling
    while sib is not None and isinstance(sib, NavigableString) and not sib.strip():
        sib = sib.previous_sibling
    return sib


def next_element(node):
    sib = node.next_sibling
    while sib is not None and isinstance(sib, NavigableString) and not sib.strip():
        sib = sib.next_sibling
    return sib


# --------------------------------------------------------------------------
# transformation passes
# --------------------------------------------------------------------------

def strip_junk(body: Tag) -> None:
    """Remove scripts, comments, Evernote anchors and other non-content."""
    for c in body.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    for t in body.find_all(["script", "style", "basefont", "meta", "link"]):
        t.decompose()
    # Evernote writes <a name="1254"/> markers. HTML has no self-closing <a>, so
    # the parser makes them swallow the rest of the page: unwrap, never remove.
    for a in body.find_all("a"):
        if a.get("name") and not a.get("href"):
            a.unwrap()


def apply_colour_classes(body: Tag) -> None:
    """Convert colour attributes/styles into a small set of semantic classes.

    Done before the styling tags are unwrapped, so the emphasis survives as a
    class on a span the reader's stylesheet can target.
    """
    for tag in body.find_all(True):
        colour = colour_of(tag)
        if not colour:
            continue
        cls = COLOUR_CLASS.get(colour)
        if cls is None:
            if colour in COLOUR_IGNORE or not re.fullmatch(r"[0-9a-f]{6}", colour):
                continue
            # An unrecognised colour still signals emphasis; keep it generic.
            cls = "hl"
        if tag.name in UNWRAP:
            # Retag as a span so unwrapping later does not lose the class.
            tag.name = "span"
        existing = tag.get("class") or []
        if cls not in existing:
            tag["class"] = existing + [cls]


def promote_headings(soup: BeautifulSoup, body: Tag) -> None:
    """Turn the oversized-font pseudo-headings into real <h2> elements."""
    candidates = [t for t in body.find_all(True) if (font_px(t) or 0) >= HEADING_PX]

    for tag in candidates:
        if tag.parent is None:  # already consumed by an ancestor
            continue
        if any((font_px(p) or 0) >= HEADING_PX for p in tag.parents if p is not body):
            continue
        if not tag.get_text(strip=True):
            continue

        # Where the element also holds body content, only the leading inline run
        # is the heading; everything from the first block child onwards stays put.
        first_block = None
        for child in tag.children:
            if isinstance(child, Tag) and child.name in BLOCK:
                first_block = child
                break

        heading = soup.new_tag("h2")

        if first_block is None:
            heading.extend(list(tag.contents))
            tag.replace_with(heading)
        else:
            lead = []
            for child in list(tag.contents):
                if child is first_block:
                    break
                lead.append(child.extract())
            heading.extend(lead)
            if heading.get_text(strip=True):
                tag.insert_before(heading)
            tag.unwrap()

        # Headings should not carry nested emphasis markup.
        for inner in heading.find_all(["b", "strong", "u", "font", "span", "div", "i", "em"]):
            inner.unwrap()


def looks_like_code(text: str) -> bool:
    """Distinguish an Octave fragment from a sentence that happens to be in Courier."""
    if not text.strip():
        return False
    if re.search(r"[=(){};\[\]]|:=|\bfor\b|\bend\b|\bfunction\b", text):
        return True
    return len(text.split()) <= 6


def mark_code(body: Tag) -> None:
    """Flag anything set in Courier New; the notes use it for Octave code.

    The author left the odd line of ordinary prose in Courier too, so the face
    alone is not enough - the text has to read like code as well.
    """
    for tag in body.find_all(True):
        family = f"{tag.get('face') or ''} {style_of(tag)}"
        if "courier" in family.lower() and looks_like_code(tag.get_text(" ", strip=True)):
            tag["data-code"] = "1"


def blocks_from_divs(soup: BeautifulSoup, body: Tag) -> None:
    """Give every <div> a real block identity instead of unwrapping it.

    A <div> holding only inline content is a line of text and becomes a <p> (or a
    <pre> where that line is code). A <div> wrapping other blocks was only ever
    layout, and is unwrapped. Unwrapping both kinds would silently run separate
    lines together, which matters for the worked examples.
    """
    for div in body.find_all("div"):
        if div.parent is None:
            continue
        if div.find(["ul", "ol", "figure", "table", "h2", "h3", "div", "p", "pre"]):
            # Wrap its own text before unwrapping, otherwise that text merges
            # into whatever inline run happens to sit next to it afterwards.
            wrap_loose_runs(soup, div)
            div.unwrap()
            continue
        if not div.get_text(strip=True):
            div.decompose()
            continue

        is_code = bool(div.find(attrs={"data-code": "1"})) or div.get("data-code") == "1"
        node = soup.new_tag("pre" if is_code else "p")
        node.extend(list(div.contents))
        div.replace_with(node)


def merge_code_blocks(soup: BeautifulSoup, body: Tag) -> None:
    """Join consecutive one-line <pre> elements into a single code block."""
    for pre in body.find_all("pre"):
        if pre.parent is None:
            continue
        while True:
            nxt = next_element(pre)
            if not (isinstance(nxt, Tag) and nxt.name == "pre"):
                break
            pre.append(NavigableString("\n"))
            for node in list(nxt.contents):
                pre.append(node.extract())
            nxt.decompose()

    # Inside a code block the colour markup is noise; only the trailing comment
    # is worth keeping distinct.
    for pre in body.find_all("pre"):
        for tag in pre.find_all(True):
            if tag.parent is None:
                continue
            if tag.name == "br":
                tag.replace_with(NavigableString("\n"))
                continue
            if tag.name in ("sub", "sup"):
                continue  # the pseudo-code carries real index notation
            is_comment = "term" in (tag.get("class") or []) and re.match(
                r"\s*[%#]", tag.get_text()
            )
            if is_comment:
                tag.name = "span"
                tag.attrs = {"class": ["comment"]}
                for inner in tag.find_all(True):
                    inner.unwrap()
            else:
                tag.unwrap()

    # Inline code inside ordinary prose keeps a <code> wrapper.
    for tag in body.find_all(attrs={"data-code": "1"}):
        if tag.find_parent("pre") is not None:
            del tag["data-code"]
            continue
        if tag.name in UNWRAP and tag.get_text(strip=True):
            tag.name = "code"
        del tag["data-code"]


def headings_from_paragraphs(soup: BeautifulSoup, body: Tag) -> None:
    """Turn a stand-alone, fully emphasised line into an <h3> sub-heading.

    Within a section the notes label sub-topics with a bold or underlined line of
    its own ("Overfitting with linear regression"). Without this they render as a
    bare <strong> adrift between two lists, reading as neither heading nor item.
    """
    for p in body.find_all("p"):
        if p.parent is None or p.find_parent("li"):
            continue
        text = p.get_text(" ", strip=True)
        if not text or len(text) > 140 or p.find(["br", "img"]):
            continue

        emphasised = [e for e in p.find_all(["b", "strong", "u"]) if e.parent is not None]
        if not emphasised:
            continue
        marked = " ".join(e.get_text(" ", strip=True) for e in emphasised)
        # Require the emphasis to cover essentially the whole line.
        if len(re.sub(r"\W", "", marked)) < len(re.sub(r"\W", "", text)):
            continue

        h3 = soup.new_tag("h3")
        h3.extend(list(p.contents))
        p.replace_with(h3)
        # Headings should not carry nested emphasis markup, but sub/sup is real
        # notation and stays.
        for inner in h3.find_all(["b", "strong", "u", "span", "em", "i", "code"]):
            inner.unwrap()


def normalise_lists(body: Tag) -> None:
    """Repair lists that contain a bare nested list rather than a list item.

    A <ul> may only contain <li>. Where the source leaves a sub-list as a direct
    child of another list, it belongs to the item just above it.
    """
    changed = True
    while changed:
        changed = False
        for lst in body.find_all(["ul", "ol"]):
            for child in lst.find_all(["ul", "ol"], recursive=False):
                target = prev_element(child)
                if isinstance(target, Tag) and target.name == "li":
                    target.append(child.extract())
                else:
                    child.unwrap()
                changed = True

    # Two adjacent sub-lists inside one item are a single list.
    for lst in body.find_all(["ul", "ol"]):
        nxt = next_element(lst)
        while isinstance(nxt, Tag) and nxt.name == lst.name and lst.parent is not None:
            for node in list(nxt.contents):
                lst.append(node.extract())
            after = next_element(nxt)
            nxt.decompose()
            nxt = after


def wrap_loose_runs(soup: BeautifulSoup, body: Tag) -> None:
    """Wrap inline content still sitting directly in the body into a block."""
    run: list = []

    def flush() -> None:
        if not run:
            return
        nodes, run[:] = list(run), []
        if not any(
            (n.strip() if isinstance(n, NavigableString) else n.get_text(strip=True) or n.name in ("img", "br"))
            for n in nodes
        ):
            return
        # A run made entirely of code is a line of code, not a paragraph.
        tags = [n for n in nodes if isinstance(n, Tag)]
        text_outside_code = "".join(
            str(n) for n in nodes if isinstance(n, NavigableString)
        ).strip()
        all_code = (
            bool(tags)
            and not text_outside_code
            and all(t.get("data-code") == "1" or t.find(attrs={"data-code": "1"}) for t in tags)
        )
        block = soup.new_tag("pre" if all_code else "p")
        nodes[0].insert_before(block)
        for n in nodes:
            block.append(n.extract())

    for child in list(body.children):
        if isinstance(child, Tag) and child.name in BLOCK_LEVEL:
            flush()
        else:
            run.append(child)
    flush()


def fix_list_nesting(body: Tag) -> None:
    """Move sub-lists out of the invisible <li> wrappers and into their parent item.

    Evernote represents a nested list as a *sibling* list item styled
    ``display: inline; list-style: none`` rather than nesting the <ul> inside the
    item it belongs to. Rendered, this looks right; structurally it is wrong, and
    it is why the source lists are so hard to read.
    """
    changed = True
    while changed:
        changed = False
        for li in body.find_all("li"):
            st = style_of(li)
            spacer = ("display: inline" in st or "display:inline" in st) and (
                "list-style: none" in st or "list-style:none" in st
            )
            if not spacer:
                # A list item whose entire content is a nested list is the same hack.
                kids = [c for c in li.children if not is_blank(c)]
                spacer = len(kids) == 1 and isinstance(kids[0], Tag) and kids[0].name in ("ul", "ol")
            if not spacer:
                continue

            # Unwrap any blockquote used purely as an indent.
            for bq in li.find_all("blockquote"):
                bq.unwrap()

            target = prev_element(li)
            while target is not None and not (isinstance(target, Tag) and target.name == "li"):
                target = prev_element(target)

            contents = [c.extract() for c in list(li.contents) if not is_blank(c)]
            if target is not None:
                for node in contents:
                    target.append(node)
            else:
                for node in contents:
                    li.insert_before(node)
            li.decompose()
            changed = True


def images_to_figures(soup: BeautifulSoup, body: Tag) -> None:
    """Give every diagram a <figure> of its own, out of the flow of the text."""
    for img in body.find_all("img"):
        for attr in list(img.attrs):
            if attr not in ("src", "alt"):
                del img[attr]
        img["alt"] = img.get("alt") or ""
        img["loading"] = "lazy"

        fig = soup.new_tag("figure")
        img.replace_with(fig)
        fig.append(img)


def drop_spacing_breaks(body: Tag) -> None:
    """Remove <br> used as vertical spacing, keeping genuine line breaks."""
    for br in body.find_all("br"):
        nxt, prv = next_element(br), prev_element(br)
        adjacent_block = any(
            isinstance(n, Tag) and n.name in ("ul", "ol", "figure", "img", "h2", "h3", "div", "table")
            for n in (nxt, prv)
        )
        if nxt is None or prv is None or adjacent_block:
            br.decompose()


def strip_styling(body: Tag) -> None:
    """Drop presentational attributes and unwrap the tags that only carried them."""
    for tag in body.find_all(True):
        for attr in ("style", "face", "size", "color", "bgcolor", "align",
                     "border", "cellpadding", "cellspacing", "width", "height",
                     "type", "valign", "background", "id"):
            if attr in tag.attrs and not (tag.name == "h2" and attr == "id"):
                del tag[attr]

    for tag in body.find_all(list(UNWRAP)):
        if tag.get("class") or tag.get("data-code"):
            # Carries a semantic colour class or the code marker; keep it as a
            # span so the meaning survives, but drop the presentational tag name.
            tag.name = "span"
            continue
        tag.unwrap()

    # Modernise the emphasis tags.
    for tag in body.find_all("b"):
        tag.name = "strong"
    for tag in body.find_all("i"):
        tag.name = "em"


def tidy(body: Tag) -> None:
    """Collapse redundant wrappers, whitespace and empty elements."""
    # Nested spans of the same class add nothing.
    for span in body.find_all("span"):
        parent = span.parent
        if (
            isinstance(parent, Tag)
            and parent.name == "span"
            and parent.get("class") == span.get("class")
            and len([c for c in parent.children if not is_blank(c)]) == 1
        ):
            span.unwrap()

    for _ in range(4):
        for tag in body.find_all(["div", "span", "strong", "em", "u", "p", "li", "ul", "ol"]):
            if tag.parent is None:
                continue
            if is_blank(tag):
                tag.decompose()

    # Normalise whitespace inside text, without touching content.
    for text in body.find_all(string=True):
        if text.find_parent(["pre", "code"]) is not None:
            continue
        new = re.sub(r"[ \t\r\f\v ]+", " ", text.replace("\n", " "))
        if new != text:
            text.replace_with(NavigableString(new))


def sectionise(soup: BeautifulSoup, body: Tag) -> list[tuple[str, str]]:
    """Wrap the run of content beneath each <h2> in a <section>; return the TOC."""
    toc: list[tuple[str, str]] = []
    used: set[str] = set()

    headings = [h for h in body.find_all("h2")]
    for h in headings:
        title = h.get_text(" ", strip=True)
        slug = base = slugify(title)
        n = 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        toc.append((slug, title))

        section = soup.new_tag("section")
        section["id"] = slug
        h.insert_before(section)
        section.append(h.extract())

        node = section.next_sibling
        while node is not None:
            nxt = node.next_sibling
            if isinstance(node, Tag) and node.name in ("h2", "section"):
                break
            section.append(node.extract())
            node = nxt
    return toc


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

def node_html(node) -> str:
    """Serialise one node. Bare text must be escaped; BeautifulSoup escapes tags."""
    if isinstance(node, NavigableString):
        return html.escape(str(node), quote=False)
    return str(node)


def indent_html(tag: Tag, level: int = 0) -> str:
    """Pretty-print with block elements on their own lines and inline runs intact."""
    pad = "  " * level
    out: list[str] = []

    block_children = {"ul", "ol", "li", "section", "figure", "h2", "h3", "p", "table", "tr"}

    # Consecutive inline nodes must stay on one line. Splitting them across lines
    # would introduce whitespace the source never had, turning "mat"+"rix" into
    # "mat rix" once the browser collapses the newline.
    inline: list[str] = []

    def flush_inline() -> None:
        if not inline:
            return
        run = re.sub(r"\s+", " ", "".join(inline)).strip()
        inline.clear()
        if run:
            out.append(pad + run)

    for child in tag.children:
        if isinstance(child, NavigableString):
            inline.append(node_html(child))
            continue
        if child.name == "pre":
            # Whitespace is significant; emit verbatim.
            flush_inline()
            out.append(pad + str(child))
            continue
        if child.name in block_children:
            flush_inline()
            attrs = "".join(
                f' {k}="{html.escape(" ".join(v) if isinstance(v, list) else str(v), quote=True)}"'
                for k, v in child.attrs.items()
            )
            inner_blocks = any(
                isinstance(g, Tag) and g.name in block_children or
                (isinstance(g, Tag) and g.name == "pre")
                for g in child.children
            )
            if inner_blocks:
                out.append(f"{pad}<{child.name}{attrs}>")
                out.append(indent_html(child, level + 1))
                out.append(f"{pad}</{child.name}>")
            else:
                inner = re.sub(r"\s+", " ", "".join(node_html(g) for g in child.children)).strip()
                out.append(f"{pad}<{child.name}{attrs}>{inner}</{child.name}>")
        else:
            inline.append(node_html(child))
    flush_inline()
    return "\n".join(x for x in out if x.strip())


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} &mdash; Stanford Machine Learning</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="topbar">
  <a href="index.html">Notes index</a>
  <a href="https://holehouse.org">holehouse.org</a>
</header>

<main>
<h1>{h1}</h1>

{nav}
{toc}
{content}

{nav_bottom}
</main>
</body>
</html>
"""


def build_nav(prev_href, prev_label, next_href, next_label, bottom=False) -> str:
    cls = "chapter-nav bottom" if bottom else "chapter-nav"
    left = f'<a href="{prev_href}" rel="prev">&larr; {html.escape(prev_label)}</a>' if prev_href else "<span></span>"
    right = f'<a href="{next_href}" rel="next">{html.escape(next_label)} &rarr;</a>' if next_href else "<span></span>"
    return (
        f'<nav class="{cls}">\n  {left}\n'
        f'  <a href="index.html">Index</a>\n  {right}\n</nav>'
    )


def build_toc(toc: list[tuple[str, str]]) -> str:
    if len(toc) < 2:
        return ""
    items = "\n".join(f'    <li><a href="#{s}">{html.escape(t)}</a></li>' for s, t in toc)
    return f'<nav class="toc" aria-label="On this page">\n  <h2>On this page</h2>\n  <ul>\n{items}\n  </ul>\n</nav>'


def convert(path: str, meta: dict) -> tuple[str, list[tuple[str, str]]]:
    raw = open(path, encoding="utf-8", errors="replace").read()
    raw = raw.lstrip("﻿")
    soup = BeautifulSoup(raw, "html5lib")
    body = soup.body

    strip_junk(body)

    h1 = body.find("h1")
    h1_text = h1.get_text(" ", strip=True) if h1 else meta["title"]
    if h1:
        h1.decompose()

    # The original nav paragraph is rebuilt from the chapter table.
    for p in body.find_all("p"):
        links = {a.get_text(strip=True).lower() for a in p.find_all("a")}
        if links & {"previous", "next", "index"}:
            p.decompose()

    apply_colour_classes(body)
    mark_code(body)
    promote_headings(soup, body)
    fix_list_nesting(body)
    images_to_figures(soup, body)
    drop_spacing_breaks(body)
    strip_styling(body)
    blocks_from_divs(soup, body)
    tidy(body)
    wrap_loose_runs(soup, body)
    merge_code_blocks(soup, body)
    headings_from_paragraphs(soup, body)
    normalise_lists(body)
    toc = sectionise(soup, body)

    content = indent_html(body, 0)
    page = PAGE.format(
        title=html.escape(h1_text),
        h1=html.escape(h1_text),
        nav=build_nav(meta["prev_href"], meta["prev_label"], meta["next_href"], meta["next_label"]),
        toc=build_toc(toc),
        content=content,
        nav_bottom=build_nav(meta["prev_href"], meta["prev_label"], meta["next_href"], meta["next_label"], bottom=True),
    )
    return page, toc
