"""One-off conversion of the chapter HTML into the Markdown sources in ``src/``.

This ran once, to bootstrap the Markdown sources from the hand-built V3 HTML.
It is kept so the derivation of the sources is on record, and so the process
can be repeated if a chapter is ever edited as HTML by mistake.

What becomes Markdown: headings, paragraphs, the nested bullet lists that make
up most of the notes, bold, italic, inline code, links, and code blocks whose
only markup is ``#`` comments. Everything else (MathML equations, figures,
tables, styled spans, sub- and superscripts, anything with a class) is copied
into the Markdown verbatim as an HTML island, which ``build.py`` passes
through untouched.

The converter is conservative: wherever Markdown syntax could be ambiguous
(bold text ending in punctuation, a link whose target contains a space), it
falls back to the verbatim HTML. ``check.py`` then proves the round trip.

Usage::

    python build/html2md.py            # write src/*.md from mlclass/*.html
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import unicodedata
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
SRC = os.path.join(SITE, "src")

sys.path.insert(0, HERE)
from build import slugify  # noqa: E402  (same slug rule as the build)

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}
INLINE_TAGS = {
    "a",
    "span",
    "strong",
    "em",
    "code",
    "sub",
    "sup",
    "math",
    "br",
    "small",
    "b",
    "i",
    "u",
    "kbd",
    "img",
}
SKIP_STEMS = {"05_Octave", "01_02_Introduction_regression_analysis_and_gr", "changelog"}

WS_RE = re.compile(r"[ \t\r\n\f]+")

# ---------------------------------------------------------------------------
# a small DOM that remembers where every element sits in the source
# ---------------------------------------------------------------------------


class Node:
    """An element in the parsed HTML.

    Attributes
    ----------
    tag : str
        Lower-case tag name.
    attrs : list[tuple[str, str | None]]
        Attributes in source order.
    start, end : int
        Offsets of the element's first and last character in the source, so
        its exact text can be sliced out for an HTML island.
    kids : list[Node | str]
        Child elements and decoded text.
    """

    def __init__(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        start: int,
        parent: Node | None,
    ) -> None:
        self.tag, self.attrs, self.start, self.parent = tag, attrs, start, parent
        self.end = start
        self.kids: list[Node | str] = []

    def get(self, name: str) -> str | None:
        """Return an attribute value, or ``None``."""
        return next((v for k, v in self.attrs if k == name), None)

    @property
    def classes(self) -> list[str]:
        """The element's classes."""
        return (self.get("class") or "").split()

    @property
    def elements(self) -> list[Node]:
        """Child elements only."""
        return [k for k in self.kids if isinstance(k, Node)]

    def text(self) -> str:
        """All descendant text, concatenated."""
        return "".join(k if isinstance(k, str) else k.text() for k in self.kids)

    def find(self, tag: str, cls: str | None = None) -> Node | None:
        """First descendant with this tag (and class), depth first."""
        for k in self.elements:
            if k.tag == tag and (cls is None or cls in k.classes):
                return k
            found = k.find(tag, cls)
            if found is not None:
                return found
        return None

    def find_all(self, tag: str) -> list[Node]:
        """Every descendant with this tag, in document order."""
        out = []
        for k in self.elements:
            if k.tag == tag:
                out.append(k)
            out.extend(k.find_all(tag))
        return out


class TreeBuilder(HTMLParser):
    """Parse HTML into ``Node`` objects that carry source offsets."""

    def __init__(self, raw: str) -> None:
        super().__init__(convert_charrefs=True)
        self.raw = raw
        self.line_starts = [0] + [m.end() for m in re.finditer(r"\n", raw)]
        self.root = Node("#root", [], 0, None)
        self.cur = self.root

    def abs_pos(self) -> int:
        """Absolute offset of the token currently being handled."""
        line, col = self.getpos()
        return self.line_starts[line - 1] + col

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        pos = self.abs_pos()
        node = Node(tag, attrs, pos, self.cur)
        self.cur.kids.append(node)
        raw_tag = self.get_starttag_text() or ""
        if tag in VOID_TAGS or raw_tag.rstrip().endswith("/>"):
            node.end = pos + len(raw_tag)
        else:
            self.cur = node

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        pos = self.abs_pos()
        node = Node(tag, attrs, pos, self.cur)
        node.end = pos + len(self.get_starttag_text() or "")
        self.cur.kids.append(node)

    def handle_endtag(self, tag: str) -> None:
        pos = self.abs_pos()
        m = re.match(r"</[^>]*>", self.raw[pos:])
        node: Node | None = self.cur
        while node is not None and node is not self.root and node.tag != tag:
            node = node.parent
        if node is None or node is self.root or node.parent is None:
            return
        node.end = pos + (len(m.group(0)) if m else 0)
        self.cur = node.parent

    def handle_data(self, data: str) -> None:
        self.cur.kids.append(data)


def parse(raw: str) -> Node:
    """Parse an HTML document.

    Parameters
    ----------
    raw : str
        Document source.

    Returns
    -------
    Node
        The root node; the document's elements are its children.
    """
    tb = TreeBuilder(raw)
    tb.feed(raw)
    tb.close()
    return tb.root


# ---------------------------------------------------------------------------
# inline conversion
# ---------------------------------------------------------------------------


def is_inline(node: Node | str) -> bool:
    """Whether a node is inline content (text, or an inline element)."""
    if isinstance(node, str):
        return True
    if node.tag == "math":
        return node.get("display") != "block"
    return node.tag in INLINE_TAGS


def is_punct(ch: str) -> bool:
    """Whether CommonMark would treat ``ch`` as punctuation."""
    return ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" or unicodedata.category(
        ch
    ).startswith("P")


def escape_text(text: str) -> str:
    """Backslash-escape the characters that would otherwise be Markdown.

    Parameters
    ----------
    text : str
        Plain text (already whitespace-collapsed).

    Returns
    -------
    str
        Text safe to place in a Markdown paragraph.
    """
    text = text.replace("\\", "\\\\")
    for ch in "*`[]<":
        text = text.replace(ch, "\\" + ch)
    text = re.sub(r"(?<![0-9A-Za-z])_|_(?![0-9A-Za-z])", r"\\_", text)
    text = re.sub(r"&(?=[A-Za-z0-9#]+;)", "&amp;", text)
    return text


def escape_line_start(line: str) -> str:
    """Escape anything that would turn the start of a line into block syntax.

    Parameters
    ----------
    line : str
        One line of paragraph text.

    Returns
    -------
    str
        The line, with a leading heading, list, quote or fence marker escaped.
    """
    if re.match(r"#{1,6}(\s|$)", line) or line.startswith(
        (">", "-", "+", "=", "~", "|", ":::")
    ):
        return "\\" + line
    m = re.match(r"(\d{1,9})([.)])(\s|$)", line)
    if m:
        return line[: m.end(1)] + "\\" + line[m.end(1) :]
    return line


class Inline:
    """Convert a run of inline nodes to Markdown text.

    Parameters
    ----------
    raw : str
        Document source, for verbatim islands.
    """

    def __init__(self, raw: str) -> None:
        self.raw = raw

    def island(self, node: Node) -> str:
        """The element's source text, as an inline island."""
        return self.raw[node.start : node.end]

    def parts(self, nodes: list[Node | str]) -> list[tuple[str, str]]:
        """Convert nodes to ``(kind, text)`` parts.

        ``kind`` is ``text`` for escaped prose, ``emph`` for a ``*``-delimited
        run, or ``html`` for a verbatim island. Two emphasis runs back to back
        would fuse their delimiters, so the second falls back to HTML.
        """
        out: list[tuple[str, str]] = []
        for node in nodes:
            if isinstance(node, str):
                out.append(("text", escape_text(WS_RE.sub(" ", node))))
            elif node.tag in ("strong", "em") and not node.attrs:
                part = self.emphasis(node)
                if part[0] == "emph" and out and out[-1][1].endswith("*"):
                    part = ("html", self.island(node))
                out.append(part)
            elif node.tag == "code" and not node.attrs:
                out.append(self.code(node))
            elif node.tag == "a" and [k for k, _ in node.attrs] == ["href"]:
                out.append(self.link(node))
            else:
                out.append(("html", self.island(node)))
        return out

    @staticmethod
    def edge(parts: list[tuple[str, str]], last: bool) -> str:
        """The effective first (or last) character of a run; islands count as letters."""
        seq = reversed(parts) if last else parts
        for kind, text in seq:
            if not text:
                continue
            return "a" if kind == "html" else (text[-1] if last else text[0])
        return ""

    def emphasis(self, node: Node) -> tuple[str, str]:
        """``<strong>`` or ``<em>`` as ``**x**`` / ``*x*`` where that is unambiguous."""
        inner = self.parts(node.kids)
        first, last = self.edge(inner, False), self.edge(inner, True)
        ok = (
            first
            and last
            and not first.isspace()
            and not last.isspace()
            and not is_punct(first)
            and not is_punct(last)
            and not any(kind == "emph" for kind, _ in inner)
        )
        if not ok:
            return ("html", self.island(node))
        mark = "**" if node.tag == "strong" else "*"
        return ("emph", mark + "".join(t for _, t in inner) + mark)

    def code(self, node: Node) -> tuple[str, str]:
        """``<code>`` as a backtick span where the content allows it."""
        text = node.text()
        if (
            not text
            or text[0] == " "
            or text[-1] == " "
            or "\n" in text
            or node.elements
        ):
            return ("html", self.island(node))
        longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
        fence = "`" * (longest + 1)
        pad = " " if text.startswith("`") or text.endswith("`") else ""
        return ("text", f"{fence}{pad}{text}{pad}{fence}")

    def link(self, node: Node) -> tuple[str, str]:
        """``<a href>`` as ``[text](href)`` where the target is a plain URL."""
        href = node.get("href") or ""
        inner = self.parts(node.kids)
        text = "".join(t for _, t in inner)
        if (
            not text.strip()
            or re.search(r"[\s()<>]", href)
            or any(k == "html" and "\n" in t for k, t in inner)
        ):
            return ("html", self.island(node))
        return ("text", f"[{text}]({href})")

    def run(self, nodes: list[Node | str]) -> str:
        """Convert a run to one line of Markdown, trimmed at both ends."""
        text = "".join(t for _, t in self.parts(nodes))
        return re.sub(r"^ +| +$", "", text)


# ---------------------------------------------------------------------------
# block conversion
# ---------------------------------------------------------------------------


class Converter:
    """Convert one chapter's ``<main>`` to Markdown.

    Parameters
    ----------
    raw : str
        Document source.
    stem : str
        File stem, for messages.
    """

    def __init__(self, raw: str, stem: str) -> None:
        self.raw = raw
        self.stem = stem
        self.inline = Inline(raw)
        self.warnings: list[str] = []

    # -- islands ------------------------------------------------------------

    def block_island(self, node: Node, indent: str) -> list[str]:
        """An element copied verbatim, as lines at the given indent."""
        src = self.raw[node.start : node.end]
        lines = src.split("\n")
        if len(lines) > 1 and "<pre" not in src:
            lines = [lines[0]] + textwrap.dedent("\n".join(lines[1:])).split("\n")
        return [indent + ln if ln.strip() else "" for ln in lines]

    # -- code -----------------------------------------------------------------

    @staticmethod
    def fenceable(pre: Node) -> bool:
        """Whether a ``<pre>`` holds only text and ``#`` comment spans."""
        for k in pre.elements:
            if (
                k.tag != "span"
                or k.classes != ["comment"]
                or k.elements
                or not k.text().startswith("#")
            ):
                return False
        return "#" not in "".join(k for k in pre.kids if isinstance(k, str))

    def containerable(self, node: Node) -> bool:
        """Whether a ``<div class>`` can be a ``:::`` container.

        Only wrappers around a code example qualify: a classed div, no bare
        text, and at least one code block that can be a fence.
        """
        if (
            node.tag != "div"
            or [k for k, _ in node.attrs] != ["class"]
            or not node.classes
        ):
            return False
        if any(isinstance(k, str) and k.strip() for k in node.kids):
            return False
        return any(
            k.tag == "pre" and not k.attrs and self.fenceable(k) for k in node.elements
        )

    def fence(self, pre: Node, indent: str) -> list[str]:
        """A ``<pre>`` as a fenced code block."""
        text = pre.text()
        longest = max((len(m) for m in re.findall(r"`{3,}", text)), default=2)
        mark = "`" * max(3, longest + 1)
        body = text[:-1] + "\n" if text.endswith("\n") else text
        lines = [indent + mark]
        lines += [indent + ln if ln else "" for ln in body.split("\n")]
        lines.append(indent + mark)
        return lines

    # -- blocks ---------------------------------------------------------------

    def blocks(self, kids: list[Node | str], indent: str, in_list: bool) -> list[str]:
        """Convert a sequence of children to Markdown lines.

        Parameters
        ----------
        kids : list[Node | str]
            Child nodes of a section or list item.
        indent : str
            Indentation for every emitted line.
        in_list : bool
            Whether the parent is a list item. Inside an item the blocks run
            together so the list stays tight; at section level they are
            separated by blank lines, which stops a block that follows a list
            from being read as a continuation of its last item.
        """
        chunks: list[list[str]] = []
        run: list[Node | str] = []

        def flush() -> None:
            if not any(isinstance(n, Node) or n.strip() for n in run):
                run.clear()
                return
            has_text = any(isinstance(n, str) and n.strip() for n in run)
            if in_list or has_text:
                if not in_list:
                    self.warnings.append(
                        f"{self.stem}: bare text at section level becomes a paragraph"
                    )
                text = self.inline.run(run)
                if text:
                    chunks.append([indent + escape_line_start(text)])
            else:
                # Inline elements sitting directly in a section, with no <p>
                # around them: each becomes a block island so no <p> is added.
                for n in run:
                    if isinstance(n, Node):
                        chunks.append(self.block_island(n, indent))
            run.clear()

        for node in kids:
            if isinstance(node, str) or is_inline(node):
                run.append(node)
                continue
            flush()
            if node.tag == "ul":
                chunks.append(self.bullet_list(node, indent))
            elif node.tag == "h3" and not node.attrs:
                chunks.append([indent + "### " + self.inline.run(node.kids)])
            elif node.tag == "p" and not node.attrs and not in_list:
                parts = self.inline.parts(node.kids)
                if all(kind == "html" for kind, text in parts if text.strip()):
                    # Nothing but islands inside: keep the <p> itself verbatim,
                    # or the build would see a lone island and drop the <p>.
                    chunks.append(self.block_island(node, indent))
                else:
                    chunks.append(
                        [indent + escape_line_start(self.inline.run(node.kids))]
                    )
            elif node.tag == "pre" and not node.attrs and self.fenceable(node):
                chunks.append(self.fence(node, indent))
            elif self.containerable(node):
                inner = self.blocks(node.kids, indent, in_list=True)
                chunks.append(
                    [indent + "::: " + " ".join(node.classes)]
                    + inner
                    + [indent + ":::"]
                )
            else:
                chunks.append(self.block_island(node, indent))
        flush()
        lines: list[str] = []
        for i, chunk in enumerate(chunks):
            if i and not in_list:
                lines.append("")
            lines.extend(chunk)
        return lines

    def bullet_list(self, ul: Node, indent: str) -> list[str]:
        """A ``<ul>`` as a bullet list, or verbatim if Markdown cannot express it."""
        for li in ul.elements:
            if li.tag != "li":
                self.warnings.append(
                    f"{self.stem}: <{li.tag}> directly inside <ul>; list kept as HTML"
                )
                return self.block_island(ul, indent)
            kids = li.kids
            uls = [
                i for i, k in enumerate(kids) if isinstance(k, Node) and k.tag == "ul"
            ]
            after = kids[uls[0] + 1 :] if uls else []
            if any(isinstance(k, Node) or k.strip() for k in after):
                self.warnings.append(
                    f"{self.stem}: content after a nested list in one item; list kept as HTML"
                )
                return self.block_island(ul, indent)
        lines: list[str] = []
        for li in ul.elements:
            body = self.blocks(li.kids, indent + "  ", in_list=True)
            if not body:
                lines.append(indent + "-")
                continue
            # The first line always goes on the marker line. For a block-first
            # item that keeps the item from starting with a lone "-", which a
            # following line could turn into a setext heading underline.
            lines.append(indent + "- " + body[0][len(indent) + 2 :])
            lines.extend(body[1:])
        return lines

    def section(self, sec: Node, used: set[str]) -> list[str]:
        """A ``<section>`` as a ``##`` heading and its content."""
        kids = sec.kids
        h2 = next(k for k in kids if isinstance(k, Node) and k.tag == "h2")
        title = h2.text()
        if h2.elements:
            self.warnings.append(f"{self.stem}: heading contains markup: {title!r}")
        slug = base = slugify(WS_RE.sub(" ", title).strip())
        n = 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        heading = "## " + self.inline.run(h2.kids)
        if sec.get("id") != slug:
            heading += f" {{#{sec.get('id')}}}"
        rest = [k for k in kids if k is not h2]
        return [heading, ""] + self.blocks(rest, "", in_list=False)

    def convert(self, main: Node) -> str:
        """The chapter body as Markdown."""
        out: list[str] = []
        used: set[str] = set()
        for node in main.elements:
            if node.tag == "h1" or (
                node.tag in ("nav", "div")
                and ({"chapter-nav", "toc"} & set(node.classes))
            ):
                continue
            if node.tag == "section":
                out.extend(self.section(node, used))
                out.append("")
            else:
                self.warnings.append(
                    f"{self.stem}: unexpected <{node.tag}> directly in <main>"
                )
                out.extend(self.block_island(node, ""))
                out.append("")
        return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# metadata and driver
# ---------------------------------------------------------------------------


def need(node: Node | None, what: str) -> Node:
    """Return ``node``, or stop with a clear message when a page lacks it."""
    if node is None:
        raise SystemExit(f"expected {what} in the page")
    return node


def quote(value: str) -> str:
    """Quote a front-matter value."""
    return '"' + value.replace('"', '\\"') + '"'


def chapter_stems() -> list[str]:
    """Chapter file stems in reading order."""
    stems = []
    for name in sorted(os.listdir(SITE)):
        if re.match(r"(\d\d|A\d)_.*\.html$", name) and name[:-5] not in SKIP_STEMS:
            stems.append(name[:-5])
    return stems


def nav_titles(stems: list[str]) -> dict[str, str]:
    """The short title each chapter is given when linked from its neighbours."""
    titles: dict[str, str] = {}
    for stem in stems:
        root = parse(open(os.path.join(SITE, stem + ".html"), encoding="utf-8").read())
        nav = need(root.find("nav", "chapter-nav"), "chapter navigation")
        for a in nav.find_all("a"):
            if a.get("rel") in ("prev", "next"):
                titles[(a.get("href") or "")[:-5]] = (
                    a.text().replace("←", "").replace("→", "").strip()
                )
    return titles


def index_entries() -> dict[str, dict[str, str]]:
    """Title, origin tag and description of each chapter from the index page."""
    root = parse(open(os.path.join(SITE, "index.html"), encoding="utf-8").read())
    entries: dict[str, dict[str, str]] = {}
    for li in need(root.find("ul", "chapter-list"), "chapter list").find_all("li"):
        a = need(li.find("a"), "chapter link")
        entries[(a.get("href") or "")[:-5]] = {
            "index_title": WS_RE.sub(
                " ", need(li.find("span", "ch-title"), "title").text()
            ).strip(),
            "origin": "2026"
            if "ch-tag-new" in need(li.find("span", "ch-tag"), "tag").classes
            else "2011",
            "description": WS_RE.sub(
                " ", need(li.find("span", "ch-desc"), "description").text()
            ).strip(),
        }
    return entries


def convert_chapter(
    stem: str, navs: dict[str, str], index: dict[str, dict[str, str]]
) -> tuple[str, list[str]]:
    """Convert one chapter file to Markdown text.

    Parameters
    ----------
    stem : str
        Chapter file stem.
    navs : dict[str, str]
        Navigation titles by stem.
    index : dict[str, dict[str, str]]
        Index-page entries by stem.

    Returns
    -------
    tuple[str, list[str]]
        The Markdown source and any warnings.
    """
    raw = open(os.path.join(SITE, stem + ".html"), encoding="utf-8").read()
    root = parse(raw)
    main = need(root.find("main"), "<main>")
    title = WS_RE.sub(" ", need(main.find("h1"), "<h1>").text()).strip()
    meta = [("title", title), ("nav_title", navs[stem])]
    entry = index.get(stem, {})
    if entry.get("index_title") and entry["index_title"] != title:
        meta.append(("index_title", entry["index_title"]))
    meta.append(("origin", entry.get("origin", "2011")))
    meta.append(("description", entry.get("description", "")))
    conv = Converter(raw, stem)
    body = conv.convert(main)
    front = "\n".join(f"{k}: {quote(v)}" for k, v in meta)
    return f"---\n{front}\n---\n\n{body}", conv.warnings


def convert_index() -> tuple[str, list[str]]:
    """Convert the index page to ``index.md``."""
    raw = open(os.path.join(SITE, "index.html"), encoding="utf-8").read()
    root = parse(raw)
    main = need(root.find("main"), "<main>")
    title = WS_RE.sub(" ", need(root.find("title"), "<title>").text()).strip()
    h1 = WS_RE.sub(" ", need(main.find("h1"), "<h1>").text()).strip()
    conv = Converter(raw, "index")
    out: list[str] = []
    for node in main.elements:
        if node.tag == "h1":
            continue
        if node.tag == "ul" and "chapter-list" in node.classes:
            out.append("<!-- chapter-list -->")
        elif node.tag == "h2" and not node.attrs:
            out.append("## " + conv.inline.run(node.kids))
        elif node.tag == "p" and not node.attrs:
            out.append(escape_line_start(conv.inline.run(node.kids)))
        else:
            out.extend(conv.block_island(node, ""))
        out.append("")
    body = "\n".join(out).rstrip("\n") + "\n"
    return f"---\ntitle: {quote(title)}\nh1: {quote(h1)}\n---\n\n{body}", conv.warnings


def main() -> int:
    """Write every source file."""
    os.makedirs(SRC, exist_ok=True)
    stems = chapter_stems()
    navs = nav_titles(stems)
    index = index_entries()
    warnings: list[str] = []
    for stem in stems:
        text, warn = convert_chapter(stem, navs, index)
        warnings.extend(warn)
        with open(os.path.join(SRC, stem + ".md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"{stem:52s} {len(text):>8,d} chars")
    text, warn = convert_index()
    warnings.extend(warn)
    with open(os.path.join(SRC, "index.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"{'index':52s} {len(text):>8,d} chars")
    for w in warnings:
        print("warning:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
