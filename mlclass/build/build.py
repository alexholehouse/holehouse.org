"""Render the Markdown sources in ``src/`` to the HTML pages served from ``mlclass/``.

The HTML is output only. Every chapter page and the notes index are generated
from a Markdown file, and running this script again on unchanged sources
produces byte-identical output.

The Markdown dialect is CommonMark with three conventions on top, all
documented in ``src/README.md``:

* **HTML islands.** Any HTML element written into the source (an equation in
  MathML, a figure, a table, a ``<sub>``) is passed through verbatim. The
  island is located before the Markdown parser runs and swapped back in
  afterwards, so nothing inside it is ever interpreted as Markdown.
* **Section ids.** Every ``##`` heading opens a ``<section>`` whose id is the
  slug of the heading text. A heading can carry an explicit id with a
  trailing ``{#my-id}``.
* **Code comments.** Inside a fenced code block, everything from a ``#`` to the
  end of the line is wrapped in ``<span class="comment">``.
* **Containers.** A block between ``::: some-class`` and ``:::`` lines becomes
  ``<div class="some-class">``. The notes use it for the ``code-eg`` wrapper
  around a code example and its explanatory note.

Usage::

    python build/build.py            # render everything into mlclass/
    python build/build.py --out DIR  # render into another directory
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.container import container_plugin

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
SRC = os.path.join(SITE, "src")

# Placeholder delimiters. Private-use code points never occur in the notes and
# are neither whitespace nor punctuation to CommonMark, so a placeholder sits
# in running text exactly the way a word would.
PH_OPEN, PH_CLOSE = "", ""
PH_RE = re.compile(f"{PH_OPEN}([BI])(\\d+){PH_CLOSE}")

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

# Block-level tags, used only to indent the generated HTML for reading.
BLOCK_TAGS = {
    "section",
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
    "nav",
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
    "details",
    "summary",
    "header",
    "main",
}
VERBATIM_TAGS = {"pre", "math", "svg"}

CHAPTER_LIST_MARKER = "<!-- chapter-list -->"

# ---------------------------------------------------------------------------
# source files
# ---------------------------------------------------------------------------


@dataclass
class Page:
    """One Markdown source file, split into its front matter and body.

    Attributes
    ----------
    stem : str
        File name without extension; also the output file name.
    meta : dict[str, str]
        Front-matter fields.
    body : str
        Markdown body.
    """

    stem: str
    meta: dict[str, str]
    body: str
    islands: list[str] = field(default_factory=list)


class BuildError(Exception):
    """A source file cannot be rendered as written."""


def parse_front_matter(text: str, stem: str) -> tuple[dict[str, str], str]:
    """Split a source file into its front matter and Markdown body.

    The front matter is the block between two ``---`` lines at the top of the
    file. Each line is ``key: value``; a value may be wrapped in double quotes,
    in which case ``\\"`` inside it is an escaped quote.

    Parameters
    ----------
    text : str
        Full contents of the source file.
    stem : str
        File stem, used in error messages.

    Returns
    -------
    tuple[dict[str, str], str]
        The fields and the body that follows the closing ``---``.
    """
    if not text.startswith("---\n"):
        raise BuildError(f"{stem}.md: missing front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise BuildError(f"{stem}.md: front matter is not closed")
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise BuildError(f"{stem}.md: bad front matter line {line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1].replace('\\"', '"')
        meta[key.strip()] = value
    return meta, text[end + 5 :]


def load_pages() -> list[Page]:
    """Read every ``src/*.md`` file except the README.

    Returns
    -------
    list[Page]
        Pages sorted by file name, which is also the reading order.
    """
    pages = []
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".md") or name == "README.md":
            continue
        stem = name[:-3]
        with open(os.path.join(SRC, name), encoding="utf-8") as fh:
            meta, body = parse_front_matter(fh.read(), stem)
        pages.append(Page(stem, meta, body))
    return pages


# ---------------------------------------------------------------------------
# HTML islands
# ---------------------------------------------------------------------------

TAG_RE = re.compile(
    r"<!--.*?-->|<(/?)([A-Za-z][A-Za-z0-9-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*)>", re.S
)
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def fenced_ranges(text: str) -> list[tuple[int, int]]:
    """Find the character ranges covered by fenced code blocks.

    Parameters
    ----------
    text : str
        Markdown body.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` offsets of each fenced block, fences included.
    """
    ranges = []
    pos = 0
    open_fence: tuple[str, int, int] | None = None
    for line in text.splitlines(keepends=True):
        m = FENCE_RE.match(line)
        if open_fence is None:
            if m:
                open_fence = (m.group(1)[0], len(m.group(1)), pos)
        elif (
            m
            and m.group(1)[0] == open_fence[0]
            and len(m.group(1)) >= open_fence[1]
            and not line[m.end() :].strip()
        ):
            ranges.append((open_fence[2], pos + len(line)))
            open_fence = None
        pos += len(line)
    if open_fence is not None:
        ranges.append((open_fence[2], len(text)))
    return ranges


def island_end(text: str, start: int) -> int:
    """Find where the HTML element starting at ``start`` ends.

    Tags are balanced by name, void elements and self-closing tags do not
    nest, and comments are single tokens.

    Parameters
    ----------
    text : str
        Markdown body.
    start : int
        Offset of the opening ``<``.

    Returns
    -------
    int
        Offset just past the closing tag.
    """
    stack: list[str] = []
    for m in TAG_RE.finditer(text, start):
        if m.group(0).startswith("<!--"):
            if not stack:
                return m.end()
            continue
        closing, name, attrs = m.group(1), m.group(2).lower(), m.group(3)
        if closing:
            while stack and stack[-1] != name:
                stack.pop()
            if stack:
                stack.pop()
            if not stack:
                return m.end()
        elif name in VOID_TAGS or attrs.rstrip().endswith("/"):
            if not stack:
                return m.end()
        else:
            stack.append(name)
    line = text.count("\n", 0, start) + 1
    tag = TAG_RE.match(text, start)
    name = tag.group(2) if tag else "?"
    raise BuildError(f"unclosed HTML element <{name}> starting on line {line}")


def protect_islands(page: Page) -> str:
    """Replace every HTML island in the body with a placeholder.

    An island whose opening tag is the first thing on its line and whose
    closing tag is the last is a *block* island; anything else is *inline*.
    The distinction only matters after rendering, when a paragraph that holds
    nothing but block islands is unwrapped.

    Parameters
    ----------
    page : Page
        The page; its ``islands`` list is filled in.

    Returns
    -------
    str
        The body with placeholders in place of the islands.
    """
    text = page.body
    skip = fenced_ranges(text)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        fence = next(((a, b) for a, b in skip if a <= i < b), None)
        if fence:
            out.append(text[i : fence[1]])
            i = fence[1]
            continue
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        if ch == "`":
            run = re.match(r"`+", text[i:]).group(0)  # type: ignore[union-attr]  # text[i] is a backtick
            close = re.compile(f"(?<!`){re.escape(run)}(?!`)").search(
                text, i + len(run)
            )
            if close:
                out.append(text[i : close.end()])
                i = close.end()
            else:
                out.append(run)
                i += len(run)
            continue
        if ch == "<" and TAG_RE.match(text, i):
            end = island_end(text, i)
            line_start = text.rfind("\n", 0, i) + 1
            line_end = text.find("\n", end)
            line_end = n if line_end < 0 else line_end
            is_block = not text[line_start:i].strip() and not text[end:line_end].strip()
            island = text[i:end]
            if is_block:
                indent = text[line_start:i]
                island = "\n".join(
                    ln[len(indent) :] if ln.startswith(indent) else ln
                    for ln in island.split("\n")
                )
            page.islands.append(island)
            out.append(
                f"{PH_OPEN}{'B' if is_block else 'I'}{len(page.islands) - 1}{PH_CLOSE}"
            )
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Turn heading text into a stable, URL-safe id.

    This is the same rule the original conversion used, so existing anchors
    keep working.

    Parameters
    ----------
    text : str
        Heading text.

    Returns
    -------
    str
        Lower-case ASCII slug.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text) or "section"


def render_code(
    self: object, tokens: list[Token], idx: int, options: dict, env: dict
) -> str:
    """Render a code block as a bare ``<pre>`` with comments marked up.

    Parameters
    ----------
    self, tokens, idx, options, env
        The markdown-it renderer signature; rules are bound as methods.

    Returns
    -------
    str
        ``<pre>...</pre>`` followed by a newline.
    """
    code = tokens[idx].content
    if code.endswith("\n"):
        code = code[:-1]
    lines = []
    for line in code.split("\n"):
        escaped = html.escape(line, quote=False)
        head, sep, comment = escaped.partition("#")
        lines.append(
            f'{head}<span class="comment">{sep}{comment}</span>' if sep else head
        )
    return "<pre>" + "\n".join(lines) + "</pre>\n"


CONTAINER_RE = re.compile(r"^[A-Za-z][\w-]*(?: [A-Za-z][\w-]*)*$")


def render_container(
    self: object, tokens: list[Token], idx: int, options: dict, env: dict
) -> str:
    """Render ``::: class`` containers as ``<div class="...">``.

    Parameters
    ----------
    self, tokens, idx, options, env
        The markdown-it renderer signature; rules are bound as methods.

    Returns
    -------
    str
        The opening or closing div tag on its own line.
    """
    tok = tokens[idx]
    if tok.nesting == 1:
        return f'<div class="{tok.info.strip()}">\n'
    return "</div>\n"


def make_renderer() -> MarkdownIt:
    """Build the markdown-it instance used for every page.

    Returns
    -------
    MarkdownIt
        CommonMark parser with raw HTML disabled, ``:::`` containers, and
        custom code rendering.
    """
    md = MarkdownIt("commonmark", {"html": False})
    md.use(
        container_plugin,
        name="div",
        validate=lambda params, markup: bool(CONTAINER_RE.match(params.strip())),
        render=render_container,
    )
    md.add_render_rule("fence", render_code)
    md.add_render_rule("code_block", render_code)
    return md


HEADING_ID_RE = re.compile(r"\s*\{#([A-Za-z0-9_-]+)\}\s*$")


def assign_heading_ids(tokens: list[Token], sectioned: bool) -> list[tuple[str, str]]:
    """Give every top-level ``h2`` an id and strip ``{#id}`` from its text.

    Parameters
    ----------
    tokens : list[Token]
        Parsed token stream, modified in place.
    sectioned : bool
        Whether the page wraps headings in sections at all.

    Returns
    -------
    list[tuple[str, str]]
        ``(id, plain text)`` for each heading, in order; the page contents.
    """
    toc: list[tuple[str, str]] = []
    used: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok.type != "heading_open" or tok.tag != "h2" or tok.level != 0:
            continue
        inline = tokens[i + 1]
        children = inline.children or []
        explicit = None
        if children:
            last = children[-1]
            if last.type == "text":
                m = HEADING_ID_RE.search(last.content)
                if m:
                    explicit = m.group(1)
                    last.content = last.content[: m.start()]
                    inline.content = HEADING_ID_RE.sub("", inline.content)
        if not sectioned:
            continue
        title = "".join(
            c.content for c in children if c.type in ("text", "code_inline")
        )
        slug = explicit or slugify(title)
        base, n = slug, 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        tok.attrSet("id", slug)
        toc.append((slug, title))
    return toc


def sectionise(body_html: str) -> str:
    """Wrap the run of content beneath each ``<h2 id>`` in a ``<section>``.

    Parameters
    ----------
    body_html : str
        Rendered body in which every h2 carries an id attribute.

    Returns
    -------
    str
        The same HTML with section wrappers and the ids moved onto them.
    """
    parts = re.split(r'(?m)^<h2 id="([^"]+)">', body_html)
    if len(parts) == 1:
        return body_html
    out = [parts[0]]
    for j in range(1, len(parts), 2):
        out.append(
            f'<section id="{parts[j]}">\n<h2>{parts[j + 1].rstrip()}\n</section>\n'
        )
    return "".join(out)


def restore_islands(rendered: str, islands: list[str]) -> str:
    """Put the HTML islands back in place of their placeholders.

    Parameters
    ----------
    rendered : str
        HTML from the Markdown renderer.
    islands : list[str]
        Island sources, indexed by placeholder number.

    Returns
    -------
    str
        HTML with the islands restored.
    """
    block_para = re.compile(f"<p>((?:\\s*{PH_OPEN}B\\d+{PH_CLOSE}\\s*)+)</p>")
    rendered = block_para.sub(lambda m: m.group(1).strip(), rendered)
    return PH_RE.sub(lambda m: islands[int(m.group(2))], rendered)


def indent_html(body_html: str) -> str:
    """Indent block structure for readability without touching content.

    Lines inside ``<pre>`` are left exactly as they are. Lines inside
    ``<math>`` and ``<svg>`` are indented but do not affect the depth.

    Parameters
    ----------
    body_html : str
        HTML with one block tag per line, as the renderer emits it.

    Returns
    -------
    str
        Indented HTML.
    """
    out: list[str] = []
    depth = 0
    verbatim: str | None = None
    open_re = re.compile(r"<([A-Za-z][A-Za-z0-9]*)(?:[\s/>])")
    close_re = re.compile(r"</([A-Za-z][A-Za-z0-9]*)>")
    counted = BLOCK_TAGS - {"pre"}
    for raw in body_html.split("\n"):
        if verbatim == "pre":
            out.append(raw)
            if "</pre>" in raw:
                verbatim = None
            continue
        line = raw.strip()
        if verbatim:
            out.append("  " * depth + line if line else "")
            if f"</{verbatim}>" in raw:
                verbatim = None
            continue
        if not line:
            out.append("")
            continue
        opens = [t.lower() for t in open_re.findall(line)]
        closes = [t.lower() for t in close_re.findall(line)]
        n_open = sum(1 for t in opens if t in counted)
        n_close = sum(1 for t in closes if t in counted)
        level = max(depth - 1, 0) if line.startswith("</") and n_close else depth
        out.append("  " * level + line)
        depth = max(depth + n_open - n_close, 0)
        for tag in VERBATIM_TAGS:
            if re.search(f"<{tag}[\\s>]", line) and f"</{tag}>" not in line:
                verbatim = tag
    return "\n".join(out)


def render_body(
    page: Page, md: MarkdownIt, sectioned: bool
) -> tuple[str, list[tuple[str, str]]]:
    """Render a page body to HTML.

    Parameters
    ----------
    page : Page
        Source page.
    md : MarkdownIt
        Shared renderer.
    sectioned : bool
        Wrap each ``##`` heading in a section and build a contents list.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        The body HTML and the ``(id, title)`` contents entries.
    """
    text = protect_islands(page)
    tokens = md.parse(text)
    toc = assign_heading_ids(tokens, sectioned)
    rendered = md.renderer.render(tokens, md.options, {})
    if sectioned:
        rendered = sectionise(rendered)
    rendered = restore_islands(rendered, page.islands)
    return indent_html(rendered).strip("\n"), toc


# ---------------------------------------------------------------------------
# page templates
# ---------------------------------------------------------------------------

CHAPTER_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} &mdash; Stanford Machine Learning</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="topbar">
  <a href="/">holehouse.org</a>
  <a href="/blog/">Blog</a>
  <a href="/mlclass/" class="current" aria-current="page">Machine learning notes</a>
</header>

<main>
<h1>{title}</h1>

{nav}
{toc}{content}

{nav_bottom}
</main>
</body>
</html>
"""

INDEX_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="topbar">
  <a href="/">holehouse.org</a>
  <a href="/blog/">Blog</a>
  <a href="/mlclass/" class="current" aria-current="page">Machine learning notes</a>
</header>

<main>
<h1>{h1}</h1>

{content}

</main>
</body>
</html>
"""


def build_nav(prev: Page | None, nxt: Page | None, bottom: bool = False) -> str:
    """Build the previous / index / next navigation bar.

    Parameters
    ----------
    prev, nxt : Page or None
        Neighbouring chapters in reading order.
    bottom : bool
        Whether this is the copy at the foot of the page.

    Returns
    -------
    str
        The ``<nav>`` element.
    """
    cls = "chapter-nav bottom" if bottom else "chapter-nav"
    left = (
        f'<a href="{prev.stem}.html" rel="prev">&larr; {html.escape(prev.meta["nav_title"], quote=False)}</a>'
        if prev
        else "<span></span>"
    )
    right = (
        f'<a href="{nxt.stem}.html" rel="next">{html.escape(nxt.meta["nav_title"], quote=False)} &rarr;</a>'
        if nxt
        else "<span></span>"
    )
    return f'<nav class="{cls}">\n  {left}\n  <a href="index.html">Index</a>\n  {right}\n</nav>'


def build_toc(toc: list[tuple[str, str]]) -> str:
    """Build the "on this page" contents list.

    Parameters
    ----------
    toc : list[tuple[str, str]]
        ``(id, title)`` per section.

    Returns
    -------
    str
        The ``<nav class="toc">`` element, or an empty string for pages with
        fewer than two sections.
    """
    if len(toc) < 2:
        return ""
    items = "\n".join(
        f'    <li><a href="#{s}">{html.escape(t, quote=False)}</a></li>' for s, t in toc
    )
    return f'<nav class="toc" aria-label="On this page">\n  <h2>On this page</h2>\n  <ul>\n{items}\n  </ul>\n</nav>\n'


def build_chapter_list(chapters: list[Page]) -> str:
    """Build the index page's list of chapters from their front matter.

    Parameters
    ----------
    chapters : list[Page]
        Chapters in reading order.

    Returns
    -------
    str
        The ``<ul class="chapter-list">`` element.
    """
    items = []
    for ch in chapters:
        origin = ch.meta.get("origin", "2011")
        tag, cls = (
            ("2026 addition", "ch-tag ch-tag-new")
            if origin == "2026"
            else ("2011 course", "ch-tag")
        )
        title = html.escape(ch.meta.get("index_title", ch.meta["title"]), quote=False)
        desc = html.escape(ch.meta.get("description", ""), quote=False)
        items.append(
            f'  <li><a href="{ch.stem}.html">\n'
            f'    <span class="ch-head"><span class="ch-title">{title}</span> <span class="{cls}">{tag}</span></span>\n'
            f'    <span class="ch-desc">{desc}</span>\n'
            f"  </a></li>"
        )
    return '<ul class="chapter-list">\n' + "\n".join(items) + "\n</ul>"


def render_chapter(
    page: Page, prev: Page | None, nxt: Page | None, md: MarkdownIt
) -> str:
    """Render one chapter page.

    Parameters
    ----------
    page : Page
        The chapter source.
    prev, nxt : Page or None
        Neighbours in reading order.
    md : MarkdownIt
        Shared renderer.

    Returns
    -------
    str
        Complete HTML document.
    """
    for key in ("title", "nav_title"):
        if key not in page.meta:
            raise BuildError(f"{page.stem}.md: front matter needs '{key}'")
    content, toc = render_body(page, md, sectioned=True)
    return CHAPTER_PAGE.format(
        title=html.escape(page.meta["title"], quote=False),
        nav=build_nav(prev, nxt),
        toc=build_toc(toc),
        content=content,
        nav_bottom=build_nav(prev, nxt, bottom=True),
    )


def render_index(page: Page, chapters: list[Page], md: MarkdownIt) -> str:
    """Render the notes index page.

    Parameters
    ----------
    page : Page
        The ``index.md`` source.
    chapters : list[Page]
        Chapters in reading order, for the generated list.
    md : MarkdownIt
        Shared renderer.

    Returns
    -------
    str
        Complete HTML document.
    """
    if CHAPTER_LIST_MARKER not in page.body:
        raise BuildError(f"index.md: expected a {CHAPTER_LIST_MARKER} line")
    page.body = page.body.replace(CHAPTER_LIST_MARKER, build_chapter_list(chapters))
    content, _ = render_body(page, md, sectioned=False)
    return INDEX_PAGE.format(
        title=html.escape(page.meta["title"], quote=False),
        h1=html.escape(page.meta.get("h1", page.meta["title"]), quote=False),
        content=content,
    )


def build(out_dir: str) -> list[str]:
    """Render every source page into ``out_dir``.

    Parameters
    ----------
    out_dir : str
        Directory to write the HTML files into.

    Returns
    -------
    list[str]
        Paths written.
    """
    md = make_renderer()
    pages = load_pages()
    chapters = [p for p in pages if p.stem != "index"]
    index = next((p for p in pages if p.stem == "index"), None)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, ch in enumerate(chapters):
        prev = chapters[i - 1] if i > 0 else None
        nxt = chapters[i + 1] if i + 1 < len(chapters) else None
        path = os.path.join(out_dir, ch.stem + ".html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_chapter(ch, prev, nxt, md))
        written.append(path)
    if index is not None:
        path = os.path.join(out_dir, "index.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_index(index, chapters, md))
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list[str] or None
        Arguments; ``None`` means ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out", default=SITE, help="output directory (default: the site folder)"
    )
    args = ap.parse_args(argv)
    try:
        written = build(args.out)
    except BuildError as exc:
        print(f"build error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(os.path.relpath(path, SITE))
    print(f"{len(written)} pages written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
