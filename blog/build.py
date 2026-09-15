"""Render the blog: Markdown posts in ``src/`` become the HTML pages in this folder.

Each post is one Markdown file with a front-matter block::

    ---
    title: "Why disordered proteins are hard to think about"
    date: 2026-09-15
    author: Alex Holehouse
    tags: proteins, teaching
    summary: "One-line description shown on the index page and in the feed."
    ---

    Body in Markdown. Raw HTML is allowed.

``title`` and ``date`` are required. ``author`` defaults to ``DEFAULT_AUTHOR``,
``tags`` may be empty, ``summary`` defaults to the first paragraph, and
``draft: true`` keeps a post out of the build. The page name is the file name
with any leading ``YYYY-MM-DD-`` stripped, so ``2026-09-15-hello.md`` is
served at ``/blog/hello.html``; set ``slug`` to override.

The build writes every post page, ``index.html`` (newest first), one page per
tag under ``tags/``, and ``feed.xml``. It is deterministic: the same sources
always produce the same bytes.

Usage::

    python blog/build.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import html
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field

from markdown_it import MarkdownIt
from markdown_it.token import Token

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

SITE_URL = "https://holehouse.org"
SITE_NAME = "holehouse.org"
BLOG_TITLE = "holehouse.org blog"
BLOG_DESCRIPTION = "Occasional writing by Alex Holehouse."
DEFAULT_AUTHOR = "Alex Holehouse"


class BuildError(Exception):
    """A post cannot be rendered as written."""


@dataclass
class Post:
    """One post, parsed and rendered.

    Attributes
    ----------
    stem : str
        Source file name without extension.
    slug : str
        Output file name without extension.
    title, author, summary : str
        Front-matter text fields.
    date : datetime.date
        Publication date.
    tags : list[str]
        Tags as written, in order.
    body : str
        Rendered HTML of the post body.
    """

    stem: str
    slug: str
    title: str
    date: dt.date
    author: str
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    body: str = ""

    @property
    def url(self) -> str:
        """Site-relative URL of the post page."""
        return f"/blog/{self.slug}.html"


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def parse_front_matter(text: str, stem: str) -> tuple[dict[str, str], str]:
    """Split a post into its front matter and Markdown body.

    Parameters
    ----------
    text : str
        Full file contents.
    stem : str
        File stem, for error messages.

    Returns
    -------
    tuple[dict[str, str], str]
        Fields (values unquoted) and the body after the closing ``---``.
    """
    if not text.startswith("---\n"):
        raise BuildError(f"{stem}.md: missing front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise BuildError(f"{stem}.md: front matter is not closed")
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise BuildError(f"{stem}.md: bad front matter line {line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].replace('\\"', '"')
        meta[key.strip().lower()] = value
    return meta, text[end + 5 :]


def parse_tags(value: str) -> list[str]:
    """Split a ``tags`` value into tags.

    Accepts ``a, b, c`` or ``[a, b, c]``, with optional quotes.

    Parameters
    ----------
    value : str
        Raw front-matter value.

    Returns
    -------
    list[str]
        Tags, de-duplicated and in the order written.
    """
    value = value.strip().strip("[]")
    tags: list[str] = []
    for part in value.split(","):
        tag = part.strip().strip("\"'").strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def slugify(text: str) -> str:
    """Turn text into a lower-case ASCII slug.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    str
        Words joined by hyphens, or ``post`` if nothing survives.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text) or "post"


def load_posts(md: MarkdownIt) -> list[Post]:
    """Read and render every post in ``src/``.

    Parameters
    ----------
    md : MarkdownIt
        Shared renderer.

    Returns
    -------
    list[Post]
        Published posts, newest first.
    """
    posts: list[Post] = []
    seen: dict[str, str] = {}
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".md") or name == "README.md":
            continue
        stem = name[:-3]
        with open(os.path.join(SRC, name), encoding="utf-8") as fh:
            meta, body = parse_front_matter(fh.read(), stem)
        if meta.get("draft", "").lower() in ("true", "yes", "1"):
            continue
        for key in ("title", "date"):
            if not meta.get(key):
                raise BuildError(f"{stem}.md: front matter needs '{key}'")
        try:
            date = dt.date.fromisoformat(meta["date"])
        except ValueError as exc:
            raise BuildError(
                f"{stem}.md: date must be YYYY-MM-DD, got {meta['date']!r}"
            ) from exc
        slug = meta.get("slug") or re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)
        slug = slugify(slug)
        if slug in seen:
            raise BuildError(
                f"{stem}.md and {seen[slug]}.md both want the page name {slug!r}"
            )
        seen[slug] = stem
        body_html, first_paragraph = render_body(md, body)
        posts.append(
            Post(
                stem=stem,
                slug=slug,
                title=meta["title"],
                date=date,
                author=meta.get("author") or DEFAULT_AUTHOR,
                tags=parse_tags(meta.get("tags", "")),
                summary=meta.get("summary") or first_paragraph,
                body=body_html,
            )
        )
    posts.sort(key=lambda p: (p.date, p.stem), reverse=True)
    return posts


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def make_renderer() -> MarkdownIt:
    """Build the Markdown renderer.

    Returns
    -------
    MarkdownIt
        CommonMark plus tables and strikethrough, with raw HTML allowed.
    """
    md = MarkdownIt("commonmark", {"html": True})
    md.enable(["table", "strikethrough"])
    return md


def plain_text(tokens: list[Token] | None) -> str:
    """Flatten inline tokens to their text.

    Parameters
    ----------
    tokens : list[Token] or None
        Children of an inline token.

    Returns
    -------
    str
        Text content with markup removed.
    """
    out = []
    for tok in tokens or []:
        if tok.type in ("text", "code_inline"):
            out.append(tok.content)
        elif tok.type == "softbreak":
            out.append(" ")
        elif tok.children:
            out.append(plain_text(tok.children))
    return "".join(out)


def render_body(md: MarkdownIt, body: str) -> tuple[str, str]:
    """Render a post body, giving headings anchor ids.

    Parameters
    ----------
    md : MarkdownIt
        Shared renderer.
    body : str
        Markdown source.

    Returns
    -------
    tuple[str, str]
        The HTML and the plain text of the first paragraph.
    """
    tokens = md.parse(body)
    used: set[str] = set()
    first_paragraph = ""
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag in ("h2", "h3") and tok.level == 0:
            base = slug = slugify(plain_text(tokens[i + 1].children))
            n = 2
            while slug in used:
                slug, n = f"{base}-{n}", n + 1
            used.add(slug)
            tok.attrSet("id", slug)
        elif tok.type == "paragraph_open" and tok.level == 0 and not first_paragraph:
            first_paragraph = re.sub(
                r"\s+", " ", plain_text(tokens[i + 1].children)
            ).strip()
    return md.renderer.render(tokens, md.options, {}), first_paragraph


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="/style.css">
<link rel="alternate" type="application/rss+xml" title="{blog_title}" href="/blog/feed.xml">
</head>
<body>
<header class="topbar">
  <a href="/">holehouse.org</a>
  <a href="/blog/" class="current" aria-current="page">Blog</a>
  <a href="/mlclass/">Machine learning notes</a>
</header>

<main>
{content}
</main>
</body>
</html>
"""


def esc(text: str) -> str:
    """HTML-escape text for element content."""
    return html.escape(text, quote=False)


def long_date(date: dt.date) -> str:
    """Format a date as ``15 September 2026``."""
    return f"{date.day} {date.strftime('%B %Y')}"


def tag_links(tags: list[str]) -> str:
    """Render tags as links to their pages."""
    return "".join(
        f' <a class="tag" href="/blog/tags/{slugify(t)}.html">{esc(t)}</a>'
        for t in tags
    )


def page(title: str, content: str) -> str:
    """Wrap content in the site shell."""
    return PAGE.format(
        title=esc(title),
        blog_title=esc(BLOG_TITLE),
        site_name=esc(SITE_NAME),
        content=content,
    )


def post_page(post: Post) -> str:
    """The HTML page for one post."""
    meta = f"{long_date(post.date)} &middot; {esc(post.author)}" + (
        " &middot;" + tag_links(post.tags) if post.tags else ""
    )
    content = (
        f'<article class="post">\n<h1>{esc(post.title)}</h1>\n<p class="post-meta">{meta}</p>\n\n'
        f"{post.body.rstrip()}\n</article>\n\n"
        '<nav class="post-nav"><a href="/blog/">&larr; All posts</a></nav>'
    )
    return page(f"{post.title} — {SITE_NAME}", content)


def post_list(posts: list[Post]) -> str:
    """A list of posts, newest first."""
    if not posts:
        return '<p class="note">Nothing here yet.</p>'
    items = []
    for p in posts:
        meta = long_date(p.date) + (" &middot;" + tag_links(p.tags) if p.tags else "")
        summary = (
            f'\n    <p class="post-summary">{esc(p.summary)}</p>' if p.summary else ""
        )
        items.append(
            f'  <li>\n    <a class="post-title" href="{p.url}">{esc(p.title)}</a>\n'
            f'    <span class="post-meta">{meta}</span>{summary}\n  </li>'
        )
    return '<ul class="post-list">\n' + "\n".join(items) + "\n</ul>"


def index_page(posts: list[Post]) -> str:
    """The blog front page."""
    all_tags = sorted({t for p in posts for t in p.tags}, key=str.lower)
    tags = (
        f'<p class="post-meta">Tags:{tag_links(all_tags)}</p>\n\n' if all_tags else ""
    )
    return page(BLOG_TITLE, f"<h1>Blog</h1>\n\n{tags}{post_list(posts)}")


def tag_page(tag: str, posts: list[Post]) -> str:
    """The page listing every post with one tag."""
    content = (
        f"<h1>Posts tagged &ldquo;{esc(tag)}&rdquo;</h1>\n\n{post_list(posts)}\n\n"
        '<nav class="post-nav"><a href="/blog/">&larr; All posts</a></nav>'
    )
    return page(f"{tag} — {BLOG_TITLE}", content)


def feed(posts: list[Post]) -> str:
    """An RSS 2.0 feed of every post."""
    items = []
    for p in posts:
        when = dt.datetime.combine(p.date, dt.time(), tzinfo=dt.timezone.utc)
        items.append(
            "  <item>\n"
            f"    <title>{html.escape(p.title)}</title>\n"
            f"    <link>{SITE_URL}{p.url}</link>\n"
            f'    <guid isPermaLink="true">{SITE_URL}{p.url}</guid>\n'
            f"    <pubDate>{email.utils.format_datetime(when)}</pubDate>\n"
            f"    <author>{html.escape(p.author)}</author>\n"
            f"    <description>{html.escape(p.summary)}</description>\n"
            "  </item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0">\n<channel>\n'
        f"  <title>{html.escape(BLOG_TITLE)}</title>\n  <link>{SITE_URL}/blog/</link>\n"
        f"  <description>{html.escape(BLOG_DESCRIPTION)}</description>\n"
        + "\n".join(items)
        + "\n</channel>\n</rss>\n"
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def write(path: str, text: str) -> None:
    """Write a file, creating its directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def build(out_dir: str) -> list[str]:
    """Render the whole blog into ``out_dir``.

    Parameters
    ----------
    out_dir : str
        Destination folder.

    Returns
    -------
    list[str]
        Paths written.
    """
    posts = load_posts(make_renderer())
    tags_dir = os.path.join(out_dir, "tags")
    if os.path.isdir(tags_dir):
        for name in os.listdir(tags_dir):
            if name.endswith(".html"):
                os.remove(os.path.join(tags_dir, name))
    written = []
    for p in posts:
        path = os.path.join(out_dir, f"{p.slug}.html")
        write(path, post_page(p))
        written.append(path)
    by_tag: dict[str, list[Post]] = {}
    for p in posts:
        for t in p.tags:
            by_tag.setdefault(t, []).append(p)
    for t, tagged in sorted(by_tag.items(), key=lambda kv: kv[0].lower()):
        path = os.path.join(tags_dir, f"{slugify(t)}.html")
        write(path, tag_page(t, tagged))
        written.append(path)
    write(os.path.join(out_dir, "index.html"), index_page(posts))
    write(os.path.join(out_dir, "feed.xml"), feed(posts))
    written += [os.path.join(out_dir, "index.html"), os.path.join(out_dir, "feed.xml")]
    generated = {os.path.abspath(w) for w in written}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.abspath(os.path.join(out_dir, name))
        if name.endswith(".html") and path not in generated:
            print(
                f"note: {name} is not produced by any post in src/ (renamed or deleted?)"
            )
    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(
        description="Render the blog from the Markdown posts in src/."
    )
    ap.add_argument(
        "--out", default=HERE, help="output directory (default: the blog folder)"
    )
    args = ap.parse_args(argv)
    try:
        written = build(args.out)
    except BuildError as exc:
        print(f"build error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(os.path.relpath(path, HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
