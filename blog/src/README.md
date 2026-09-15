# Blog posts

Each post is one Markdown file in this folder. Run the build from the repo root and commit the sources together with the generated pages:

```
python blog/build.py        # or: python build.py, which also rebuilds the notes
```

## Writing a post

Name the file `YYYY-MM-DD-short-name.md`. The date prefix keeps the folder in order; the rest becomes the page name, so `2026-09-15-hello-world.md` is published at `/blog/hello-world.html`.

Start the file with a front-matter block:

```
---
title: "Why disordered proteins are hard to think about"
date: 2026-09-15
author: Alex Holehouse
tags: proteins, teaching
summary: "One or two sentences shown on the index page and in the feed."
---
```

| Field | Required | Notes |
|---|---|---|
| `title` | yes | Shown as the page heading and in the browser tab. |
| `date` | yes | `YYYY-MM-DD`. Posts are listed newest first. |
| `author` | no | Defaults to Alex Holehouse. |
| `tags` | no | Comma-separated. Each tag gets its own page under `/blog/tags/`. |
| `summary` | no | Defaults to the first paragraph of the post. |
| `slug` | no | Overrides the page name derived from the file name. |
| `draft` | no | `true` keeps the post out of the build entirely. |

Below the front matter, write ordinary Markdown: headings, lists, links, images, fenced code blocks, tables. Raw HTML is allowed too, so an equation in MathML or an embedded figure can go straight in. `##` and `###` headings get anchor ids from their text.

Images and other files go in this folder's parent, `blog/`, next to the generated pages, and are referenced as `/blog/figure.png`.

## What the build produces

- `blog/<name>.html` for each post
- `blog/index.html`, every post newest first, with the full tag list at the top
- `blog/tags/<tag>.html`, one per tag
- `blog/feed.xml`, an RSS feed

The build is deterministic and prints a note if a page in `blog/` no longer corresponds to any post, which happens when a post is renamed or deleted. Delete that page by hand.
