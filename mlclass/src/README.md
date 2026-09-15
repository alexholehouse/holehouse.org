# About 

To make updating and maintaining the mlclass notes easier, we converted the HTML to source Markdown, and then we can regenerate that HTML from that source.

Every chapter page and the notes index are generated from the Markdown files in this folder. The HTML in the parent folder is output; edit the Markdown here, rebuild, and commit both.

```
python build/build.py                 # from the mlclass/ folder: regenerate every page
python build/check.py OLD_DIR .       # compare two renderings, ignoring whitespace
```

The build is deterministic. Running it twice on the same sources produces byte-identical HTML, so a `git diff` after a rebuild shows only the effect of your edit.

## Front matter

Each chapter starts with a block like this:

```
---
title: "06: Logistic Regression"
nav_title: "06: Logistic regression"
origin: "2011"
description: "Classification with the sigmoid: decision boundaries, ..."
---
```

| Field | Used for |
|---|---|
| `title` | The page heading and browser title. |
| `nav_title` | The label shown when a neighboring chapter links to this one. |
| `index_title` | Optional. The title shown on the index page, when it differs from `title`. |
| `origin` | `2011` for the original course notes, `2026` for chapters written for this edition. Sets the tag on the index page. |
| `description` | The one-line description under the title on the index page. |

Chapters are ordered by file name, and that order drives the previous/next links and the index list. To add a chapter, add a file whose name sorts into the right place and give it front matter. Nothing else needs editing.

`index.md` holds the index page's prose. The line `<!-- chapter-list -->` is replaced by the generated list of chapters.

## The Markdown dialect

The body is CommonMark. Headings, bullet lists, bold, italic, inline code, links and fenced code blocks are ordinary Markdown. Four conventions sit on top.

**HTML islands.** Any HTML element written into the source is passed through to the page exactly as written. Equations (MathML), figures, tables, `<sub>`, `<sup>`, and anything with a `class` attribute are all islands. Nothing inside an island is treated as Markdown, so you never need to escape anything within one. An island that starts a line and ends it stands on its own as a block; anything else is inline. Inside a list item, indent every line of a multi-line island to the item's content column, the same as any other continuation line.

**Section ids.** Every `##` heading opens a `<section>` whose id is the heading text slugified: lower case, ASCII, words joined by hyphens. Links from other chapters point at these ids, so if you reword a heading whose id is linked to, keep the old id by appending it to the heading: `## New wording {#old-id}`.

**Code blocks.** A fenced block renders as a bare `<pre>`. Everything from a `#` to the end of a line is marked up as a comment, so keep `#` out of code that is not a comment or write that block as an HTML island instead. Fences inside list items are indented to the item's content column; blank lines inside them are fine.

**Containers.** A block between `::: some-class` and `:::` lines becomes `<div class="some-class">`. The notes use `::: code-eg` for a code example followed by its explanatory `<p class="eqn-note">`.

## Things to watch

- Lists in these notes are *tight*: no blank lines between items or between the parts of an item. A blank line inside a list makes Markdown wrap every item in `<p>`, which changes the layout.
- At section level, separate blocks with a blank line. Without one, a block that follows a list is read as part of the list's last item.
- A backslash escapes Markdown punctuation in prose: `\*`, `\_`, `\[`, `\<`. The converter that produced these files added them wherever a character would otherwise have been interpreted.
- Two lists stay as HTML islands because Markdown cannot express their structure: one in chapter 04 has figures placed between list items, and one in chapter 07 has text after a nested list inside the same item. They can be edited as HTML in place.

## Where these files came from

`build/html2md.py` produced the first version of every file here from the hand-built HTML in September 2026. `build/check.py` then confirmed that rebuilding from the Markdown reproduces every page: the same title and the same `<main>` tree once whitespace a browser ignores is normalised. The scripts that built the HTML from the original Evernote export before that are kept in `build/legacy/`.
