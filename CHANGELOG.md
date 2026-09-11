# V3 changelog

V3 is a rebuild of the V1 machine learning course notes. The source for V1 was an automatic Evernote HTML export, and the markup showed it. This version regenerates every page as clean, semantic HTML while keeping the notes themselves — the same terse bullet style, the same asides, the same diagrams.

The work came in two phases, and they were verified differently.

**The rebuild** was mechanical: the rendered text of every V3 page came out character-for-character identical to the corresponding V1 page, apart from the 37 content fixes listed under "Content fixes" below. Nothing was rewritten, summarised or dropped. A later **second review pass** (see its own section) read every chapter end to end and corrected roughly 280 further typos and errors, including some that change technical content; each of those is itemised there.

**The conversion of screenshots to real maths** came afterwards and did change what is on the page: 378 images of equations, tables and algorithms became MathML, HTML tables, code blocks and SVG diagrams, leaving 174 genuinely pictorial figures. Each converted figure was transcribed by hand, rendered, and read back; where the arithmetic could be checked independently it was checked with numpy. See "Conversion summary" for the totals and the errors this turned up.

## What changed structurally

### Markup

- **Removed the Evernote damage.** The source carried 429 `<font>` tags, 10,537 `<span>`s (many of them long runs of WebKit paste junk carrying 20+ dead CSS properties), `<basefont>` elements, and hundreds of unclosed `</font>` tags. All gone.
- **No inline styling anywhere.** Every page now has zero `style` attributes; all presentation lives in `style.css`.
- **Real headings.** Section titles were `<div style="font-size: 27px">` with a bold/underlined span inside. They are now `<h2>`, each wrapped in a `<section>` with a stable `id`. Sub-topic labels set in bold on their own line are now `<h3>` rather than a stray `<strong>` floating between two lists.
- **Properly nested lists.** This was the biggest structural problem. Evernote encoded a sub-list as a *sibling* list item styled `display: inline; list-style: none` rather than nesting it inside the item it belonged to. 267 of these were unpicked and the sub-lists moved into their parent item, so the outline structure is now real rather than merely visual. A further 14 places where a `<ul>` sat directly inside another `<ul>` (invalid HTML) were repaired.
- **Figures.** All 383 diagrams were put into `<figure>` elements, with lazy loading and the presentational `width`/`height`/`cursor` attributes stripped. Most were later replaced by real markup — see "Conversion summary" — leaving 174 that are genuinely pictures.
- **Code blocks.** Octave snippets were identified by their Courier New face and are now `<pre>` blocks, with inline references to variables and functions as `<code>`. Comments within code keep a distinct style. Lines of ordinary prose that happened to be set in Courier were left as prose.
- **Valid HTML5.** Every page passes a strict HTML5 parse. V1 had, among other things, a stray `</body></html>` in the middle of each chapter's navigation paragraph, and self-closing `<a name="..."/>` anchors that make a browser swallow the rest of the document.
- **Removed the Google AdSense scripts**, which pointed at `http://` URLs and no longer load.

### Navigation and reading

- Each chapter has an **"on this page" contents list** linking to its sections.
- **Previous / Index / Next** navigation at the top and bottom of every chapter, with the neighbouring chapter named rather than just "Previous"/"Next".
- Pages are **responsive**, **dark-mode aware**, and **print-friendly**.
- The **top bar stays pinned** to the top of the window as you scroll, so the index link is always one click away in these long chapters. It is `position: sticky` rather than `fixed`, so it stays in the document flow and nothing underneath needs compensating padding. Following a contents link would otherwise scroll the section heading underneath the bar, so section anchors carry a matching `scroll-margin-top` — every anchor now lands about 13px clear of it.
- Nested list levels get distinct markers so the four- and five-level-deep outlines stay readable.
- The author's colour coding is preserved as semantic classes rather than hard-coded colours: `.term` for the dark blue used to introduce a key term, plus red/green/purple highlight classes.

### Missing content restored

- **Chapter 10, "Advice for applying machine learning techniques", was missing entirely from V1.** The index linked to it but the file did not exist in the folder. It has been restored.
- **167 of the 383 diagrams were missing from V1's image folders.** Two folders — for chapters 01/02 and 04 — were completely empty, and chapters 17 and 18 had no image folder at all, so 43 diagrams in the opening chapters and every diagram in the last two chapters were broken. All were restored from the published copy of the notes. Five images in chapter 18 could not be: they are absent from the published copy too, and those figures have been removed. Every image reference that remains resolves locally.
- **Image folder names now match the `src` attributes exactly.** V1 referenced `08_Neural_Networks_Representation_files/` while the folder on disk was `08_neural_networks_representation_files/`. This works on macOS, which is case-insensitive, but breaks on any case-sensitive web server.
- `resume.css`, an unrelated CV template that V1 carried around, has been dropped.
- **The `.rar` and `.zip` download archives have been fetched** and sit alongside the notes. Both are the originals from April 2015 and contain the **V1** content — 421 files, the same 383 diagrams, and the old Evernote markup — so they are *not* a package of this rewrite. The index deliberately does not link them; see "Archives" below.

### Archives

`Machine_learning_complete.rar` (21.0 MB) and `Machine_learning_complete.zip` (21.1 MB) were downloaded from the published site to complete the asset set. Verified: both are valid archives at their full expected size, and the image set inside is identical to V3's — nothing in them was missing from this folder.

They are stale by design: they package the pre-rewrite pages. If you want a download link on the index again, the archives should be regenerated from the V3 files first.

## Lectures 1 and 2 split into separate chapters

The Evernote export ran lectures 1 and 2 together on a single page, `01_02_Introduction_regression_analysis_and_gr.html`. They divide cleanly at a real boundary, so they are now two chapters:

- **`01_Introduction.html`** — Introduction to the course, What is machine learning?, Supervised learning, Unsupervised learning (4 sections, 6 diagrams)
- **`02_Linear_Regression_with_One_Variable.html`** — Linear Regression, the cost function, a deeper look at it, gradient descent, and linear regression with gradient descent (5 sections, 6 diagrams, 13 equation blocks, 1 table)

This matches the actual lecture boundary in the course: lecture 1 is the introduction and the supervised/unsupervised distinction; lecture 2 is linear regression with one variable.

Each chapter has its own "on this page" contents list, its own previous/next navigation, and its own image directory (`01_Introduction_files/` and `02_Linear_Regression_with_One_Variable_files/`) matching the convention every other chapter follows. The index lists them separately, and chapter 03's previous-link now points at chapter 02.

All nine sections are present exactly once and in their original order; no text was edited in the split. The combined page has been removed so there is only one source of truth.

**One section was written for chapter 02**: a short "What this chapter covers" opener. Standing alone, the chapter previously began with "Housing price data example used earlier", which no longer had an antecedent on its own page. The new section says what the chapter is for — the first actual algorithm — and names the three pieces (hypothesis, cost function, gradient descent) plus the point that the hypothesis → cost → minimize pattern recurs for every algorithm that follows. It is the only prose added to any original chapter, it is written in the same terse bullet style as the rest, and it introduces no new technical content: everything in it is stated properly later in the chapter.

## Real maths instead of screenshots (every chapter with maths in it)

In the original notes every equation is a screenshot — a scan of a slide or a photo of handwriting. Those images don't scale, don't reflow on a phone, can't be selected or searched, and are invisible to a screen reader.

Chapter 02 now renders its equations as **native MathML**. No JavaScript, no library, no external assets — every current browser renders it directly, so the pages stay self-contained and work offline exactly as before.

**14 screenshots replaced**, and the equations now scale with the text, reflow, select, search, and read aloud:

| Was | Now |
| --- | --- |
| `Image [6].png` | The training-set table, as a real `<table>` (m = 47) |
| `Image [7].png` | h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x |
| `Image [8].png` | The summed squared error |
| `Image [9].png` | The cost function J(θ<sub>0</sub>,θ<sub>1</sub>) |
| `Image [11].png` | minimize over θ<sub>0</sub>,θ<sub>1</sub> of J |
| `Image [12].png` | The simplified hypothesis, its cost function, and its minimization |
| `Image [17].png` | The gradient descent update rule |
| `Image [18].png` | The derivative term |
| `Image [19].png` | The simultaneous-update algorithm (temp0 / temp1) |
| `Image [20].png` | The derivative term for the simplified cost |
| `Image [21].png` | The partial derivative, expanded |
| `Image [22].png` | The j = 0 and j = 1 cases |
| `Image [24].png` | The design matrix X |
| `Image [25].png` | The target vector y |

### Chapter 03 — every figure converted

Chapter 03 is linear algebra review, and every one of its **14 figures was a matrix or an equation** — no plots, no genuine diagrams. All 14 are now MathML, so the chapter contains **no images at all**: matrices, indexing (A<sub>11</sub>, A<sub>12</sub>, …), vectors, addition, scalar multiplication, matrix-by-vector and matrix-by-matrix multiplication, the house-price worked examples, identity matrices, the inverse, and the transpose.

Several were annotated whiteboard photographs where the annotation — coloured circles, arrows — only pointed at elements the written maths already named. Those became the matrix plus the equations that were written beside it, which states the same thing without needing the arrows.

**All the arithmetic was verified independently** against NumPy rather than trusted from the transcription: every sum, product, inverse and transpose in the chapter recomputes correctly. One thing this surfaced is that the three-hypotheses example rounds its predictions to whole numbers on the original slide (486, 410, 692 where the exact values are 486, 410.4, 691.6). The rounded figures are kept as the author had them, with a note saying so, so a reader who recomputes doesn't think it's an error.

### Chapter 18 — 3 of 17 figures converted, and 5 empty figures removed

The application chapter, and almost entirely photographs: street scenes with the text boxed, pedestrians boxed, patches of characters, real training data against synthetic. Fourteen of the seventeen real images are photographs and stay exactly as they are — there is nothing to convert them into.

The three that are not photographs are all flowcharts, and those are now inline SVG in the same style as the diagrams drawn for chapters 20 and 21: the OCR pipeline, the same pipeline marked up for ceiling analysis, and the face recognition pipeline. They take their colours from the page theme, scale with the text, and carry a `<title>` and `<desc>` so a screen reader can describe them.

#### Five figures whose images never existed

This chapter carried five `<figure>` elements whose `<img>` had **no `src` at all** — invisible holes in the layout. They are not damage from this rewrite: the same five empty images are in V1, and in the published copy of the notes on holehouse.org, which was checked directly. The images were lost before publication and cannot be recovered from anywhere, so the empty figures have been removed rather than left as gaps.

Finding them also exposed a gap in this rewrite's own checking. The link validator tested every image that *had* a `src`, so an image with no `src` slipped past it silently. The check now covers both, and the whole set was re-tested: 174 images across 23 pages, every one of them resolving.

## Conversion summary

Across the eighteen content chapters, **378 screenshots became 174**. Every equation, table, algorithm and flowchart that could sensibly become markup now is; what remains is genuinely pictorial — scatter plots, decision boundaries, network diagrams, distributions, and photographs.

Three chapters ended with no screenshots at all: **03** (linear algebra), **16** (recommender systems) and **19** (course summary, which never had any). The chapters that kept the most are the ones that argue visually — **12** kept 34 vector and margin diagrams, **15** kept 23 distribution plots.

Source errors were found and corrected in **eight of the eighteen chapters** — 04, 07, 09, 10, 12, 14, 16 and 17 — nearly all of them caught by transcribing a figure next to its neighbours and noticing the two disagreed. Chapter 15 was checked just as closely and had none, which is worth recording alongside the rest.

The recurring one is the logistic cost function, which needed attention in five separate chapters. That it recurs in the same form each time is a strong sign it propagated through the source slides rather than being mistyped once.

Where a slide and its neighbours disagreed, the deciding evidence was always something independent of the figures: the arithmetic the prose performs on a table, a matrix shape, or a mean stated three slides later. Those checks are recorded in each chapter's section above.

### Chapter 17 — 19 of 33 figures converted

Fourteen figures stay, and they are all genuinely pictures: the learning curves, the contour plots that show batch gradient descent walking smoothly into the minimum while the stochastic version wanders around it, the convergence plots at different averaging windows, and the map-reduce diagram of a training set split across four machines. The nineteen converted are the updates — batch, stochastic, mini-batch and online — the cost functions behind them, and the map-reduce partial sums.

#### Two figures needed more than transcription

- **The logistic training cost is written without brackets.** Read literally, `−1/m Σ y log h − (1−y) log(1−h)` puts the second logarithm outside both the sum and the −1/m, which is not the cost function. Bracketed here. This is the fifth chapter in which this same expression has needed attention, in one form or another.
- **One figure is a cropped fragment.** The slide crops the term `(h(x⁽ⁱ⁾) − y⁽ⁱ⁾)x_j⁽ⁱ⁾` out of the line above it and the crop loses the opening bracket, leaving an unmatched closing one. Balanced here. The surrounding prose ("The term … is the same as that found in the summation for batch gradient descent") makes clear what it is meant to be.

### Chapter 16 — all 32 figures converted

The most convertible chapter in the set, and the first with **no screenshots left at all**. Of its 32 figures, 31 are tables, matrices or equations; the only one that is a picture in any sense is a decorative star-rating scale sitting beside the first table, which is dropped rather than reproduced.

#### One error, unusually well evidenced

Four figures print the movie rating table with a "?" for Dave's rating of *Swords vs. karate*. The Y matrix that encodes that exact table gives the same cell as **0** — and figure [25] gets both wrong together, printing the table with "?" and the matrix with 0 side by side *in a single image*.

The mean normalisation later in the chapter settles which is right. μ₅ is stated as 1.25, which is the mean of (0, 0, 5, 0). Were the cell unrated, μ₅ would be the mean of (0, 0, 5) = 1.667, and the normalised row would carry a "?" where the figure prints −1.25. The tables are corrected to 0, with a note.

Every number in the mean-normalisation worked example was checked against numpy before transcription: the five row means, all twenty entries of the normalised matrix, and which cells stay unrated. All matched.

#### A bug worth recording

The negative entries in the normalised matrix first rendered as −.5, −.25 and a bare minus sign, instead of −2.5, −2.25 and −2. The cause was `str.lstrip("-" + MINUS)` used to take a magnitude: `lstrip` takes a *set* of characters, and `MINUS` is the entity string `&#x2212;`, so the call stripped any leading `-`, `&`, `#`, `x`, `2`, `1` or `;` — eating the digits as well as the sign. It is a silent corruption that produces plausible-looking numbers, and only rendering the page caught it. The magnitude now comes from a slice, and the finished matrix is asserted against the verified values.

### Chapter 15 — 13 of 36 figures converted

The largest chapter after the SVMs, and the one where keeping images is most clearly right. This chapter argues by *showing distributions*: the 3D bumps, the contour plots, the scatter-plus-marginals panels that demonstrate why two independent Gaussians cannot capture a correlated cloud. Those 23 figures are the explanation, not decoration around it. The thirteen that became markup are the Gaussian itself, the parameter estimates, both forms of the anomaly detection algorithm, and the multivariate Gaussian with its covariance matrix.

**No errors found.** Worth stating plainly, since most chapters so far have had one. The two figures that state the multivariate Gaussian twice — one handwritten, one typeset — agree with each other exactly, and so do the two statements of the algorithm. Every index and limit checks out.

Two figures were more than equations. The **three-step algorithm** mixes numbered prose with displayed maths, and the **diagonal-covariance figure** is an equation followed by a hand-drawn matrix with zeros sketched into the off-diagonal corners; that one is now a proper matrix with σ²₁ … σ²ₙ down the diagonal, which makes the chapter's actual point — that the original model *is* the multivariate Gaussian with the off-diagonal entries forced to zero — visible rather than implied by a sketch.

The `|Σ|` figure gained a note. It is the determinant of the covariance matrix, not an absolute value, and at that size the notation is easy to misread.

### Chapter 14 — 9 of 21 figures converted, 1 supplemented

Twelve figures are plots — 2D scatters with a projection line, 3D point clouds, the plane-through-a-cube picture — and those stay. The nine that became markup are an unusually mixed set for one chapter: two data tables, one line of Octave, five equations and a block of pseudocode. A thirteenth figure, the projection diagram, keeps its plot and gains the equation `z = U_reduce^T x` that was written across it.

#### One error

Figure [12] gives the covariance matrix as a sum running to **n**, the number of features, where it must run to **m**, the number of training examples. The `1/m` in front of it already says as much, and the chapter's own prose says it twice over: Σ "is an [n x n] matrix", built from x⁽ⁱ⁾ which is "[n x 1]". What makes this one easy to miss is that the shape doesn't give it away — summing n outer products of n-vectors is still [n×n]. Only the averaging is wrong. Corrected, with a note.

#### Two layout fixes this chapter forced

- **The country feature table has eight columns** and cannot shrink below the width its content needs, so on a phone it pushed the whole page sideways by 243px. Wide tables now sit in their own scroll container: the table scrolls, the page doesn't.
- **Unit qualifiers in table headers** ("trillions of US$", "Gini as percentage") inherited the header's bold and competed with the column names. They are now set lighter and smaller beneath the name.

### Piecewise blocks realigned (chapters 06 and 10)

Chrome's MathML Core honours neither `columnalign` on `<mtd>` nor a CSS `text-align` on it, so every row of a column centres itself over the widest row. For a matrix that is the right behaviour and nobody notices. For a piecewise definition it is wrong, and how wrong depends entirely on how much the rows differ in width — which is why it went unnoticed for so long.

An audit of every multi-row `<mtable>` in the set, measuring where each row actually starts, turned up two real cases:

- **Chapter 10's `err()`** — a long condition above the single word "otherwise" — was **130px out of line**. This is the one previously recorded above as fixed; it was not.
- **Chapter 06's `Cost()`**, which appears twice, was about 14px out.

Both are now laid out with a CSS grid instead: the maths inside each cell is still MathML, only the two-dimensional arrangement moved. They align the way they always should have.

The audit's remaining hits were checked and left alone — they are matrices, vectors and a two-line `min` under-script, where centring is conventional typesetting rather than a defect.

### Chapter 13 — 5 of 14 figures converted

Nine figures carry the chapter's actual argument — what a cluster looks like, what a local optimum looks like, where the elbow falls — and those stay. Five become markup: the K-means algorithm, the assignment step, the distortion cost function, the objective it minimises, and the random-restart loop.

**The two most important figures in the chapter are pseudocode rather than equations**, and they went through MathML rather than a `<pre>` block. A `<pre>` was the obvious first choice, matching the Octave snippets elsewhere in these notes, but the pseudocode mixes prose with subscripted symbols and one of them is μ*K* — Unicode has a subscript small k but no subscript capital K, so a plain-text rendering would have had to write it as `mu_K` and break the run of real symbols.

That created its own problem: an `<mtable>` centred every line of the pseudocode over the longest one, which for indented code is unreadable. The fix is one block-level `<math>` element per line — a block box starts at the left margin, so alignment is real — with a leading `<mspace>` doing the indenting. This is the same Chrome limitation described above, met from a different direction.

One note was added rather than transcribed. The assignment-step figure shows `min` over k with an arrow drawn to c⁽ⁱ⁾; what actually gets stored is not the distance but the k attaining it, so the operation is strictly an argmin. The equation is rendered as the slide has it, with a note saying so.

### Chapter 12 — 20 of 54 figures converted

The largest chapter in the set, and the split is lopsided in a way that reflects the material rather than the effort available. The large-margin sections argue geometrically — vector diagrams, projections, scatter plots with candidate boundaries drawn on — and those are pictures in the proper sense. 34 of them stay exactly as they are. The other 20 are pure equations and are now MathML: the logistic hypothesis and cost, the derivation of the SVM cost from it, the large-margin optimisation problem, the norm derivation, the Gaussian kernel, and the kernelised cost function.

#### Three errors, all found by transcribing figures next to their neighbours

- **Two sums indexed over the wrong variable.** Figures [10] and [11] write the regularisation penalty as a sum over `i = 1 … n` while the summand is θ*j* — the index and the summand disagree, so as printed the expression is meaningless. Figure [9], two slides earlier, has the same sum indexed correctly over *j*. Both corrected.
- **A constraint with the wrong condition.** Figure [31] gives the second constraint as `if y⁽ⁱ⁾ = 1`, which is the case the *first* constraint already covers — so as written the problem says nothing about negative examples at all. It must be `y⁽ⁱ⁾ = 0`, as the two equivalent figures [13] and [22] both have it. Corrected.

One further figure, [51], is not wrong but is ambiguous: it drops the square brackets that [11] puts around the two cost terms, leaving it unclear how far the sum extends. The brackets are restored so the kernelised cost matches the form the chapter established before kernels were introduced.

#### Two MathML lessons worth recording

Both were caught by rendering rather than by reading the markup, and both were silent:

- **A bare `&#x2212;` dropped into an `<mrow>` is discarded by the renderer.** The negative exponent vanished from the sigmoid, leaving `e^θᵀx` — a different and wrong function — with no error anywhere. A minus sign has to be an `<mo>` element, and `form="prefix"` is what makes it unary rather than binary.
- **Parentheses are stretchy by default**, so a pair sharing a row with a tall fraction grows to match it. `h(x)` ended up with parentheses twice the height of the x inside them. Non-stretchy parentheses are now the default in this chapter, with tall ones used only where a `log` genuinely wraps a fraction.

Figure [25] was also reworked. It renders the same expression as [24]; the difference on the slide is a red annotation identifying the square root as ‖θ‖. Reproducing the equation twice would have said nothing, so it now shows that identification as its own step, which is what makes the derivation from a sum of squares to a norm follow.

### Chapter 11 — 3 of 5 figures converted, plus the F₁ score

The smallest chapter so far, and it splits cleanly: three figures carry text or numbers, two are genuine plots (the precision/recall trade-off curve and the accuracy-against-training-set-size study) and stay as images.

- **The spam example** — two emails set against each other — is now real text in a two-column layout rather than a photograph of a slide. The deliberate misspellings are the entire point of the example, so they are transcribed exactly as the slide has them: `w4tchs`, `Med1cine`, `M0rgages`. The slide underlines the first two; the third it leaves unmarked, so a note points it out. Substituting digits for letters is precisely how spam evades a filter matching on exact words, which is why "misspelled word" is proposed as a feature on the very next slide — that connection was invisible while the example was an image.
- **The bag-of-words feature vector** is now a MathML column with its word labels alongside, as the handwriting has them.
- **The three-algorithm precision/recall comparison** is a real table. Its numbers were checked against the arithmetic the prose immediately performs on them: the stated averages of 0.45, 0.4 and 0.51 come out exactly, the average does pick algorithm 3 (which is the chapter's complaint about averaging), and F₁ picks algorithm 1 (which is the chapter's point in introducing it).

**The F₁ score was converted too**, though it was never a screenshot — it sat in the prose as the ASCII string `= 2 * (PR/ [P + R])`. It is the one real formula in the chapter, and leaving it as ASCII while every other chapter renders its maths properly would have been inconsistent. It is now written in the conventional single-fraction form, 2PR/(P+R), which is the same quantity.

#### A layout bug this chapter exposed

The side-by-side email example is a CSS grid, and grid items default to `min-width: auto` — so the long `From:` address inside the `<pre>` forced its track wider than a phone screen and scrolled the whole page sideways, cutting off the heading and the contents box. Fixed with `min-width: 0` on the grid children, which lets the existing `pre-wrap` do its job. All 23 pages were then measured at 480px: none overflows horizontally.

### Chapter 10 — 10 of 16 figures converted

Converted: the regularised cost; the 70:30 train/test split, now a real `<table>` with a column marking which set each row belongs to; the test error in both squared-error and logistic form; the misclassification error `err()` as a piecewise definition; the test error built from it; the ten candidate polynomial models; the training, cross-validation and test errors side by side; the regularised model with its cost; and the training error on its own.

Kept as images: the bias/variance fit plots, the λ-sweep plots, the error-against-polynomial-degree diagnostics, and the two learning curves.

#### Three more source errors

- **A sum to the wrong limit, twice.** Two slides write the regularisation penalty as a sum to **m**, the number of training examples, where it must run to **n**, the number of features. The first sum in the same expression already runs to m over examples, so as printed the two indices collide. Both corrected — and cross-checked afterwards, so chapter 10's penalty terms now agree with chapter 07's.
- **The logistic `log(1 − h)` slip again**, this time in the logistic test error. That makes three chapters carrying the same mistake (07, 09 and 10), which suggests it propagated through the source slides rather than being a one-off transcription slip.

#### Two MathML rendering fixes

The piecewise `err()` definition needed a trailing space inside `<mtext>if </mtext>` removed — it is collapsed away, so "if" ran into h(x).

Its second problem, "otherwise" sitting centred rather than aligned under the condition above it, was **not** fixed at the time, though it was recorded here as fixed. Moving `columnalign` from the `<mtable>` to each `<mtd>` changed nothing, and the measurement that appeared to confirm otherwise was taken against a standalone test file rather than the page itself. See "Piecewise blocks realigned" below for the actual fix.

### Chapter 09 — 24 replaced, 1 supplemented, 11 left alone

The largest conversion so far. Replaced: the K-vector form of y with its class labels; the regularised logistic cost; the neural network cost function (which appears three times) along with each of its halves; the partial derivative being sought (twice); forward propagation written out step by step; the δ terms for the hidden layers; the training set; the whole back-propagation accumulation — initialise Δ, loop over examples, accumulate in element and vectorised form, and form the D terms; the numerical gradient for each parameter; and two Octave blocks (the `fminunc` call and the gradient-checking loop).

Supplemented: the matrix-dimension figure keeps its small network sketch and gains the dimensions as text.

Kept as images: the annotated back-propagation diagrams — the networks with δ terms flowing backwards — plus the network architectures and the derivative-approximation plot. These are the heart of the chapter and markup cannot reproduce them.

#### A slide error in the per-example cost

One slide writes the per-example cost as

> cost(i) = y⁽ⁱ⁾ log h(x⁽ⁱ⁾) + (1 − y⁽ⁱ⁾) log h(x⁽ⁱ⁾)

with the second logarithm missing its `1 −`. As written the two terms are the same function, which cannot be right. The correct form appears on another slide in this same chapter, and in chapters 06 and 07. Corrected, with a note.

#### Long equations now break across lines

The neural network cost function is genuinely long — cross-entropy over m examples and K outputs, plus a triple-summed regularisation term. As a single line it ran about 115px past the text column, so a reader had to scroll the equation sideways to see the end of it.

Those three equations now break before the regularisation term, with the continuation indented, the way a textbook sets them. Measured afterwards: **no equation in any converted chapter overflows the text column** at the standard width.

### Chapter 08 — 3 replaced, 4 supplemented, 13 left alone

Chapter 08 inverts the ratio of every chapter before it. It is mostly **network diagrams** — circles, wires and weight labels — which markup cannot reproduce, so most figures stay exactly as they are.

**Replaced (3):** the feature vector and layer-2 z vector; the final hypothesis computed from the layer-2 activations; and the one-hot encoding of y.

**Supplemented (4):** four figures are diagrams that carry substantial maths the prose never restates. Those keep their diagram and gain the maths as text underneath:

- The **four layer-2 activation equations** — a<sub>1</sub><sup>(2)</sup> through the hypothesis. The prose says only "the equations below", so until now these existed purely as pixels. The same figure appears twice in the chapter, in both model-representation sections, and both are introduced as the equations to read, so both were supplemented.
- The **AND truth table** and the **XNOR truth table**, both now real `<table>` elements.

**The logic-gate arithmetic was verified against the sigmoid** rather than transcribed on trust: AND (−30, 20, 20), NOR (10, −20, −20) and OR (−10, 20, 20) each produce the truth table shown, and the composed network gives XNOR — 1 exactly when the inputs agree.

#### Tables are now set in the sans stack

The XNOR table is a grid of 0s and 1s, and Georgia's oldstyle zero sits at x-height — in a truth table specifically, "0" was genuinely hard to tell from "o". Tables now use the same sans stack as the equation notes, with `tabular-nums` so numeric columns line up. Table captions also moved above the table where they belong. This improves the training-set tables in chapters 02 and 04 at the same time.

### Chapter 07 — 14 of 17 figures converted

Converted: penalising θ<sub>3</sub> and θ<sub>4</sub> by hand; the regularised cost function, with and without the minimisation; gradient descent with θ<sub>0</sub> split out; the regularised update rule and its grouped form; the shrinkage factor on its own; the regularised normal equation; the logistic cost function and the term that regularises it; logistic gradient descent before and after regularisation; and both `costFunction` skeletons — the second with every regularised term written out.

Kept as images: the three under/over-fitting curve plots, the three classification-boundary plots, and the house-price fit.

#### Three source slides had mistakes

This chapter needed more care than the others. Transcribing faithfully would have reproduced broken maths, so each was corrected:

- **A dropped bracket.** One slide writes `log h(x⁽ⁱ⁾` with no closing bracket. The corrected transcription now matches chapter 06's version of the same equation **character for character** — verified by comparing the two rendered expressions programmatically.
- **The same expression mangled differently.** Another slide opens a bracket after the first `log` and omits the pair around the second, giving `log 1 − h(x⁽ⁱ⁾)` where the whole of `1 − h(x⁽ⁱ⁾)` is the argument. Corrected to the standard cross-entropy form.
- **A struck-out index.** Two slides show `j = 0, 1, 2, …, n` with the 0 crossed out in red, because θ<sub>0</sub> is updated on its own line and never penalised. Strike-through doesn't survive transcription, so the index is written as it is meant to be read, with a note explaining why θ<sub>0</sub> is separate.

#### And one error in the author's own text

The prose introducing the regularised normal equation says the added matrix "is the n+1 identity matrix". It isn't — the figure beside it clearly shows **0 in the top-left** and 1s down the rest of the diagonal, which is the whole point: it is what lets θ<sub>0</sub> escape the penalty. The sentence contradicted both the adjacent figure and the rest of the chapter, which repeatedly says θ<sub>0</sub> is not penalised. It now reads "the identity matrix with the top-left entry set to 0".

#### A note on `jVal` versus `jval`

Chapter 06's prose and code use `jval`/`THETA`; chapter 07's use `jVal`/`theta`. Each chapter is now internally consistent with its own text, which is what matters for a reader following one chapter. The difference between them is the author's own and has been left alone.

### Chapter 06 — 14 of 25 figures converted

Converted: the logistic hypothesis (three times, where it recurs); the training-set setup with its feature vector and y ∈ {0,1}; the linear-regression cost shown for comparison; the cost rewritten with a `Cost()` term, twice; the piecewise `Cost()` definition with its brace intact; the assembled cost function; the combined single-expression form using the y and (1−y) trick; gradient descent written out and in compact form; what you need to be able to compute; the worked example for the advanced optimizers; and the Octave `costFunction` skeleton.

Kept as images — the shape is the content: the linear-regression-with-thresholding plot, both sigmoid curves, the four decision-boundary plots, the two cost-curve plots, and the two multiclass scatters.

**The `costFunction` skeleton is now a real code block with maths inside it** — Octave in monospace, and the derivative terms as inline MathML rather than a picture of one. This is the first place in the notes where code and maths sit in the same figure, and separating them means both are now selectable and searchable.

#### A naming inconsistency this surfaced

Converting the skeleton put two spellings of the same function into one chapter. The author's own text block and prose use `jval` and `THETA` — the prose says, by name, "two return values from costFunction are **jval**". Professor Ng's slide, which the screenshot showed, uses `jVal` and `theta`.

Transcribing the slide faithfully would have left the chapter contradicting itself and the prose pointing at an identifier that no longer appeared. The converted block therefore uses the author's spelling, so the chapter is internally consistent and every reference resolves.

### Chapter 04 — 12 of 17 figures converted

Chapter 04 splits: **12 figures were equations, algorithm blocks or a data table** and are now MathML; **5 are genuine plots and stay as images.**

Converted: the cost function for n features; the gradient descent algorithm in three forms (n features, the n = 1 case, and the general n ≥ 1 rule, each with its `Repeat { … }` structure intact); the derivative term; the feature-scaling formula; the derivative of J to be set to zero; the four-feature training set as a real `<table>`; the normal equation (which appears twice); the normal equation with the data substituted in; and the construction of the design matrix from transposed training examples.

Kept as images, because the shape of the curve *is* the content: the contour map of slow convergence on unscaled features, the J(θ)-against-iterations convergence plot, the two learning-rate diagnostics (α too large, and overshooting), and the polynomial-regression scatter.

The four-feature table was cross-checked against the same dataset as it appears in chapters 01/02 and 03 — all three now agree exactly, including the price of the 852 ft² house that the earlier fix corrected to 178.

### Chapter 03 — worked NumPy beside the maths

Each of chapter 03's 14 equations is now followed by the **NumPy that performs it**, so the chapter reads as both a reference and something you can run. The maths is unchanged; the code sits underneath it in a block whose left rule continues the equation's accent line, so the pair reads as one unit.

The snippets form a **single continuous session** — `A` is built in the first and indexed in the second, `X` is built for the house-price example and reused for the three-hypothesis one — so the chapter can be followed top to bottom in a REPL.

**None of the code or its output was typed by hand.** Every snippet is executed and every value it claims is asserted before it is written into the page, and the check is repeated afterwards by extracting the code back out of the published HTML and running it end to end. The chapter's code therefore cannot drift from what it says it produces. (Verified against NumPy 2.3.1.)

Alongside the operations, the code notes the places where NumPy differs from the notation in the notes — the traps worth knowing rather than trivia:

- **NumPy indexes from 0**, so the notes' A<sub>32</sub> is `A[2, 1]`. The notes already warn that indexing convention varies; this is the concrete case.
- **`@` is matrix multiplication, `*` is element by element.** The single most common source of wrong answers when translating this maths into code.
- **A 1-D array is neither a row nor a column** — `y.shape` is `(4,)`, not `(4, 1)`. Reshape when the distinction matters.
- **Prefer `np.linalg.solve(C, b)` to `inv(C) @ b`** — faster and numerically better behaved; a singular matrix raises `LinAlgError`.

One wording change came out of proofreading the rendered page rather than the source: an equation note read "1s down the diagonal, 0s elsewhere", and Georgia sets 0 as an x-height oldstyle figure, so "0s" read as "os". It now says "ones down the diagonal, zeros everywhere else".

**12 images were deliberately kept** — every plot, contour map, 3-D surface and hand-drawn intuition diagram. Those carry information that markup cannot reproduce, so converting them would lose meaning rather than gain it. The 14 replaced files are still on disk, unreferenced, in case they're wanted for comparison.

Equations sit in a bordered block that picks up the page theme, so they read as displayed maths in both light and dark mode.

Equation notes are set in the sans stack rather than the body serif. Georgia ships only oldstyle figures and has no lining-figure feature to switch on, so a subscript zero rendered at x-height and read as a letter o — unworkable in notes that discuss &#x3B8;<sub>0</sub>. The sans face fixes the digits and has the side benefit of marking these notes as editorial additions rather than the author's text.

### One content fix this surfaced

Transcribing the maths exposed a disagreement inside the chapter: the training-set table gives the 852 ft² house a price of **178**, while the y vector in `Image [25].png` gives **172**. They encode the same dataset, so one had to be wrong; Professor Ng's slide uses 178, and the table appears earlier in the same chapter. The vector is now **178**, and the chapter is self-consistent.

## New chapters

Two chapters were written for this version. Neither is part of Professor Ng's 2011 course, and each says so in its own opening section so a reader is never misled about provenance.

- **20: Modern Deep Learning** — a high-level tour of what happened after the course: large language models and how pretraining becomes an assistant, the ChatGPT / Claude / Gemini families, agentic workflows (tool use, RAG, MCP, sub-agents), diffusion models, AlphaFold, and protein language models (ESM). Deliberately shallow, and it closes by tying each topic back to the chapters that already cover its failure modes. Version numbers are avoided throughout, since they date within months and the concepts don't.
- **21: Attention** — the one genuinely new mechanism, covered properly: attention as a differentiable soft lookup, scaled dot-product attention and why the √d<sub>k</sub> scaling is there, self-attention, positional encoding, masking, multi-head attention, the transformer block, and the O(n²) cost.

Both follow the house style — the same terse nested bullets, the same `.term` highlighting for key terms, the same section structure and "on this page" contents. The diagrams are inline SVG rather than scanned slides, drawn to take their colours from the theme so they work in light and dark mode; they are styled by a `figure.diagram` block added to `style.css`.

The index lists both with a note that they were added later, and the navigation chain now runs 19 → 20 → 21.

## Chapter 05 rebuilt as Probability and Bayes' Rule; the NumPy primer became Appendix 1

The chapter 05 slot — originally an unwritten programming tutorial, lately the NumPy primer — now holds the chapter the rest of the notes were quietly assuming: **Probability and Bayes' Rule**. The primer moved unchanged to **Appendix 1** (`A1_Python_and_NumPy.html`), chained after chapter 26 and listed at the foot of the index; its note explains the move.

The new chapter 05 covers four ideas, each pinned to where the notes already use it:

- **The vocabulary and the two rules** — joint/conditional/marginal, the product rule (chapter 22's chain-rule factorisation in embryo) and the sum rule.
- **Bayes' rule**, derived in two lines, with the rare-disease worked example using chapter 11's own 0.5% prevalence: a 99%-sensitive test yields a 9% posterior, computed on the page — and precision unmasked as a posterior, recall as a likelihood.
- **MLE and MAP** — the origin story of two cost functions (chapter 02's squared error as Gaussian MLE, chapter 06's cross-entropy as Bernoulli MLE), and the thing the notes never said aloud, now said: **L2 regularization is MAP estimation with a Gaussian prior**, with λ as the noise-to-prior variance ratio. Chapter 07 now points here.
- **Generative vs discriminative, and Naive Bayes** — the debt-payer. A complete Bernoulli Naive Bayes spam classifier on chapter 11's own vocabulary (andrew/buy/deal/discount/now), with Laplace smoothing derived as a prior and log-space computation motivated by an underflow demo; it scores "buy discount now" at 0.971 spam and "andrew … now" at 0.013, all verified and page-executable. The closing section shows Naive Bayes' log-odds are linear — logistic regression's generative twin — and frames the which-wins question as chapter 10's bias/variance trade, slotting into chapter 11's algorithm comparison. Chapter 11's "Naive Bayes — cover later" now reads "covered in chapter 05".

Cross-links added in chapters 06 (MLE), 07 (MAP), 11 (the debt) and 15 (the independence assumption); navigation and index rewired. Full set: 29 pages, parse clean, no broken links, 72 code blocks execute with zero failures.

## Chapter 26 — Convolutional Neural Networks

A new chapter filling the most conspicuous gap in the set: the notes went from fully-connected networks (08–09) straight to transformers (21), skipping the architecture that started the deep learning era. The way in is a promise chapter 18 already made — its pedestrian and text detectors crop a patch, classify, slide, repeat; a CNN *is* that idea with the detector learned and the sliding built into the arithmetic. Chapter 18's sliding-window section now points forward to it.

The chapter covers: why fully-connected fails on images (chapter 08's own 50-million-feature number, plus the deeper point that an FC net is permutation-blind); the convolution operation with its equation and output-size formula; a **ten-line NumPy `conv2d`** whose hand-built vertical-edge filter lights up exactly on the edge of a test image ([0, 3, 3, 0] rows, computed on the page); the three ideas (local connectivity, weight sharing — 6,250× fewer parameters in the worked comparison — and translation equivariance, framed via chapter 24's invariance-by-construction move); max pooling in three lines; the conv–ReLU–pool architecture with an SVG pipeline diagram and a LeNet-style bookkeeping table (28×28 digits, 30,134 parameters total — less than a single 784→100 FC layer); training as pure reuse (chapter 09 backprop, chapter 20 toolkit, chapter 18 augmentation); the 2012 AlexNet discontinuity; sliding windows collapsing into a single forward pass; and CNNs vs transformers as an inductive-bias question, with vision transformers and chapter 10's bias/variance frame deciding which wins at which data budget.

Navigation chained (25 → 26), index entry added, all three new code blocks execute from the page, full set re-validated: 28 pages, parse clean, no broken links, no overflow.

## Chapter 20 — the modern training toolkit added

Chapter 20 was the map of modern deep learning; it now also carries the one set of details that belongs to it: **what changed between the 2011 training recipe and the modern one**. The notes train sigmoid networks with plain gradient descent (chapters 06–09) — which is precisely why deep networks didn't work in 2011 — and nothing in the set explained the fixes. Five new sections, sitting between "What actually changed" (which gains training know-how as its fourth item) and the LLM overview, make chapter 20 the bridge between chapter 09's networks and chapters 21–25:

- **Activations** — the vanishing-gradient argument from the sigmoid's own derivative bound (g′ = a(1−a) ≤ ¼, the identity chapter 09 verified), with (¼)³⁰ ≈ 9×10⁻¹⁹ computed on the page; ReLU, dead units, leaky/GELU.
- **Initialization** — chapter 09's "small random values" stress-tested at depth: a 40-layer propagation demo where w = 0.01 kills the signal (std 8×10⁻³⁹), He's √(2/n) keeps it alive, and 3× He explodes it. The variance argument is chapter 21's √d_k reasoning reused.
- **Optimizers** — momentum, RMSProp and Adam with their update equations, framed as chapter 04's contour problem moved inside the network. The code block runs an honest race: on a mildly ill-conditioned bowl, tuned momentum wins (37 steps vs GD's 141) and Adam is unremarkable (172); at condition number 10⁴, Adam wins by 300× (220 steps vs 67,370) without retuning — robustness, not raw speed, is why it became the default.
- **Schedules and warmup** — why chapter 02's "no schedule needed" argument fails under chapter 17's mini-batch noise; cosine decay and warmup.
- **Dropout and batch normalization** — inverted dropout with its expectation checked numerically; batch norm as chapter 04's mean-normalization formula verbatim plus learned γ, β, with layer norm (chapter 21) as the same equation on the other axis. Closes with a 2011-recipe vs modern-recipe table, row by row.

Cross-links updated: chapter 22's Adam mention now points here, and chapter 21's layer-norm section names its batch-wise sibling. All four new code blocks execute from the page; every number was harness-verified first.

## Chapters 22–25 — the modern-topics survey broken out into real chapters

Chapter 20's survey sections on diffusion models, AlphaFold and ESM were excised and replaced by four full chapters, each in the notes' own style (terse bullets, native MathML, verified code, theme-aware SVG diagrams), each pointing back constantly at the 2011 material it builds on. Chapter 20 keeps its role as the map — its LLM overview, post-training and agentic sections stand — with its note, contents list and summary rewritten to hand off to the new chapters.

- **22: Large Language Models.** The mathematics behind chapter 20's overview: the chain-rule factorisation, softmax over a vocabulary, and cross-entropy as maximum likelihood — with the sigmoid shown to be two-class softmax both algebraically and numerically (chapter 06's hypothesis falls out of the identity). Covers autoregressive and masked objectives with a full comparison table, temperature, perplexity, and scaling laws. Three verified code blocks; ties to chapters 03, 06, 09, 10, 17 and 21 throughout.
- **23: Diffusion Models.** The forward process, the closed-form jump via ᾱ, the ε-prediction objective (chapter 02's squared error), and the sampling update — then a **complete diffusion model in ~40 lines of NumPy**: bimodal 1-D data, chapter 08's two-layer network trained with chapter 09's backprop written out by hand, chapter 17's mini-batches. Executed straight from the page it trains in 0.6s and turns pure noise into two clean modes at ±1.9 with a near-empty gap — every quoted number reproduced by running the published code. The diffusion diagram moved here from chapter 20.
- **24: AlphaFold2.** The architecture piece by piece: MSAs and coevolution (chapter 14's covariance as the structural fingerprint), the MSA and pair representations, the Evoformer's four mechanisms (pair-biased row attention — chapter 21's score plus a b_ij term — column attention, outer-product mean, triangle updates), the structure module with rigid frames and Invariant Point Attention, FAPE and the auxiliary losses (including the masked-MSA head — chapter 22's objective inside AlphaFold), recycling, self-distillation, pLDDT/PAE, limits, and AlphaFold3's diffusion decoder (chapter 23). New SVG architecture diagram.
- **25: ESM2 and ESM-C.** Chapter 22's masked objective on a 20-letter alphabet, and why proteins settle the AR-vs-MLM comparison in favour of masking. ESM2 as the reference family (with ESMFold), ESM-C as the efficiency successor, a comparison table, and the usage maths: mean-pooled embeddings, the masked-marginal log-odds variant score (chapter 06's log-odds meeting chapter 15's anomaly logic), and pseudo-perplexity.

Navigation was rewired (21 → 22 → 23 → 24 → 25), the index gained four entries, and the whole set re-validated: 27 pages, strict parse clean, no broken links, no overflow at 480/760px, and all code blocks in the new chapters execute from the page.

## Python throughout — MATLAB and Octave retired

The original course taught its programming in Octave/MATLAB, and the notes carried that through: Octave code snippets, references to `fminunc`, `pinv`, `svd`, `reshape`, `det` and `hist` by their Octave names, and a chapter 05 that was an empty placeholder for an Octave tutorial that was never written. All of it is now Python.

### What was converted

- **Every code snippet** became working NumPy/SciPy: the normal equation (`np.linalg.pinv(X.T @ X) @ X.T @ y`), the `fminunc` calls (`scipy.optimize.minimize` with `jac=True`), the cost-function templates (Python functions returning `(jval, gradient)`, with the inline maths kept), parameter unrolling (`np.concatenate`/`.reshape`), the gradient-checking loop, the PCA pipeline (`np.linalg.svd`, `U[:, :k]`), and the cocktail-party one-liner in chapter 01.
- **Every prose reference** was reworded — "In MATLAB or octave we can implement this as follows", "Octave indexes from 1" (NumPy indexes from 0, with the off-by-one flagged), "use the hist command" (`plt.hist`), the libsvm/liblinear mention (scikit-learn's `SVC`/`LinearSVC`), and the index page's note about what the notes cover.
- **Chapter 05 was rebuilt.** The original slot held only "These notes have not been done". It is now a short NumPy primer — arrays and shapes, `@`/`.T`/`*`, broadcasting, 0-indexing, and the `np.linalg` + `scipy.optimize.minimize` toolbox the other chapters' code uses — under the filename `05_Python_and_NumPy.html`, with the index and neighbouring chapters' navigation updated. Its code blocks are verified like all the others.

### Worked Python beside the maths, in every chapter

Chapter 03 already had NumPy code alongside its linear algebra. That treatment now runs across the board: **34 new code blocks over chapters 02–17** (55 on the pages in total, counting chapter 03's originals and the new chapter 05), each inserted directly under the equation it recapitulates, in the same visual style.

The blocks are not decorations — each one recomputes something the notes assert:

- **Ch 02** reproduces the worked cost values J(1) = 0, J(0.5) ≈ 0.58, J(0) ≈ 2.3 from the prose, runs one simultaneous update, and runs gradient descent to the exact fit.
- **Ch 04** solves the chapter's own four-house table by normal equation — reproducing all four prices to machine precision.
- **Ch 06** verifies cost(θ = 0) = log 2, fits the boundary at x ≈ 2.5, and runs the (θ−5)² example through `minimize` to (5, 5).
- **Ch 07** demonstrates the (1 − αλ/m) shrink factor and shows regularization repairing a genuinely singular XᵀX built from duplicated features.
- **Ch 08** forward-propagates the AND gate and the full XNOR network, reproducing both truth tables.
- **Ch 09** confirms g′(z) = a(1−a) against a numerical derivative, round-trips the 231-element unroll, and gradient-checks a hand-differentiable cost.
- **Ch 10** runs a real train/test split, the err() expression, and the model-selection loop — where d = 9 halves the training error while the cross-validation error explodes from 0.027 to 2.25.
- **Ch 11** scores the three-algorithm table both ways: the average picks algorithm 3, F₁ picks algorithm 1.
- **Ch 12** computes cost₁/cost₀, the Gaussian kernel (1 on the landmark, 4×10⁻⁶ far away), and the worked θᵀf prediction.
- **Ch 13** runs K-means to convergence on six points and computes the distortion.
- **Ch 14** recovers the y = 2x direction from correlated data and shows one component retaining 99.96% of the variance.
- **Ch 15** fits μ and σ² from a thousand samples, and verifies numerically that the diagonal-Σ multivariate Gaussian equals the product of per-feature Gaussians.
- **Ch 16** computes Alice's 4.95-star prediction and the full mean-normalisation matrix with `nanmean`.
- **Ch 17** runs stochastic gradient descent one example at a time to the same answer, and verifies the four map-reduce partial sums equal the single 400-example sum.

**Verification:** every value quoted in a code comment was computed by a harness before insertion, and after insertion all 55 blocks were extracted back out of the pages and executed — zero failures. The full set re-validates clean: strict HTML5 parse, no broken links or anchors, no page overflow at 480px or 760px.

## Second review pass — a read-through of every chapter

After the conversion work was finished, every chapter was read end to end looking for errors, inconsistencies and missing information that the earlier passes — which were focused on structure and on figures — had not touched. This pass made roughly 280 further corrections across chapters 01–18. The two added chapters (20, 21) were checked and needed nothing; chapter 19 and the index needed one capitalisation change between them.

Most of the fixes are plain typos in the prose ("definer breast cancer", "takes steps to converse", "Merecer's Theorem", "predict the prize", "aspect ration", and a systematic *chose* → *choose* affecting 30 sentences — the two genuinely past-tense uses were left alone). Those are not itemised individually. The corrections that change technical content are:

- **Ch 01** — "Tom Michel (1999)" → **Tom Mitchell (1998)**, the source of the famous E/T/P definition; the tumour class list ran 0, 1, 2, then "type 4" → **type 3**.
- **Ch 02** — the example pair "(x^i, y^j)" → **(x^i, y^i)**; the claim that moving up a slope makes the cost "a bigger number" corrected — the positive-slope case *decreases* θ₁, mirroring the bullet above it; the normal equation was called a "numeric method" in three places — it is the **analytic** solution (gradient descent is the numerical one).
- **Ch 04** — `pinv(X'*x)*x'*y` → **`pinv(X'*X)*X'*y`** (Octave is case-sensitive; as printed the code used a different variable); the section heading "gradient descent vs. **feature scaling**" → "vs. **the normal equation**", which is what the section compares; "m ≤ n (m is much larger than n)" → "(**n** is much larger than **m**)", matching its own example of m = 10, n = 100. "Gradient **Decent**" in a heading → Descent.
- **Ch 06** — "generates a value where is always either 0 or 1" → "a value which is always **between 0 and 1**" (the sigmoid's whole point); the non-linear-boundary hypothesis was missing its **θ₂x₂ term** while the parameter vector [-1,0,0,1,1] has five entries; "stuck in a **global** minimum" → **local**; the claimed "underlying **Gaussian** assumption" behind the MLE derivation → **Bernoulli** (Gaussian belongs to linear regression's squared error); the third one-vs-all classifier separates squares from "crosses and **square**" → "crosses and **triangle**".
- **Ch 07** — "Sum for every θ (i.e. j = **0** to n)" → **j = 1 to n**, as the surrounding notes say three times; `@costfunction` → `@costFunction`.
- **Ch 08** — "100 x 100 **RB** → 50 000 000 features": the number is right for grayscale quadratic features (n = 10,000, n²/2), so the stray "RB" (suggesting RGB, nine times more) was dropped.
- **Ch 09** — the training set ended "(x^**n**, y^m)" with no closing brace → "(x^m, y^m)}"; "back through the network from layer L-1 down to layer" was cut off mid-sentence → "down to layer **2**"; `resape` → **`reshape`**; the accumulator update read "Δ := Δ^l + …" → "**Δ^l** := Δ^l + …".
- **Ch 11** — the two threshold examples were internally inconsistent: "predict 1 if h ≥ 0.8, predict 0 if h < **0.2**" (leaving 0.2–0.8 unclassified) → "< **0.8**", and likewise "< 0.7" → "< **0.3**"; the heading "Error metrics for skewed **analysis**" → "skewed **classes**", the term the section itself uses.
- **Ch 12** — the kernel hypothesis "returns 1 if the weighted sum is **less** than or equal to 0" → **greater**; f₂ and f₃ were defined against the wrong landmarks (f₂ labelled l¹, f₃ labelled l² with distance to l¹) → each is the similarity to **its own** landmark; the worked example concluded "0.5 is greater than **1**" → "greater than **0**", the actual decision rule.
- **Ch 13** — "the second part minimizes the **J** variables" → the **μ** variables (J is what both parts minimise).
- **Ch 14** — "**Principle** Component Analysis" → **Principal**, in both headings, the contents list and the prose; the linear-regression comparison "minimize the straight line between a point and a squared line" untangled to "minimize the squared distance between a point and the line".
- **Ch 15** — "parameterized by the mean and **squared variance**" → variance (σ² *is* the variance); "determine parameters for each of your **examples**" → **features** (μⱼ and σⱼ² are per-feature); the heading "Developing and evaluating **and** anomaly detection system" → "**an** anomaly detection system".
- **Ch 16** — "the regularization term goes from k=1 through to **m**" → **n**, matching the equation beside it; the three prose references to the parameter *matrix* as lower-case θ → capital **Θ**, as the maths defines it.
- **Ch 17** — "stochastic and batch gradient descent are just specific forms of **batch** gradient descent" → "of **mini-batch** gradient descent (b = 1 and b = m)".
- **Ch 18** — "Multi-class **characterization** problem" → classification; "we can **qualitatively** show what the upside would be" → **quantitatively** — putting a number on it is the whole point of ceiling analysis.

One notation clean-up: chapters 08 and 09 used the look-alike glyph **Ɵ** (U+019F, a Latin "o with middle tilde") 75 times where every equation, caption and other chapter writes **Θ**. Chapter 02 had two more. All are now real thetas, so searching the notes for θ or Θ actually finds them.

Sentences that were garbled but decipherable ("You would risk making a ridiculous hugely impact your classification boundary", "Say we have two new data points new data-point has the values") were repaired to their evident intent, with the surrounding bullets as the guide. Truncated sentences whose ending was recoverable from context were completed ("Monitoring machines in data [centers]", "see data in a helpful [way]"); the author's voice, asides and jokes were left exactly as written.

## Content fixes

37 corrections from the original rebuild pass. Typos and slips only — no rephrasing, and no changes to the technical content of the notes. (The second review pass above came later and goes further.)

### Chapter 01/02 — Introduction, Regression Analysis and Gradient Descent

| Original | Corrected |
| --- | --- |
| understand **cotour** plots | understand **contour** plots |
| center of the **countour** plot | center of the **contour** plot |
| Do the following until **covergence** | Do the following until **convergence** |
| Two key **functins** | Two key **functions** |
| towards the **mimum** (down) will **greate** a negative derivative | towards the **minimum** (down) will **create** a negative derivative |
| complex **hyothesis** with two **pariables** | complex **hypothesis** with two **variables** |
| linear regression **modles** | linear regression **models** |
| because of the **summartion** term | because of the **summation** term |
| turn records **in knowledges** | turn records **into knowledge** |
| **Samuels** wrote a checkers playing program | **Samuel** wrote a checkers playing program |

The last one is a name, not a typo: the checkers program was Arthur Samuel's.

### Chapter 04 — Linear Regression with Multiple Variables

| Original | Corrected |
| --- | --- |
| partial derivative **of of** the θ vector | partial derivative **of** the θ vector |
| Very hard to **tel** in advance | Very hard to **tell** in advance |

### Chapter 06 — Logistic Regression

| Original | Corrected |
| --- | --- |
| and **thats** exactly correct | and **that's** exactly correct |
| `function [jval, `**`gradent`**`] = costFunction(THETA)` | `function [jval, `**`gradient`**`] = costFunction(THETA)` |
| `[optTheta, `**`funtionVal`**`, exitFlag]` | `[optTheta, `**`functionVal`**`, exitFlag]` |
| `options= optimset(...)` | `options = optimset(...)` |
| `initialTheta= zeros(2,1)` | `initialTheta = zeros(2,1)` |
| `exitFlag]= fminunc(...)` | `exitFlag] = fminunc(...)` |

The last three are spacing around `=` in the Octave example, which had no space on the left-hand side.

### Chapter 07 — Regularization

| Original | Corrected |
| --- | --- |
| doesn't extend **to to** the lambda term | doesn't extend **to** the lambda term |

### Chapter 08 — Neural Networks: Representation

| Original | Corrected |
| --- | --- |
| is a 3x1 **vecor** | is a 3x1 **vector** |

### Chapter 09 — Neural Networks: Learning

| Original | Corrected |
| --- | --- |
| the **Backproc** implementation is **correc** | the **Backprop** implementation is **correct** |

### Chapter 11 — Machine Learning System Design

| Original | Corrected |
| --- | --- |
| we **have have** a higher recall | we **have** a higher recall |

### Chapter 12 — Support Vector Machines

| Original | Corrected |
| --- | --- |
| power is using **diferent** kernels | power is using **different** kernels |
| **Disussed** more later | **Discussed** more later |
| Gaussian Kernel this **evalues** to 1 | Gaussian Kernel this **evaluates** to 1 |
| why the SVM **choses** this hypothesis | why the SVM **chooses** this hypothesis |
| by contrast is **the the chosen** by the SVM | by contrast is **the one chosen** by the SVM |
| Project a line from x1 **on to to** the θ vector | Project a line from x1 **on to** the θ vector |

### Chapter 13 — Clustering

| Original | Corrected |
| --- | --- |
| index of the **closes** variable of cluster centroid **closes** to x<sup>i</sup> | index of the **closest** variable of cluster centroid **closest** to x<sup>i</sup> |
| that value is **one the the** clusters | that value is **one of the** clusters |
| you don't get **a a** nice line | you don't get **a** nice line |

### Chapter 15 — Anomaly Detection

| Original | Corrected |
| --- | --- |
| the mean (n-**dimenisonal** vector) | the mean (n-**dimensional** vector) |
| if you think something **iss** anomalous | if you think something **is** anomalous |
| concentric circles around **the the** means | concentric circles around **the** means |

### Chapter 17 — Large Scale Machine Learning

| Original | Corrected |
| --- | --- |
| greater number of **entires** per average | greater number of **entries** per average |
| **Send to to** a centralized master server | **Send to** a centralized master server |

### Chapter 18 — Application Example: Photo OCR

| Original | Corrected |
| --- | --- |
| makes it a bit **easer** | makes it a bit **easier** |

## Deliberately left alone

- **"neurone" / "neurones"** in chapter 08. This is the British spelling and consistent with "colour" and "tumour" elsewhere in the notes.
- **"Simplez!"** in chapter 08, which reads as deliberate.
- **"yeses as nos"** in chapter 06 — correct as written.
- **"hard(ish...)"**, **"Nada."** and similar asides, which are the author's voice.
- The five corrections already recorded in the original changelog were made in the V1 text long ago and carry through unchanged. The historical table is preserved in `changelog.html`.

## Verification

The rebuild was checked three ways, and all three pass on all 18 chapters:

1. **Text equivalence.** The rendered text of each V3 page, with whitespace normalised the way a browser collapses it, is identical to the V1 page once the 37 fixes above are applied to the V1 text. This catches both dropped content and whitespace introduced or lost between inline elements.
2. **Image equivalence.** Every page references the same images, in the same order, as its V1 counterpart. All 383 resolve to a file on disk, and no image on disk is unreferenced.
3. **Validity.** Every page passes a strict HTML5 parse, has no inline styles or legacy tags, and every internal link, section anchor and image path resolves.
