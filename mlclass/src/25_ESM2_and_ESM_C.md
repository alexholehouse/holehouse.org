---
title: "25: ESM2 and ESM-C"
nav_title: "25: ESM2 and ESM-C"
origin: "2026"
description: "Protein language models: masked residue prediction, what the embeddings capture, and how to use them in practice."
---

## A note on this chapter

- An added chapter, and the one that closes the modern-biology loop
  - Chapter 22 set up masked language models and said their strength is representations; this chapter is that claim, applied to biology
  - Everything here is chapter 22's MLM section with a 20-letter alphabet - the same objective, the same softmax, the same cross-entropy

## Proteins as language

- The observation
  - A protein sequence is a string over an alphabet of 20 amino acids
  - We have billions of them, unlabelled, from sequencing everything in sight
  - So do to proteins exactly what chapter 22 did to text
- Why the masked objective rather than next-token (the chapter 22 comparison, settled by the domain)
  - A protein is not written left to right - residue 40 constrains residue 8 exactly as much as the reverse
  - There is no natural generation order to exploit, and the product we want is a representation per residue, not completions
  - Both favour the bidirectional, masked side of chapter 22's table
- What "grammar" means here
  - Which residues are plausible at a position is constrained by structure, stability and function
  - A model forced to predict masked residues well must internalise those constraints - the same argument as chapter 22's "to predict the next token you must model what produced it"

## The objective: masked residues

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo stretchy="false">(</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo><mo>=</mo><mo>&#x2212;</mo><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>M</mi><mo stretchy="false">|</mo></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>&#x2208;</mo><mi>M</mi></mrow></munder><mi>log</mi><mspace width="0.15em"/><msub><mi>p</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>&#x2223;</mo><msub><mi>x</mi><mrow><mo>&#x2216;</mo><mi>M</mi></mrow></msub><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">Chapter 22's masked loss, verbatim &mdash; only the vocabulary changed: ~33 tokens (20 amino acids plus specials) instead of 100,000 word-pieces. M is the masked ~15% of positions; following BERT, most are replaced by a mask token, a few by random residues, a few left alone.</span>
  </div>

- The architecture is chapter 21's transformer encoder, unmasked attention in both directions
  - Every residue attends to every residue - contacts 200 positions apart are one attention step away, which is the whole sales pitch of attention over recurrence
- No structures, no labels, no alignments anywhere in the objective
  - Unlike AlphaFold2 (chapter 24), which is handed an MSA per query, ESM sees single sequences - evolution's statistics enter only through the training set containing millions of related sequences

## ESM2

- The reference protein language model family (Meta AI; the 2023 *Science* paper)
  - Trained on tens of millions of unique sequences from UniRef
  - A family of sizes - 8M, 35M, 150M, 650M, 3B and 15B parameters - trained identically, which made it a clean scaling-law study (chapter 22) for biology
  - Rotary positional encoding - the "relative position" style flagged in chapter 21
- What emerges, with nobody asking
  - Attention maps line up with residue-residue *contact maps* - the coevolution signal chapter 24's Evoformer extracts from an explicit alignment, recovered here from single sequences
  - Secondary structure, binding sites and disorder become linearly readable from the embeddings
  - And it scales: the bigger the model, the lower the masked-residue perplexity, and the better the implied structure - capability tracking the training loss, exactly as in chapter 22
- ESMFold
  - A structure head on top of ESM2-3B: predicts 3D coordinates from a *single sequence*, no MSA search
  - An order of magnitude faster than AlphaFold2 and somewhat less accurate - the trade that makes metagenomic-scale folding (hundreds of millions of predictions) feasible, and the right tool where alignments don't exist (chapter 24's shallow-MSA limit)

## ESM-C

- The successor family (ESM Cambrian, EvolutionaryScale, late 2024 - the ESM team, spun out of Meta)
  - Not a new idea: the same masked-residue objective, deliberately
  - What changed is everything around it - several-fold more training data (UniRef plus large metagenomic collections, billions of sequences), longer training, and a modernised transformer recipe (rotary positions retained, gated feed-forward layers, streamlined normalisation)
- The headline is efficiency, not a new ceiling
  - ESM-C 300M gives representations comparable to ESM2 650M; ESM-C 600M is comparable to (and often better than) ESM2 3B
  - Roughly the same quality at a fraction of the parameters and compute - which matters because these models are run constantly as embedding machines, not trained once
- Read through this course's lens
  - Chapter 22's scaling laws have data and parameters as separate axes: ESM2 climbed the parameter axis, ESM-C shows how far the data-and-recipe axes go at fixed size
  - The same lesson as chapter 11's "more data beats a cleverer algorithm", rediscovered at the billion-parameter scale

## Comparing ESM2 and ESM-C

<div class="table-wrap"><table>
  <caption>Same objective, same interface &mdash; the differences are data, recipe and efficiency.</caption>
  <tr><th scope="col"></th><th scope="col">ESM2 (2022-23)</th><th scope="col">ESM-C (2024)</th></tr>
  <tr><th scope="row">Developer</th><td>Meta AI</td><td>EvolutionaryScale</td></tr>
  <tr><th scope="row">Objective</th><td>masked residues (chapter 22 MLM)</td><td>identical</td></tr>
  <tr><th scope="row">Sizes</th><td>8M &ndash; 15B parameters</td><td>300M, 600M, 6B</td></tr>
  <tr><th scope="row">Training data</th><td>UniRef, tens of millions of sequences</td><td>several-fold larger, heavy metagenomic fraction</td></tr>
  <tr><th scope="row">Architecture</th><td>transformer encoder, rotary positions</td><td>same skeleton, modernised recipe</td></tr>
  <tr><th scope="row">Rule of thumb</th><td>the established baseline</td><td>ESM2-quality embeddings at ~&frac12; to &frac15; the size</td></tr>
  <tr><th scope="row">Structure head</th><td>ESMFold (on the 3B model)</td><td>none - embeddings are the product</td></tr>
</table></div>

- Practical guidance
  - Starting fresh and wanting embeddings: ESM-C, smallest size that meets your accuracy needs
  - Reproducing or comparing against the literature: ESM2, which remains the common reference point
  - Single-sequence structure prediction: ESMFold, since ESM-C ships no folding head

## Using the models: the maths

- 1\) Embeddings - the workhorse
  - The final layer gives one d-dimensional vector per residue; averaging them gives one vector per protein

<div class="eqn">
<math display="block"><mrow><mi>z</mi><mo>=</mo><mfrac><mn>1</mn><mi>r</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>r</mi></munderover><msub><mi>h</mi><mi>i</mi></msub></mrow></math>
<span class="eqn-note">Mean-pooling the per-residue vectors h<sub>i</sub>. The fixed-length z then feeds a small supervised model &mdash; chapter 06's logistic regression on a few hundred labelled examples is often enough, which changes what a small lab can attempt. This is the "learned features replacing hand-built features" thread that runs from chapter 08 through chapter 16, completed.</span>
  </div>

- 2\) Zero-shot variant effect prediction
  - Mask position i, read out the model's distribution over residues there, and compare mutant against wildtype as a log-odds ratio - chapter 06's favourite quantity

<div class="eqn">
<math display="block"><mrow><mi>s</mi><mo stretchy="false">(</mo><mtext>mut</mtext><mo stretchy="false">)</mo><mo>=</mo><mi>log</mi><mspace width="0.15em"/><msub><mi>p</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>=</mo><mtext>mut</mtext><mo>&#x2223;</mo><msub><mi>x</mi><mrow><mo>&#x2216;</mo><mi>i</mi></mrow></msub><mo stretchy="false">)</mo><mo>&#x2212;</mo><mi>log</mi><mspace width="0.15em"/><msub><mi>p</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>=</mo><mtext>wt</mtext><mo>&#x2223;</mo><msub><mi>x</mi><mrow><mo>&#x2216;</mo><mi>i</mi></mrow></msub><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">The masked-marginal score: how much less plausible does the model find the mutant than what evolution kept? Strongly negative predicts damage. No labelled variant data is used at any point &mdash; it is chapter 15's anomaly-detection logic (flag what the density model finds improbable) with a learned p.</span>
  </div>

::: code-eg
```
import numpy as np

# toy numbers: the model's masked distribution at one buried position,
# where the wildtype is leucine (L)
p = {"L": 0.62, "I": 0.21, "V": 0.11, "P": 0.002}

np.log(p["I"] / p["L"])   # -1.08 - conservative substitution: mildly suspect
np.log(p["V"] / p["L"])   # -1.73 - similar, a little worse
np.log(p["P"] / p["L"])   # -5.74 - proline in a buried helix: strongly deleterious
```
<p class="eqn-note">The scoring rule on illustrative numbers: chemically similar residues score near zero, and the substitution that breaks the local structure scores far below. Real use sums this over every mutated position.</p>
:::

- 3\) Pseudo-perplexity - evaluating an MLM
  - Chapter 22's perplexity needs the chain rule, which an MLM doesn't have
  - The stand-in: mask each position in turn, score the true residue, and exponentiate the average - r forward passes per sequence

<div class="eqn">
<math display="block"><mrow><mtext>pPPL</mtext><mo>=</mo><mi>exp</mi><mo form="prefix">(</mo><mo>&#x2212;</mo><mfrac><mn>1</mn><mi>r</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>r</mi></munderover><mi>log</mi><mspace width="0.15em"/><msub><mi>p</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>&#x2223;</mo><msub><mi>x</mi><mrow><mo>&#x2216;</mo><mi>i</mi></mrow></msub><mo stretchy="false">)</mo><mo form="postfix">)</mo></mrow></math>
<span class="eqn-note">Between 1 (certain) and 20 (uniform over the amino acids). Well-modelled protein families sit far below 20, and a sequence's pseudo-perplexity tracks how natural the model finds it &mdash; low pPPL designs are likelier to fold.</span>
  </div>

## Caveats

- The model has learned the distribution of sequences evolution actually sampled
  - That is a biased sample of what is physically possible, and these models weaken the further you get from well-populated families
- Evaluation leaks easily
  - Random train/test splits put homologous sequences on both sides, measuring memorization and calling it generalization - chapter 10's discipline needs sequence-identity-aware splits here, not row-shuffling
- A high score is plausibility under evolution's distribution, not a guarantee of function - wet-lab validation is still the test set that matters

## Summary

- Protein language models are chapter 22's masked language models on a 20-letter alphabet - the domain has no reading direction, so the bidirectional objective wins by default
- ESM2: the reference family, 8M to 15B parameters, where contacts and structure emerged unasked from masked-residue prediction - and ESMFold turned that into single-sequence structure prediction
- ESM-C: the same objective on far more data with a modern recipe - ESM2-level embeddings at a fraction of the size; efficiency, not a new objective, is the contribution
- The three ways to use them: mean-pooled embeddings feeding a small supervised model (chapter 06), masked-marginal log-odds for zero-shot variant effects (chapters 06 and 15), and pseudo-perplexity for how natural a sequence is (chapter 22)
- The caveats are the course's own: a biased training distribution, and evaluation that only means something when the split respects homology (chapter 10)
