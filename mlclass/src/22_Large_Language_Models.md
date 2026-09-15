---
title: "22: Large Language Models"
nav_title: "22: Large language models"
origin: "2026"
description: "Tokens and embeddings, autoregressive and masked objectives, sampling, perplexity, and training at scale."
---

## A note on this chapter

- Like chapters 20 and 21, this is an addition - there is no 2011 lecture behind it
  - Chapter 20 gives the high-level tour of large language models; this chapter is the mathematics
  - It assumes chapter 21 (attention) and leans constantly on chapters 06 and 09
- The punchline, stated up front
  - A language model is chapter 06's classifier with a vocabulary-sized output, applied over and over
  - Almost every equation below is one we have already seen, wearing bigger numbers

## The language modelling problem

- A language model assigns a probability to a sequence of tokens
  - Good sentences should get more probability than word salad
  - That single requirement, pushed hard enough, is where everything else comes from
- A joint distribution over sequences is unmanageable directly
  - Even a 10-token sentence over a 50,000-word vocabulary has 50,000<sup>10</sup> possible values
  - So factorize it with the chain rule of probability - no approximation involved

<div class="eqn">
<math display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><msub><mi>w</mi><mn>1</mn></msub><mo>,</mo><msub><mi>w</mi><mn>2</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>w</mi><mi>T</mi></msub><mo stretchy="false">)</mo><mo>=</mo><munderover><mo>&#x220F;</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover><mi>p</mi><mo stretchy="false">(</mo><msub><mi>w</mi><mi>t</mi></msub><mo>&#x2223;</mo><msub><mi>w</mi><mn>1</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>w</mi><mrow><mi>t</mi><mo>&#x2212;</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">The chain rule: the probability of a sequence is the product of one next-token prediction per position. It turns "model language" into "predict the next token, repeatedly" &mdash; a supervised learning problem with the labels built in, which is the self-supervision idea from chapter 20.</span>
  </div>

- Each factor is a classification problem
  - Input: the tokens so far. Output: a distribution over the vocabulary
  - Exactly the multiclass setup of chapters 06 and 08, with |V| classes instead of 4

## Tokens and embeddings

- Tokens
  - Text is split into word-pieces from a fixed vocabulary V, typically 50,000-200,000 entries
  - Common words are one token; rare words split into several ("tokenization" → token + ization)
  - Fixes the output dimension, and nothing is ever out of vocabulary
- Embeddings
  - Token i becomes a one-hot vector e<sub>i</sub> - all zeros with a 1 in position i, exactly the one-hot class vectors from chapter 08's multiclass section
  - Multiply by a learned embedding matrix E, which is \[|V| x d\]

<div class="eqn">
<math display="block"><mrow><msub><mi>x</mi><mi>i</mi></msub><mo>=</mo><msup><mi>E</mi><mi>T</mi></msup><msub><mi>e</mi><mi>i</mi></msub></mrow></math>
<span class="eqn-note">A one-hot vector times a matrix just selects row i of E &mdash; chapter 03's matrix-vector multiplication doing dictionary lookup. Each token gets a learned d-dimensional vector (d is typically 1,000-10,000), and tokens that behave similarly end up with similar vectors, because that is what reduces prediction error.</span>
  </div>

- Position is added to each embedding before any attention happens - chapter 21's positional encoding, needed for exactly the reason given there
- The sequence of embedded, position-tagged vectors then runs through the stack of transformer blocks from chapter 21

## Autoregressive models

- The GPT family, and every current chat assistant, are autoregressive (AR) models
  - They model the chain-rule factors directly: always predict the next token from what came before
- The output head
  - The transformer turns the context into a vector; one final matrix maps it to |V| numbers - the logits z
  - Softmax turns logits into a distribution

<div class="eqn">
<math display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><msub><mi>w</mi><mi>t</mi></msub><mo>=</mo><mi>v</mi><mo>&#x2223;</mo><msub><mi>w</mi><mrow><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><msup><mi>e</mi><msub><mi>z</mi><mi>v</mi></msub></msup></mrow><mrow><munderover><mo>&#x2211;</mo><mrow><msup><mi>v</mi><mo>&#x2032;</mo></msup><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><mi>V</mi><mo stretchy="false">|</mo></mrow></munderover><msup><mi>e</mi><msub><mi>z</mi><msup><mi>v</mi><mo>&#x2032;</mo></msup></msub></msup></mrow></mfrac></mrow></math>
<span class="eqn-note">Softmax: exponentiate every logit and normalize. Positive, sums to 1, and the biggest logit gets the biggest share &mdash; the same function that turned scores into weights inside attention (chapter 21).</span>
  </div>

- Softmax is the sigmoid, grown up
  - Chapter 06's sigmoid is exactly softmax over two classes with the second logit fixed at 0

<div class="eqn">
<math display="block"><mrow><mfrac><mrow><msup><mi>e</mi><mi>z</mi></msup></mrow><mrow><msup><mi>e</mi><mi>z</mi></msup><mo>+</mo><msup><mi>e</mi><mn>0</mn></msup></mrow></mfrac><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>&#x2212;</mo><mi>z</mi></mrow></msup></mrow></mfrac><mo>=</mo><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">Divide top and bottom by e<sup>z</sup> and chapter 06's hypothesis falls out. Binary classification was the |V| = 2 case all along.</span>
  </div>

::: code-eg
```
import numpy as np

def softmax(z):
    z = z - z.max()            # subtract the max first - numerical safety,
    e = np.exp(z)              # and it changes nothing (top and bottom scale alike)
    return e / e.sum()

softmax(np.array([1.3, 0.0]))[0]   # 0.785835
1 / (1 + np.exp(-1.3))             # 0.785835 - the sigmoid, identically
```
<p class="eqn-note">The identity checked numerically: two-class softmax with the second logit at 0 <em>is</em> chapter 06's sigmoid.</p>
:::

- The training objective
  - Maximise the log-probability of the training text; equivalently minimise the average negative log-likelihood

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo stretchy="false">(</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo><mo>=</mo><mo>&#x2212;</mo><mfrac><mn>1</mn><mi>T</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover><mi>log</mi><mspace width="0.15em"/><msub><mi>p</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>w</mi><mi>t</mi></msub><mo>&#x2223;</mo><msub><mi>w</mi><mrow><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">Cross-entropy loss. For two classes this is literally chapter 06's cost function &mdash; y log h + (1 &minus; y) log(1 &minus; h) is the |V| = 2 case &mdash; and it is maximum likelihood for the same reason: there the labels were modelled as Bernoulli, here as categorical. One objective, from spam filters to chat assistants.</span>
  </div>

::: code-eg
```
vocab  = ["mat", "dog", "moon", "sofa", "banana"]
logits = np.array([3.2, 1.1, 0.4, 2.4, -1.0])   # "the cat sat on the ___"
p = softmax(logits)
# mat 0.607, sofa 0.273, dog 0.074, moon 0.037, banana 0.009

-np.log(p[0])   # 0.499 - the loss if the actual next word is "mat"
-np.log(p[4])   # 4.699 - the loss if it is "banana": rare surprises cost a lot
```
<p class="eqn-note">One next-token prediction, scored. The loss is small when the model put probability on what actually happened, and large when it was surprised &mdash; averaging this over trillions of tokens is the whole of pretraining.</p>
:::

- Why one pass trains every position
  - With chapter 21's causal mask, position t can only see positions 1 to t
  - So a single forward pass over a T-token document yields T separate next-token predictions, each scored against the token that actually follows
  - Every token of the corpus is a training example - no labelling budget, which is what chapters 10 and 11 spent so much care rationing

## Masked language models

- The BERT family - and chapter 25's protein models - train differently
  - Hide a random subset of tokens (typically 15%); predict each hidden token from *everything else*

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo stretchy="false">(</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo><mo>=</mo><mo>&#x2212;</mo><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>M</mi><mo stretchy="false">|</mo></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mi>t</mi><mo>&#x2208;</mo><mi>M</mi></mrow></munder><mi>log</mi><mspace width="0.15em"/><msub><mi>p</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>w</mi><mi>t</mi></msub><mo>&#x2223;</mo><msub><mi>w</mi><mrow><mo>&#x2216;</mo><mi>M</mi></mrow></msub><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">M is the set of masked positions. Same softmax, same cross-entropy &mdash; the only changes are which positions are scored and what the model is allowed to look at.</span>
  </div>

- The crucial architectural difference is the attention mask
  - No causal mask: every position attends to every position, in both directions
  - "The \_\_\_ sat on the mat" - the right context is often what settles the answer
- What you give up
  - The chain rule decomposition is gone - the products of masked conditionals do not multiply into a coherent p(sequence)
  - So an MLM cannot generate text by construction; it fills in blanks
  - And only the masked 15% of positions produce a training signal per pass, against 100% for the AR objective
- What you get
  - Representations built from both directions at once - each position's vector summarises its full context
  - Which is exactly what you want when the goal is embeddings for a downstream task rather than generation

## Comparing the two objectives

<div class="table-wrap"><table>
  <caption>The same architecture and the same cross-entropy loss &mdash; the objective and the mask are the entire difference.</caption>
  <tr><th scope="col"></th><th scope="col">Autoregressive (GPT-style)</th><th scope="col">Masked (BERT-style)</th></tr>
  <tr><th scope="row">Predicts</th><td>the next token, from the left context</td><td>masked tokens, from both sides</td></tr>
  <tr><th scope="row">Attention mask</th><td>causal (chapter 21)</td><td>none - fully bidirectional</td></tr>
  <tr><th scope="row">Models p(sequence)?</th><td>yes, exactly, via the chain rule</td><td>no - conditionals don't assemble into a joint</td></tr>
  <tr><th scope="row">Training signal per pass</th><td>every position</td><td>the masked ~15%</td></tr>
  <tr><th scope="row">Can generate?</th><td>natively, one token at a time</td><td>not directly</td></tr>
  <tr><th scope="row">Strongest at</th><td>generation: writing, code, dialogue</td><td>representations: embeddings, classification, retrieval</td></tr>
  <tr><th scope="row">Examples</th><td>GPT, Claude, Gemini, Llama</td><td>BERT, RoBERTa, ESM (chapter 25)</td></tr>
</table></div>

- Why the assistants are all autoregressive
  - Assistants must generate, and AR models are exact generative models of the sequence distribution
  - The denser training signal also pays at scale
- Why MLMs did not disappear
  - When the product is an embedding - search, similarity, features for a small downstream model (chapter 10's workflow) - bidirectional context wins
  - Protein models stayed masked for exactly this reason, plus one more: a protein is not written left to right, so there is no natural generation order to exploit - chapter 25

## Generation, temperature and sampling

- Generation from an AR model is the chain rule run forwards
  - Predict a distribution, pick a token, append it, repeat
  - Each chosen token becomes context for the next prediction
- How to pick from the distribution
  - Greedy - always take the argmax. Deterministic, and often repetitive and dull
  - Sampling - draw from p. Faithful to the model, but its rare-token tail produces occasional nonsense
  - Temperature - reshape the distribution before sampling

<div class="eqn">
<math display="block"><mrow><msub><mi>p</mi><mi>v</mi></msub><mo>&#x221D;</mo><msup><mi>e</mi><mrow><msub><mi>z</mi><mi>v</mi></msub><mo>/</mo><mi>&#x3C4;</mi></mrow></msup></mrow></math>
<span class="eqn-note">Divide the logits by a temperature &tau; before the softmax. &tau; &rarr; 0 approaches greedy; &tau; = 1 is the model's own distribution; &tau; &gt; 1 flattens it towards uniform.</span>
  </div>

::: code-eg
```
for tau in (0.5, 1.0, 2.0):
    pt = softmax(logits / tau)
    # tau=0.5: top prob 0.819  - sharpened, nearly greedy
    # tau=1.0: top prob 0.607  - the model as trained
    # tau=2.0: top prob 0.419  - flattened, more adventurous
```
<p class="eqn-note">The same five logits from above at three temperatures. One knob trades reliability against variety, with no retraining.</p>
:::

- In practice the tail is also cut before sampling (top-k or top-p), which removes most of the nonsense at little cost

## Measuring a language model: perplexity

<div class="eqn">
<math display="block"><mrow><mtext>PPL</mtext><mo>=</mo><mi>exp</mi><mo form="prefix">(</mo><mo>&#x2212;</mo><mfrac><mn>1</mn><mi>T</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover><mi>log</mi><mspace width="0.15em"/><msub><mi>p</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>w</mi><mi>t</mi></msub><mo>&#x2223;</mo><msub><mi>w</mi><mrow><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo><mo form="postfix">)</mo></mrow></math>
<span class="eqn-note">The exponential of the average loss. Interpretation: the model is, on average, as uncertain as if it were choosing uniformly between PPL tokens. A model that knows nothing about a 10-token vocabulary scores exactly 10; a good English model scores under 10 on a 100,000-token vocabulary.</span>
  </div>

- This is the single-real-number evaluation metric of chapter 11, for language models
  - Lower is better; 1 would be a model that is never surprised
- The chapter 10 warning applies with full force
  - Perplexity and benchmark scores are only meaningful on text the model has not trained on
  - When the training set is a crawl of the internet, guaranteeing that is genuinely hard - contamination is the field's version of evaluating on the training set

## Training at scale

- The optimization is the one we already know
  - Mini-batch gradient descent (chapter 17) on the cross-entropy loss, over trillions of tokens
  - In practice Adam (chapter 20) plus weight decay, which is chapter 07's regularization under its other name
  - Backpropagation through the transformer stack is chapter 09's algorithm; the residual connections exist to keep its gradients alive (chapter 21)
- Scaling laws
  - The loss falls as a power law in parameters N, data D and compute

<div class="eqn">
<math display="block"><mrow><mi>L</mi><mo stretchy="false">(</mo><mi>N</mi><mo stretchy="false">)</mo><mo>&#x2248;</mo><msup><mrow><mo stretchy="false">(</mo><mfrac><msub><mi>N</mi><mi>c</mi></msub><mi>N</mi></mfrac><mo stretchy="false">)</mo></mrow><msub><mi>&#x3B1;</mi><mi>N</mi></msub></msup></mrow></math>
<span class="eqn-note">An empirical regularity, not a theorem: straight lines on log-log plots over many orders of magnitude, with matching laws for data and compute. Predictable enough to budget a training run in advance &mdash; and the same fitting-a-curve-to-observations exercise as chapter 04's polynomial regression, applied to the models themselves.</span>
  </div>

- The practical corollary
  - For a fixed compute budget there is an optimal balance of model size and data - training a smaller model on more tokens often beats a bigger model trained short
  - Getting this trade-off right is a bias/variance argument (chapter 10) conducted with power laws
- From raw model to assistant
  - Everything above produces a next-token predictor, not a helpful conversationalist
  - The post-training stages - supervised fine-tuning, preference learning, reasoning training - are covered in chapter 20 and not repeated here

## Summary

- Factorize p(sequence) with the chain rule; each factor is multiclass classification over the vocabulary
- Autoregressive models predict the next token under a causal mask - exact generative models, every position a training example
- Masked models predict hidden tokens from both directions - better representations, no generation
- Both use the same loss: softmax plus cross-entropy, which is chapter 06's maximum-likelihood cost with |V| classes; the sigmoid is its two-class special case
- Temperature reshapes the output distribution at generation time; perplexity is the exponential of the loss, and chapter 10's held-out discipline decides whether it means anything
- Training is chapter 17's mini-batch descent with chapter 09's backpropagation, at a scale set by empirical power laws
- The through-line: nothing in this chapter required a new idea beyond attention - the 2011 toolkit, scaled up, is the modern language model
