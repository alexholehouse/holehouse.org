---
title: "20: Modern Deep Learning"
nav_title: "20: Modern deep learning"
origin: "2026"
description: "What changed after 2011: ReLU, initialization, Adam, dropout and batch normalization, and how today's assistants are trained."
---

## A note on this chapter

- Chapters 01 to 19 are my write-up of Professor Ng's 2011 course - this one isn't
- It's a high-level tour of what happened next
  - The course predates essentially all of it
  - But almost none of it is conceptually new - it's the same machinery at a scale nobody had tried yet
- Deliberately shallow
  - Enough to know what each thing is, roughly how it works, and what it's for
  - The details live in their own chapters now: attention (21), the mathematics of LLMs (22), diffusion models (23), AlphaFold2 (24) and protein language models (25)
    - This chapter is the map - plus the one set of details that belongs here: the training toolkit that separates 2011's networks from deep ones
- I've avoided version numbers throughout
  - They date within months, and the concepts don't

## What actually changed

- Four things arrived together, and none of them alone would have been enough
  - **<span class="term">Scale</span>** - models with billions to trillions of parameters, trained on a large fraction of the public internet
  - **<span class="term">Hardware</span>** - GPUs, then accelerators built specifically for this, making the matrix multiplies of chapter 03 fast enough to be practical at that size
  - **<span class="term">Architecture</span>** - the transformer (chapter 21), which unlike the recurrent models before it parallelizes across the whole sequence
  - **<span class="term">Training know-how</span>** - the numerical fixes that keep signals and gradients alive at depth: new activations, matched initialization, adaptive optimizers, normalization. The five sections below cover these properly - they are the bridge between chapter 09's networks and everything after
- The consequence that mattered most was a change in *how you use* a model
  - Old way - collect labelled data for your task, train a model for that task
  - New way - take a model someone else trained on an enormous unlabelled corpus, and adapt it
  - This is **<span class="term">transfer learning</span>**, and it's why a lab with no ML budget can now do serious ML
- Where the labels went
  - Labelled data was always the bottleneck - chapters 10 and 11 are largely about spending a scarce labelling budget well
  - The trick is to invent a label the data already contains: hide part of the input, predict it from the rest
  - Next word, masked word, noised image - the supervision comes free with the data
  - Usually called **<span class="term">self-supervised learning</span>**, and it's the single idea underneath everything below

## Activations: from sigmoid to ReLU

- The notes' networks (chapters 08 and 09) use sigmoid units throughout - and that is a large part of why they could not go deep
  - Backpropagation multiplies one derivative factor per layer (chapter 09)
  - The sigmoid's derivative is g′(z) = a(1 − a) - the identity chapter 09 verified numerically - and it is *at most* 1/4, at z = 0
  - Away from zero it saturates and the derivative is nearly nothing - the same flat-region problem chapter 21 met in softmax

<div class="eqn">
<math display="block"><mrow><msup><mi>g</mi><mo>&#x2032;</mo></msup><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><mi>g</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>&#x2264;</mo><mfrac><mn>1</mn><mn>4</mn></mfrac></mrow></math>
<span class="eqn-note">The best case for a sigmoid layer. Chain 30 of them and the gradient reaching the early layers carries a factor of at most (1/4)<sup>30</sup> &mdash; the <span class="term">vanishing gradient</span> problem. The early layers stop learning not because of a bug but because of arithmetic.</span>
  </div>

::: code-eg
```
import numpy as np

0.25 ** 30   # 8.7e-19 - the best-case gradient factor through 30 sigmoid layers
1.0  ** 30   # 1.0     - the same product for ReLU units on their active side
```
<p class="eqn-note">The whole story in two lines: sigmoid depth multiplies the gradient into oblivion; ReLU depth leaves it alone.</p>
:::

- The fix is almost embarrassingly blunt: the <span class="term">rectified linear unit</span>
  - g(z) = max(0, z) - identity for positive input, zero otherwise
  - Derivative is exactly 1 on the active side, exactly 0 on the inactive side - no saturation, no shrinking factor, and cheaper than an exponential
  - Not differentiable at exactly 0, which in practice matters not at all (pick either side)
- Its one failure mode, and the descendants
  - A unit whose input is always negative outputs 0 forever and its gradient is 0 forever - a <span class="term">dead unit</span>
  - Leaky ReLU gives the negative side a small slope so nothing can die; GELU (a smooth blend of the two regimes) is the default inside transformers (chapters 21-22)
  - All of them keep the property that matters: a derivative near 1 over most of the working range

## Initialization: keeping the signal alive

- Chapter 09 said: initialize to "small random values" - random for symmetry-breaking, which stands. But *how small* becomes critical with depth
  - Each layer multiplies the signal's variance by (number of inputs) × Var(w) - the same sum-of-independent-terms argument chapter 21 used to justify dividing by √d<sub>k</sub>
  - If that factor is below 1 the activations shrink geometrically; above 1 they explode geometrically. Forty layers turn "slightly off" into "gone"

<div class="eqn">
<math display="block"><mrow><mtext>Var</mtext><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>2</mn><msub><mi>n</mi><mtext>in</mtext></msub></mfrac></mrow></math>
<span class="eqn-note">He initialization for ReLU layers: variance 2 over the number of inputs, the 2 compensating for ReLU zeroing half its inputs. (Xavier initialization, 1/n<sub>in</sub>, is the same argument for symmetric activations.) Chosen so the layer's output variance equals its input variance &mdash; the multiply-per-layer factor is pinned at 1 by construction.</span>
  </div>

::: code-eg
```
rng = np.random.default_rng(0)
n = 256                                     # 256 units per layer

def depth_test(scale, act):
    x = rng.standard_normal(n)
    for _ in range(40):                     # forty layers deep
        W = rng.standard_normal((n, n)) * scale
        x = act(W @ x)
    return x.std()

relu = lambda z: np.maximum(0, z)

depth_test(0.01,              relu)   # 8e-39  - "small values": the signal is gone
depth_test(np.sqrt(2 / n),    relu)   # 0.27   - He init: still alive at layer 40
depth_test(3 * np.sqrt(2 / n), relu)  # 6e+18  - too big: exploded instead
```
<p class="eqn-note">Chapter 09's advice, stress-tested at depth. The window between vanishing and exploding is narrow, and the He formula puts you in it for any width automatically.</p>
:::

## Optimizers: momentum, RMSProp, Adam

- The problem, which these notes have already drawn
  - Chapter 04's contour plot: badly scaled features make long thin valleys, and gradient descent zig-zags across the steep direction while crawling along the shallow one
  - The learning rate is capped by the steepest direction, so the shallow direction sets the runtime
  - Feature scaling fixed this for the *inputs*; inside a deep network the same mismatch reappears at every layer, where you cannot hand-scale it away
- <span class="term">Momentum</span> - remember the direction you have been moving
  - Keep a running velocity; the zig-zag components cancel in the average, the consistent component accumulates

<div class="eqn">
<math display="block"><mrow><mi>v</mi><mo>:=</mo><mi>&#x3B2;</mi><mi>v</mi><mo>+</mo><mo>&#x2207;</mo><mi>J</mi><mo stretchy="false">(</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo><mo>,</mo><mspace width="1.2em"/><mi>&#x3B8;</mi><mo>:=</mo><mi>&#x3B8;</mi><mo>&#x2212;</mo><mi>&#x3B1;</mi><mi>v</mi></mrow></math>
<span class="eqn-note">&beta; &asymp; 0.9: each step is mostly the previous step plus a gradient nudge &mdash; a heavy ball rolling downhill rather than a walker re-deciding direction from scratch.</span>
  </div>

- <span class="term">RMSProp</span> - give every parameter its own learning rate
  - Track a running average of each parameter's squared gradient and divide by its square root - parameters with habitually large gradients get small steps, and vice versa

<div class="eqn">
<math display="block"><mrow><mi>s</mi><mo>:=</mo><mi>&#x3C1;</mi><mi>s</mi><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><mi>&#x3C1;</mi><mo stretchy="false">)</mo><msup><mrow><mo stretchy="false">(</mo><mo>&#x2207;</mo><mi>J</mi><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>,</mo><mspace width="1.2em"/><mi>&#x3B8;</mi><mo>:=</mo><mi>&#x3B8;</mi><mo>&#x2212;</mo><mfrac><mi>&#x3B1;</mi><mrow><msqrt><mi>s</mi></msqrt><mo>+</mo><mi>&#x3B5;</mi></mrow></mfrac><mo>&#x2207;</mo><mi>J</mi></mrow></math>
<span class="eqn-note">All operations element-wise. This is feature scaling (chapter 04) applied to the <em>gradient</em>, continuously, per parameter &mdash; the fix moved from preprocessing into the optimizer itself.</span>
  </div>

- <span class="term">Adam</span> - both at once, plus a correction
  - A momentum-style average m of the gradient, an RMSProp-style average v of its square, and a correction for the early steps when both averages are still warming up from zero

<div class="eqn">
<math display="block"><mrow><mi>m</mi><mo>:=</mo><msub><mi>&#x3B2;</mi><mn>1</mn></msub><mi>m</mi><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>&#x3B2;</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mo>&#x2207;</mo><mi>J</mi><mo>,</mo><mspace width="1.2em"/><mi>v</mi><mo>:=</mo><msub><mi>&#x3B2;</mi><mn>2</mn></msub><mi>v</mi><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>&#x3B2;</mi><mn>2</mn></msub><mo stretchy="false">)</mo><msup><mrow><mo stretchy="false">(</mo><mo>&#x2207;</mo><mi>J</mi><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
<math display="block"><mrow><mover><mi>m</mi><mo>^</mo></mover><mo>=</mo><mfrac><mi>m</mi><mrow><mn>1</mn><mo>&#x2212;</mo><msubsup><mi>&#x3B2;</mi><mn>1</mn><mi>t</mi></msubsup></mrow></mfrac><mo>,</mo><mspace width="1em"/><mover><mi>v</mi><mo>^</mo></mover><mo>=</mo><mfrac><mi>v</mi><mrow><mn>1</mn><mo>&#x2212;</mo><msubsup><mi>&#x3B2;</mi><mn>2</mn><mi>t</mi></msubsup></mrow></mfrac><mo>,</mo><mspace width="1em"/><mi>&#x3B8;</mi><mo>:=</mo><mi>&#x3B8;</mi><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mover><mi>m</mi><mo>^</mo></mover><mrow><msqrt><mover><mi>v</mi><mo>^</mo></mover></msqrt><mo>+</mo><mi>&#x3B5;</mi></mrow></mfrac></mrow></math>
<span class="eqn-note">Defaults &beta;<sub>1</sub> = 0.9, &beta;<sub>2</sub> = 0.999 barely ever change &mdash; which is the actual selling point: it works out of the box across wildly different problems, at the cost of one extra stored value per parameter for each of m and v. This is the optimizer behind chapters 22-25.</span>
  </div>

::: code-eg
```
def race(update, L, tol=1e-8, steps=100_000):
    theta, state = np.array([10.0, 1.0]), {}
    for k in range(1, steps + 1):          # J = theta1^2/2 + L*theta2^2/2
        grad = np.array([theta[0], L * theta[1]])
        theta = update(theta, grad, state, k)
        if theta[0]**2 / 2 + L * theta[1]**2 / 2 < tol:
            return k

def gd(t, g, s, k):                        # best stable learning rate
    return t - (2 / (L + 1)) * g

def momentum(t, g, s, k):
    s['v'] = (4/9) * s.get('v', 0) + g     # textbook-optimal beta for L = 25
    return t - (1/9) * s['v']

def adam(t, g, s, k):
    s['m'] = 0.9   * s.get('m', 0) + 0.1   * g
    s['v'] = 0.999 * s.get('v', 0) + 0.001 * g * g
    m_hat = s['m'] / (1 - 0.9 ** k)
    v_hat = s['v'] / (1 - 0.999 ** k)
    return t - 0.5 * m_hat / (np.sqrt(v_hat) + 1e-8)

L = 25                       # a mildly elongated bowl - chapter 04's contours
race(gd, 25)                 # 141 steps
race(momentum, 25)           #  37 steps - the sqrt(condition-number) speedup
race(adam, 25)               # 172 steps - unremarkable here...

L = 10_000                   # ...but make the scales wildly different and
race(gd, 10_000)             # 67,370 steps - capped by the steep axis
race(adam, 10_000)           #    220 steps - per-parameter scaling wins
```
<p class="eqn-note">An honest race. On a clean, mildly ill-conditioned bowl, well-tuned momentum is the fastest thing there is and Adam is nothing special. Blow the conditioning out to 10<sup>4</sup> &mdash; the un-scaled-features regime of chapter 04, which deep networks recreate internally &mdash; and Adam wins by 300&times; <em>without retuning anything</em>. Robustness, not raw speed, is why it became the default.</p>
:::

## Learning-rate schedules and warmup

- Chapter 02 argued no schedule is needed: near a minimum the gradient shrinks, so steps shrink themselves
  - True for batch descent on a smooth bowl. With mini-batches (chapter 17) it fails: gradient noise does not shrink as you converge, so a constant rate leaves you rattling around the minimum - chapter 17 already met this as SGD "wandering"
- <span class="term">Decay</span> - end low
  - Chapter 17's fix was α = c<sub>1</sub>/(t + c<sub>2</sub>); the modern default is cosine decay - a smooth run from the peak rate down to near zero over the planned training length
  - Same idea either way: big steps to cross the landscape early, small steps to settle late
- <span class="term">Warmup</span> - start low too
  - The first few hundred steps ramp the rate up from zero
  - Early on, Adam's running averages are built from a handful of noisy mini-batches, and a full-size step taken on garbage statistics can wreck the network before training starts - warmup lets the estimates settle first
- So the standard schedule is a ramp up then a long cosine down - and its *length* is a hyperparameter chosen on validation data, exactly the chapter 10 procedure

## Dropout and batch normalization

- <span class="term">Dropout</span> - regularization by sabotage
  - During training, independently zero each hidden unit with probability 1 − p (keep with probability p), a fresh coin flip every example
  - No unit can rely on a specific partner existing, so the network cannot build the brittle co-adapted features that overfitting (chapters 07 and 10) is made of
  - Equivalent view: you are training a huge ensemble of thinned networks that share weights, and averaging them at test time

<div class="eqn">
<math display="block"><mrow><mi>a</mi><mo>:=</mo><mfrac><mrow><mi>a</mi><mo>&#x2299;</mo><mi>mask</mi></mrow><mi>p</mi></mfrac></mrow></math>
<span class="eqn-note">Inverted dropout: divide by the keep-probability during training so the expected activation is unchanged &mdash; then test time needs no adjustment at all, just switch the mask off.</span>
  </div>

::: code-eg
```
a = np.ones(100_000) * 2.0            # an activation of 2.0, many trials
p = 0.8                               # keep 80% of units
mask = rng.random(100_000) < p
(a * mask / p).mean()                 # 2.0 - the expectation survives
```
<p class="eqn-note">The scaling checked: dropping a fifth of the units while dividing by 0.8 leaves the expected signal exactly where it was.</p>
:::

- <span class="term">Batch normalization</span> - feature scaling, moved inside the network
  - Chapter 04 standardized the *inputs* so gradient descent saw round contours. A deep network un-does that favour internally: each layer's inputs are the previous layer's outputs, with whatever mean and scale training has drifted them to
  - So re-standardize between layers, using the current mini-batch's own statistics

<div class="eqn">
<math display="block"><mrow><mover><mi>x</mi><mo>^</mo></mover><mo>=</mo><mfrac><mrow><mi>x</mi><mo>&#x2212;</mo><msub><mi>&#x3BC;</mi><mi>B</mi></msub></mrow><msqrt><msubsup><mi>&#x3C3;</mi><mi>B</mi><mn>2</mn></msubsup><mo>+</mo><mi>&#x3B5;</mi></msqrt></mfrac><mo>,</mo><mspace width="1.2em"/><mi>y</mi><mo>=</mo><mi>&#x3B3;</mi><mover><mi>x</mi><mo>^</mo></mover><mo>+</mo><mi>&#x3B2;</mi></mrow></math>
<span class="eqn-note">Chapter 04's mean-normalization formula, verbatim &mdash; &mu;<sub>B</sub> and &sigma;<sub>B</sub> are the mini-batch's mean and standard deviation &mdash; plus a learned scale &gamma; and shift &beta; so the network can undo the normalization wherever it turns out to be unhelpful. At test time the batch statistics are replaced by running averages kept during training.</span>
  </div>

- What it buys
  - Much higher usable learning rates and far less sensitivity to initialization - training that simply works where it used to diverge
  - A mild regularization side-effect, since each example's normalization depends on which batch-mates it drew
- Its sibling
  - Layer normalization (chapter 21) computes the same statistics per *example* across features instead of per feature across the batch - no batch dependence, which is why transformers use it
  - Batch norm rules convolutional vision models; layer norm rules sequence models. Same equation, different axis
- The recipe shift, side by side

<div class="table-wrap"><table>
  <caption>The 2011 recipe (chapters 06-09) against the modern one &mdash; every row is a small fix, and depth needs all of them at once.</caption>
  <tr><th scope="col"></th><th scope="col">These notes, 2011</th><th scope="col">Modern practice</th></tr>
  <tr><th scope="row">Activation</th><td>sigmoid / tanh</td><td>ReLU family (GELU in transformers)</td></tr>
  <tr><th scope="row">Initialization</th><td>"small random values"</td><td>He / Xavier - variance matched to width</td></tr>
  <tr><th scope="row">Optimizer</th><td>batch gradient descent, fminunc</td><td>Adam (or SGD + momentum) on mini-batches</td></tr>
  <tr><th scope="row">Learning rate</th><td>one constant &alpha;, chosen by plot</td><td>warmup then cosine decay</td></tr>
  <tr><th scope="row">Regularization</th><td>L2 penalty (chapter 07)</td><td>weight decay + dropout + early stopping + data augmentation (chapter 18)</td></tr>
  <tr><th scope="row">Normalization</th><td>inputs only (chapter 04)</td><td>batch/layer norm between layers</td></tr>
  <tr><th scope="row">Feasible depth</th><td>a few layers</td><td>hundreds</td></tr>
</table></div>

- Nothing in this section is conceptually deep, and that is the point
  - The 2011 course had the right objective, the right algorithm and the right architecture idea; what was missing was a handful of numerical fixes that keep signals and gradients alive at depth
  - Those fixes, plus the hardware and data above, are the gap between chapter 09's networks and everything in chapters 21-25

## Large language models

<p class="note">This section is the overview; chapter 22 does the mathematics properly.</p>

- The training objective is almost insultingly simple
  - Given a run of text, predict the next token
  - A **<span class="term">token</span>** is roughly a word-piece - common words are one token, rare ones split into several
  - That's it. Softmax over the vocabulary, cross-entropy loss - the multi-class classification of chapter 09, with a vocabulary of maybe 100,000 classes
- Why that produces something so much more general than it sounds
  - To predict the next token well across the whole internet, you have to model whatever generated it
  - Finishing "the capital of France is" needs a fact; finishing a proof needs the argument; finishing a function body needs the code to typecheck
  - So syntax, facts, reasoning patterns and style all fall out of one objective, because all of them reduce prediction error
- **<span class="term">Scaling laws</span>** - the empirical finding that drove the whole build-out
  - Loss falls predictably as a power law in model size, data and compute
  - Predictably enough to plan a training run before doing it, which is what made the capital expenditure defensible
  - Note this is an empirical regularity over many orders of magnitude, not a theorem - it holds until it doesn't
- **<span class="term">In-context learning</span>** - the surprise
  - Put a few worked examples in the prompt and the model does the task, with no gradient step and no weight change
  - Nobody trained for this - it emerged from scale
  - Practically it means the interface to the model is *text*, not a training pipeline

### What they're used for

- Writing, editing, summarizing, translating
- Code - generation, review, refactoring, explanation; probably the strongest commercial use
- Extraction - pulling structure out of unstructured text, which used to be a bespoke NLP project each time
- Classification with no training set, by simply describing the classes
- An interface layer - natural language over an API, a database, or a pile of documents

## How a raw model becomes an assistant

- A model straight out of pretraining is not a chatbot
  - It continues text. Ask it a question and a plausible continuation is *a list of similar questions*
  - It's a model of the corpus, not an assistant - being helpful was never the objective
- So there's a second stage, usually called **<span class="term">post-training</span>**
  - **<span class="term">Supervised fine-tuning</span>**
    - Continue training on curated examples of instructions and good responses
    - Ordinary supervised learning - the model learns the *shape* of being asked and answering
  - **<span class="term">Learning from preferences</span>**
    - Show humans two candidate responses, ask which is better
    - Train a **<span class="term">reward model</span>** to predict that judgement, then optimize the model against it - this is RLHF
    - Why preferences rather than labels: "which of these is better" is a question people can answer reliably, "write the ideal response" is not
  - **<span class="term">Reasoning training</span>**
    - More recent, and the reason for the recent step-change on maths and code
    - Reward the model for reaching a verifiably correct answer, letting it work at length first
    - The model learns to spend more computation on harder problems - test-time compute becomes a dial you can turn
- **<span class="hl-red">Alignment</span>** is the open problem here
  - The objective is a proxy for what we want, and optimizing hard against a proxy is how you get a model that games it
  - Same failure mode as any badly-chosen cost function, with more consequences

<figure class="diagram">
  <svg viewBox="0 0 620 130" role="img" aria-label="Pipeline from a large unlabelled corpus through pretraining, supervised fine-tuning, and preference or reasoning training to a deployed assistant">
    <rect class="box" x="10" y="44" width="94" height="40" rx="4"/>
    <text class="lbl" x="24" y="62">internet-</text>
    <text class="lbl" x="24" y="76">scale text</text>
    <path class="arrow" d="M104 64 L136 64"/>
    <polygon class="arrow-head" points="136,64 129,60 129,68"/>
    <rect class="box-accent" x="140" y="44" width="118" height="40" rx="4"/>
    <text class="lbl-key" x="152" y="62">pretraining</text>
    <text class="lbl-sm" x="152" y="77">predict next token</text>
    <path class="arrow" d="M258 64 L276 64"/>
    <polygon class="arrow-head" points="276,64 269,60 269,68"/>
    <rect class="box" x="280" y="44" width="96" height="40" rx="4"/>
    <text class="lbl" x="292" y="62">fine-tuning</text>
    <text class="lbl-sm" x="292" y="77">curated examples</text>
    <path class="arrow" d="M376 64 L408 64"/>
    <polygon class="arrow-head" points="408,64 401,60 401,68"/>
    <rect class="box" x="412" y="44" width="110" height="40" rx="4"/>
    <text class="lbl" x="424" y="62">preferences /</text>
    <text class="lbl" x="424" y="77">reasoning</text>
    <path class="arrow" d="M522 64 L554 64"/>
    <polygon class="arrow-head" points="554,64 547,60 547,68"/>
    <text class="lbl-key" x="558" y="68">assistant</text>
    <text class="lbl-sm" x="140" y="30">months, enormous cost</text>
    <text class="lbl-sm" x="300" y="112">comparatively cheap - this is where behaviour is shaped</text>
  </svg>
  <figcaption>Almost all the capability comes from stage one; almost all the behaviour from the stages after it.</figcaption>
</figure>

## The assistants - ChatGPT, Claude, Gemini {#the-assistants}

- All three are the same recipe - a large transformer, pretrained on text, then post-trained into an assistant
  - They differ in training data, in post-training method and emphasis, and in the surrounding product
  - Not in any deep architectural sense that would matter to this chapter
- **<span class="term">ChatGPT</span>** (OpenAI)
  - The one that made this public in late 2022, built on the GPT model family
  - Its real contribution was the interface - the underlying capability existed before, but nobody had put a chat box on it
- **<span class="term">Claude</span>** (Anthropic)
  - Model families named Opus, Sonnet and Haiku - roughly most capable, balanced, and fastest
  - Notable for **<span class="term">Constitutional AI</span>** - the model critiques and revises its own outputs against an explicit written set of principles, so some of the human feedback loop is replaced by AI feedback against a stated standard
  - Strong on long-context work and on code
- **<span class="term">Gemini</span>** (Google DeepMind)
  - Natively multimodal - text, images, audio and video handled by one model rather than bolted together
  - Long context windows, and tight integration with Google's own products
- Two things worth understanding about all of them
  - **<span class="term">Context window</span>** - how much text the model can attend to at once. This is the practical constraint you feel most often, and it's bounded by the O(n<sup>2</sup>) cost of attention (chapter 21)
  - **<span class="term">Multimodality</span>** - images, audio and video get turned into token sequences too, so the same machinery applies. Nothing conceptually new; the pipeline just has more front ends

## Agentic workflows

- The shift from "model that answers" to "model that does"
  - Give the model a set of **<span class="term">tools</span>** - run code, search, read a file, call an API
  - It emits a structured call, your code executes it, the result goes back into the context, and it continues
  - Loop until the task is done
- Why this is more than a convenience
  - It closes the loop with reality - the model can now check its own work rather than assert it
  - It fixes the things a language model is inherently bad at, by delegating them: arithmetic to a calculator, current facts to a search, correctness to a test suite
  - And it makes the work incremental - errors surface at step three rather than at the end
- The pieces you'll hear named
  - **<span class="term">Tool use</span>** / function calling - the model outputs a structured call rather than prose
  - **<span class="term">RAG</span>** (retrieval-augmented generation) - fetch relevant documents, put them in the context, answer from them. Grounds answers in a source you control, and gets around the context window for large corpora
  - **<span class="term">MCP</span>** (Model Context Protocol) - an open standard for how tools and data sources describe themselves to models, so integrations aren't rebuilt per vendor
  - **<span class="term">Sub-agents</span>** - delegating a self-contained piece of work to a separate context, which keeps the main one small
- What they're used for
  - Coding agents that read a repository, make changes and run the tests
  - Research - search, read, cross-check, synthesize
  - Data work - write the query, run it, plot the result, notice it looks wrong, fix it
  - Anything that was a script you'd have written by hand, where the steps aren't known in advance
- **<span class="hl-red">The failure mode to design around</span>**
  - Errors compound - a 95% reliable step is 60% reliable after ten of them
  - So the engineering is mostly about verification, recovery and keeping the human in the loop where it matters, not about the prompt

<figure class="diagram">
  <svg viewBox="0 0 580 170" role="img" aria-label="The agent loop: the model chooses a tool, the tool executes, its result returns to the context, and the loop repeats until the task is done">
    <rect class="box" x="12" y="62" width="70" height="36" rx="4"/>
    <text class="lbl" x="34" y="84">task</text>
    <path class="arrow" d="M82 80 L114 80"/>
    <polygon class="arrow-head" points="114,80 107,76 107,84"/>
    <rect class="box-accent" x="118" y="58" width="96" height="44" rx="4"/>
    <text class="lbl-key" x="140" y="76">model</text>
    <text class="lbl-sm" x="130" y="92">picks a tool</text>
    <path class="arrow" d="M214 80 L262 80"/>
    <polygon class="arrow-head" points="262,80 255,76 255,84"/>
    <rect class="box" x="266" y="58" width="104" height="44" rx="4"/>
    <text class="lbl" x="278" y="76">run code /</text>
    <text class="lbl" x="278" y="92">search / read</text>
    <path class="arrow" d="M318 102 C 318 140, 166 140, 166 106"/>
    <polygon class="arrow-head" points="166,102 162,111 171,111"/>
    <text class="lbl-sm" x="196" y="155">result returns to the context</text>
    <path class="arrow" d="M370 80 L414 80"/>
    <polygon class="arrow-head" points="414,80 407,76 407,84"/>
    <rect class="box" x="418" y="58" width="60" height="44" rx="4"/>
    <text class="lbl" x="430" y="84">done?</text>
    <path class="arrow" d="M478 80 L506 80"/>
    <polygon class="arrow-head" points="506,80 499,76 499,84"/>
    <text class="lbl-key" x="510" y="84">answer</text>
  </svg>
  <figcaption>The loop - and every pass through it is a chance to catch an error, or to compound one.</figcaption>
</figure>

## What to be sceptical about

- **<span class="hl-red">Evaluation is much harder than it looks</span>**
  - Chapter 10's discipline - train/validation/test, held out properly - matters more than ever, and is followed less
  - When the training set is "the internet", your test set is probably in it. Benchmark contamination is pervasive and often undetectable
  - In biology the leak is subtler: random splits of sequence data put homologues on both sides, so you measure memorization and call it generalization
- Fluency is not correctness
  - These models are optimized to produce plausible continuations, and a confident wrong answer is exactly as fluent as a right one
  - There's no internal signal that separates the two for you - hence tools, retrieval and verification
- Bias in, bias out - unchanged since chapter 11, just at larger scale and harder to inspect
- Cost and access
  - Pretraining a frontier model is out of reach for almost everyone
  - But *using* one is cheap, and fine-tuning an open one is very achievable - which is the practically important fact

## Summary

- One idea underlies nearly all of it
  - Invent a supervised task the unlabelled data answers for itself, train an enormous model on it, then adapt
  - Next token for text, masked residue for proteins, added noise for images
- **<span class="term">LLMs</span>** - next-token prediction at scale, post-trained into assistants; used for text, code, extraction, and as a natural-language interface to everything else
- **<span class="term">Agentic workflows</span>** - give the model tools and a loop, so it can act and check rather than only answer
- **<span class="term">Diffusion</span>** - learn to remove a little noise, run it backwards to generate; now chapter 23, with a working NumPy model
- **<span class="term">AlphaFold2</span>** - sequence to structure via evolutionary covariation; now chapter 24, piece by piece
- **<span class="term">ESM</span>** - masked language modelling on protein sequences; now chapter 25, ESM2 and ESM-C compared
- What carries over from the rest of these notes
  - Everything. Gradient descent, regularization, bias and variance, the train/validation/test discipline, feature scaling, softmax, evaluation metrics for skewed data
  - The models got much bigger; the failure modes are the ones in chapters 07, 10 and 11
  - The one genuinely new mechanism is attention (chapter 21) - and chapters 22-25 show how far it travels
