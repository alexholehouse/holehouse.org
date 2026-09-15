---
title: "21: Attention"
nav_title: "21: Attention"
origin: "2026"
description: "Attention as a soft lookup, self-attention, positional encoding, multi-head attention, and the transformer block."
---

## A note on this chapter

- Everything up to chapter 19 is my write-up of Professor Ng's 2011 course - this chapter is not
- Attention post-dates the course by some years
  - The mechanism in the form used here comes from *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau et al., 2014) and *Attention Is All You Need* (Vaswani et al., 2017)
  - So there's no lecture behind this one, and the diagrams are my own
- I've added it because attention is the piece that connects these notes to essentially everything that has happened since
  - It builds directly on things covered earlier - logistic units, softmax, vectorization, learned weight matrices
  - If you've worked through chapters 08 and 09 you already have what you need

## Why attention?

- Say we want to map one sequence to another
  - Translate a French sentence to English
  - Summarize a paragraph
  - Predict the next word
- The obvious approach is an **<span class="term">encoder-decoder</span>**
  - Encoder reads the input sequence one step at a time, updating a hidden state
  - Final hidden state is a fixed-length vector - the "meaning" of the input
  - Decoder generates the output sequence from that vector

### The problem with this

- Everything has to squeeze through one fixed-length vector
  - This is the **<span class="hl-red">bottleneck</span>**
  - A 5 word sentence and a 50 word sentence get the same number of numbers to describe them
  - Performance degrades badly as sentences get longer - which is exactly what you'd expect
- Long-range dependencies get lost
  - If word 40 depends on word 2, that information has to survive 38 update steps
  - Signal decays, gradients vanish - the same problem we saw with deep networks in chapter 09, just stretched along time instead of depth
- It's inherently sequential
  - Step t needs step t-1, so you can't parallelize over the sequence
  - Given the whole point of vectorization (chapter 04) was to stop looping, this hurts

<figure class="diagram">
  <svg viewBox="0 0 620 186" role="img" aria-label="A fixed-length context vector bottleneck compared with attention, where the decoder reads every encoder state">
    <text class="lbl-sm" x="10" y="16">Without attention - one fixed vector carries everything</text>
    <rect class="box" x="14" y="28" width="52" height="30" rx="4"/>
    <rect class="box" x="76" y="28" width="52" height="30" rx="4"/>
    <rect class="box" x="138" y="28" width="52" height="30" rx="4"/>
    <rect class="box" x="200" y="28" width="52" height="30" rx="4"/>
    <text class="lbl" x="32" y="48">h1</text>
    <text class="lbl" x="94" y="48">h2</text>
    <text class="lbl" x="156" y="48">h3</text>
    <text class="lbl" x="218" y="48">h4</text>
    <path class="arrow" d="M252 43 L300 43"/>
    <polygon class="arrow-head" points="300,43 293,39 293,47"/>
    <rect class="box-accent" x="304" y="28" width="46" height="30" rx="4"/>
    <text class="lbl-key" x="313" y="48">c</text>
    <path class="arrow" d="M350 43 L398 43"/>
    <polygon class="arrow-head" points="398,43 391,39 391,47"/>
    <rect class="box" x="402" y="28" width="52" height="30" rx="4"/>
    <text class="lbl" x="418" y="48">dec</text>
    <text class="lbl-sm" x="462" y="47">everything</text>
    <text class="lbl-sm" x="462" y="60">via one c</text>

    <text class="lbl-sm" x="10" y="100">With attention - the decoder looks back at every state</text>
    <rect class="box" x="14" y="140" width="52" height="30" rx="4"/>
    <rect class="box" x="76" y="140" width="52" height="30" rx="4"/>
    <rect class="box" x="138" y="140" width="52" height="30" rx="4"/>
    <rect class="box" x="200" y="140" width="52" height="30" rx="4"/>
    <text class="lbl" x="32" y="160">h1</text>
    <text class="lbl" x="94" y="160">h2</text>
    <text class="lbl" x="156" y="160">h3</text>
    <text class="lbl" x="218" y="160">h4</text>
    <path class="arrow" d="M40 140 C 130 118, 350 118, 414 146"/>
    <path class="arrow" d="M102 140 C 180 122, 356 122, 416 147"/>
    <path class="arrow" d="M164 140 C 220 126, 362 126, 418 148"/>
    <path class="arrow" d="M226 140 C 264 130, 368 130, 420 149"/>
    <polygon class="arrow-head" points="422,149 413,145 414,154"/>
    <rect class="box" x="402" y="140" width="52" height="30" rx="4"/>
    <text class="lbl" x="418" y="160">dec</text>
    <text class="lbl-sm" x="462" y="152">weights the</text>
    <text class="lbl-sm" x="462" y="165">states it needs</text>
  </svg>
  <figcaption>The bottleneck, and what attention replaces it with.</figcaption>
</figure>

- The fix is almost embarrassingly simple
  - Don't force everything through one vector
  - Keep *all* the encoder states around
  - At each output step, let the decoder decide which of them to look at
- That "decide which to look at, and by how much" is **<span class="term">attention</span>**

## Attention as a soft lookup

- The cleanest way to think about attention is as a dictionary lookup that has been made differentiable
- An ordinary lookup
  - You have a set of **<span class="term">key</span>**-**<span class="term">value</span>** pairs
  - You supply a **<span class="term">query</span>**
  - You find the key that matches, and return its value
  - This is a **<span class="hl-red">hard</span>** lookup - one key wins, everything else returns nothing
- The problem with a hard lookup is that it isn't differentiable
  - "Which key matched" is a step function of the query
  - Derivative is zero everywhere and undefined at the jump - no gradient, nothing to learn from
- So we soften it
  - Score the query against *every* key - how well does each one match?
  - Turn those scores into weights that are positive and sum to 1 (softmax, exactly as in chapter 09)
  - Return the **<span class="term">weighted average</span>** of all the values
- Now everything is smooth
  - Nudge the query slightly, the weights shift slightly, the output shifts slightly
  - Which means we can backpropagate through it and learn what to attend to
- If one score is much larger than the rest, softmax puts nearly all the weight there
  - So a soft lookup can approximate a hard one when it wants to
  - We get the behaviour of a lookup and the trainability of a smooth function

<figure class="diagram">
  <svg viewBox="0 0 560 200" role="img" aria-label="A query is scored against each key, the scores are softmaxed into weights, and the values are averaged using those weights">
    <text class="lbl-key" x="14" y="100">query q</text>
    <path class="arrow" d="M76 96 C 110 96, 110 40, 146 40"/>
    <path class="arrow" d="M76 98 C 110 98, 110 76, 146 76"/>
    <path class="arrow" d="M76 100 C 110 100, 110 112, 146 112"/>
    <path class="arrow" d="M76 102 C 110 102, 110 148, 146 148"/>
    <rect class="box" x="150" y="26" width="46" height="28" rx="4"/>
    <rect class="box" x="150" y="62" width="46" height="28" rx="4"/>
    <rect class="box" x="150" y="98" width="46" height="28" rx="4"/>
    <rect class="box" x="150" y="134" width="46" height="28" rx="4"/>
    <text class="lbl" x="166" y="45">k1</text>
    <text class="lbl" x="166" y="81">k2</text>
    <text class="lbl" x="166" y="117">k3</text>
    <text class="lbl" x="166" y="153">k4</text>
    <text class="lbl-sm" x="206" y="45">score</text>
    <text class="lbl-sm" x="206" y="81">score</text>
    <text class="lbl-sm" x="206" y="117">score</text>
    <text class="lbl-sm" x="206" y="153">score</text>
    <path class="arrow" d="M246 40 L286 40"/>
    <path class="arrow" d="M246 76 L286 76"/>
    <path class="arrow" d="M246 112 L286 112"/>
    <path class="arrow" d="M246 148 L286 148"/>
    <text class="lbl-key" x="252" y="16">softmax</text>
    <text class="lbl-key" x="292" y="45">0.7</text>
    <text class="lbl" x="292" y="81">0.2</text>
    <text class="lbl" x="292" y="117">0.05</text>
    <text class="lbl" x="292" y="153">0.05</text>
    <text class="lbl-sm" x="286" y="178">sum to 1</text>
    <rect class="box" x="344" y="26" width="46" height="28" rx="4"/>
    <rect class="box" x="344" y="62" width="46" height="28" rx="4"/>
    <rect class="box" x="344" y="98" width="46" height="28" rx="4"/>
    <rect class="box" x="344" y="134" width="46" height="28" rx="4"/>
    <text class="lbl" x="360" y="45">v1</text>
    <text class="lbl" x="360" y="81">v2</text>
    <text class="lbl" x="360" y="117">v3</text>
    <text class="lbl" x="360" y="153">v4</text>
    <path class="arrow" d="M390 40 C 430 40, 430 92, 464 96"/>
    <path class="arrow" d="M390 76 C 430 76, 430 94, 464 98"/>
    <path class="arrow" d="M390 112 C 430 112, 430 100, 464 100"/>
    <path class="arrow" d="M390 148 C 430 148, 430 104, 464 102"/>
    <polygon class="arrow-head" points="468,99 459,95 460,104"/>
    <text class="lbl-key" x="474" y="103">output</text>
  </svg>
  <figcaption>Score against every key, softmax the scores, average the values.</figcaption>
</figure>

## Scaled dot-product attention

- Now we need to pick an actual scoring function
- Simplest sensible choice - the dot product
  - score(q, k<sub>i</sub>) = q · k<sub>i</sub>
  - Large when the two vectors point the same way, near zero when orthogonal
  - Which is exactly the notion of "similarity" we want
  - And it's a matrix multiply, so it's fast - no extra parameters at all

### Why divide by the square root of d<sub>k</sub>?

- There's a problem with raw dot products as vectors get longer
  - Suppose the components of q and k are independent with mean 0 and variance 1
  - Their dot product is a sum of d<sub>k</sub> such products
  - So it has mean 0 and variance d<sub>k</sub> - i.e. typical magnitude grows like √d<sub>k</sub>
- With d<sub>k</sub> = 64 the scores are routinely in the tens
  - Feed those into softmax and it **<span class="hl-red">saturates</span>** - one weight goes to ~1, the rest to ~0
  - In the flat region the gradient is vanishingly small
  - Same problem as a saturated sigmoid back in chapter 06, for the same reason
- So divide the scores by √d<sub>k</sub>
  - Puts the variance back to about 1 regardless of dimension
  - Keeps softmax in the part of its range where it has useful gradients
  - It's a small detail that matters a lot in practice

### The whole thing in matrix form

- Stack the queries, keys and values as rows of matrices - Q, K and V
  - Q is \[n x d<sub>k</sub>\] for n queries
  - K is \[m x d<sub>k</sub>\], V is \[m x d<sub>v</sub>\] for m key-value pairs
- Then the entire operation for every query at once is

<pre>Attention(Q, K, V) = softmax( QK<sup>T</sup> / &radic;d<sub>k</sub> ) V</pre>

- Worth walking through the shapes, since this is where it clicks
  - QK<sup>T</sup> is \[n x m\] - every query scored against every key
  - Divide by √d<sub>k</sub>, then softmax *along each row* - so each query's weights sum to 1
  - Multiply by V, which is \[m x d<sub>v</sub>\] - gives \[n x d<sub>v</sub>\]
  - One output vector per query, each a weighted average of the values
- Note there are no learned parameters in this function
  - It's two matrix multiplies and a softmax
  - All the learning lives in how Q, K and V get produced - which is next

<figure class="diagram">
  <svg viewBox="0 0 600 120" role="img" aria-label="Data flow of scaled dot-product attention: Q and K are multiplied, scaled, masked, softmaxed, then multiplied by V">
    <text class="lbl-key" x="14" y="64">Q, K</text>
    <path class="arrow" d="M58 60 L92 60"/>
    <polygon class="arrow-head" points="92,60 85,56 85,64"/>
    <rect class="box" x="96" y="44" width="66" height="32" rx="4"/>
    <text class="lbl" x="110" y="64">QK</text>
    <text class="lbl-sm" x="136" y="59">T</text>
    <path class="arrow" d="M162 60 L192 60"/>
    <polygon class="arrow-head" points="192,60 185,56 185,64"/>
    <rect class="box" x="196" y="44" width="60" height="32" rx="4"/>
    <text class="lbl" x="206" y="64">&divide; &radic;d</text>
    <text class="lbl-sm" x="240" y="68">k</text>
    <path class="arrow" d="M256 60 L286 60"/>
    <polygon class="arrow-head" points="286,60 279,56 279,64"/>
    <rect class="box" x="290" y="44" width="56" height="32" rx="4"/>
    <text class="lbl" x="300" y="64">mask</text>
    <text class="lbl-sm" x="292" y="94">optional</text>
    <path class="arrow" d="M346 60 L376 60"/>
    <polygon class="arrow-head" points="376,60 369,56 369,64"/>
    <rect class="box-accent" x="380" y="44" width="72" height="32" rx="4"/>
    <text class="lbl-key" x="390" y="64">softmax</text>
    <path class="arrow" d="M452 60 L482 60"/>
    <polygon class="arrow-head" points="482,60 475,56 475,64"/>
    <rect class="box" x="486" y="44" width="52" height="32" rx="4"/>
    <text class="lbl" x="502" y="64">&times; V</text>
    <path class="arrow" d="M538 60 L566 60"/>
    <polygon class="arrow-head" points="566,60 559,56 559,64"/>
    <text class="lbl-key" x="546" y="30">out</text>
  </svg>
  <figcaption>Scaled dot-product attention, end to end.</figcaption>
</figure>

## Self-attention

- So far Q, K and V were handed to us - in **<span class="term">self-attention</span>** the sequence produces all three itself
- Start with the sequence as a matrix X, one row per token
  - Learn three weight matrices - W<sup>Q</sup>, W<sup>K</sup> and W<sup>V</sup>
  - Q = X W<sup>Q</sup>
  - K = X W<sup>K</sup>
  - V = X W<sup>V</sup>
- Each token therefore emits three different vectors, and it's worth being clear about what each is for
  - **<span class="term">Query</span>** - what this token is looking for
  - **<span class="term">Key</span>** - what this token offers to others looking for something
  - **<span class="term">Value</span>** - what this token actually contributes if attended to
- Splitting key from value is the subtle part, and it's easy to skip past
  - What makes a token *findable* need not be what makes it *useful*
  - A pronoun might match on grammatical role but contribute semantic content
  - Keeping them separate lets the model learn the two independently
- Then run scaled dot-product attention on those three
  - Every token attends to every token, including itself
  - Output is a new sequence, same length, where each position now mixes in information from wherever it found it useful
- Note what we've bought
  - Any position can reach any other in **<span class="hl-green">one step</span>** - no decay over 38 timesteps
  - Every position is computed independently, so the whole sequence goes through at once - fully parallel
  - Both of the RNN problems from the first section, gone

### Cross-attention

- Same machinery, different sources
  - Q comes from one sequence, K and V from another
  - This is the original translation use - decoder queries the encoder
- Self-attention is just the case where all three come from the same place

## Positional encoding

- There's a problem with what we just built, and it's a real one
  - Attention is a weighted sum over the whole sequence
  - Addition doesn't care about order
  - So shuffle the input tokens and you get the same outputs, just shuffled to match
- The mechanism is **<span class="hl-red">permutation equivariant</span>** - it has no idea what order anything came in
  - "the cat sat on the mat" and "the mat sat on the cat" are identical to it
  - Clearly not acceptable for language
- Fix is to put the position into the input
  - Build a vector that encodes "this is position i"
  - Add it to the token's embedding before any attention happens
  - Now two identical words at different positions have different representations, and the dot products can pick up on it
- Two broad approaches
  - **<span class="term">Learned</span>** - just an embedding table indexed by position, trained like any other parameter
  - **<span class="term">Fixed</span>** - sinusoids of different frequencies, which is what the original paper used
  - Sinusoids have the nice property that they're defined for any position, so they extend past the longest training sequence
- Modern systems mostly encode *relative* position instead, but the principle is the same - attention needs to be told about order, because it cannot work it out

## Masking

- Sometimes a query must not be allowed to see certain positions
- **<span class="term">Causal masking</span>** - for generation
  - When predicting token t we may only use tokens 1 to t
  - Otherwise the model can read the answer off the input - it learns nothing and fails completely at generation time when the future isn't there
  - This is the same trap as evaluating on your training set (chapter 10), just hidden inside the architecture
- **<span class="term">Padding masking</span>** - for batching
  - Sequences in a batch have different lengths, so short ones get padded
  - Padding is meaningless and must not be attended to
- Implementation is a neat trick
  - Set the disallowed scores to -∞ *before* the softmax
  - e<sup>-&infin;</sup> = 0, so those positions get exactly zero weight
  - And the remaining weights still sum to 1 automatically, because softmax normalizes over what's left
  - No renormalization step needed - it falls out

<figure class="diagram">
  <svg viewBox="0 0 420 190" role="img" aria-label="A causal attention mask shown as a lower triangular grid, where each query may only attend to positions at or before itself">
    <text class="lbl-sm" x="96" y="18">keys (what we attend to)</text>
    <text class="lbl-sm" x="14" y="40">queries</text>
    <rect class="cell-on" x="96" y="30" width="30" height="30"/>
    <rect class="cell-off" x="128" y="30" width="30" height="30"/>
    <rect class="cell-off" x="160" y="30" width="30" height="30"/>
    <rect class="cell-off" x="192" y="30" width="30" height="30"/>
    <rect class="cell-on" x="96" y="62" width="30" height="30"/>
    <rect class="cell-on" x="128" y="62" width="30" height="30"/>
    <rect class="cell-off" x="160" y="62" width="30" height="30"/>
    <rect class="cell-off" x="192" y="62" width="30" height="30"/>
    <rect class="cell-on" x="96" y="94" width="30" height="30"/>
    <rect class="cell-on" x="128" y="94" width="30" height="30"/>
    <rect class="cell-on" x="160" y="94" width="30" height="30"/>
    <rect class="cell-off" x="192" y="94" width="30" height="30"/>
    <rect class="cell-on" x="96" y="126" width="30" height="30"/>
    <rect class="cell-on" x="128" y="126" width="30" height="30"/>
    <rect class="cell-on" x="160" y="126" width="30" height="30"/>
    <rect class="cell-on" x="192" y="126" width="30" height="30"/>
    <text class="lbl-sm" x="78" y="50">1</text>
    <text class="lbl-sm" x="78" y="82">2</text>
    <text class="lbl-sm" x="78" y="114">3</text>
    <text class="lbl-sm" x="78" y="146">4</text>
    <text class="lbl-sm" x="106" y="174">1</text>
    <text class="lbl-sm" x="138" y="174">2</text>
    <text class="lbl-sm" x="170" y="174">3</text>
    <text class="lbl-sm" x="202" y="174">4</text>
    <rect class="cell-on" x="248" y="46" width="14" height="14"/>
    <text class="lbl-sm" x="270" y="58">allowed</text>
    <rect class="cell-off" x="248" y="76" width="14" height="14"/>
    <text class="lbl-sm" x="270" y="88">masked to -inf</text>
    <text class="lbl-sm" x="248" y="122">query 3 may use</text>
    <text class="lbl-sm" x="248" y="136">keys 1, 2 and 3</text>
    <text class="lbl-sm" x="248" y="150">but never 4</text>
  </svg>
  <figcaption>A causal mask - each query sees only itself and what came before.</figcaption>
</figure>

## Multi-head attention

- One attention operation has a limitation
  - It produces one set of weights per query
  - So it can express one notion of relevance at a time
  - But a word may relate to different words for entirely different reasons - one syntactic, one semantic
  - Averaging those into a single weighting loses both
- So run several attention operations in parallel - each is a **<span class="term">head</span>**
  - Each head gets its own W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>
  - So each learns its own idea of what to look for
  - Concatenate the outputs and pass through one more learned matrix W<sup>O</sup>
- The cost is not what you might expect
  - With h heads, each head works in dimension d<sub>model</sub> / h rather than d<sub>model</sub>
  - e.g. d<sub>model</sub> = 512 with 8 heads gives 64 dimensions per head
  - Total computation is about the same as one full-width head
  - So we get several independent views essentially for free
- In a trained model heads do specialise, and visibly so
  - Some track syntactic relations, some resolve pronouns, some just attend to the previous token
  - Though it's easy to over-read this - plenty of heads do nothing interpretable at all, and many can be pruned with little loss

<figure class="diagram">
  <svg viewBox="0 0 560 170" role="img" aria-label="Multi-head attention splits the input into several heads, each running attention in a smaller dimension, then concatenates and projects the results">
    <rect class="box" x="14" y="62" width="48" height="34" rx="4"/>
    <text class="lbl" x="32" y="84">X</text>
    <path class="arrow" d="M62 79 C 90 79, 90 28, 126 28"/>
    <path class="arrow" d="M62 79 C 90 79, 90 62, 126 62"/>
    <path class="arrow" d="M62 79 C 90 79, 90 96, 126 96"/>
    <path class="arrow" d="M62 79 C 90 79, 90 132, 126 132"/>
    <rect class="box-accent" x="130" y="14" width="104" height="28" rx="4"/>
    <rect class="box-accent" x="130" y="48" width="104" height="28" rx="4"/>
    <rect class="box-accent" x="130" y="82" width="104" height="28" rx="4"/>
    <rect class="box-accent" x="130" y="118" width="104" height="28" rx="4"/>
    <text class="lbl-key" x="140" y="33">head 1</text>
    <text class="lbl-key" x="140" y="67">head 2</text>
    <text class="lbl-key" x="140" y="101">head 3</text>
    <text class="lbl-key" x="140" y="137">head h</text>
    <text class="lbl-sm" x="192" y="33">d/h</text>
    <text class="lbl-sm" x="192" y="67">d/h</text>
    <text class="lbl-sm" x="192" y="101">d/h</text>
    <text class="lbl-sm" x="192" y="137">d/h</text>
    <path class="arrow" d="M234 28 C 270 28, 270 72, 300 76"/>
    <path class="arrow" d="M234 62 C 270 62, 270 76, 300 78"/>
    <path class="arrow" d="M234 96 C 270 96, 270 82, 300 80"/>
    <path class="arrow" d="M234 132 C 270 132, 270 86, 300 82"/>
    <polygon class="arrow-head" points="304,79 295,75 296,84"/>
    <rect class="box" x="308" y="62" width="76" height="34" rx="4"/>
    <text class="lbl" x="318" y="84">concat</text>
    <path class="arrow" d="M384 79 L414 79"/>
    <polygon class="arrow-head" points="414,79 407,75 407,83"/>
    <rect class="box" x="418" y="62" width="60" height="34" rx="4"/>
    <text class="lbl" x="436" y="84">W</text>
    <text class="lbl-sm" x="452" y="78">O</text>
    <path class="arrow" d="M478 79 L508 79"/>
    <polygon class="arrow-head" points="508,79 501,75 501,83"/>
    <text class="lbl-key" x="512" y="84">out</text>
  </svg>
  <figcaption>Several heads, each in a smaller dimension, concatenated and projected.</figcaption>
</figure>

## Attention inside a transformer block

- Attention on its own isn't a network - it's one layer in a repeating block
- A block is
  - Multi-head self-attention
  - A position-wise feed-forward network - an ordinary two-layer net applied to each position separately, with the same weights
  - A residual connection around each of those two
  - Layer normalization
- Why each of the extra pieces is there
  - **<span class="term">Residual connections</span>** - output = x + f(x)
    - Gives gradients a direct path back through the whole stack
    - Without them, deep stacks simply don't train - same story as chapter 09
  - **<span class="term">Layer normalization</span>**
    - Keeps activations at a sane scale across the layer
    - Feature scaling from chapter 04, applied inside the network rather than to the inputs
    - The batch-wise sibling of this idea - batch normalization - is in chapter 20
  - **<span class="term">Feed-forward network</span>**
    - Attention is a weighted *average* - it moves information around but is linear in V
    - You need somewhere to actually transform it non-linearly, and this is that place
    - Usually widened by 4x internally, and it holds most of the parameters
- Stack these blocks and you have a transformer
  - Attention mixes information between positions
  - The feed-forward layer processes it at each position
  - Repeat - and essentially every large model today is this, made deeper and wider

## The cost of attention

- The catch is in the score matrix
  - QK<sup>T</sup> is \[n x n\] for a sequence of length n
  - So compute and memory both scale as **<span class="hl-red">O(n<sup>2</sup>)</span>**
  - Double the sequence length, quadruple the cost
- Which is fine at n = 512 and ruinous at n = 100,000
  - This is the single biggest constraint on how much context a model can take
  - It's the reason context length is a headline number people quote
- Broad approaches to it
  - **<span class="term">Sparse attention</span>** - don't attend to everything; use local windows, or a few global tokens
  - **<span class="term">Low-rank / kernel methods</span>** - approximate the softmax so you never form the n x n matrix
  - **<span class="term">IO-aware exact attention</span>** - keep the maths exact but never write the full matrix to slow memory (this is what FlashAttention does, and it's the one that's been most widely adopted)
- Worth noting the third one won not by approximating better but by taking the hardware seriously - a good reminder that the bottleneck isn't always where the maths says it is

## Summary

- Attention is a differentiable lookup
  - Score a query against every key, softmax to weights, average the values
  - Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V
- Self-attention has the sequence generate its own Q, K and V through learned matrices
  - Any position reaches any other in one step
  - The whole sequence computes in parallel
- Because it's order-blind, position has to be added to the input explicitly
- Masking with -∞ before the softmax restricts what a query may see, and keeps the weights normalized for free
- Multiple heads give several independent notions of relevance at roughly the cost of one
- The n<sup>2</sup> score matrix is the fundamental cost, and most of the engineering effort goes there
- The through-line from the rest of these notes
  - Softmax from chapter 09, feature scaling from chapter 04, vectorization from chapter 04, the vanishing gradient problem from chapter 09
  - None of the ingredients are new - what's new is arranging them so the model decides for itself what to look at
