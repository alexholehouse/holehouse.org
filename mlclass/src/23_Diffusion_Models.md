---
title: "23: Diffusion Models"
nav_title: "23: Diffusion models"
origin: "2026"
description: "Destroying data with noise and learning to reverse it, with a complete diffusion model in NumPy."
---

## A note on this chapter

- An addition, like chapters 20-22 - the course predates all of this
  - Chapter 20 used to hold a short survey of diffusion; this chapter replaces it with the actual mathematics and a working model
- Everything needed is already in these notes
  - Gaussians from chapter 15, squared-error cost from chapter 02, a small neural network from chapter 08 trained with chapter 09's backpropagation, mini-batch descent from chapter 17
  - The NumPy model below is genuinely nothing but those pieces

## The idea

- The dominant approach to generating images, video and - increasingly - molecular structures
- Two processes, one fixed and one learned
  - <span class="term">Forward process</span> - take a real example and add a little Gaussian noise. Repeat a few hundred times. You end up with pure noise, and it's destroyed the same way every time
  - <span class="term">Reverse process</span> - train a network to undo one step of that: given a noisy example, predict the noise that was added
  - Then generate by starting from pure noise and running the learned reverse step repeatedly, until a sample falls out

<figure class="diagram">
  <svg viewBox="0 0 560 140" role="img" aria-label="The forward diffusion process adds noise to an image step by step; the learned reverse process removes it">
    <rect class="box" x="14" y="46" width="60" height="46" rx="4"/>
    <text class="lbl" x="30" y="74">image</text>
    <rect class="box" x="140" y="46" width="60" height="46" rx="4"/>
    <text class="lbl-sm" x="152" y="74">noisier</text>
    <rect class="box" x="266" y="46" width="60" height="46" rx="4"/>
    <text class="lbl-sm" x="278" y="74">noisier</text>
    <rect class="box" x="392" y="46" width="60" height="46" rx="4"/>
    <text class="lbl" x="404" y="74">noise</text>
    <path class="arrow" d="M74 58 L136 58"/>
    <polygon class="arrow-head" points="136,58 129,54 129,62"/>
    <path class="arrow" d="M200 58 L262 58"/>
    <polygon class="arrow-head" points="262,58 255,54 255,62"/>
    <path class="arrow" d="M326 58 L388 58"/>
    <polygon class="arrow-head" points="388,58 381,54 381,62"/>
    <text class="lbl-sm" x="150" y="34">forward - add noise (fixed, not learned)</text>
    <path class="arrow" d="M388 82 L330 82"/>
    <polygon class="arrow-head" points="330,82 337,78 337,86"/>
    <path class="arrow" d="M262 82 L204 82"/>
    <polygon class="arrow-head" points="204,82 211,78 211,86"/>
    <path class="arrow" d="M136 82 L78 82"/>
    <polygon class="arrow-head" points="78,82 85,78 85,86"/>
    <text class="lbl-key" x="150" y="118">reverse - learned denoiser, run to generate</text>
  </svg>
  <figcaption>Destroy structure in small fixed steps; learn to undo one of them; run it backwards to generate.</figcaption>
</figure>

- Why go the long way round
  - Going from noise to a photograph in one jump is a very hard function to learn
  - Going from slightly-noisier to slightly-less-noisy is an easy one - and you can compose it as many times as you like
  - The generation is decomposed into many small, individually-easy steps. That's the whole trick

## The forward process

- Fix a <span class="term">noise schedule</span> - a sequence of small variances β<sub>1</sub>, …, β<sub>T</sub>
  - Each step shrinks the signal slightly and adds a little Gaussian noise

<div class="eqn">
<math display="block"><mrow><mi>q</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo>&#x2223;</mo><msub><mi>x</mi><mrow><mi>t</mi><mo>&#x2212;</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo><mo>=</mo><mi>N</mi><mo form="prefix">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo>;</mo><mspace width="0.2em"/><msqrt><mn>1</mn><mo>&#x2212;</mo><msub><mi>&#x3B2;</mi><mi>t</mi></msub></msqrt><mspace width="0.15em"/><msub><mi>x</mi><mrow><mi>t</mi><mo>&#x2212;</mo><mn>1</mn></mrow></msub><mo>,</mo><mspace width="0.2em"/><msub><mi>&#x3B2;</mi><mi>t</mi></msub><mi>I</mi><mo form="postfix">)</mo></mrow></math>
<span class="eqn-note">One forward step: a Gaussian (chapter 15) centred just below the current value, with variance &beta;<sub>t</sub>. Nothing is learned here &mdash; the destruction is by design.</span>
  </div>

- The property that makes training practical
  - A chain of Gaussians is a Gaussian, so you can jump straight from x<sub>0</sub> to any step t in closed form - no need to simulate the chain

<div class="eqn">
<math display="block"><mrow><msub><mi>x</mi><mi>t</mi></msub><mo>=</mo><msqrt><msub><mover><mi>&#x3B1;</mi><mo>&#x00AF;</mo></mover><mi>t</mi></msub></msqrt><mspace width="0.15em"/><msub><mi>x</mi><mn>0</mn></msub><mo>+</mo><msqrt><mn>1</mn><mo>&#x2212;</mo><msub><mover><mi>&#x3B1;</mi><mo>&#x00AF;</mo></mover><mi>t</mi></msub></msqrt><mspace width="0.15em"/><mi>&#x3B5;</mi><mo>,</mo><mspace width="1em"/><mi>&#x3B5;</mi><mo>&#x223C;</mo><mi>N</mi><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mi>I</mi><mo stretchy="false">)</mo></mrow></math>
<math display="block"><mrow><msub><mover><mi>&#x3B1;</mi><mo>&#x00AF;</mo></mover><mi>t</mi></msub><mo>=</mo><munderover><mo>&#x220F;</mo><mrow><mi>s</mi><mo>=</mo><mn>1</mn></mrow><mi>t</mi></munderover><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>&#x3B2;</mi><mi>s</mi></msub><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">&alpha;&#x304;<sub>t</sub> is the surviving fraction of the original signal. It starts near 1 and decays towards 0, so x<sub>T</sub> is (nearly) pure standard Gaussian noise whatever x<sub>0</sub> was &mdash; every data point is destroyed to the same known distribution, which is exactly what lets generation start from that distribution.</span>
  </div>

## Learning to reverse it

- The network's one job: look at a noisy example and the step number, and predict the noise that was mixed in
  - Call it ε<sub>&theta;</sub>(x<sub>t</sub>, t)
  - Training data is free in unlimited quantities: take a real x<sub>0</sub>, pick a random t, draw ε, mix them with the closed form above - now you know the right answer exactly
  - Self-supervision again (chapter 20): the label is manufactured from the data itself

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo stretchy="false">(</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo><mo>=</mo><mi>E</mi><mo form="prefix">[</mo><msup><mrow><mo stretchy="false">&#x2016;</mo><mi>&#x3B5;</mi><mo>&#x2212;</mo><msub><mi>&#x3B5;</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo>,</mo><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup><mo form="postfix">]</mo></mrow></math>
<span class="eqn-note">The entire training objective is squared error &mdash; chapter 02's cost function, with the "right answer" being the noise we ourselves added. No adversary, no likelihood gymnastics; this simplicity is a large part of why diffusion took over.</span>
  </div>

- Generation runs the chain backwards
  - Start from x<sub>T</sub> ∼ N(0, I), and at each step subtract out (a correctly-scaled portion of) the predicted noise, then add back a little fresh noise

<div class="eqn">
<math display="block"><mrow><msub><mi>x</mi><mrow><mi>t</mi><mo>&#x2212;</mo><mn>1</mn></mrow></msub><mo>=</mo><mfrac><mn>1</mn><msqrt><mn>1</mn><mo>&#x2212;</mo><msub><mi>&#x3B2;</mi><mi>t</mi></msub></msqrt></mfrac><mo form="prefix">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo>&#x2212;</mo><mfrac><msub><mi>&#x3B2;</mi><mi>t</mi></msub><msqrt><mn>1</mn><mo>&#x2212;</mo><msub><mover><mi>&#x3B1;</mi><mo>&#x00AF;</mo></mover><mi>t</mi></msub></msqrt></mfrac><mspace width="0.15em"/><msub><mi>&#x3B5;</mi><mi>&#x3B8;</mi></msub><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo>,</mo><mi>t</mi><mo stretchy="false">)</mo><mo form="postfix">)</mo><mo>+</mo><msqrt><msub><mi>&#x3B2;</mi><mi>t</mi></msub></msqrt><mspace width="0.15em"/><mi>z</mi><mo>,</mo><mspace width="1em"/><mi>z</mi><mo>&#x223C;</mo><mi>N</mi><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mi>I</mi><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">One reverse step. The fresh noise z looks paradoxical &mdash; we are trying to remove noise &mdash; but it is what makes the reverse process a distribution rather than a single trajectory: run it twice and you get two different samples. At the final step (t = 1) no z is added.</span>
  </div>

## A complete diffusion model in NumPy

- The pieces above, assembled and run
  - Data: one-dimensional, a mixture of two Gaussians - half the points near −2, half near +2
  - Deliberately chosen so success is checkable: samples from a trained model must come out bimodal, and nothing simpler than a real generative model produces that from pure noise
- First, the data and the forward process

::: code-eg
```
import numpy as np

rng = np.random.default_rng(0)

def sample_data(n):                # half near -2, half near +2
    modes = rng.choice([-2.0, 2.0], size=n)
    return modes + 0.3 * rng.standard_normal(n)

T = 50
beta = np.linspace(1e-3, 0.2, T)   # the noise schedule
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)      # alpha_bar[-1] = 0.0045: 0.45% of signal left

x0  = sample_data(10000)
eps = rng.standard_normal(10000)
xT  = np.sqrt(alpha_bar[-1]) * x0 + np.sqrt(1 - alpha_bar[-1]) * eps
xT.mean(), xT.std()                # (0.001, 1.0) - the data is gone;
                                   # every x0 ends as standard normal noise
```
<p class="eqn-note">The closed-form jump verified: after 50 steps the bimodal data is indistinguishable from N(0, 1), which is exactly what the &alpha;&#x304;<sub>t</sub> equation promised.</p>
:::

- The noise predictor - chapter 08's network, in miniature
  - Two inputs (the noisy value, and the step t scaled to \[0, 1\]), one hidden layer of tanh units, one output: the predicted noise

::: code-eg
```
H = 64
W1 = rng.standard_normal((H, 2)) * 0.5   # chapter 09: random init, small values
b1 = np.zeros(H)
W2 = rng.standard_normal(H) * 0.5
b2 = 0.0

def predict(x, t_frac):
    a1 = np.tanh(W1 @ np.vstack([x, t_frac]) + b1[:, None])   # hidden layer
    return W2 @ a1 + b2, a1
```
<p class="eqn-note">&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) as sixty-four hidden units. Feeding t in as an input is what lets one network learn to denoise at every noise level at once.</p>
:::

- Training - manufacture (noisy input, true noise) pairs and do mini-batch gradient descent on the squared error
  - The gradient computation is chapter 09's backpropagation written out by hand for a two-layer net

::: code-eg
```
lr = 1e-2
for step in range(4000):                      # mini-batch descent, chapter 17
    x0  = sample_data(256)
    t   = rng.integers(0, T, size=256)        # a random step per example
    eps = rng.standard_normal(256)
    xt  = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps

    eps_hat, a1 = predict(xt, t / T)
    d = eps_hat - eps                         # the error to send backwards

    gW2 = a1 @ d / len(d)                     # backprop, chapter 09 in miniature:
    gb2 = d.mean()                            # output-layer gradients...
    da1 = np.outer(W2, d) * (1 - a1 ** 2)     # ...delta for the hidden layer...
    gW1 = da1 @ np.vstack([xt, t / T]).T / len(d)
    gb1 = da1.mean(axis=1)                    # ...and its gradients

    W1 -= lr * gW1; b1 -= lr * gb1            # simultaneous update, chapter 02
    W2 -= lr * gW2; b2 -= lr * gb2
# training loss falls from 4.1 to ~0.35 over the 4000 steps (under a second)
```
<p class="eqn-note">The whole training loop. Note what it never does: it never sees a "generated sample is good/bad" signal. It only ever learns to predict added noise, and generation quality follows from that alone.</p>
:::

- Generation - start from pure noise, run the reverse update 50 times

::: code-eg
```
n = 4000
x = rng.standard_normal(n)                    # x_T: pure noise, no data in sight
for t in range(T - 1, -1, -1):
    eps_hat, _ = predict(x, np.full(n, t / T))
    x = (x - beta[t] / np.sqrt(1 - alpha_bar[t]) * eps_hat) / np.sqrt(alpha[t])
    if t > 0:
        x = x + np.sqrt(beta[t]) * rng.standard_normal(n)   # the fresh z

(x < 0).mean()                     # 0.517          - half in each mode, as in the data
x[x < 0].mean(), x[x >= 0].mean()  # (-1.89, 1.87)  - mode centres (true: -2, +2)
x[x < 0].std(),  x[x >= 0].std()   # (0.43, 0.47)   - mode widths  (true: 0.3)
(np.abs(x) < 1).mean()             # 0.054          - the gap is nearly empty
```
<p class="eqn-note">Four thousand samples of standard normal noise, pushed backwards through the learned denoiser, come out <em>bimodal</em>: two clean modes at &plusmn;1.9 with an almost-empty gap between them. Slightly blurrier than the truth (widths 0.45 against 0.3) &mdash; about right for a sixty-four-unit network &mdash; but this is real generation: a distribution nothing in the sampling loop ever saw, reconstructed from noise.</p>
:::

- Scaling this up to images changes nothing conceptual
  - x becomes a million-pixel array instead of one number; the two-layer net becomes a large U-Net or transformer; T becomes ~1000
  - The schedule, the closed-form jump, the squared-error objective and the sampling loop are exactly the ones above

## Conditioning and guidance

- Everything so far generates unconditionally - real systems generate *from a prompt*
  - <span class="term">Conditioning</span> - feed a text prompt (encoded by a language model) into the denoiser as an extra input, so ε<sub>&theta;</sub>(x<sub>t</sub>, t, prompt) is steered towards matching images. The prompt attends into the denoiser via cross-attention - chapter 21
  - <span class="term">Classifier-free guidance</span> - run the denoiser with and without the prompt and extrapolate away from the unconditioned prediction. Exaggerates prompt adherence, at some cost in diversity
  - <span class="term">Latent diffusion</span> - run the whole process in a learned compressed space rather than on pixels, which is what made it cheap enough to use widely. Chapter 14's dimensionality-reduction argument, doing real work: denoise where the data actually varies

## What they're used for {#what-they-are-used-for}

- Image and video generation and editing - inpainting, upscaling, style transfer
- Increasingly structures rather than pictures - protein backbone design (RFdiffusion and relatives), small molecules, materials
  - AlphaFold3 replaced its final coordinate module with a diffusion process - chapter 24
- Anything where you want to sample from a complicated distribution rather than predict a single answer
  - Regression (chapter 02) gives you the conditional mean; diffusion gives you draws from the whole conditional distribution

## Summary

- Forward: destroy data with T small fixed Gaussian steps; a closed form jumps to any noise level directly
- Reverse: train a network to predict the added noise - squared error against a label you manufactured, so supervision is free
- Generate: start from pure noise and apply the reverse update T times, adding a little fresh noise each step
- The NumPy model above does all of this in ~40 lines - chapter 15's Gaussians, chapter 02's cost, chapter 08's network, chapter 09's backprop, chapter 17's mini-batches - and turns noise into a bimodal distribution it was never shown
- Conditioning brings in the prompt through cross-attention (chapter 21); latent diffusion moves the whole game into a compressed space (chapter 14)
- The deep idea: don't learn the hard leap from noise to data - learn one easy step, and take it many times
