---
title: "05: Probability and Bayes' Rule"
nav_title: "05: Probability and Bayes' rule"
origin: "2026"
description: "The probability the course assumes but never teaches: Bayes' rule, maximum likelihood, and naive Bayes. Replaces the programming chapter that was never written."
---

## A note on this chapter

- An addition, slotted into the original numbering
  - This slot held an unwritten programming tutorial (that material now lives in Appendix 1, as a NumPy primer)
  - Probability belongs here instead, because everything from chapter 06 onwards quietly relies on it: maximum likelihood in 06, Gaussians in 15, priors-in-disguise in 07 - and Bayes' theorem itself is never actually stated
- It also pays a debt
  - Chapter 11 compares Naive Bayes against other spam classifiers with a "cover later" that never came. This is the later
- Short by design - four ideas, each of which the rest of the notes will use
  - Conditional probability and Bayes' rule; maximum likelihood and MAP; generative vs discriminative; Naive Bayes

## The vocabulary

- A probability distribution assigns a number in \[0, 1\] to each outcome, and the numbers sum to 1
  - P(A, B) - the <span class="term">joint</span>: both things happen
  - P(A | B) - the <span class="term">conditional</span>: A, given that B happened
  - P(A) - the <span class="term">marginal</span>: A, averaging over everything else
- Two rules connect them, and they are the only machinery this chapter needs

<div class="eqn">
<math display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>,</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>&#x2223;</mo><mi>B</mi><mo stretchy="false">)</mo><mspace width="0.15em"/><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">The <span class="term">product rule</span>: "both" is "one of them" times "the other, given the first". Chaining it across a whole sequence is exactly how chapter 22 factorises a sentence.</span>
  </div>

<div class="eqn">
<math display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mo>&#x2211;</mo><mi>A</mi></munder><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo>&#x2223;</mo><mi>A</mi><mo stretchy="false">)</mo><mspace width="0.15em"/><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">The <span class="term">sum rule</span> (marginalization): the total probability of B is its probability under each scenario A, weighted by how likely each scenario is.</span>
  </div>

- <span class="term">Independence</span>
  - A and B are independent when P(A, B) = P(A)P(B) - knowing one tells you nothing about the other
  - Almost nothing interesting is truly independent; the useful move is *assuming* it anyway and knowing what you paid - which is the whole story of Naive Bayes below, and of chapter 15's per-feature Gaussians

## Bayes' rule

- Write the product rule both ways round - P(A, B) = P(A | B)P(B) = P(B | A)P(A) - and divide
  - That's the entire derivation. Two lines, no further assumptions

<div class="eqn">
<math display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>&#x2223;</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo>&#x2223;</mo><mi>A</mi><mo stretchy="false">)</mo><mspace width="0.15em"/><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></mfrac></mrow></math>
<span class="eqn-note">Bayes' rule. Read as a machine for reversing conditionals: from "how likely is the evidence, given the hypothesis" to "how likely is the hypothesis, given the evidence" &mdash; which is the direction you actually want, and the direction experiments don't give you.</span>
  </div>

- The named parts, because the rest of the notes use these words
  - P(A) - the <span class="term">prior</span>: what you believed before the evidence
  - P(B | A) - the <span class="term">likelihood</span>: how well the hypothesis predicts the evidence
  - P(A | B) - the <span class="term">posterior</span>: what you should believe after
  - P(B) - the evidence, computed by the sum rule; it just normalizes, so posterior ∝ likelihood × prior

## A worked example: the rare disease

- Chapter 11's cancer-screening numbers, taken seriously
  - 0.5% of patients have the disease; the test catches 99% of true cases, and wrongly flags 5% of healthy patients
  - Your test is positive. How worried should you be?

::: code-eg
```
p_disease = 0.005          # the prior - chapter 11's 0.5% prevalence
sens = 0.99                # P(positive | disease)   - the likelihood
spec = 0.95                # P(negative | healthy)

# the sum rule: total probability of a positive test
p_pos = sens * p_disease + (1 - spec) * (1 - p_disease)   # 0.0547

# Bayes' rule
sens * p_disease / p_pos   # 0.0905 - about 9%, not 99%
```
<p class="eqn-note">A 99%-sensitive test and the posterior is still only 9% &mdash; because the 5% false-positive rate acting on the enormous healthy majority swamps the true cases. The prior does most of the work when classes are skewed.</p>
:::

- This is chapter 11's skewed-classes lesson, in its native language
  - Precision - "of everything we flagged, how much was real" - *is* a posterior: P(y = 1 | flagged)
  - Recall is a likelihood: P(flagged | y = 1). Chapter 11's whole error-metric machinery is Bayes' rule wearing engineering clothes

## Maximum likelihood and MAP

- Fitting parameters is a Bayes' rule question: what should we believe about θ, given data?
  - posterior ∝ likelihood × prior, now over parameters: p(θ | data) ∝ p(data | θ) p(θ)
- <span class="term">Maximum likelihood</span> (MLE): ignore the prior, pick the θ that makes the data most probable

<div class="eqn">
<math display="block"><mrow><mover><mi>&#x3B8;</mi><mo>^</mo></mover><mo>=</mo><munder><mo movablelimits="false">arg&#x2009;max</mo><mi>&#x3B8;</mi></munder><mspace width="0.4em"/><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mi>log</mi><mspace width="0.15em"/><mi>p</mi><mo stretchy="false">(</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2223;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>;</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">Log because products of many probabilities underflow and sums don't &mdash; and because maximising a log maximises the original. This single line is where two of the notes' cost functions come from: model y as Bernoulli and you get chapter 06's cross-entropy; model the residuals as Gaussian and you get chapter 02's squared error. Neither cost was an aesthetic choice &mdash; both are MLE under a stated noise model.</span>
  </div>

- <span class="term">MAP</span> (maximum a posteriori): keep the prior

<div class="eqn">
<math display="block"><mrow><mover><mi>&#x3B8;</mi><mo>^</mo></mover><mo>=</mo><munder><mo movablelimits="false">arg&#x2009;max</mo><mi>&#x3B8;</mi></munder><mspace width="0.4em"/><mrow><mo stretchy="false">[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mi>log</mi><mspace width="0.15em"/><mi>p</mi><mo stretchy="false">(</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2223;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>;</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo><mo>+</mo><mi>log</mi><mspace width="0.15em"/><mi>p</mi><mo stretchy="false">(</mo><mi>&#x3B8;</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo></mrow></mrow></math>
<span class="eqn-note">One extra term. Now choose a Gaussian prior for the parameters &mdash; &theta;<sub>j</sub> &#x223C; N(0, &tau;&sup2;), "parameters are probably small" &mdash; and log p(&theta;) is a constant minus &Sigma;&theta;<sub>j</sub>&sup2;/2&tau;&sup2;. That is <em>chapter 07's regularization penalty</em>, derived rather than bolted on: L2 regularization is MAP estimation with a Gaussian prior, and &lambda; is just the noise-to-prior variance ratio. Chapter 07 said "prefer small parameters" as an instinct; this is the instinct as a theorem.</span>
  </div>

- The practical consequences fall out immediately
  - Big data swamps the prior: the likelihood has m terms and the prior has one, so MLE and MAP agree as m grows - regularization matters most exactly when data is scarce, which chapter 10's learning curves showed empirically
  - A tighter prior (smaller τ) is a bigger λ - more shrinkage, more bias, less variance: chapter 10's dial, given a probabilistic handle

## Generative vs discriminative classifiers {#generative-vs-discriminative}

- Two ways to build P(y | x), and the split organises half of machine learning
  - <span class="term">Discriminative</span> - model P(y | x) directly. Chapter 06's logistic regression: fit the boundary, say nothing about how x arises
  - <span class="term">Generative</span> - model each class's data, P(x | y), plus the class frequencies P(y), and let Bayes' rule flip the conditional at prediction time

<div class="eqn">
<math display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>y</mi><mo>&#x2223;</mo><mi>x</mi><mo stretchy="false">)</mo><mo>&#x221D;</mo><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2223;</mo><mi>y</mi><mo stretchy="false">)</mo><mspace width="0.15em"/><mi>P</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">Generative classification in one line: score each class by "how typical is this x of the class" times "how common is the class", and normalize. The rare-disease example above is exactly this with two classes.</span>
  </div>

- The trade
  - Generative models answer more questions - they can generate data, score how unusual an example is (chapter 15's anomaly detection is a generative model used without labels), and handle missing features by marginalizing
  - But they spend capacity modelling x, which the classification task never asked for; the discriminative model puts everything into the boundary
  - Model each class with chapter 15's multivariate Gaussian and you get Gaussian discriminant analysis - and the decision boundary comes out linear, closely related to what logistic regression fits directly

## Naive Bayes

- The generative recipe hits a wall on high-dimensional x
  - Chapter 11's spam representation: a 10,000-entry bitmap of which vocabulary words appear. P(x | y) is a distribution over 2<sup>10,000</sup> bitmaps - unlearnable
- The naive assumption: features are independent, *given the class*

<div class="eqn">
<math display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2223;</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><munderover><mo>&#x220F;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>P</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>j</mi></msub><mo>&#x2223;</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">One number per word per class &mdash; "how often does <em>discount</em> appear in spam" &mdash; instead of a distribution over every possible email. 20,000 parameters instead of 2<sup>10,000</sup>. The assumption is false (words travel in packs), which is why it's called naive; it works anyway because classification only needs the scores <em>ordered</em> correctly, not calibrated. Chapter 15 made the identical assumption when it multiplied per-feature Gaussians.</span>
  </div>

- Fitting is just counting, plus one guard
  - P(x<sub>j</sub> = 1 | spam) = fraction of spam emails containing word j - counting is the MLE here
  - The guard: a word never seen in training spam would get probability 0, and one zero annihilates the whole product. <span class="term">Laplace smoothing</span> adds one phantom occurrence of each outcome: (count + 1)/(total + 2)
  - Smoothing *is* a prior - the MAP section above, applied to counts
- And always work in logs - the products underflow otherwise
  - 0.1<sup>400</sup> is exactly 0.0 in floating point; 400 log 0.1 = −921 is a perfectly good number

::: code-eg
```
import numpy as np

vocab = ["andrew", "buy", "deal", "discount", "now"]   # chapter 11's vocabulary

X = np.array([[0,1,1,1,1],      # emails as chapter 11's bitmaps -
              [0,1,0,1,1],      # four spam:  buy/deal/discount/now
              [0,1,1,0,1],
              [0,1,1,1,0],
              [1,0,0,0,1],      # four ham: andrew, sometimes buy/deal/now
              [1,0,1,0,0],
              [1,0,0,0,0],
              [1,1,0,0,1]])
y = np.array([1,1,1,1, 0,0,0,0])

phi_y    = y.mean()                                    # P(spam) = 0.5
phi_spam = (X[y==1].sum(0) + 1) / ((y==1).sum() + 2)   # per-word rates,
phi_ham  = (X[y==0].sum(0) + 1) / ((y==0).sum() + 2)   # Laplace-smoothed

def p_spam(x):                     # Bayes' rule, in logs
    ls = np.log(phi_y)   + (x*np.log(phi_spam) + (1-x)*np.log(1-phi_spam)).sum()
    lh = np.log(1-phi_y) + (x*np.log(phi_ham)  + (1-x)*np.log(1-phi_ham)).sum()
    return 1 / (1 + np.exp(lh - ls))

p_spam(np.array([0,1,0,1,1]))   # 0.971 - "buy discount now": spam
p_spam(np.array([1,0,0,0,1]))   # 0.013 - "andrew ... now": not spam
```
<p class="eqn-note">A complete spam classifier: count, smooth, apply Bayes' rule in log space. Note the last line &mdash; converting a log-odds difference to a probability is chapter 06's sigmoid, which is no coincidence (next section). Real systems differ only in vocabulary size.</p>
:::

## Naive Bayes and logistic regression

- The two spam classifiers of these notes are a matched pair
  - Work out Naive Bayes' log-odds, log P(spam | x) − log P(ham | x), and it is *linear in x* - a weighted sum of word indicators plus a constant, pushed through a sigmoid
  - That is exactly logistic regression's form (chapter 06). Same hypothesis class; different way of choosing the weights
  - Naive Bayes picks them by counting each feature separately under its independence assumption; logistic regression picks them by directly optimising the classification objective, jointly
- Which wins is a data-budget question - and the course's own instructor wrote the definitive paper on it
  - Naive Bayes reaches its best performance very quickly (counting is easy to estimate from little data) but its ceiling is lower, since the independence assumption caps it
  - Logistic regression starts worse and ends better - given enough data, fitting the boundary directly beats fitting a wrong model of the data
  - This is chapter 10's bias/variance story: the generative assumption is bias you trade for variance - and it slots straight into chapter 11's algorithm-comparison plot, where Naive Bayes was one of the four contenders
- Rule of thumb
  - Tiny labelled set, or you need a baseline in an afternoon: Naive Bayes (chapter 11's "build a quick and dirty first version" advice, made concrete)
  - Real amounts of data: discriminative wins, and everything from chapter 06 to chapter 22 is on that side of the line

## Where Bayes lives in these notes

- Now visible in hindsight, chapters that were speaking probability all along
  - Chapter 02's squared error - MLE under Gaussian noise
  - Chapter 06's cost - MLE under a Bernoulli model, as its notes say; its sigmoid converts log-odds to probability, as above
  - Chapter 07's regularization - MAP with a Gaussian prior
  - Chapter 15's anomaly detection - a generative model P(x), used with a threshold instead of a second class
  - Chapter 15's per-feature product - the Naive Bayes independence assumption without labels
  - Chapter 22's cross-entropy - MLE under a categorical model, at internet scale
  - Chapter 25's variant scoring - a log-odds ratio of masked conditionals

## Summary

- Two rules - product and sum - generate everything here; Bayes' rule is the product rule divided by itself
- Posterior ∝ likelihood × prior: the machine for reversing conditionals, and with skewed classes the prior dominates - a 99% test can mean a 9% posterior
- MLE maximises the likelihood and is where chapters 02 and 06 got their cost functions; MAP adds a prior, and a Gaussian prior *is* chapter 07's L2 penalty
- Discriminative models fit P(y | x) directly; generative models fit P(x | y)P(y) and flip it with Bayes - buying generation, anomaly scores and missing-data handling at the price of modelling x
- Naive Bayes assumes feature independence within a class, turning an impossible density estimate into counting - smoothed by a prior, computed in logs
- Its log-odds are linear, making it logistic regression's generative twin: faster to its (lower) ceiling, overtaken as data grows - the promised backstory to chapter 11's comparison
