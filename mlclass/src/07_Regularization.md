---
title: "07: Regularization"
nav_title: "07: Regularization"
origin: "2011"
description: "Overfitting, and how a penalty on the parameters tames it in both linear and logistic regression."
---

## The problem of overfitting

- So far we've seen a few algorithms - work well for many applications, but can suffer from the problem of overfitting
- What is overfitting?
- What is regularization and how does it help

### Overfitting with linear regression

- Using our house pricing example again
  - Fit a linear function to the data - not a great model
    - This is **<span class="term">underfitting</span>** - also known as <span class="term"><strong>high bias</strong></span>
    - Bias is a historic/technical one - if we're fitting a straight line to the data we have a strong preconception that there should be a linear fit
      - In this case, this is not correct, but a straight line can't help being straight!
  - Fit a quadratic function
    - Works well
  - Fit a 4th order polynomial
    - Now curve fits through all five examples
      - Seems to do a good job fitting the training set
      - But, despite fitting the data we've provided very well, this is actually not such a good model
    - This is **<span class="term">overfitting</span>** - also known as **<span class="term">high variance</span>**
  - Algorithm has high variance
    - High variance - if fitting high order polynomial then the hypothesis can basically fit any data
    - Space of hypothesis is too large

<figure><img alt="" loading="lazy" src="07_Regularization_files/Image.png"/></figure>

- To recap, if we have too many features then the learned hypothesis may give a cost function of exactly zero
  - But this tries too hard to fit the training set
  - Fails to provide a *general* solution - **<span class="hl-red">unable to generalize</span>** (apply to new examples)

### Overfitting with logistic regression

- Same thing can happen to logistic regression
  - Sigmoidal function is an underfit
  - But a high order polynomial gives an overfitting (high variance hypothesis)

<figure><img alt="" loading="lazy" src="07_Regularization_files/Image [1].png"/></figure>

### Addressing overfitting

- Later we'll look at identifying when overfitting and underfitting is occurring
- Earlier we just plotted a higher order function - saw that it looks "too curvy"
  - Plotting hypothesis is one way to decide, but doesn't always work
  - Often have lots of features - here it's not just a case of selecting a degree polynomial, but also harder to plot the data and visualize to decide what features to keep and which to drop
  - If you have lots of features and little data - overfitting can be a problem
- How do we deal with this?
  - 1\) **Reduce number of features**
    - Manually select which features to keep
    - Model selection algorithms are discussed later (good for reducing number of features)
    - But, in reducing the number of features we lose some information
      - Ideally select those features which minimize data loss, but even so, some info is lost
  - 2\) **<span class="term">Regularization</span>**
    - Keep all features, but reduce magnitude of parameters θ
    - Works well when we have a lot of features, each of which contributes a bit to predicting y

## Cost function optimization for regularization

- Penalize and make some of the θ parameters really small
  - e.g. here θ<sub>3</sub> and θ<sub>4</sub>

<div class="eqn">
<math display="block"><mrow><munder><mi>min</mi><mi>&#x3B8;</mi></munder><mspace width="0.35em"/><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><mn>1000</mn><msubsup><mi>&#x3B8;</mi><mn>3</mn><mn>2</mn></msubsup><mo>+</mo><mn>1000</mn><msubsup><mi>&#x3B8;</mi><mn>4</mn><mn>2</mn></msubsup></mrow></math>
</div>

- The addition in blue is a modification of our cost function to help penalize θ<sub>3</sub> and θ<sub>4</sub>
  - So here we end up with θ<sub>3</sub> and θ<sub>4</sub> being close to zero (because the constants are massive)
  - So we're basically left with a quadratic function

<figure><img alt="" loading="lazy" src="07_Regularization_files/Image [3].png"/></figure>

- In this example, we penalized two of the parameter values
  - More generally, regularization is as follows
- Regularization
  - Small values for parameters corresponds to a simpler hypothesis (you effectively get rid of some of the terms)
  - A simpler hypothesis is less prone to overfitting
- Another example
  - Have 100 features x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>100</sub>
  - Unlike the polynomial example, we don't know what are the high order terms
    - How do we pick the ones to shrink?
  - With regularization, take cost function and modify it to shrink all the parameters
    - Add a term at the end
      - This regularization term shrinks every parameter
      - By convention you don't penalize θ<sub>0</sub> - minimization is from θ<sub>1</sub> onwards

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><mi>&#x3BB;</mi><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3B8;</mi><mi>j</mi><mn>2</mn></msubsup><mo>]</mo></mrow></math>
<span class="eqn-note">The penalty runs over &#x3B8;<sub>1</sub>, &#x3B8;<sub>2</sub>, &#x2026;, &#x3B8;<sub>n</sub> &mdash; by convention &#x3B8;<sub>0</sub> is not penalised.</span>
</div>

<ul>
  <li>In practice, if you include θ<sub>0</sub> it has little impact</li>
  <li>
    <strong>λ </strong>is the <strong><span class="term">regularization parameter</span></strong>
    <ul>
      <li>Chapter 05 derives this penalty as MAP estimation with a Gaussian prior - "prefer small parameters", stated probabilistically</li>
    </ul>
    <ul>
      <li>
        Controls a trade off between our two goals
        <ul>
          <li>1) Want to fit the training set well</li>
          <li>2) Want to keep parameters small</li>
        </ul>
      </li>
    </ul>
  </li>
  <li>
    With our example, using the <strong><span class="term">regularized objective</span></strong> (i.e. the cost function with the regularization term) you get a much smoother curve which fits the data and gives a much better hypothesis
    <ul>
      <li>
        If <strong>λ </strong>is very large we end up penalizing ALL the parameters (θ<sub>1</sub>, θ<sub>2</sub> etc.) so all the parameters end up being close to zero
        <ul>
          <li>
            If this happens, it's like we got rid of all the terms in the hypothesis
            <ul>
              <li>The result here is then underfitting</li>
            </ul>
          </li>
          <li>So this hypothesis is too biased because of the absence of any parameters (effectively)</li>
        </ul>
      </li>
    </ul>
  </li>
  <li>
    So, <strong>λ </strong>should be chosen carefully - not too big...
    <ul>
      <li>We look at some automatic ways to select <strong>λ </strong>later in the course</li>
    </ul>
  </li>
</ul>

## Regularized linear regression

- Previously, we looked at two algorithms for linear regression
  - Gradient descent
  - Normal equation
- Our linear regression with regularization is shown below

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><mi>&#x3BB;</mi><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3B8;</mi><mi>j</mi><mn>2</mn></msubsup><mo>]</mo></mrow></math>
<math display="block"><mrow><munder><mi>min</mi><mi>&#x3B8;</mi></munder><mspace width="0.35em"/><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>
</div>

::: code-eg
```
import numpy as np

def J(theta, X, y, lam):
    m = len(y)
    err = ((X @ theta - y) ** 2).sum()
    penalty = lam * (theta[1:] ** 2).sum()   # theta[1:] - theta0 is not penalised
    return (err + penalty) / (2 * m)
```
<p class="eqn-note">The regularized cost. The penalty slices from theta[1:], leaving θ<sub>0</sub> out &mdash; matching the sum from j = 1.</p>
:::

- Previously, gradient descent would repeatedly update the parameters θ<sub>j</sub>, where j = 0,1,2...n simultaneously
  - Shown below

<div class="eqn">
<math display="block"><mtext>Repeat {</mtext></math>
<math display="block"><mrow><mspace width="1.6em"/><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mn>0</mn><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>
<math display="block"><mrow><mspace width="1.6em"/><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>
<math display="block"><mtext>}</mtext></math>
<span class="eqn-note">j = 1, 2, 3, &#x2026;, n. &#x3B8;<sub>0</sub> has its own line above because it is never penalised &mdash; on the original slide the 0 is struck through in this index.</span>
</div>

- We've got the θ<sub>0</sub> update here shown explicitly
  - This is because for regularization we don't penalize θ<sub>0 </sub>so treat it slightly differently
- How do we regularize these two rules?
  - Take the term and add λ/m \* θ<sub>j</sub>
    - Sum for every θ (i.e. j = 1 to n)
  - This gives regularization for gradient descent
- We can show using calculus that the equation given below is the partial derivative of the regularized J(θ)

<div class="eqn">
<math display="block"><mrow><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mo>[</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>+</mo><mfrac><mi>&#x3BB;</mi><mi>m</mi></mfrac><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>]</mo></mrow></math>
<span class="eqn-note">j = 1, 2, 3, &#x2026;, n. The bracketed term is the partial derivative of the regularised J(&#x3B8;).</span>
</div>

- The update for θ<sub>j </sub>
  - θ<sub>j</sub> gets updated to
    - θ<sub>j </sub>- α \* \[a big term which also depends on θ<sub>j</sub>\]<sub> </sub>
- So if you group the θ<sub>j </sub>terms together

<div class="eqn">
<math display="block"><mrow><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mi>&#x3BB;</mi><mi>m</mi></mfrac><mo>)</mo><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>
</div>

- The term
  <div class="eqn">
  <math display="block"><mrow><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mi>&#x3BB;</mi><mi>m</mi></mfrac><mo>)</mo></mrow></math>
  <span class="eqn-note">Slightly less than 1, so each update shrinks &#x3B8;<sub>j</sub> a little before the usual gradient step.</span>
  </div>
  ::: code-eg
  ```
  alpha, lam, m = 0.05, 10, 10
  1 - alpha * lam / m   # 0.95  : theta_j is multiplied by this every step

  alpha, lam, m = 0.01, 1, 10
  1 - alpha * lam / m   # 0.999 : closer to 1 for gentler regularization
  ```
  <p class="eqn-note">The shrink factor for two settings &mdash; every update scales θ<sub>j</sub> by a number just under 1 before the usual gradient step.</p>
  :::
  - Is going to be a number less than 1 usually
  - Usually learning rate is small and m is large
    - So this typically evaluates to (1 - a small number)
    - So the term is often around 0.99 to 0.95
- This in effect means θ<sub>j </sub>gets multiplied by 0.99
  - Means the squared norm of θ<sub>j </sub>gets a little smaller
  - The second term is exactly the same as the original gradient descent

## Regularization with the normal equation

- Normal equation is the other linear regression model
  - Minimize the J(θ) using the normal equation
  - To use regularization we add a term (+ λ \[n+1 x n+1\]) to the equation
    - \[n+1 x n+1\] is the identity matrix with the top-left entry set to 0 — so θ<sub>0</sub> escapes the penalty, as everywhere else in this chapter

<div class="eqn">
<math display="block"><mrow><mi>&#x3B8;</mi><mo>=</mo><msup><mrow><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>+</mo><mi>&#x3BB;</mi><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mi>&#x2026;</mi></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd><mtd><mi>&#x2026;</mi></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd><mtd><mi>&#x2026;</mi></mtd></mtr><mtr><mtd><mi>&#x22EE;</mi></mtd><mtd><mi>&#x22EE;</mi></mtd><mtd><mi>&#x22EE;</mi></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>)</mo></mrow><mrow><mo>&#x2212;</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi>T</mi></msup><mi>y</mi></mrow></math>
<math display="block"><mrow><mtext>e.g. if n = 2:</mtext><mspace width="0.6em"/><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<span class="eqn-note">An [n+1 x n+1] matrix &mdash; the identity with the top-left entry set to 0, so that &#x3B8;<sub>0</sub> escapes the penalty.</span>
</div>

::: code-eg
```
X = np.array([[1, 2, 2],   # x2 is an exact copy of x1,
              [1, 3, 3],   # so X.T @ X is singular
              [1, 4, 4],
              [1, 5, 5]], dtype=float)
y = np.array([2, 3, 4, 5], dtype=float)

np.linalg.det(X.T @ X)   # 0.0 - not invertible

L = np.eye(3)
L[0, 0] = 0              # the modified identity from the equation above
theta = np.linalg.solve(X.T @ X + 1.0 * L, X.T @ y)
X @ theta                # [2.14, 3.05, 3.95, 4.86]  - a sensible fit anyway
```
<p class="eqn-note">Redundant features make X<sup>T</sup>X singular &mdash; exactly the non-invertibility case from chapter 04. Adding λ times the modified identity makes the system solvable again.</p>
:::

### Regularization for logistic regression

- We saw earlier that logistic regression can be prone to overfitting with lots of features
- Logistic regression cost function is as follows;

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>)</mo><mo>]</mo></mrow></math>
</div>

- To modify it we have to add an extra term
  <div class="eqn">
  <math display="block"><mrow><mo>+</mo><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3B8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
  </div>
- This has the effect of penalizing the parameters θ<sub>1</sub>, θ<sub>2</sub> up to θ<sub>n </sub>
  - Means, like with linear regression, we can get what appears to be a better fitting lower order hypothesis
- How do we implement this?
  - Original logistic regression with gradient descent function was as follows
    <div class="eqn">
    <math display="block"><mrow><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>
    <span class="eqn-note">j = 0, 1, 2, 3, &#x2026;, n</span>
    </div>
- Again, to modify the algorithm we simply need to modify the update rule for θ<sub>1</sub>, onwards
  - Looks cosmetically the same as linear regression, except obviously the hypothesis is very different
    <div class="eqn">
    <math display="block"><mrow><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mi>&#x3BB;</mi><mi>m</mi></mfrac><mo>)</mo><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>
    <span class="eqn-note">Cosmetically the same as regularised linear regression &mdash; but h<sub>&#x3B8;</sub>(x) is the sigmoid here, so it is a different algorithm.</span>
    </div>

## Advanced optimization of regularized linear regression

- As before, define a cost_function which takes a θ parameter and gives jVal and gradient back

<pre>def cost_function(theta):

    jVal = [code to compute <math><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];

    gradient[0] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];

    gradient[1] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];

    gradient[2] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];
    <math><mrow><mi>&#x22EE;</mi></mrow></math>
    gradient[n] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];</pre>

- use **<code class="hl-green">scipy.optimize.minimize</code>**
  - Pass it the **<code class="hl-green">cost_function</code>** object
  - Minimizes in an optimized manner using the cost function
- <code class="hl-green"><strong>jVal</strong></code>
  - Need code to compute J(θ)
    - Need to include regularization term
- Gradient
  - Needs to be the partial derivative of J(θ) with respect to θ<sub>i</sub>
  - Adding the appropriate term here is also necessary

<pre>def cost_function(theta):

    jVal = [code to compute <math><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];
      <math><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>)</mo><mo>]</mo><mo>+</mo><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3B8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>

    gradient[0] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];
      <math><mrow><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mn>0</mn><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>

    gradient[1] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];
      <math><mrow><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mn>1</mn><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>+</mo><mfrac><mi>&#x3BB;</mi><mi>m</mi></mfrac><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mrow></math>

    gradient[2] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];
      <math><mrow><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mn>2</mn><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>+</mo><mfrac><mi>&#x3BB;</mi><mi>m</mi></mfrac><msub><mi>&#x3B8;</mi><mn>2</mn></msub></mrow></math>
    <math><mrow><mi>&#x22EE;</mi></mrow></math>
    gradient[n] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];</pre>

- Ensure summation doesn't extend to the lambda term!
  - It doesn't, but, you know, don't be daft!
