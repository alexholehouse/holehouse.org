---
title: "10: Advice for applying Machine Learning"
nav_title: "10: Applying ML"
index_title: "10: Advice for applying machine learning techniques"
origin: "2011"
description: "Diagnosing what is wrong with a model: train, validation and test sets, bias versus variance, and learning curves."
---

## Deciding what to try next

- We now know many techniques
  - But, there is a big difference between someone who knows an algorithm vs. someone less familiar and doesn't understand how to apply them
  - Make sure you know how to choose the best avenues to explore the various techniques
  - Here we focus deciding what avenues to try

### Debugging a learning algorithm

- So, say you've implemented regularized linear regression to predict housing prices

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><mi>&#x3BB;</mi><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3B8;</mi><mi>j</mi><mn>2</mn></msubsup><mo>]</mo></mrow></math>
<span class="eqn-note">The original slide sums the penalty to m; it runs to n, the number of features &mdash; m is already used by the sum over training examples.</span>
</div>

- Trained it
- But, when you test on new data you find it makes unacceptably large errors in its predictions
- :-(
- What should you try next?
  - There are many things you can do;
    - **<span class="hl-red">Get more training data</span>**
      - Sometimes more data doesn't help
      - Often it does though, although you should always do some preliminary testing to make sure more data will actually make a difference (discussed later)
    - **<span class="hl-red">Try a smaller set of features</span>**
      - Carefully select small subset
      - You can do this by hand, or use some dimensionality reduction technique (e.g. PCA - we'll get to this later)
    - **<span class="hl-red">Try getting additional features</span>**
      - Sometimes this isn't helpful
      - LOOK at the data
      - Can be very time consuming
    - **<span class="hl-red">Adding polynomial features</span>**
      - You're grasping at straws, aren't you...
    - **<span class="hl-red">Building your own, new, better features</span>** based on your knowledge of the problem
      - Can be risky if you accidentally over fit your data by creating new features which are inherently specific/relevant to your training data
    - **<span class="hl-red">Try decreasing or increasing </span>**<span class="hl-red"><strong>λ</strong></span>
      - Change how important the regularization term is in your calculations
  - These changes can become MAJOR projects/headaches (6 months +)
    - Sadly, most common method for choosing one of these examples is to go by gut feeling (randomly)
    - Many times, see people spend huge amounts of time only to discover that the avenue is fruitless
      - No apples, pears, or any other fruit. Nada.
  - There are some simple techniques which can let you rule out half the things on the list
    - Save you a lot of time!
- **Machine learning diagnostics**
  - Tests you can run to see what is/what isn't working for an algorithm
  - See what you can change to improve an algorithm's performance
  - These can take time to implement and understand (week)
    - But, they can also save you spending months going down an avenue which will *never* work

## Evaluating a hypothesis

- When we fit parameters to training data, try and minimize the error
  - We might think a low error is good - doesn't necessarily mean a good parameter set
    - Could, in fact, be indicative of overfitting
    - This means your model will fail to generalize
  - How do you tell if a hypothesis is overfitting?
    - Could plot h<sub>θ</sub>(x)
    - But with lots of features may be impossible to plot
- Standard way to evaluate a hypothesis is
  - Split data into two portions
    - 1st portion is **<span class="term">training set</span>**
    - 2nd portion is **<span class="term">test set</span>**
  - Typical split might be 70:30 (training:test)

<table>
  <caption>A typical 70:30 split &mdash; the first 70% becomes the training set, the rest the test set</caption>
  <tr><th scope="col">Size</th><th scope="col">Price</th><th scope="col">Set</th></tr>
  <tr><td>2104</td><td>400</td><td>training</td></tr><tr><td>1600</td><td>330</td><td>training</td></tr><tr><td>2400</td><td>369</td><td>training</td></tr><tr><td>1416</td><td>232</td><td>training</td></tr><tr><td>3000</td><td>540</td><td>training</td></tr><tr><td>1985</td><td>300</td><td>training</td></tr><tr><td>1534</td><td>315</td><td>training</td></tr>
  <tr><td>1427</td><td>199</td><td>test</td></tr><tr><td>1380</td><td>212</td><td>test</td></tr><tr><td>1494</td><td>243</td><td>test</td></tr>
</table>

<p class="eqn-note">The training set gives (x<sup>(1)</sup>, y<sup>(1)</sup>) &#x2026;
(x<sup>(m)</sup>, y<sup>(m)</sup>); the test set gives
(x<sub>test</sub><sup>(1)</sup>, y<sub>test</sub><sup>(1)</sup>) &#x2026;
(x<sub>test</sub><sup>(m<sub>test</sub>)</sup>, y<sub>test</sub><sup>(m<sub>test</sub>)</sup>).</p>

- NB if data is ordered, send a random percentage
  - (Or randomly order, then send data)
  - Data is typically ordered in some way anyway
- So a typical **<span class="term">train and test scheme</span>** would be
  - 1\) Learn parameters θ from training data, minimizing J(θ) using 70% of the training data
  - 2\) Compute the test error
    - J<sub>test</sub>(θ) = average square error as measured on the test set
      <div class="eqn">
      <math display="block"><mrow><msub><mi>J</mi><mi>test</mi></msub><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><msub><mi>m</mi><mi>test</mi></msub></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>m</mi><mi>test</mi></msub></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msubsup><mi>x</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo form="postfix">)</mo><mo>&#x2212;</mo><msubsup><mi>y</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
      </div>
      ::: code-eg
      ```
      import numpy as np

      x = np.arange(1.0, 11)                   # ten examples on a line...
      y = 2 * x + 1 + np.array([ 0.2, -0.1,  0.3, -0.2,  0.1,
                                -0.3,  0.2, -0.1,  0.3, -0.2])   # ...plus noise

      X = np.c_[np.ones(7), x[:7]]             # first 70% is the training set
      theta = np.linalg.pinv(X.T @ X) @ X.T @ y[:7]
      theta            # [1.114, 1.979] - close to the true (1, 2)

      X_test = np.c_[np.ones(3), x[7:]]        # last 30% is the test set
      ((X_test @ theta - y[7:]) ** 2).mean() / 2   # J_test = 0.0259
      ```
      <p class="eqn-note">Fit on the 70%, score on the held-out 30%. The test error is small because a straight line genuinely describes this data.</p>
      :::
  - This is the definition of the **<span class="term">test set error</span>**
- What about if we were using logistic regression
  - The same, learn using 70% of the data, test with the remaining 30%
    <div class="eqn">
    <math display="block"><mrow><msub><mi>J</mi><mi>test</mi></msub><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><msub><mi>m</mi><mi>test</mi></msub></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>m</mi><mi>test</mi></msub></munderover><msubsup><mi>y</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msubsup><mi>x</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo form="postfix">)</mo><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msubsup><mi>y</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msubsup><mi>x</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo form="postfix">)</mo><mo>)</mo></mrow></math>
    <span class="eqn-note">The original slide writes the second logarithm without its 1 &#x2212; ; corrected here, as in chapters 07 and 09.</span>
    </div>
  - Sometimes there a better way - misclassification error (0/1 misclassification)
    - We define the error as follows
      <div class="eqn">
      <div class="cases">
      <math display="block"><mrow><mi>err</mi><mo stretchy="false">(</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo></mrow></math>
      <span class="cases-brace" aria-hidden="true">{</span>
      <div class="cases-rows"><div class="cases-row"><math display="block"><mrow><mn>1</mn></mrow></math><math display="block"><mrow><mtext>if</mtext><mspace width="0.35em"/><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>&#x2265;</mo><mn>0.5</mn><mo>,</mo><mspace width="0.4em"/><mi>y</mi><mo>=</mo><mn>0</mn><mspace width="0.6em"/><mtext>or</mtext><mspace width="0.4em"/><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>&lt;</mo><mn>0.5</mn><mo>,</mo><mspace width="0.4em"/><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></div><div class="cases-row"><math display="block"><mrow><mn>0</mn></mrow></math><math display="block"><mrow><mtext>otherwise</mtext></mrow></math></div></div>
      </div>
      <span class="eqn-note">1 when the prediction and the label disagree, 0 when they agree.</span>
      </div>
      ::: code-eg
      ```
      h = np.array([0.9, 0.4, 0.6, 0.2])   # predicted probabilities
      y = np.array([1,   1,   0,   0])

      err = (h >= 0.5) != (y == 1)         # [False, True, True, False]
      err.mean()                           # 0.5 - half the test set is mislabelled
      ```
      <p class="eqn-note">The err() definition as one boolean expression: a 1 wherever the thresholded prediction and the label disagree, then averaged.</p>
      :::
    - Then the test error is<br/>
      <div class="eqn">
      <math display="block"><mrow><mtext>Test error</mtext><mo>=</mo><mfrac><mn>1</mn><msub><mi>m</mi><mi>test</mi></msub></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>m</mi><mi>test</mi></msub></munderover><mi>err</mi><mo form="prefix">(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msubsup><mi>x</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo form="postfix">)</mo><mo>,</mo><msubsup><mi>y</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo form="postfix">)</mo></mrow></math>
      </div>
      - i.e. it's the fraction in the test set the hypothesis mislabels
- These are the standard techniques for evaluating a learned hypothesis

## Model selection and training validation test sets

- How to choose regularization parameter or degree of polynomial (**<span class="term">model selection problems</span>**)
- We've already seen the problem of overfitting
  - More generally, this is why training set error is a poor predictor of hypothesis accuracy for new data (generalization)
- Model selection problem
  - Try to choose the degree for a polynomial to fit data
    <div class="eqn">
    <math display="block"><mrow><mn>1.</mn><mspace width="0.6em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>+</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mi>x</mi></mrow></math>
    <math display="block"><mrow><mn>2.</mn><mspace width="0.6em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>+</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mi>x</mi><mo>+</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><msup><mi>x</mi><mn>2</mn></msup></mrow></math>
    <math display="block"><mrow><mn>3.</mn><mspace width="0.6em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>+</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mi>x</mi><mo>+</mo><mi>&#x22EF;</mi><mo>+</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><msup><mi>x</mi><mn>3</mn></msup></mrow></math>
    <math display="block"><mrow><mspace width="0.6em"/><mi>&#x22EE;</mi></mrow></math>
    <math display="block"><mrow><mn>10.</mn><mspace width="0.6em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>+</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mi>x</mi><mo>+</mo><mi>&#x22EF;</mi><mo>+</mo><msub><mi>&#x3B8;</mi><mn>10</mn></msub><msup><mi>x</mi><mn>10</mn></msup></mrow></math>
    <span class="eqn-note">Fit each, then choose d by cross-validation error.</span>
    </div>
  - d = what degree of polynomial do you want to pick
    - An additional parameter to try and determine your training set
      - d =1 (linear)
      - d=2 (quadratic)
      - ...
      - d=10
    - Choose a model, fit that model and get an estimate of how well you hypothesis will generalize
  - You could
    - Take model 1, minimize with training data which generates a parameter vector θ<sup>1</sup> (where d =1)
    - Take model 2, do the same, get a *different* θ<sup>2</sup> (where d = 2)
    - And so on
    - Take these parameters and look at the test set error for each using the previous formula
      - J**<sub>test</sub>**(θ<sup>1</sup>)
      - J**<sub>test</sub>**(θ<sup>2</sup>)
      - ...
      - J**<sub>test</sub>**(θ<sup>10</sup>)
  - You could then
    - See which model has the lowest test set error
  - Say, for example, d=5 is the lowest
    - Now take the d=5 model and say, how well does it generalize?
      - You could use J**<sub>test</sub>**(θ<sup>5</sup>)
      - BUT, this is going to be an optimistic estimate of generalization error, because our parameter is fit to that test set (i.e. specifically chose it because the test set error is small)
      - So not a good way to evaluate if it will generalize
  - To address this problem, we do something a bit different for model selection
- Improved model selection
  - Given a training set instead split into three pieces
    - 1 - **<span class="term">Training set</span>** (60%) - m values
    - 2 - **<span class="term">Cross validation</span>** (CV) set (20%)m**<sub>cv</sub>**
    - 3 - **<span class="term">Test set</span>** (20%) m**<sub>test</sub>**
  - As before, we can calculate
    - Training error
    - Cross validation error
    - Test error
      <div class="eqn">
      <math display="block"><mrow><mtext>Training error:</mtext><mspace width="1em"/><msub><mi>J</mi><mi>train</mi></msub><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
      <math display="block"><mrow><mtext>Cross validation error:</mtext><mspace width="1em"/><msub><mi>J</mi><mi>cv</mi></msub><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><msub><mi>m</mi><mi>cv</mi></msub></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>m</mi><mi>cv</mi></msub></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msubsup><mi>x</mi><mi>cv</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo form="postfix">)</mo><mo>&#x2212;</mo><msubsup><mi>y</mi><mi>cv</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
      <math display="block"><mrow><mtext>Test error:</mtext><mspace width="1em"/><msub><mi>J</mi><mi>test</mi></msub><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><msub><mi>m</mi><mi>test</mi></msub></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>m</mi><mi>test</mi></msub></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msubsup><mi>x</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo form="postfix">)</mo><mo>&#x2212;</mo><msubsup><mi>y</mi><mi>test</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
      </div>
      ::: code-eg
      ```
      rng = np.random.default_rng(0)
      x = np.linspace(0, 1, 30)
      y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, 30)   # a curve plus noise
      idx = rng.permutation(30)
      tr, cv = idx[:18], idx[18:]          # 60:40 train / cross-validation split

      for d in (1, 3, 9):
          c = np.polyfit(x[tr], y[tr], d)
          J_train = ((np.polyval(c, x[tr]) - y[tr]) ** 2).mean() / 2
          J_cv    = ((np.polyval(c, x[cv]) - y[cv]) ** 2).mean() / 2
          # d=1: J_train=0.100  J_cv=0.134   underfit - both high
          # d=3: J_train=0.023  J_cv=0.027   the sweet spot
          # d=9: J_train=0.014  J_cv=2.250   overfit - train falls, cv explodes
      ```
      <p class="eqn-note">Model selection run for real: the degree-9 polynomial has the best training error of the three and by far the worst cross-validation error &mdash; which is the entire argument for selecting d on a set the fit never saw.</p>
      :::
  - So
    - Minimize cost function for each of the models as before
    - Test these hypothesis on the cross validation set to generate the cross validation error
    - Pick the hypothesis with the lowest cross validation error
      - e.g. pick θ<sup>5</sup>
    - Finally
      - Estimate generalization error of model using the test set
- Final note
  - In machine learning as practiced today - many people will select the model using the test set and then check the model is OK for generalization using the test error (which we've said is bad because it gives a biased analysis)
    - With a MASSIVE test set this is maybe OK
  - But considered much better practice to have separate training and validation sets

## Diagnosis - bias vs. variance

- If you get bad results usually because of one of
  - **<span class="term">High bias</span>** - under fitting problem
  - **<span class="term">High variance</span>** - over fitting problem
- Important to work out which is the problem
  - Knowing which will help let you improve the algorithm
- Bias/variance shown graphically below

<figure><img alt="" loading="lazy" src="10_Advice_for_applying_machine_learning_files/Image [8].png"/></figure>

- The degree of a model will increase as you move towards overfitting
- Lets define training and cross validation error as before
- Now plot
  - x = degree of polynomial d
  - y = error for both training and cross validation (two lines)
    - CV error and test set error will be very similar
      <figure><img alt="" loading="lazy" src="10_Advice_for_applying_machine_learning_files/Image [9].png"/></figure>
    - This plot helps us understand the error
  - We want to minimize both errors
    - Which is why that d=2 model is the sweet spot
- How do we apply this for diagnostics
  - If cv error is high we're either at the high or the low end of d
    <figure><img alt="" loading="lazy" src="10_Advice_for_applying_machine_learning_files/Image [10].png"/></figure>
  - if d is too small --> this probably corresponds to a high bias problem
  - if d is too large --> this probably corresponds to a high variance problem
- **<span class="hl-red">For the high bias case, we find both cross validation and training error are high</span>**
  - Doesn't fit training data well
  - Doesn't generalize either
- **<span class="hl-red">For high variance, we find the cross validation error is high but training error is low</span>**
  - So we suffer from overfitting (training is low, cross validation is high)
  - i.e. training set fits well
  - But generalizes poorly

## Regularization and bias/variance

- How is bias and variance affected by regularization?

<div class="eqn">
<math display="block"><mrow><mtext>Model:</mtext><mspace width="0.8em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>+</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mi>x</mi><mo>+</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><msup><mi>x</mi><mn>3</mn></msup><mo>+</mo><msub><mi>&#x3B8;</mi><mn>4</mn></msub><msup><mi>x</mi><mn>4</mn></msup></mrow></math>
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3B8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
<span class="eqn-note">As above, the penalty sums to n rather than m on the original slide.</span>
</div>

- The equation above describes fitting a high order polynomial with regularization (used to keep parameter values small)
  - Consider three cases
    - **λ = large**
      - All θ values are heavily penalized
      - So most parameters end up being close to zero
      - So hypothesis ends up being close to 0
      - So **<span class="hl-red">high bias -&gt; under fitting data</span>**
    - **λ = intermediate**
      - Only this value gives the fitting which is reasonable
    - **λ = small**
      - Lambda = 0
      - So we make the regularization term 0
      - So **<span class="hl-red">high variance -&gt; Get overfitting</span>** (minimal regularization means it obviously doesn't do what it's meant to)

<figure><img alt="" loading="lazy" src="10_Advice_for_applying_machine_learning_files/Image [12].png"/></figure>

- How can we automatically choose a good value for λ?
  - To do this we define another function J**<sub>train</sub>**(θ) which is the optimization function *without* the regularization term (average squared errors)
    <div class="eqn">
    <math display="block"><mrow><msub><mi>J</mi><mi>train</mi></msub><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
    <span class="eqn-note">The optimisation objective without the regularisation term &mdash; just the average squared error.</span>
    </div>
  - Define cross validation error and test set errors as before (i.e. without regularization term)
    - So they are 1/2 average squared error of various sets
- <strong>Choosing </strong>**λ**
  - Have a set or range of values to use
  - Often increment by factors of 2 so
    - model(1)= λ = 0
    - model(2)= λ = 0.01
    - model(3)= λ = 0.02
    - model(4) = λ = 0.04
    - model(5) = λ = 0.08<br/> .<br/> .<br/> .
    - model(p) = λ = 10
  - This gives a number of models which have different λ
  - With these models
    - Take each one (p<sup>th</sup>)
    - Minimize the cost function
    - This will generate some parameter vector
      - Call this θ<sup>(p)</sup>
    - So now we have a set of parameter vectors corresponding to models with different λ values
  - Take all of the hypothesis and use the cross validation set to validate them
    - Measure average squared error on cross validation set
    - Pick the model which gives the lowest error
    - Say we pick θ<sup>(5)</sup>
  - Finally, take the one we've selected (θ<sup>(5)</sup>) and test it with the test set
- **Bias/variance as a function of λ**
  - Plot λ vs.
    - J**<sub>train</sub>**
      - When λ is small you get a small value (regularization basically goes to 0)
      - When λ is large you get a large value corresponding to high bias
    - J**<sub>cv</sub>**
      - When λ is small we see high variance
        - Too small a value means we over fit the data
      - When λ is large we end up underfitting, so this is bias
        - So cross validation error is high
  - Such a plot can help show you you're picking a good value for λ

## Learning curves

- A learning curve is often useful to plot for algorithmic sanity checking or improving performance
- What is a learning curve?
  - Plot J**<sub>train</sub>** (average squared error on training set) or J**<sub>cv</sub>** (average squared error on cross validation set)
  - Plot against m (number of training examples)
    - m is a constant
    - So artificially reduce m and recalculate errors with the smaller training set sizes
  - J**<sub>train</sub>**
    - Error on smaller sample sizes is smaller (as less variance to accommodate)
    - So as m grows error grows
  - J**<sub>cv</sub>**
    - Error on cross validation set
    - When you have a tiny training set you generalize badly
    - But as training set grows your hypothesis generalizes better
    - So cv error will decrease as m increases

<figure><img alt="" loading="lazy" src="10_Advice_for_applying_machine_learning_files/Image [14].png"/></figure>

- What do these curves look like if you have
  - **High bias**
    - e.g. setting straight line to data
    - J**<sub>train</sub>**
      - Training error is small at first and grows
      - Training error becomes close to cross validation
      - So the performance of the cross validation and training set end up being similar (but very poor)
    - J**<sub>cv</sub>**
      - Straight line fit is similar for a few vs. a lot of data
      - So it doesn't generalize any better with lots of data because the function just doesn't fit the data
        - No increase in data will help it fit
    - The problem with high bias is because cross validation and training error are both high
    - Also implies that if a learning algorithm has high bias as we get more examples the cross validation error doesn't decrease
      - **So if an algorithm is already suffering from high bias, <span class="hl-red">more data does not help</span>**
      - So knowing if you're suffering from high bias is good!
      - In other words, high bias is a problem with the underlying way you're modeling your data
        - So more data won't improve that model
        - It's too simplistic
  - **High variance**
    - e.g. high order polynomial
    - J**<sub>train</sub>**
      - When set is small, training error is small too
      - As training set sizes increases, value is still small
      - But slowly increases (in a near linear fashion)
      - Error is still low
    - J**<sub>cv</sub>**
      - Error remains high, even when you have a moderate number of examples
      - Because the problem with high variance (overfitting) is your model doesn't generalize
    - An indicative diagnostic that you have high variance is that there's a big gap between training error and cross validation error
    - If a learning algorithm is suffering from high variance, more data is probably going to help
      <figure><img alt="" loading="lazy" src="10_Advice_for_applying_machine_learning_files/Image [15].png"/></figure>
      - **So if an algorithm is already suffering from high variance, <span class="hl-green">more data will probably</span><span class="hl-green"> help</span>**
        - Maybe
  - These are clean curves
  - In reality the curves you get are far dirtier
  - But, learning curve plotting can help diagnose the problems your algorithm will be suffering from

## What to do next (revisited)

- How do these ideas help us choose how we approach a problem?
  - Original example
    - Trained a learning algorithm (regularized linear regression)
    - But, when you test on new data you find it makes unacceptably large errors in its predictions
    - What should you try next?
  - How do we decide what to do?
    - **Get more examples** --> helps to fix high variance
      - Not good if you have high bias<br/>
    - **Smaller set of features** --> fixes high variance (overfitting)
      - Not good if you have high bias<br/>
    - **Try adding additional features** --> fixes high bias (because hypothesis is too simple, make hypothesis more specific)<br/>
    - **Add polynomial terms** --> fixes high bias problem<br/>
    - <strong>Decreasing </strong>**λ** --> fixes high bias<br/>
    - <strong>Increasing </strong>**λ** --> fixes high variance<br/>
- Relating it all back to neural networks - selecting a network architecture
  - One option is to use a small neural network
    - Few (maybe one) hidden layer and few hidden units
    - Such networks are prone to under fitting
    - But they are computationally cheaper
  - Larger network
    - More hidden layers
      - How do you decide that a larger network is good?
  - Using a single hidden layer is good default
    - Also try with 1, 2, 3, see which performs best on cross validation set
    - So like before, take three sets (training, cross validation, test)
  - More units
    - This is computationally expensive
    - Prone to over-fitting
      - Use regularization to address over fitting
