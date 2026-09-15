---
title: "06: Logistic Regression"
nav_title: "06: Logistic regression"
origin: "2011"
description: "Classification with the sigmoid: decision boundaries, the log-loss cost, and one-vs-all for more than two classes."
---

## Classification

- Where y is a discrete value
  - Develop the logistic regression algorithm to determine what class a new input should fall into
- Classification problems
  - Email -> spam/not spam?
  - Online transactions -> fraudulent?
  - Tumor -> Malignant/benign
- Variable in these problems is Y
  - Y is either 0 or 1
    - 0 = negative class (absence of something)
    - 1 = positive class (presence of something)
- Start with **<span class="term">binary class problems</span>**
  - Later look at multiclass classification problem, although this is just an extension of binary classification
- How do we develop a classification algorithm?
  - Tumour size vs malignancy (0 or 1)
  - We *could* use linear regression
    - Then threshold the classifier output (i.e. anything over some value is yes, else no)
    - In our example below linear regression with thresholding seems to work

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image.png"/></figure>

- We can see above this does a reasonable job of stratifying the data points into one of two classes
  - But what if we had a single Yes with a very small tumour
  - This would lead to classifying all the existing yeses as nos
- Another issues with linear regression
  - We know Y is 0 or 1
  - Hypothesis can give values larger than 1 or less than 0
- So, logistic regression generates a value which is always between 0 and 1
  - Logistic regression is a **<span class="term">classification algorithm</span>** - don't be confused

## Hypothesis representation

- What function is used to represent our hypothesis in classification
- We want our classifier to output values between 0 and 1
  - When using linear regression we did h<sub>θ</sub>(x) = (θ<sup><em>T</em></sup> x)
  - For classification hypothesis representation we do h<sub>θ</sub>(x) = g((θ<sup><em>T</em></sup> x))
    - Where we define g(z)
      - z is a real number
    - g(z) = 1/(1 + e<sup><em>-z</em></sup>)
      - This is the **<span class="term">sigmoid function</span>**, or the **<span class="term">logistic function</span>**
    - If we combine these equations we can write out the hypothesis as
      <div class="eqn">
      <math display="block"><mrow><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>&#x2212;</mo><msup><mi>&#x3B8;</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mrow></math>
      </div>
      ::: code-eg
      ```
      import numpy as np

      def g(z):   # the sigmoid function
          return 1 / (1 + np.exp(-z))

      g(0)     # 0.5      - the crossing point
      g(10)    # 0.99995  - asymptotes towards 1
      g(-10)   # 0.00005  - asymptotes towards 0
      ```
      <p class="eqn-note">Crosses 0.5 at z = 0 and saturates in both directions &mdash; the shape drawn above.</p>
      :::
- What does the sigmoid function look like
- Crosses 0.5 at the origin, then flattens out
  - Asymptotes at 0 and 1

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [2].png"/></figure>

- Given this we need to fit θ to our data

### Interpreting hypothesis output

- <span class="hl-red">When our hypothesis (h<sub>θ</sub>(x)) outputs a number, we treat that value as the estimated probability that y=1 on input x</span>
  - Example
    - If X is a feature vector with x<sub>0</sub> = 1 (as always) and x<sub>1</sub> = tumourSize
    - h<sub>θ</sub>(x) = 0.7
      - Tells a patient they have a 70% chance of a tumor being malignant
  - We can write this using the following notation
    - h<sub>θ</sub>(x) = P(y=1|x ; θ)
  - What does this mean?
    - Probability that y=1, given x, parameterized by θ
- Since this is a binary classification task we know y = 0 or 1
  - So the following must be true
    - P(y=1|x ; θ) + P(y=0|x ; θ) = 1
    - P(y=0|x ; θ) = 1 - P(y=1|x ; θ)

## Decision boundary

- Gives a better sense of what the hypothesis function is computing
- Better understand of what the hypothesis function looks like
  - One way of using the sigmoid function is;
    - When the probability of y being 1 is greater than 0.5 then we can predict y = 1
    - Else we predict y = 0
  - When is it exactly that h<sub>θ</sub>(x) is greater than 0.5?
    - Look at sigmoid function
      - g(z) is greater than or equal to 0.5 when z is greater than or equal to 0
        <figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [3].png"/></figure>
    - So <span class="hl-red">if z is positive, g(z) is greater than 0.5</span>
      - z = (θ<sup><em>T</em></sup> x)
    - So when
      - θ<sup><em>T</em></sup> x >= 0
    - Then h<sub>θ</sub> >= 0.5
- So what we've shown is that the hypothesis predicts y = 1 when θ<sup><em>T</em></sup> x >= 0
  - The corollary of that when θ<sup><em>T</em></sup> x \<= 0 then the hypothesis predicts y = 0
  - Let's use this to better understand how the hypothesis makes its predictions

### Decision boundary

- h<sub>θ</sub>(x) = g(θ<sub>0</sub> + θ<sub>1</sub>x<sub>1 </sub>+ θ<sub>2</sub>x<sub>2</sub>)

<strong><u><figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [4].png"/></figure></u></strong>

- So, for example
  - θ<sub>0</sub> = -3
  - θ<sub>1</sub> = 1
  - θ<sub>2</sub> = 1
- So our parameter vector is a column vector with the above values
  - So, θ<sup><em>T</em></sup> is a row vector = \[-3,1,1\]
- What does this mean?
  - The z here becomes θ<sup><em>T</em></sup> x
  - We predict "y = 1" if
    - \-3x<sub>0</sub> + 1x<sub>1</sub> + 1x<sub>2</sub> >= 0
    - \-3 + x<sub>1</sub> + x<sub>2</sub> >= 0
- We can also re-write this as
  - If (x<sub>1</sub> + x<sub>2</sub> >= 3) then we predict y = 1
  - If we plot
    - x<sub>1</sub> + x<sub>2</sub> = 3 we graphically plot our **<span class="term">decision boundary</span>**

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [5].png"/></figure>

- Means we have these two regions on the graph
  - Blue = false
  - Magenta = true
  - Line = decision boundary
    - Concretely, the straight line is the set of points where h<sub>θ</sub>(x) = 0.5 exactly
  - The decision boundary is a property of the hypothesis
    - Means we can create the boundary with the hypothesis and parameters without any data
      - Later, we use the data to determine the parameter values
    - i.e. y = 1 if
      - 5 - x<sub>1</sub> > 0
      - 5 > x<sub>1</sub>

## Non-linear decision boundaries

- Get logistic regression to fit a complex non-linear data set
  - Like polynomial regression add higher order terms
  - So say we have
    - h<sub>θ</sub>(x) = g(θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> + θ<sub>3</sub>x<sub>1</sub><sup>2</sup> + θ<sub>4</sub>x<sub>2</sub><sup>2</sup>)
    - We take the transpose of the θ vector times the input vector
      - Say θ<sup>T</sup> was \[-1,0,0,1,1\] then we say;
      - Predict that "y = 1" *if*
        - \-1 + x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> >= 0<br/> or
        - x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> >= 1
      - If we plot x<sub>1</sub><sup>2</sup> + x<sub>2</sub><sup>2</sup> = 1
        - This gives us a circle with a radius of 1 around 0

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [6].png"/></figure>

- Mean we can build more complex decision boundaries by fitting complex parameters to this (relatively) simple hypothesis
- More complex decision boundaries?
  - By using higher order polynomial terms, we can get even more complex decision boundaries

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [7].png"/></figure>

## Cost function for logistic regression

- Fit θ parameters
- Define the optimization objective for the cost function we use to fit the parameters
  - Training set of *m* training examples
    - Each example is an n+1 length column vector

<div class="eqn">
<math display="block"><mrow><mtext>Training set:</mtext><mspace width="0.5em"/><mo>{</mo><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>1</mi><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>1</mi><mo>)</mo></mrow></msup><mo>)</mo><mo>,</mo><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>2</mi><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>2</mi><mo>)</mo></mrow></msup><mo>)</mo><mo>,</mo><mi>&#x2026;</mi><mo>,</mo><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></msup><mo>)</mo><mo>}</mo></mrow></math>
<math display="block"><mrow><mi>x</mi><mo>&#x2208;</mo><mrow><mo>[</mo><mtable><mtr><mtd><msub><mi>x</mi><mn>0</mn></msub></mtd></mtr><mtr><mtd><msub><mi>x</mi><mn>1</mn></msub></mtd></mtr><mtr><mtd><mi>&#x22EE;</mi></mtd></mtr><mtr><mtd><msub><mi>x</mi><mi>n</mi></msub></mtd></mtr></mtable><mo>]</mo></mrow><mspace width="2em"/><msub><mi>x</mi><mn>0</mn></msub><mo>=</mo><mn>1</mn><mo>,</mo><mspace width="0.5em"/><mi>y</mi><mo>&#x2208;</mo><mo stretchy="false">{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math>
<math display="block"><mrow><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>&#x2212;</mo><msup><mi>&#x3B8;</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mrow></math>
<span class="eqn-note">m examples, each a column vector of n+1 features.</span>
</div>

- This is the situation
  - Set of m training examples
  - Each example is a feature vector which is n+1 dimensional
  - x<sub>0</sub> = 1
  - y ∈ {0,1}
  - Hypothesis is based on parameters (θ)
    - Given the training set how do we choose/fit θ?
- Linear regression uses the following function to determine θ

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- Instead of writing the squared error term, we can write
  - If we define "cost()" as;
    - cost(h<sub>θ</sub>(x<sup>i</sup>), y) = 1/2(h<sub>θ</sub>(x<sup>i</sup>) - y<sup>i</sup>)<sup>2</sup>
    - Which evaluates to the cost for an individual example using the same measure as used in linear regression
  - We can **<span class="hl-red">redefine J(</span>**<strong><span class="hl-red">θ) as</span></strong>
    <div class="eqn">
    <math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mi>Cost</mi><mo form="prefix">(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>,</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo></mrow></math>
    </div>
    - Which, appropriately, is the sum of all the individual costs over the training data (i.e. the same as linear regression)
- To further simplify it we can get rid of the superscripts
  - So
    <div class="eqn">
    <math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mi>Cost</mi><mo form="prefix">(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>,</mo><mi>y</mi><mo form="postfix">)</mo></mrow></math>
    </div>
- What does this actually mean?
  - This is the cost you want the learning algorithm to pay if the outcome is h<sub>θ</sub>(x) and the actual outcome is y
  - If we use this function for logistic regression this is a **<span class="term">non-convex function</span>** for parameter optimization
    - Could work....
- What do we mean by non convex?
  - We have some function - J(θ) - for determining the parameters
  - Our hypothesis function has a non-linearity (sigmoid function of h<sub>θ</sub>(x) )
    - This is a complicated non-linear function
  - If you take h<sub>θ</sub>(x) and plug it into the Cost() function, and then plug the Cost() function into J(θ) and plot J(θ) we find many local optimum -> *non convex function*
  - Why is this a problem
    - Lots of local minima mean gradient descent may not find the global optimum - may get stuck in a local minimum
  - We would like a convex function so if you run gradient descent you converge to a global minimum

### A convex logistic regression cost function

- To get around this we need a different, convex Cost() function which means we can apply gradient descent

<div class="eqn">
<div class="cases">
<math display="block"><mrow><mi>Cost</mi><mo stretchy="false">(</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo></mrow></math>
<span class="cases-brace" aria-hidden="true">{</span>
<div class="cases-rows"><div class="cases-row"><math display="block"><mrow><mo form="prefix">&#x2212;</mo><mi>log</mi><mo stretchy="false">(</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math><math display="block"><mrow><mtext>if</mtext><mspace width="0.35em"/><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></div><div class="cases-row"><math display="block"><mrow><mo form="prefix">&#x2212;</mo><mi>log</mi><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math><math display="block"><mrow><mtext>if</mtext><mspace width="0.35em"/><mi>y</mi><mo>=</mo><mn>0</mn></mrow></math></div></div>
</div>
</div>

- **<span class="hl-red">This is our logistic regression cost function</span>**
  - This is the penalty the algorithm pays
  - Plot the function
- Plot y = 1
  - So h<sub>θ</sub>(x) evaluates as -log(h<sub>θ</sub>(x))

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [13].png"/></figure>

- So when we're right, cost function is 0
  - Else it slowly increases cost function as we become "more" wrong
  - X axis is what we predict
  - Y axis is the cost associated with that prediction
- This cost functions has some interesting properties
  - If y = 1 and h<sub>θ</sub>(x) = 1
    - If hypothesis predicts exactly 1 and that's exactly correct then that corresponds to 0 (exactly, not nearly 0)
  - As h<sub>θ</sub>(x) goes to 0
    - Cost goes to infinity
    - This captures the intuition that if h<sub>θ</sub>(x) = 0 (predict <em>P </em>(y=1|x; θ) = 0) but y = 1 this will penalize the learning algorithm with a massive cost
- What about if y = 0
- then cost is evaluated as -log(1- h<sub>θ</sub>( x ))
  - Just get inverse of the other function

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [14].png"/></figure>

- Now it goes to plus infinity as h<sub>θ</sub>(x) goes to 1
- With our particular cost functions J(θ) is going to be convex and avoid local minimum

## Simplified cost function and gradient descent

- Define a simpler way to write the cost function and apply gradient descent to the logistic regression
  - By the end should be able to implement a fully functional logistic regression function
- Logistic regression cost function is as follows

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mi>Cost</mi><mo form="prefix">(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>,</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo></mrow></math>
<div class="cases">
<math display="block"><mrow><mi>Cost</mi><mo stretchy="false">(</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo></mrow></math>
<span class="cases-brace" aria-hidden="true">{</span>
<div class="cases-rows"><div class="cases-row"><math display="block"><mrow><mo form="prefix">&#x2212;</mo><mi>log</mi><mo stretchy="false">(</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math><math display="block"><mrow><mtext>if</mtext><mspace width="0.35em"/><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></div><div class="cases-row"><math display="block"><mrow><mo form="prefix">&#x2212;</mo><mi>log</mi><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math><math display="block"><mrow><mtext>if</mtext><mspace width="0.35em"/><mi>y</mi><mo>=</mo><mn>0</mn></mrow></math></div></div>
</div>
<span class="eqn-note">y is always 0 or 1.</span>
</div>

- This is the cost for a single example
  - For binary classification problems y is always 0 or 1
    - Because of this, we can have a simpler way to write the cost function
      - Rather than writing cost function on two lines/two cases
      - Can compress them into one equation - more efficient
  - Can write cost function is
    - <span class="hl-red"><strong>cost(h<sub>θ</sub>(x),y) = -ylog( h<sub>θ</sub>(x) ) - (1-y)log( 1- h<sub>θ</sub>(x) ) </strong></span>
      - This equation is a more compact version of the two cases above
  - We know that there are only two possible cases
    - y = 1
      - Then our equation simplifies to
        - \-log(h<sub>θ</sub>(x)) - (0)log(1 - h<sub>θ</sub>(x))
          - \-log(h<sub>θ</sub>(x))
          - Which is what we had before when y = 1
    - y = 0
      - Then our equation simplifies to
        - \-(0)log(h<sub>θ</sub>(x)) - (1)log(1 - h<sub>θ</sub>(x))
        - \= -log(1- h<sub>θ</sub>(x))
        - Which is what we had before when y = 0
    - Clever!
- So, in summary, our cost function for the θ parameters can be defined as

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>)</mo><mo>]</mo></mrow></math>
<span class="eqn-note">Because y is always 0 or 1, one of the two terms is always zero &mdash; so this single expression covers both cases.</span>
</div>

::: code-eg
```
def J(theta, X, y):
    h = g(X @ theta)
    return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()

X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)
y = np.array([0, 0, 1, 1], dtype=float)

J(np.zeros(2), X, y)   # 0.6931 = log(2): with theta = 0 every h is 0.5
```
<p class="eqn-note">The compressed cost, vectorized. With θ = 0 the hypothesis says 0.5 for everything, and the cost is exactly log 2 per example.</p>
:::

- Why do we choose this function when other cost functions exist?
  - This cost function can be derived from statistics using the principle of **<span class="term">maximum likelihood estimation</span>** (chapter 05)
    - Note this does mean there's an underlying probabilistic assumption - the labels, given x, are modelled as Bernoulli
  - Also has the nice property that it's convex
- To fit parameters θ:
  - Find parameters θ which minimize J(θ)
  - This means we have a set of parameters to use in our model for future predictions
- Then, if we're given some new example with set of features x, we can take the θ which we generated, and output our prediction using
  <div class="eqn">
  <math display="block"><mrow><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>&#x2212;</mo><msup><mi>&#x3B8;</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mrow></math>
  </div>
  - This result is
    - p(y=1 | x ; θ)
      - Probability y = 1, given x, parameterized by θ

### How to minimize the logistic regression cost function

- Now we need to figure out how to minimize J(θ)
  - Use gradient descent as before
  - Repeatedly update each parameter using a learning rate

<div class="eqn">
<math display="block"><mtext>Repeat {</mtext></math>
<math display="block"><mrow><mspace width="1.6em"/><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>
<math display="block"><mtext>}</mtext></math>
<span class="eqn-note">simultaneously update all &#x3B8;<sub>j</sub></span>
</div>

::: code-eg
```
theta = np.zeros(2)
for _ in range(20000):
    h = g(X @ theta)
    theta = theta - 0.5 * X.T @ (h - y) / len(y)

theta                  # [-25.6, 10.3]
-theta[0] / theta[1]   # 2.48 -> the decision boundary sits between x = 2 and x = 3
np.round(g(X @ theta), 3)   # [0., 0.007, 0.995, 1.]  - the fit
```
<p class="eqn-note">The same update as linear regression, but h is the sigmoid. The boundary lands at x &asymp; 2.5 &mdash; between the last y = 0 example and the first y = 1.</p>
:::

- If you had <em>n </em>features, you would have an n+1 column vector for θ
- This equation is the same as the linear regression rule
  - The only difference is that our definition for the hypothesis has changed
- Previously, we spoke about how to monitor gradient descent to check it's working
  - Can do the same thing here for logistic regression
- When implementing logistic regression with gradient descent, we have to update all the θ values (θ<sub>0</sub> to θ<sub>n</sub>) simultaneously
  - Could use a for loop
  - Better would be a vectorized implementation
- Feature scaling for gradient descent for logistic regression also applies here

## Advanced optimization

- Previously we looked at gradient descent for minimizing the cost function
- Here look at advanced concepts for minimizing the cost function for logistic regression
  - Good for large machine learning problems (e.g. huge feature set)
- <em>What is gradient descent actually doing?</em>
  - We have some cost function J(θ), and we want to minimize it
  - We need to write code which can take θ as input and compute the following
    - J(θ)
    - Partial derivative of J(θ) with respect to j (where j=0 to j = n)

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>
<math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>
<span class="eqn-note">for j = 0, 1, &#x2026;, n</span>
</div>

- Given code that can do these two things
  - Gradient descent repeatedly does the following update

<strong><u><div class="eqn">
<math display="block"><mrow><mtext>Repeat {</mtext><mspace width="0.5em"/><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mspace width="0.5em"/><mtext>}</mtext></mrow></math>
</div></u></strong>

- So update each j in θ sequentially
- So, we must;
  - Supply code to compute J(θ) and the derivatives
  - Then plug these values into gradient descent
- Alternatively, instead of gradient descent to minimize the cost function we could use
  - **<span class="term">Conjugate gradient</span>**
  - **<span class="term">BFGS</span>** (Broyden-Fletcher-Goldfarb-Shanno)
  - **<span class="term">L-BFGS</span>** (Limited memory - BFGS)
- These are more optimized algorithms which take that same input and minimize the cost function
- These are <em>very </em>complicated algorithms
- Some properties
  - **Advantages**
    - No need to manually pick alpha (learning rate)
      - Have a clever inner loop (line search algorithm) which tries a bunch of alpha values and picks a good one
    - Often faster than gradient descent
      - Do more than just pick a good learning rate
    - Can be used successfully without understanding their complexity
  - **Disadvantages**
    - Could make debugging more difficult
    - Should not be implemented yourself
    - Different libraries may use different implementations - may hit performance

### Using advanced cost minimization algorithms

- How to use algorithms
  - Say we have the following example

<div class="eqn">
<math display="block"><mrow><mi>&#x3B8;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mtd></mtr><mtr><mtd><msub><mi>&#x3B8;</mi><mn>2</mn></msub></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><msup><mrow><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>&#x2212;</mo><mn>5</mn><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><msup><mrow><mo>(</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>&#x2212;</mo><mn>5</mn><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
<math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mn>2</mn><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>&#x2212;</mo><mn>5</mn><mo>)</mo></mrow></math>
<math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mn>2</mn><mo>(</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>&#x2212;</mo><mn>5</mn><mo>)</mo></mrow></math>
</div>

::: code-eg
```
from scipy.optimize import minimize

def cost_function(theta):
    jval = ((theta - 5) ** 2).sum()
    gradient = 2 * (theta - 5)
    return jval, gradient

res = minimize(cost_function, np.zeros(2), jac=True)
res.x   # [5., 5.]  - the minimum, exactly where the derivatives hit zero
```
<p class="eqn-note">The worked example above, run for real: minimize drives both parameters to 5 without us ever picking a learning rate.</p>
:::

- Example above
  - θ<sub>1</sub> and θ<sub>2</sub> (two parameters)
  - Cost function here is J(θ) = (θ<sub>1</sub> - 5)<sup><em>2</em></sup> + ( θ<sub>2</sub> - 5)<sup><em>2</em></sup>
  - The derivatives of the J(θ) with respect to either θ<sub>1</sub> and θ<sub>2</sub> turns out to be the 2(θ<sub>i</sub> - 5)
- First we need to define our cost function, which should have the following signature

```
def cost_function(theta):
    ...
    return jval, gradient
```

- Input for the cost function is **<code class="hl-green">theta</code>**, which is a vector of the θ parameters
- Two return values from <code class="hl-green"><strong>cost_function</strong></code> are
  - **<code class="hl-green">jval</code>**
    - How we compute the cost function θ (the underived cost function)
      - In this case = (θ<sub>1</sub> - 5)<sup><em>2</em></sup> + (θ<sub>2</sub> - 5)<sup><em>2</em></sup>
  - **gradient**
    - 2 by 1 vector
    - 2 elements are the two partial derivative terms
    - i.e. this is an n-dimensional vector
      - Each indexed value gives the partial derivatives for the partial derivative of J(θ) with respect to θ<sub>i</sub>
      - Where i is the index position in the **gradient** vector
- With the cost function implemented, we can call the advanced algorithm using

```
from scipy.optimize import minimize

initial_theta = np.zeros(2)                  # initialize the theta values
res = minimize(cost_function, initial_theta, # run the algorithm
               jac=True, options={'maxiter': 100})
opt_theta, function_val = res.x, res.fun
```

- Here
  - <code class="hl-green"><strong>options</strong></code> is a dictionary giving options for the algorithm
  - **<code class="hl-green">minimize</code>**
    - SciPy's general-purpose cost function minimizer (scipy.optimize.minimize)
  - **<code class="hl-green">cost_function</code>** is passed as an object - Python functions are first-class values, so no special syntax is needed
- For this implementation
  - <code class="hl-green"><strong>jac=True</strong></code> tells minimize that cost_function returns the gradient along with the cost
- How do we apply this to logistic regression?
  - Here we have a vector

<div class="eqn">
<math display="block"><mrow><mi>THETA</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><msub><mi>&#x3B8;</mi><mn>0</mn></msub></mtd></mtr><mtr><mtd><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mtd></mtr><mtr><mtd><mi>&#x22EE;</mi></mtd></mtr><mtr><mtd><msub><mi>&#x3B8;</mi><mi>n</mi></msub></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
</div>

<pre>def cost_function(theta):

    jval = [code to compute <math><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];

    gradient[0] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];

    gradient[1] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];
    <math><mrow><mi>&#x22EE;</mi></mrow></math>
    gradient[n] = [code to compute <math><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>];</pre>

- Here
  - theta is a n+1 dimensional column vector
  - NumPy indexes from 0, so gradient\[0\] is the θ<sub>0</sub> entry
- Write a cost function which captures the cost function for logistic regression

## Multiclass classification problems

- Getting logistic regression for multiclass classification using **<span class="term">one vs. all</span>**
- Multiclass - more than yes or no (1 or 0)
  - Classification with multiple classes for assignment

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [23].png"/></figure>

- Given a dataset with three classes, how do we get a learning algorithm to work?
  - Use one vs. all classification make binary classification work for multiclass classification
- **One vs. all classification**
  - Split the training set into three separate binary classification problems
    - i.e. create a new fake training set
      - Triangle (1) vs crosses and squares (0) h<sub>θ</sub><sup><em>1</em></sup>(x)
        - P(y=1 | x<sub>1</sub>; θ)
      - Crosses (1) vs triangle and square (0) h<sub>θ</sub><sup><em>2</em></sup>(x)
        - P(y=1 | x<sub>2</sub>; θ)
      - Square (1) vs crosses and triangle (0) h<sub>θ</sub><sup><em>3</em></sup>(x)
        - P(y=1 | x<sub>3</sub>; θ)

<figure><img alt="" loading="lazy" src="06_Logistic_Regression_files/Image [24].png"/></figure>

- **Overall**
  - Train a logistic regression classifier h<sub>θ</sub><sup><em>(i)</em></sup>(x) for each class i to predict the probability that y = i
  - On a new input, *x* to make a prediction, pick the class *i* that maximizes the probability that h<sub>θ</sub><sup><em>(i)</em></sup>(x) = 1
