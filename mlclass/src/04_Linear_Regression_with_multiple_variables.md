---
title: "04: Linear Regression with Multiple Variables"
nav_title: "04: Multivariate regression"
origin: "2011"
description: "Many features at once: vectorized gradient descent, feature scaling, learning rates, polynomial features and the normal equation."
---

## Linear regression with multiple features

*New version of linear regression with multiple features*

- Multiple variables = multiple features
- In original version we had
  - X = house size, use this to predict
  - y = house price
- If in a new scheme we have more variables (such as number of bedrooms, number floors, age of the home)
  - x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>,<sub> </sub>x<sub>4 </sub>are the four features
    - x<sub>1</sub> - size (feet squared)
    - x<sub>2</sub> - Number of bedrooms
    - x<sub>3</sub> - Number of floors
    - x<sub>4</sub> - Age of home (years)
  - y is the output variable (price)
- More notation
  - **<span class="term">n </span>**
    - number of features (n = 4)
  - <strong><span class="term">m</span> </strong>
    - number of examples (i.e. number of rows in a table)
  - **<span class="term">x<sup>i</sup> </span>**
    - vector of the input for an example (so a vector of the four parameters for the i<sup>th </sup>input example)
    - i is an index into the training set
    - So
      - x is an n-dimensional feature vector
      - x<sup>3</sup> is, for example, the 3rd house, and contains the four features associated with that house
  - **<span class="term">x<sub>j</sub><sup>i </sup></span>**
    - The value of feature j in the ith training example
    - So
      - x<sub>2</sub><sup>3 </sup>is, for example, the number of bedrooms in the third house
- Now we have multiple features
  - What is the form of our hypothesis?
  - Previously our hypothesis took the form;
    - h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x
      - Here we have two parameters (theta 1 and theta 2) determined by our cost function
      - One variable x
  - Now we have multiple features
    - h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> + θ<sub>3</sub>x<sub>3</sub> + θ<sub>4</sub>x<sub>4</sub>
  - For example
    - h<sub>θ</sub>(x) = 80 + 0.1x<sub>1</sub> + 0.01x<sub>2</sub> + 3x<sub>3</sub> - 2x<sub>4</sub>
      - An example of a hypothesis which is trying to predict the price of a house
      - Parameters are still determined through a cost function
  - For convenience of notation, x<sub>0</sub> = 1
    - For every example i you have an additional 0th feature for each example
    - So now your **<span class="term">feature vector</span>** is n + 1 dimensional feature vector indexed from 0
      - This is a column vector called x
      - Each example has a column vector associated with it
      - So let's say we have a new example called "X"
    - <span class="term"><strong>Parameters</strong></span> are also in a 0 indexed n+1 dimensional vector
      - This is also a column vector called θ
      - This vector is the same for each example
  - Considering this, hypothesis can be written
    - h<sub>θ</sub>(x) = θ<sub>0</sub>x<sub>0</sub> + θ<sub>1</sub>x<sub>1</sub> + θ<sub>2</sub>x<sub>2</sub> + θ<sub>3</sub>x<sub>3</sub> + θ<sub>4</sub>x<sub>4</sub>
  - If we do
    - h<sub>θ</sub>(x) =θ*<sup>T</sup>* X
      - θ*<sup>T </sup>*is an \[1 x n+1\] matrix
      - In other words, because θ*<sup> </sup>*is a column vector, the transposition operation transforms it into a row vector
      - So before
        - θ*<sup> </sup>*was a matrix \[n + 1 x 1\]
      - Now
        - θ*<sup>T </sup>*is a matrix \[1 x n+1\]
      - Which means the inner dimensions of θ*<sup>T</sup>* and X match, so they can be multiplied together as
        - \[1 x n+1\] \* \[n+1 x 1\]
        - \= h<sub>θ</sub>(x)
        - So, in other words, the transpose of our parameter vector \* an input example X gives you a predicted hypothesis which is \[1 x 1\] dimensions (i.e. a single value)
    - This x<sub>0</sub> = 1 lets us write this like this
  - This is an example of multivariate linear regression

## Gradient descent for multiple variables

- Fitting parameters for the hypothesis with gradient descent
  - Parameters are θ<sub>0</sub> to θ<sub>n</sub>
  - Instead of thinking about this as n separate values, think about the parameters as a single vector (θ)
    - Where θ is n+1 dimensional
- Our cost function is

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- Similarly, instead of thinking of J as a function of the n+1 numbers, J() is just a function of the parameter vector
  - J(θ)
- **Gradient descent**
  <div class="eqn">
  <math display="block"><mtext>Repeat {</mtext></math>
  <math display="block"><mrow><mspace width="1.6em"/><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub><mo form="postfix">)</mo></mrow></math>
  <math display="block"><mtext>}</mtext></math>
  <span class="eqn-note">simultaneously update for every j = 0, &#x2026;, n</span>
  </div>
- Once again, this is
  - θ<sub>j</sub> = θ<sub>j</sub> - learning rate (α) times the partial derivative of J(θ) with respect to θ<sub>J(...)</sub>
  - We do this through a **simultaneous update** of every θ<sub>j</sub> value
- Implementing this algorithm
  - When n = 1

<div class="eqn">
<math display="block"><mtext>Repeat {</mtext></math>
<math display="block"><mrow><mspace width="1.6em"/><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mn>0</mn></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow></mrow></math>
<math display="block"><mrow><mspace width="1.6em"/><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></math>
<math display="block"><mtext>}</mtext></math>
<span class="eqn-note">simultaneously update &#x3B8;<sub>0</sub> and &#x3B8;<sub>1</sub>. The first update's sum is &#x2202;/&#x2202;&#x3B8;<sub>0</sub> J(&#x3B8;).</span>
</div>

- Above, we have slightly different update rules for θ<sub>0</sub> and θ<sub>1</sub>
  - Actually they're the same, except the end has a previously undefined x<sub>0</sub><sup>(i)</sup> as 1, so wasn't shown
- We now have an almost identical rule for multivariate gradient descent

<div class="eqn">
<math display="block"><mtext>New algorithm (n &#x2265; 1) &mdash; Repeat {</mtext></math>
<math display="block"><mrow><mspace width="1.6em"/><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2254;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub><mo>&#x2212;</mo><mi>&#x3B1;</mi><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mrow></math>
<math display="block"><mtext>}</mtext></math>
<span class="eqn-note">simultaneously update &#x3B8;<sub>j</sub> for j = 0, &#x2026;, n. The sum is &#x2202;/&#x2202;&#x3B8;<sub>j</sub> J(&#x3B8;).</span>
</div>

::: code-eg
```
import numpy as np

grad = X.T @ (X @ theta - y) / m   # every partial derivative at once
theta = theta - alpha * grad       # every parameter updated at once
```
<p class="eqn-note">The whole update rule in two lines: X.T @ (X @ theta &minus; y) computes the sum in the equation above for every j simultaneously.</p>
:::

- What's going on here?
  - We're doing this for each j (0 until n) as a simultaneous update (like when n = 1)
  - So, we re-set θ<sub>j</sub> to
    - θ<sub>j</sub> minus the learning rate (α) times the partial derivative of the θ vector with respect to θ<sub>j</sub>
    - In non-calculus words, this means that we do
      - Learning rate
      - Times 1/m (makes the maths easier)
      - Times the sum of
        - The hypothesis taking in the variable vector, minus the actual value, times the j-th value in that variable vector for EACH example
  - It's important to remember that
    <div class="eqn">
    <math display="block"><mrow><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><msubsup><mi>x</mi><mi>j</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>=</mo><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>
    </div>
- These algorithm are highly similar

## Gradient Descent in practice: 1 Feature Scaling

- Having covered the theory, we now move on to learn about some of the practical tricks
- Feature scaling
  - If you have a problem with multiple features
  - You should make sure those features have a similar scale
    - Means gradient descent will converge more quickly
  - e.g.
    - x1 = size (0 - 2000 feet)
    - x2 = number of bedrooms (1-5)
    - Means the contours generated if we plot θ<sub>1</sub> vs. θ<sub>2</sub> give a very tall and thin shape due to the huge range difference
  - Running gradient descent on this kind of cost function can take a long time to find the global minimum

<figure><img alt="" loading="lazy" src="04_Linear_Regression_with_multiple_variables_files/Image [5].png"/></figure>

- Pathological input to gradient descent
  - So we need to rescale this input so it's more effective
  - So, if you define each value from x1 and x2 by dividing by the max for each feature
  - Contours become more like circles (as scaled between 0 and 1)
- May want to get everything into -1 to +1 range (approximately)
  - Want to avoid large ranges, small ranges or very different ranges from one another
  - Rule of thumb regarding acceptable ranges
    - \-3 to +3 is generally fine - any bigger bad
    - \-1/3 to +1/3 is ok - any smaller bad
- Can do **<span class="term">mean normalization</span>**
  - Take a feature x<sub>i</sub>
    - Replace it by (x<sub>i</sub> - mean)/max
    - So your values all have an average of about 0

<div class="eqn">
<math display="block"><mrow><msub><mi>x</mi><mi>i</mi></msub><mo>&#x2190;</mo><mfrac><mrow><msub><mi>x</mi><mi>i</mi></msub><mo>&#x2212;</mo><msub><mi>&#x3BC;</mi><mi>i</mi></msub></mrow><mrow><msub><mi>s</mi><mi>i</mi></msub></mrow></mfrac></mrow></math>
<span class="eqn-note">&#x3BC;<sub>i</sub> is the average value of x<sub>i</sub> in the training set. s<sub>i</sub> is the range (max &#x2212; min), or the standard deviation.</span>
</div>

::: code-eg
```
mu = X[:, 1:].mean(axis=0)   # per-feature mean (skip the x0 column of 1s)
s  = X[:, 1:].std(axis=0)    # per-feature spread
X[:, 1:] = (X[:, 1:] - mu) / s   # every feature now centred, on a similar scale
```
<p class="eqn-note">Mean normalization and feature scaling in one step &mdash; broadcasting applies the subtraction and division to every row at once.</p>
:::

- Instead of max can also use standard deviation

## Learning Rate α

- Focus on the learning rate (α)
- Topics
  - Update rule
  - Debugging
  - How to choose α

### Make sure gradient descent is working

<ul>
  <li>
    Plot min J(θ) vs. no of iterations
    <ul>
      <li>(i.e. plotting J(θ) over the course of gradient descent</li>
    </ul>
  </li>
  <li>If gradient descent is working then J(θ) should decrease after every iteration</li>
  <li>
    Can also show if you're not making huge gains after a certain number
    <ul>
      <li>Can apply heuristics to reduce number of iterations if need be</li>
      <li>If, for example, after 1000 iterations you reduce the parameters by nearly nothing you could choose to only run 1000 iterations in the future</li>
      <li>Make sure you don't accidentally hard-code thresholds like this in and then forget about why they're there though!</li>
    </ul>
  </li>
  <figure><img alt="" loading="lazy" src="04_Linear_Regression_with_multiple_variables_files/Image [7].png"/></figure>
  <li>
    Number of iterations varies a lot
    <ul>
      <li>30 iterations</li>
      <li>3000 iterations</li>
      <li>3,000,000 iterations</li>
      <li>Very hard to tell in advance how many iterations will be needed</li>
      <li>Can often make a guess based a plot like this after the first 100 or so iterations</li>
    </ul>
  </li>
  <li>
    Automatic convergence tests
    <ul>
      <li>
        Check if J(θ) changes by a small threshold or less
        <ul>
          <li>Choosing this threshold is hard</li>
          <li>
            So often easier to check for a straight line
            <ul>
              <li>Why? - Because we're seeing the straightness in the context of the whole algorithm</li>
              <li>Could you design an automatic checker which calculates a threshold based on the system's preceding progress?</li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
  </li>
  <li>
    Checking it's working
    <ul>
      <li>
        If you plot J(θ) vs iterations and see the value is increasing - means you probably need a smaller α
        <ul>
          <li>Cause is because you're minimizing a function which looks like this</li>
        </ul>
      </li>
    </ul>
  </li>
  <figure><img alt="" loading="lazy" src="04_Linear_Regression_with_multiple_variables_files/Image [8].png"/></figure>
  <li>But you overshoot, so reduce learning rate so you actually reach the minimum (green line)</li>
  <figure><img alt="" loading="lazy" src="04_Linear_Regression_with_multiple_variables_files/Image [9].png"/></figure>
  <li>So, use a smaller α</li>
  <li>
    Another problem might be if J(θ) looks like a series of waves
    <ul>
      <li>Here again, you need a smaller α</li>
    </ul>
  </li>
  <li>
    However
    <ul>
      <li>If α is small enough, J(θ) will decrease on every iteration</li>
      <li>
        BUT, if α is too small then rate is too slow
        <ul>
          <li>A less steep incline is indicative of a slow convergence, because we're decreasing by less on each iteration than a steeper slope</li>
        </ul>
      </li>
    </ul>
  </li>
  <li>
    Typically
    <ul>
      <li>Try a range of alpha values</li>
      <li>Plot J(θ) vs number of iterations for each version of alpha</li>
      <li>
        Go for roughly threefold increases
        <ul>
          <li>0.001, 0.003, 0.01, 0.03. 0.1, 0.3</li>
        </ul>
      </li>
    </ul>
  </li>
</ul>

## Features and polynomial regression

- Choice of features and how you can get different learning algorithms by choosing appropriate features
- Polynomial regression for non-linear function
- Example
  - House price prediction
    - Two features
      - Frontage - width of the plot of land along road (x<sub>1</sub>)
      - Depth - depth away from road (x<sub>2</sub>)
  - You don't have to use just two features
    - **<span class="hl-red">Can create new features</span>**
  - Might decide that an important feature is the land area
    - So, create a new feature = frontage \* depth (x<sub>3</sub>)
    - h(x) = θ<sub>0</sub> + θ<sub>1</sub>x<sub>3</sub>
      - Area is a better indicator
  - Often, by defining new features you may get a better model
- Polynomial regression
  - May fit the data better
  - θ<sub>0</sub> + θ<sub>1</sub>x + θ<sub>2</sub>x<sup>2</sup> e.g. here we have a quadratic function
  - For housing data could use a quadratic function
    - But may not fit the data so well - inflection point means housing prices decrease when size gets really big
    - So instead must use a cubic function

<figure><img alt="" loading="lazy" src="04_Linear_Regression_with_multiple_variables_files/Image [10].png"/></figure>

- How do we fit the model to this data
  - To map our old linear hypothesis and cost functions to these polynomial descriptions the easy thing to do is set
    - x<sub>1</sub> = x
    - x<sub>2</sub> = x<sup>2</sup>
    - x<sub>3</sub> = x<sup>3</sup>
  - By selecting the features like this and applying the linear regression algorithms you can do polynomial linear regression
  - Remember, feature scaling becomes even more important here
- Instead of a conventional polynomial you could do variable ^(1/something) - i.e. square root, cubed root etc
- Lots of features - later look at developing an algorithm to choose the best features

## Normal equation

- For some linear regression problems the normal equation provides a better solution
- So far we've been using gradient descent
  - Iterative algorithm which takes steps to converge
- Normal equation solves θ analytically
  - Solve for the optimum value of theta
- Has some advantages and disadvantages

### How does it work?

- Simplified cost function
  - J(θ) = aθ<sup>2 </sup>+ bθ + c
    - θ is just a real number, not a vector
  - Cost function is a quadratic function
  - How do you minimize this?
    - Do
      - <div class="eqn">
        <math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><mi>&#x3B8;</mi></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo></mrow></math>
        </div>
        - Take derivative of J(θ) with respect to θ
        - Set that derivative equal to 0
        - Allows you to solve for the value of θ which minimizes J(θ)
- In our more complex problems;
  - Here θ is an n+1 dimensional vector of real numbers
  - Cost function is a function of the vector value
    - How do we minimize this function
      - Take the partial derivative of J(θ) with respect to θ<sub>j </sub>and set to 0 for every j
      - Do that and solve for θ<sub>0</sub> to θ<sub>n</sub>
      - This would give the values of θ which minimize J(θ)
  - If you work through the calculus and the solution, the derivation is pretty complex
    - Not going to go through here
    - Instead, what do you need to know to implement this process

### Example of normal equation

<table>
  <caption>Training set &mdash; m = 4 examples, n = 4 features</caption>
  <tr>
    <th scope="col">Size in feet<sup>2</sup> (x<sub>1</sub>)</th>
    <th scope="col">Number of bedrooms (x<sub>2</sub>)</th>
    <th scope="col">Number of floors (x<sub>3</sub>)</th>
    <th scope="col">Age of home in years (x<sub>4</sub>)</th>
    <th scope="col">Price ($1000) (y)</th>
  </tr>
  <tr><td>2104</td><td>5</td><td>1</td><td>45</td><td>460</td></tr>
  <tr><td>1416</td><td>3</td><td>2</td><td>40</td><td>232</td></tr>
  <tr><td>1534</td><td>3</td><td>2</td><td>30</td><td>315</td></tr>
  <tr><td>852</td><td>2</td><td>1</td><td>36</td><td>178</td></tr>
</table>

- Here
  - m = 4
  - n = 4
- To implement the normal equation
  - Take examples
  - Add an extra column (x<sub>0</sub> feature)
  - Construct a matrix (X - **<span class="term">the design matrix</span>**) which contains all the training data features in an \[m x n+1\] matrix
  - Do something similar for y
    - Construct a column vector y vector \[m x 1\] matrix
  - Using the following equation (X transpose \* X) inverse times X transpose y<br/>
    <div class="eqn">
    <math display="block"><mrow><mi>&#x3B8;</mi><mo>=</mo><msup><mrow><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>)</mo></mrow><mrow><mo>&#x2212;</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi>T</mi></msup><mi>y</mi></mrow></math>
    </div>

<div class="eqn">
<math display="block"><mrow><msup><mrow><mo>(</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>2104</mn></mtd><mtd><mn>1416</mn></mtd><mtd><mn>1534</mn></mtd><mtd><mn>852</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>45</mn></mtd><mtd><mn>40</mn></mtd><mtd><mn>30</mn></mtd><mtd><mn>36</mn></mtd></mtr></mtable><mo>]</mo></mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2104</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>45</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1416</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>40</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1534</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>30</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>852</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>36</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>)</mo></mrow><mrow><mo>&#x2212;</mo><mn>1</mn></mrow></msup><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>2104</mn></mtd><mtd><mn>1416</mn></mtd><mtd><mn>1534</mn></mtd><mtd><mn>852</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>45</mn></mtd><mtd><mn>40</mn></mtd><mtd><mn>30</mn></mtd><mtd><mn>36</mn></mtd></mtr></mtable><mo>]</mo></mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>460</mn></mtd></mtr><mtr><mtd><mn>232</mn></mtd></mtr><mtr><mtd><mn>315</mn></mtd></mtr><mtr><mtd><mn>178</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<span class="eqn-note">That is X<sup>T</sup>X inverted, times X<sup>T</sup>, times y &mdash; the normal equation with the table above substituted in.</span>
</div>

::: code-eg
```
X = np.array([[1, 2104, 5, 1, 45],
              [1, 1416, 3, 2, 40],
              [1, 1534, 3, 2, 30],
              [1,  852, 2, 1, 36]], dtype=float)
y = np.array([460, 232, 315, 178], dtype=float)

theta = np.linalg.pinv(X.T @ X) @ X.T @ y

X @ theta   # [460., 232., 315., 178.]  -> reproduces every price exactly
```
<p class="eqn-note">The table above, solved in one line. With only four examples and five parameters the fit is exact &mdash; X @ theta returns the y column to machine precision.</p>
:::

- If you compute this, you get the value of theta which minimize the cost function

### General case

- Have m training examples and n features
  - The <span class="term">design matrix</span> (X)
    - Each training example is a n+1 dimensional feature column vector
    - X is constructed by taking each training example, determining its transpose (i.e. column -> row) and using it for a row in the design matrix
    - This creates an \[m x (n+1)\] matrix
      <div class="eqn">
      <math display="block"><mrow><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><msubsup><mi>x</mi><mn>0</mn><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mn>1</mn><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mn>2</mn><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mtd></mtr><mtr><mtd><mi>&#x22EE;</mi></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mi>n</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup></mtd></mtr></mtable><mo>]</mo></mrow><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msup></mrow></math>
      <math display="block"><mrow><mi>X</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><msup><mrow><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>1</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mi>T</mi></msup></mtd></mtr><mtr><mtd><msup><mrow><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>2</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mi>T</mi></msup></mtd></mtr><mtr><mtd><mi>&#x22EE;</mi></mtd></mtr><mtr><mtd><msup><mrow><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mi>T</mi></msup></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      <span class="eqn-note">Each training example is a column vector; its transpose becomes one row of the design matrix, giving an [m x (n+1)] matrix.</span>
      </div>
  - **<span class="term">Vector y</span>**
    - Used by taking all the y values into a column vector

<div class="eqn">
<math display="block"><mrow><mi>&#x3B8;</mi><mo>=</mo><msup><mrow><mo>(</mo><msup><mi>X</mi><mi>T</mi></msup><mi>X</mi><mo>)</mo></mrow><mrow><mo>&#x2212;</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi>T</mi></msup><mi>y</mi></mrow></math>
</div>

- What is this equation?!
  - (X<sup><em>T</em></sup> \* X)<sup>-1</sup>
    - What is this --> the inverse of the matrix (X<sup><em>T</em> </sup>\* X)
      - i.e. A = X<sup><em>T</em> </sup>X
      - A<sup>-1 </sup>= (X<sup><em>T</em> </sup>X)<sup>-1</sup>
- In Python you could do;<br/><br/><code class="hl-green"><strong> theta = np.linalg.pinv(X.T @ X) @ X.T @ y</strong></code><br/>
  - X.T is the notation for X transpose
  - np.linalg.pinv is a function for the (pseudo) inverse of a matrix
- In a previous lecture discussed feature scaling
  - If you're using the normal equation then no need for feature scaling

### When should you use gradient descent and when should you use the normal equation?

- <strong><em>Gradient descent</em></strong>
  - Need to choose learning rate
  - Needs many iterations - could make it slower
  - Works well even when *n* is massive (millions)
    - Better suited to big data
    - What is a big *n* though
      - 100 or even a 1000 is still (relatively) small
      - If n is 10 000 then look at using gradient descent
- <strong><em>Normal equation</em></strong>
  - No need to choose a learning rate
  - No need to iterate, check for convergence etc.
  - Normal equation needs to compute (X<sup><em>T</em> </sup>X)<sup>-1</sup>
    - This is the inverse of an n x n matrix
    - With most implementations computing a matrix inverse grows by O(n<sup>3 </sup>)
      - So not great
  - Slow if *n* is large
    - Can be much slower

## Normal equation and non-invertibility

- Advanced concept
  - Often asked about, but quite advanced, perhaps optional material
  - Phenomenon worth understanding, but not probably necessary
- When computing (X<sup><em>T</em> </sup>X)<sup>-1</sup> \* X<sup><em>T</em> </sup>\* y)
  - What if (X<sup><em>T</em> </sup>X) is non-invertible (singular/degenerate)
    - Only some matrices are invertible
    - This should be quite a rare problem
      - NumPy can invert matrices using
        - np.linalg.pinv (pseudo inverse)
          - This gets the right value even if (X<sup><em>T</em> </sup>X) is non-invertible
        - np.linalg.inv (inverse)
  - What does it mean for (X<sup><em>T</em> </sup>X) to be non-invertible
    - Normally two common causes
      - **<span class="term">Redundant features</span>** in learning model
        - e.g.
          - x<sub>1</sub> = size in feet
          - x<sub>2</sub> = size in meters squared
      - **<span class="term">Too many features</span>**
        - e.g. m \<= n (n is much larger than m)
          - m = 10
          - n = 100
        - Trying to fit 101 parameters from 10 training examples
        - Sometimes work, but not always a good idea
        - Not enough data
        - Later look at *why* this may be too little data
        - To solve this we
          - Delete features
          - Use **<span class="term">regularization</span>** (lets you use lots of features for a small training set)
  - If you find (X<sup><em>T</em> </sup>X) to be non-invertible
    - Look at features --> are features linearly dependent?
      - So just delete one, will solve problem
