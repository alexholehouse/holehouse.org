---
title: "02: Linear Regression with One Variable"
nav_title: "02: Linear regression"
origin: "2011"
description: "Fitting a straight line: the hypothesis, the cost function, and gradient descent from first principles."
---

## What this chapter covers

- Chapter 01 was about what machine learning is - this is our first actual algorithm
  - **Linear regression with one variable** - fit a straight line to some data
  - Deliberately the simplest useful thing, because the point is the machinery around it
- Three pieces, and they show up again for essentially every algorithm in these notes
  - **Hypothesis** - the function which does the predicting
  - **Cost function** - a single number saying how wrong that function currently is
  - **Gradient descent** - the procedure which changes the parameters to make that number smaller
- Worth getting comfortable here rather than rushing on
  - This is the one problem simple enough to picture completely - two parameters, so the cost function is a surface you can actually look at
  - That stops being true almost immediately afterwards
  - But hypothesis -> cost -> minimize is exactly the pattern used for logistic regression, neural networks and everything after

## Linear Regression

- Housing price data example used earlier
  - Supervised learning regression problem
- What do we start with?
  - Training set (this is your data set)
  - Notation (*used throughout the course*)
    - m = number of **<span class="term">training examples</span>**
    - x's = input variables / features
    - y's = output variable "target" variables
      - (x,y) - single training example
      - (x<sup>i</sup>, y<sup>i</sup>) - specific example (i<sup>th</sup> training example)
        - i is an index to training set

<table>
  <caption>Training set — m = 47 examples</caption>
  <tbody><tr><th scope="col">Size in feet<sup>2</sup> (x)</th><th scope="col">Price ($) in 1000's (y)</th></tr>
  <tr><td>2104</td><td>460</td></tr>
  <tr><td>1416</td><td>232</td></tr>
  <tr><td>1534</td><td>315</td></tr>
  <tr><td>852</td><td>178</td></tr>
  <tr><td>…</td><td>…</td></tr>
</tbody></table>

- With our training set defined - how do we used it?
  - Take training set
  - Pass into a learning algorithm
  - Algorithm outputs a function (denoted <em>h </em>) (h = **<span class="term">hypothesis</span>**)
    - This function takes an input (e.g. size of new house)
    - Tries to output the estimated value of Y
- How do we represent hypothesis <em>h </em>?
  - Going to present h as;
    - h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>x
      - h(x) (shorthand)

<div class="eqn">
<math display="block"><mrow><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>1</mn></msub><mi>x</mi></mrow></math>
</div>

- What does this mean?
  - Means Y is a linear function of x!
  - θ<sub>i</sub> are **<span class="term">parameters</span>**
    - θ<sub>0</sub> is zero condition
    - θ<sub>1</sub> is gradient
- This kind of function is a linear regression with one variable
  - Also called **<span class="term">univariate linear regression</span>**
- So in summary
  - A hypothesis takes in some variable
  - Uses parameters determined by a learning system
  - Outputs a prediction based on that input

## Linear regression - implementation (cost function)

- A cost function lets us figure out how to fit the best straight line to our data
- Choosing values for θ<sub>i</sub> (parameters)
  - Different values give you different functions
  - If θ<sub>0</sub> is 1.5 and θ<sub>1</sub> is 0 then we get straight line parallel with X along 1.5 @ y
  - If θ<sub>1</sub> is > 0 then we get a positive slope
- Based on our training set we want to generate parameters which make the straight line
  - Chosen these parameters so h<sub>θ</sub>(x) is close to y for our training examples
    - Basically, uses xs in training set with h<sub>θ</sub>(x) to give output which is as close to the actual y value as possible
    - Think of h<sub>θ</sub>(x) as a "y imitator" - it tries to convert the x into y, and considering we already have y we can evaluate how well h<sub>θ</sub>(x) does this
- To formalize this;
  - We want to solve a **<span class="term">minimization problem</span>**
  - Minimize (h<sub>θ</sub>(x) - y)<sup>2 </sup>
    - i.e. minimize the difference between h(x) and y for each/any/every example
  - Sum this over the training set

<div class="eqn">
<math display="block"><mrow><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>−</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- Minimize squared difference between predicted house price and actual house price
  - 1/2m
    - 1/m - means we determine the average
    - 1/2m the 2 makes the math a bit easier, and doesn't change the constants we determine at all (i.e. half the smallest value is still the smallest value!)
  - Minimizing θ<sub>0</sub>/θ<sub>1</sub> means we get the values of θ<sub>0</sub> and θ<sub>1</sub> which find on average the minimal deviation of x from y when we use those parameters in our hypothesis function
- More cleanly, this is a cost function

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>−</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- And we want to minimize this cost function
  - Our cost function is (because of the summation term) inherently looking at ALL the data in the training set at any time
- **So to recap**
  - **Hypothesis** - is like your prediction machine, throw in an *x* value, get a putative *y* value
    <figure><img alt="" loading="lazy" src="02_Linear_Regression_with_One_Variable_files/Image [10].png"/></figure>
  - **Cost** - is a way to, using your training data, determine values for your θ values which make the hypothesis as accurate as possible
    <div class="eqn">
    <math display="block"><mrow><munder><mi>minimize</mi><mrow><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub></mrow></munder><mspace width="0.35em"></mspace><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo></mrow></math>
    <span class="eqn-note">J(θ<sub>0</sub>,θ<sub>1</sub>) is the cost function.</span>
    </div>
    - This cost function is also called the squared error cost function
      - This cost function is reasonable choice for most regression functions
      - Probably most commonly used function
  - In case J(θ<sub>0</sub>,θ<sub>1</sub>) is a bit abstract, going into what it does, why it works and how we use it in the coming sections

### Cost function - a deeper look

- Lets consider some intuition about the cost function and why we want to use it
  - The cost function determines parameters
  - The value associated with the parameters determines how your hypothesis behaves, with different values generate different
- Simplified hypothesis
  - Assumes θ<sub>0</sub> = 0

<div class="eqn">
<math display="block"><mrow><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><msub><mi>θ</mi><mn>1</mn></msub><mi>x</mi><mspace width="2em"></mspace><mo>(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>=</mo><mn>0</mn><mo>)</mo></mrow></math>
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>−</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
<math display="block"><mrow><munder><mi>minimize</mi><mrow><msub><mi>θ</mi><mn>1</mn></msub></mrow></munder><mspace width="0.35em"></mspace><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo></mrow></math>
</div>

::: code-eg
```
import numpy as np

x = np.array([1, 2, 3])   # the simplified data set used above
y = np.array([1, 2, 3])
m = 3

def J(theta1):            # cost with theta0 = 0
    return ((theta1 * x - y) ** 2).sum() / (2 * m)

J(1)      # 0.0    - a perfect fit
J(0.5)    # 0.583  - the ~0.58 read off the plot above
J(0)      # 2.333  - the ~2.3 read off the plot above
```
<p class="eqn-note">The three worked values from the bullets above, computed rather than read off the plot.</p>
:::

- Cost function and goal here are very similar to when we have θ<sub>0</sub>, but with a simpler parameter
  - Simplified hypothesis makes visualizing cost function J() a bit easier
- So hypothesis passes through 0,0
- Two key functions we want to understand
  - h<sub>θ</sub>(x)
    - Hypothesis is a function of x - function of what the size of the house is
  - J(θ<sub>1</sub>)
    - Is a function of the parameter of θ<sub>1</sub>
  - So for example
    - θ<sub>1</sub> = 1
    - J(θ<sub>1</sub>) = 0
  - Plot
    - θ<sub>1</sub> vs J(θ<sub>1</sub>)
    - Data
      - 1\)
        - θ<sub>1</sub> = 1
        - J(θ<sub>1</sub>) = 0
      - 2\)
        - θ<sub>1</sub> = 0.5
        - J(θ<sub>1</sub>) = ~0.58
      - 3\)
        - θ<sub>1</sub> = 0
        - J(θ<sub>1</sub>) = ~2.3
- If we compute a range of values plot
  - J(θ<sub>1</sub>) vs θ<sub>1</sub> we get a polynomial (looks like a quadratic)
    <figure><img alt="" loading="lazy" src="02_Linear_Regression_with_One_Variable_files/Image [13].png"/></figure>
- The optimization objective for the learning algorithm is find the value of θ<sub>1</sub> which minimizes J(θ<sub>1</sub>)
  - So, here θ<sub>1</sub> = 1 is the best value for θ<sub>1</sub>

## A deeper insight into the cost function - simplified cost function

- Assume you're familiar with contour plots or contour figures
  - Using same cost function, hypothesis and goal as previously
  - It's OK to skip parts of this section if you don't understand contour plots
- Using our original complex hypothesis with two variables,
  - So cost function is
    - J(θ<sub>0</sub>, θ<sub>1</sub>)
- Example,
  - Say
    - θ<sub>0</sub> = 50
    - θ<sub>1</sub> = 0.06
  - Previously we plotted our cost function by plotting
    - θ<sub>1</sub> vs J(θ<sub>1</sub>)
  - Now we have two parameters
    - Plot becomes a bit more complicated
    - Generates a 3D surface plot where axis are
      - X = θ<sub>1</sub>
      - Z = θ<sub>0</sub>
      - Y = J(θ<sub>0</sub>,θ<sub>1</sub>)

<figure><img alt="" loading="lazy" src="02_Linear_Regression_with_One_Variable_files/Image [14].png"/></figure>

- We can see that the height (y) indicates the value of the cost function, so find where y is at a minimum<br/>
- Instead of a surface plot we can use a **<span class="term">contour figures/plots</span>**
  - Set of ellipses in different colors
  - Each colour is the same value of J(θ<sub>0</sub>, θ<sub>1</sub>), but obviously plot to different locations because θ<sub>1</sub> and θ<sub>0</sub> will vary
  - Imagine a bowl shape function coming out of the screen so the middle is the concentric circles

<figure><img alt="" loading="lazy" src="02_Linear_Regression_with_One_Variable_files/Image [15].png"/></figure>

- Each point (like the red one above) represents a pair of parameter values for θ<sub>0</sub> and θ<sub>1</sub>
  - Our example here put the values at
    - θ<sub>0</sub> = ~800
    - θ<sub>1</sub> = ~-0.15
  - Not a good fit
    - i.e. these parameters give a value on our contour plot far from the center
  - If we have
    - θ<sub>0</sub> = ~360
    - θ<sub>1</sub> = 0
    - This gives a better hypothesis, but still not great - not in the center of the contour plot
  - Finally we find the minimum, which gives the best hypothesis
- Doing this by eye/hand is a pain in the ass
  - What we really want is an efficient algorithm for finding the minimum for θ<sub>0</sub> and θ<sub>1</sub>

## Gradient descent algorithm

- Minimize cost function J
- Gradient descent
  - Used all over machine learning for minimization
- Start by looking at a general J() function
- Problem
  - We have J(θ<sub>0</sub>, θ<sub>1</sub>)
  - We want to get **<span class="term">min J(θ<sub>0</sub>, θ<sub>1</sub>)</span>**
- Gradient descent applies to more general functions
  - J(θ<sub>0</sub>, θ<sub>1</sub>, θ<sub>2</sub> .... θ<sub>n</sub>)
  - min J(θ<sub>0</sub>, θ<sub>1</sub>, θ<sub>2</sub> .... θ<sub>n</sub>)

### How does it work?

- Start with initial guesses
  - Start at 0,0 (or any other value)
  - Keeping changing θ<sub>0</sub> and θ<sub>1</sub> a little bit to try and reduce J(θ<sub>0</sub>,θ<sub>1</sub>)
- Each time you change the parameters, you select the gradient which reduces J(θ<sub>0</sub>,θ<sub>1</sub>) the most possible
- Repeat
- Do so until you converge to a local minimum
- Has an interesting property
  - Where you start can determine which minimum you end up
    <figure><img alt="" loading="lazy" src="02_Linear_Regression_with_One_Variable_files/Image [16].png"/></figure>
  - Here we can see one initialization point led to one local minimum
  - The other led to a different one

### A more formal definition

- Do the following until convergence

<div class="eqn">
<math display="block"><mrow><msub><mi>θ</mi><mi>j</mi></msub><mo>≔</mo><msub><mi>θ</mi><mi>j</mi></msub><mo>−</mo><mi>α</mi><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo></mrow></math>
<span class="eqn-note">for j = 0 and j = 1</span>
</div>

- What does this all mean?
  - Update θ<sub>j</sub> by setting it to (θ<sub>j</sub> - α) times the partial derivative of the cost function with respect to θ<sub>j</sub>
- Notation
  - :=
    - Denotes assignment
    - NB a = b is a *truth assertion*
  - α (alpha)
    - Is a number called the **<span class="term">learning rate</span>**
    - Controls how big a step you take
      - If α is big have an aggressive gradient descent
      - If α is small take tiny steps
- Derivative term<br/>
  <div class="eqn">
  <math display="block"><mrow><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo></mrow></math>
  </div>
  - Not going to talk about it now, derive it later
- There is a subtlety about how this gradient descent algorithm is implemented
  - Do this for θ<sub>0</sub> and θ<sub>1</sub>
  - For j = 0 and j = 1 means we **<span class="hl-red">simultaneously </span>**update both
  - How do we do this?
    - Compute the right hand side for both θ<sub>0 </sub>and θ<sub>1</sub>
      - So we need a temp value
    - Then, update θ<sub>0 </sub>and θ<sub>1</sub> at the same time
    - We show this graphically below

<div class="eqn">
<math display="block"><mrow><mi>temp0</mi><mo>≔</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>−</mo><mi>α</mi><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mn>0</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo></mrow></math>
<math display="block"><mrow><mi>temp1</mi><mo>≔</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>−</mo><mi>α</mi><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mn>1</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo></mrow></math>
<math display="block"><mrow><msub><mi>θ</mi><mn>0</mn></msub><mo>≔</mo><mi>temp0</mi></mrow></math>
<math display="block"><mrow><msub><mi>θ</mi><mn>1</mn></msub><mo>≔</mo><mi>temp1</mi></mrow></math>
<span class="eqn-note">Both parameters are updated from the <em>old</em> values — compute both temps first.</span>
</div>

::: code-eg
```
alpha = 0.1
h = theta0 + theta1 * x   # hypothesis for every example

temp0 = theta0 - alpha * (h - y).mean()
temp1 = theta1 - alpha * ((h - y) * x).mean()
theta0, theta1 = temp0, temp1   # update together, at the end
```
<p class="eqn-note">One simultaneous update, exactly as the equations above &mdash; both temps are computed from the old values before either parameter changes.</p>
:::

- If you implement the non-simultaneous update it's not gradient descent, and will behave weirdly
  - But it might look sort of right - so it's important to remember this!

### Understanding the algorithm

- To understand gradient descent, we'll return to a simpler function where we minimize one parameter to help explain the algorithm in more detail
  - min θ<sub>1</sub> J(θ<sub>1</sub>) where θ<sub>1</sub> is a real number
- Two key terms in the algorithm
  - Alpha
  - Derivative term
- Notation nuances
  - Partial derivative vs. derivative
    - Use partial derivative when we have multiple variables but only derive with respect to one
    - Use derivative when we are deriving with respect to all the variables
- Derivative term
  <div class="eqn">
  <math display="block"><mrow><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo></mrow></math>
  </div>
  - Derivative says
    - Lets take the tangent at the point and look at the slope of the line
    - So moving towards the minimum (down) will create a negative derivative, alpha is always positive, so will update J(θ<sub>1</sub>) to a smaller value
    - Similarly, if the slope is positive the derivative term is positive, so θ<sub>1</sub> is made smaller - again moving towards the minimum
- Alpha term (α)
  - What happens if alpha is too small or too large
  - Too small
    - Take baby steps
    - Takes too long
  - Too large
    - Can overshoot the minimum and fail to converge
- When you get to a local minimum
  - Gradient of tangent/derivative is 0
  - So derivative term = 0
  - alpha \* 0 = 0
  - So θ<sub>1</sub> = θ<sub>1</sub>- 0
  - So θ<sub>1</sub> remains the same
- As you approach the global minimum the derivative term gets smaller, so your update gets smaller, even if alpha is fixed
  - Means as the algorithm runs you take smaller steps as you approach the minimum
  - So no need to change alpha over time

## Linear regression with gradient descent

- Apply gradient descent to minimize the squared error cost function J(θ<sub>0</sub>, θ<sub>1</sub>)
- Now we have a partial derivative

<div class="eqn">
<math display="block"><mrow><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo><mo>=</mo><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mspace width="0.25em"></mspace><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>−</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
<math display="block"><mrow><mo>=</mo><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>j</mi></msub></mrow></mfrac><mspace width="0.25em"></mspace><mfrac><mn>1</mn><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo>(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>+</mo><msub><mi>θ</mi><mn>1</mn></msub><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>−</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- So here we're just expanding out the first expression
  - J(θ<sub>0</sub>, θ<sub>1</sub>) = 1/2m....
  - h<sub>θ</sub>(x) = θ<sub>0</sub> + θ<sub>1</sub>\*x
- So we need to determine the derivative for each parameter - i.e.
  - When j = 0
  - When j = 1
- Figure out what this partial derivative is for the θ<sub>0</sub> and θ<sub>1</sub> case
  - When we derive this expression in terms of j = 0 and j = 1 we get the following

<div class="eqn">
<math display="block"><mrow><mi>j</mi><mo>=</mo><mn>0</mn><mo>:</mo><mspace width="0.5em"></mspace><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mn>0</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>−</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow></mrow></math>
<math display="block"><mrow><mi>j</mi><mo>=</mo><mn>1</mn><mo>:</mo><mspace width="0.5em"></mspace><mfrac><mrow><mo>∂</mo></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mn>1</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>,</mo><msub><mi>θ</mi><mn>1</mn></msub><mo form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo>(</mo><msub><mi>h</mi><mi>θ</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>−</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mo>⋅</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup></mrow></math>
</div>

::: code-eg
```
theta = np.zeros(2)
for _ in range(2000):
    h = theta[0] + theta[1] * x
    grad0 = (h - y).mean()           # the j = 0 derivative
    grad1 = ((h - y) * x).mean()     # the j = 1 derivative
    theta = theta - 0.1 * np.array([grad0, grad1])

theta   # [0., 1.]  -> h(x) = 0 + 1x, the perfect fit, found by descent
```
<p class="eqn-note">Batch gradient descent run to convergence on the simple data set: it lands on θ<sub>0</sub> = 0, θ<sub>1</sub> = 1 with cost zero.</p>
:::

- To check this you need to know multivariate calculus
  - So we can plug these values back into the gradient descent algorithm
- How does it work
  - Risk of meeting different local optimum
  - The linear regression cost function is always a **<span class="term">convex function</span>** - always has a single minimum
    - Bowl shaped
    - One global optima
      - So gradient descent will always converge to global optima
  - In action
    - Initialize values to
      - θ<sub>0</sub> = 900
      - θ<sub>1</sub> = -0.1

<figure><img alt="" loading="lazy" src="02_Linear_Regression_with_One_Variable_files/Image [23].png"/></figure>

- End up at a global minimum
- This is actually **<span class="term">Batch Gradient Descent</span>**
  - Refers to the fact that over each step you look at all the training data
    - Each step compute over m training examples
  - Sometimes non-batch versions exist, which look at small data subsets
    - We'll look at other forms of gradient descent (to use when m is too large) later in the course
- There exists an analytical solution for finding the minimum directly
  - **<span class="term">Normal equations</span>** method
  - Gradient descent scales better to large data sets though
  - Used in lots of contexts and machine learning

**What's next - important extensions**<br/>*Two extension to the algorithm*

- **1) Normal equation for analytic solution**
  - To solve the minimization problem we can solve it \[ min J(θ<sub>0</sub>, θ<sub>1</sub>) \] exactly using an analytic method which avoids the iterative approach used by gradient descent
  - Normal equations method
  - Has advantages and disadvantages
    - Advantage
      - No longer an alpha term
      - Can be much faster for some problems
    - Disadvantage
      - Much more complicated
  - We discuss the normal equation in the **<span class="hl">linear regression with multiple features</span>** section
- **2) We can learn with a larger number of features**
  - So may have other parameters which contribute towards a price
    - e.g. with houses
      - Size
      - Age
      - Number bedrooms
      - Number floors
    - x1, x2, x3, x4
  - With multiple features becomes hard to plot
    - Can't really plot in more than 3 dimensions
    - Notation becomes more complicated too
      - Best way to get around this is the notation of linear algebra
      - Gives notation and set of things you can do with matrices and vectors
      - e.g. Matrix

<div class="eqn">
<math display="block"><mrow><mi>X</mi><mo>=</mo><mrow><mo>[</mo><mtable>
<mtr><mtd><mn>2104</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>45</mn></mtd></mtr>
<mtr><mtd><mn>1416</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>40</mn></mtd></mtr>
<mtr><mtd><mn>1534</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>30</mn></mtd></mtr>
<mtr><mtd><mn>852</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>36</mn></mtd></mtr>
</mtable><mo>]</mo></mrow></mrow></math>
</div>

<div class="eqn">
<math display="block"><mrow><mi>y</mi><mo>=</mo><mrow><mo>[</mo><mtable>
<mtr><mtd><mn>460</mn></mtd></mtr>
<mtr><mtd><mn>232</mn></mtd></mtr>
<mtr><mtd><mn>315</mn></mtd></mtr>
<mtr><mtd><mn>178</mn></mtd></mtr>
</mtable><mo>]</mo></mrow></mrow></math>
</div>

- We see here this matrix shows us
  - Size
  - Number of bedrooms
  - Number floors
  - Age of home
- All in one variable
  - Block of numbers, take all data organized into one big block
- Vector
  - Shown as *y*
  - Shows us the prices
- Need linear algebra for more complex linear regression models
- Linear algebra is good for making computationally efficient models (as seen later too)
  - Provide a good way to work with large sets of data sets
  - Typically vectorization of a problem is a common optimization technique
