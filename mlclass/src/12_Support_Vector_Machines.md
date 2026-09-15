---
title: "12: Support Vector Machines (SVMs)"
nav_title: "12: SVMs"
index_title: "12: Support Vector Machines"
origin: "2011"
description: "Large-margin classification, the kernel trick for non-linear boundaries, and when to prefer an SVM over logistic regression."
---

## Support Vector Machine (SVM) - Optimization objective

- So far, we've seen a range of different algorithms
  - With supervised learning algorithms - performance is pretty similar
    - What matters more often is;
      - The amount of training data
      - Skill of applying algorithms
- One final supervised learning algorithm that is widely used - **<span class="term">support vector machine (SVM)</span>**
  - Compared to both logistic regression and neural networks, a SVM sometimes gives a cleaner way of learning non-linear functions
  - Later in the course we'll do a survey of different supervised learning algorithms

### An alternative view of logistic regression

- Start with logistic regression, see how we can modify it to get the SVM
  - As before, the logistic regression hypothesis is as follows
    <div class="eqn">
    <math display="block"><mrow><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo form="prefix">&#x2212;</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac></mrow></math>
    </div>
  - And the sigmoid activation function looks like this
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [1].png"/></figure>
  - In order to explain the math, we use z as defined above
- What do we want logistic regression to do?
  - We have an example where y = 1
    - Then we hope h<sub>θ</sub>(x) is close to 1
    - With h<sub>θ</sub>(x) close to 1, (θ<sup><em>T</em></sup> x) must be **<span class="term">much larger</span>** than 0
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [2].png"/></figure>
  - Similarly, when y = 0
    - Then we hope h<sub>θ</sub>(x) is close to 0
    - With h<sub>θ</sub>(x) close to 0, (θ<sup><em>T</em></sup> x) must be **<span class="term">much less</span>** than 0
  - This is our classic view of logistic regression
    - Let's consider another way of thinking about the problem
- Alternative view of logistic regression
  - If you look at cost function, each example contributes a term like the one below to the overall cost function
    <div class="eqn">
    <math display="block"><mrow><mo form="prefix">&#x2212;</mo><mo stretchy="false">(</mo><mi>y</mi><mspace width="0.2em"/><mi>log</mi><mspace width="0.2em"/><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><mi>y</mi><mo stretchy="false">)</mo><mspace width="0.2em"/><mi>log</mi><mo form="prefix">(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo form="postfix">)</mo><mo form="postfix">)</mo></mrow></math>
    </div>
    - For the overall cost function, we sum over all the training examples using the above function, and have a 1/m term
- If you then plug in the hypothesis definition (h<sub>θ</sub>(x)), you get an expanded cost function equation;
  <div class="eqn">
  <math display="block"><mrow><mo>=</mo><mo form="prefix">&#x2212;</mo><mi>y</mi><mspace width="0.2em"/><mi>log</mi><mspace width="0.25em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo form="prefix">&#x2212;</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac><mo>&#x2212;</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><mi>y</mi><mo stretchy="false">)</mo><mspace width="0.2em"/><mi>log</mi><mo form="prefix">(</mo><mn>1</mn><mo>&#x2212;</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo form="prefix">&#x2212;</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><mi>x</mi></mrow></msup></mrow></mfrac><mo form="postfix">)</mo></mrow></math>
  </div>
  - So each training example contributes that term to the cost function for logistic regression
- If y = 1 then only the first term in the objective matters
  - If we plot the functions vs. z we get the following graph
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [5].png"/></figure>
    - This plot shows the cost contribution of an example when y = 1 given z
      - So if z is big, the cost is low - this is good!
      - But if z is 0 or negative the cost contribution is high
      - This is why, when logistic regression sees a positive example, it tries to set θ<sup><em>T</em></sup> x to be a very large term
- If y = 0 then only the second term matters
  - We can again plot it and get a similar graph
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [6].png"/></figure>
    - Same deal, if z is small then the cost is low
      - But if z is large then the cost is massive

### SVM cost functions from logistic regression cost functions

- To build a SVM we must redefine our cost functions
  - When y = 1
    - Take the y = 1 function and create a new cost function
    - Instead of a curved line create two straight lines (magenta) which acts as an approximation to the logistic regression y = 1 function
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [7].png"/></figure>
      - Take point (1) on the z axis
        - Flat from 1 onwards
        - Grows when we reach 1 or a lower number
      - This means we have two straight lines
        - Flat when cost is 0
        - Straight growing line after 1
    - So this is the new y=1 cost function
      - Gives the SVM a computational advantage and an easier optimization problem
      - We call this function <span class="term"><strong>cost</strong><strong><sub>1</sub>(z)</strong> </span>
- Similarly
  - When y = 0
    - Do the equivalent with the y=0 function plot
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [8].png"/></figure>
    - We call this function <span class="term"><strong>cost</strong><strong><sub>0</sub>(z)</strong></span>
- So here we define the two cost function terms for our SVM graphically
  - How do we implement this?

### The complete SVM cost function

- As a comparison/reminder we have logistic regression below
  <div class="eqn">
  <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>&#x3b8;</mi></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><mo form="prefix">[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mi>log</mi><mspace width="0.2em"/><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo form="postfix">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mi>log</mi><mo form="prefix">(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3b8;</mi></msub><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo form="postfix">)</mo><mo form="postfix">)</mo><mo form="postfix">]</mo><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
  </div>
  - If this looks unfamiliar its because we previously had the - sign outside the expression
- For the SVM we take our two logistic regression y=1 and y=0 terms described previously and replace with
  - cost<sub>1</sub>(θ<sup><em>T</em></sup> x)
  - cost<sub>0</sub>(θ<sup><em>T</em></sup> x)
- So we get
  <div class="eqn">
  <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>&#x3b8;</mi></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mo form="prefix">[</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><msub><mtext>cost</mtext><mn>1</mn></msub><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><msub><mtext>cost</mtext><mn>0</mn></msub><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo form="postfix">]</mo><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
  <span class="eqn-note">The slide indexes the regularisation sum over i = 1 &#x2026; n while its summand is &#x3b8;<sub>j</sub>; the index has to be j, as it is on the previous slide. Corrected here.</span>
  </div>

### SVM notation is slightly different

- In convention with SVM notation we rename a few things here
- <u>1) Get rid of the 1/m terms</u>
  - This is just a slightly different convention
  - By removing 1/m we should get the same optimal values for
    - 1/m is a constant, so should get same optimization
    - e.g. say you have a minimization problem which minimizes to u = 5
      - If your cost function \* by a constant, you still generate the minimal value
      - That minimal value is different, but that's irrelevant
- <u>2) For logistic regression we had two terms;</u>
  - Training data set term (i.e. that we sum over m) = **A**
  - Regularization term (i.e. that we sum over n) = **B**
    - So we could describe it as A + λB
    - Need some way to deal with the trade-off between regularization and data set terms
    - Set different values for λ to parametrize this trade-off
  - Instead of parameterizing this as A + λB
    - For SVMs the convention is to use a different parameter called C
    - So do CA + B
    - If C were equal to 1/λ then the two functions (CA + B and A + λB) would give the same value
- So, our overall equation is
  <div class="eqn">
  <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>&#x3b8;</mi></munder><mspace width="0.35em"/><mi>C</mi><mspace width="0.2em"/><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mo form="prefix">[</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><msub><mtext>cost</mtext><mn>1</mn></msub><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><msub><mtext>cost</mtext><mn>0</mn></msub><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo form="postfix">]</mo><mo>+</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
  <span class="eqn-note">Same correction to the regularisation index as on the previous slide.</span>
  </div>
- Unlike logistic, h<sub>θ</sub>(x) doesn't give us a probability, but instead we get a direct prediction of 1 or 0
  - So if θ<sup><em>T</em></sup> x is equal to or greater than 0 --> h<sub>θ</sub>(x) = 1
  - Else --> h<sub>θ</sub>(x) = 0

## Large margin intuition

- Sometimes people refer to SVM as **<span class="term">large margin classifiers</span>**
  - We'll consider what that means and what an SVM hypothesis looks like
  - The SVM cost function is as above, and we've drawn out the cost terms below
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [12].png"/></figure>
    <div class="eqn">
    <math display="block"><mtable><mtr><mtd columnalign="left"><mtext>If y = 1, we want</mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>&#x3b8;</mi><mi>T</mi></msup><mi>x</mi><mo>&#x2265;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>(not just</mtext><mspace width="0.3em"/><mo>&#x2265;</mo><mn>0</mn><mtext>)</mtext></mtd></mtr><mtr><mtd columnalign="left"><mtext>If y = 0, we want</mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>&#x3b8;</mi><mi>T</mi></msup><mi>x</mi><mo>&#x2264;</mo><mo form="prefix">&#x2212;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>(not just</mtext><mspace width="0.3em"/><mo>&lt;</mo><mn>0</mn><mtext>)</mtext></mtd></mtr></mtable></math>
    <span class="eqn-note">Left: y = 1, where the cost is zero once &#x3b8;<sup>T</sup>x reaches 1. Right: y = 0, where it is zero once &#x3b8;<sup>T</sup>x drops to &#x2212;1. The SVM asks for a margin, not merely the right side of zero.</span>
    </div>
    ::: code-eg
    ```
    import numpy as np

    def cost1(z):   # the y = 1 cost: zero once z >= 1
        return np.maximum(0, 1 - z)

    def cost0(z):   # the y = 0 cost: zero once z <= -1
        return np.maximum(0, 1 + z)

    cost1(np.array([-1.0, 0, 1, 2]))   # [2., 1., 0., 0.]
    cost0(np.array([-2.0, -1, 0, 1]))  # [0., 0., 1., 2.]
    ```
    <p class="eqn-note">The two straight-line costs drawn above: flat at zero beyond the margin, growing linearly inside it.</p>
    :::
  - Left is cost<sub>1</sub> and right is cost<sub>0</sub>
  - What does it take to make terms small
    - If y =1
      - cost<sub>1</sub>(z) = 0 only when z >= 1
    - If y = 0
      - cost<sub>0</sub>(z) = 0 only when z \<= -1
  - Interesting property of SVM
    - If you have a positive example, you only really *need* z to be greater or equal to 0
      - If this is the case then you predict 1
    - SVM wants a bit more than that - doesn't want to \*just\* get it right, but have the value be quite a bit bigger than zero
      - Throws in an extra safety margin factor
- Logistic regression does something similar
- What are the consequences of this?
  - Consider a case where we set C to be huge
    - C = 100,000
    - So considering we're minimizing CA + B
      - If C is huge we're going to pick an A value so that A is equal to zero
      - What is the optimization problem here - how do we make A = 0?
    - Making A = 0
      - If y = 1
        - Then to make our "A" term 0 need to find a value of θ so (θ<sup><em>T</em></sup> x) is greater than or equal to 1
      - Similarly, if y = 0
        - Then we want to make "A" = 0 so we need to find a value of θ so (θ<sup><em>T</em></sup> x) is equal to or less than -1
    - So - if we think of our optimization problem a way to ensure that this first "A" term is equal to 0, we re-factor our optimization problem into just minimizing the "B" (regularization) term, because
      - When A = 0 --> A\*C = 0
    - So we're minimizing B, under the constraints shown below
      <div class="eqn">
      <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>&#x3b8;</mi></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
      <math display="block"><mtable><mtr><mtd columnalign="left"><mtext>s.t.</mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2265;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>if</mtext><mspace width="0.35em"/><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mn>1</mn></mtd></mtr><mtr><mtd columnalign="left"><mtext></mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2264;</mo><mo form="prefix">&#x2212;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>if</mtext><mspace width="0.35em"/><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mn>0</mn></mtd></mtr></mtable></math>
      </div>
  - Turns out when you solve this problem you get interesting decision boundaries
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [14].png"/></figure>
  - The green and magenta lines are functional decision boundaries which could be chosen by logistic regression
    - But they probably don't generalize too well
  - The black line, by contrast is the one chosen by the SVM because of this safety net imposed by the optimization graph
    - More robust separator
  - Mathematically, that black line has a larger minimum distance (margin) from any of the training examples
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [15].png"/></figure>
  - By separating with the largest margin you incorporate robustness into your decision making process
- We looked at this at when C is very large
  - SVM is more sophisticated than the large margin might look
    - If you were just using large margin then SVM would be very sensitive to outliers
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [16].png"/></figure>
    - You would risk letting a single ridiculous outlier hugely impact your classification boundary
      - A single example might not represent a good reason to change an algorithm
      - If C is very large then we *do* use this quite naive maximize the margin approach
        <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [17].png"/></figure>
      - So we'd change the black to the magenta
    - But if C is reasonably small, or a not too large, then you stick with the black decision boundary
  - What about non-linearly separable data?
    - Then SVM still does the right thing if you use a normal size C
    - So the idea of SVM being a large margin classifier is only really relevant when you have no outliers and you have easily linearly separable data
  - Means we ignore a few outliers

## Large margin classification mathematics (optional)

### Vector inner products

- Have two (2D) vectors u and v - what is the inner product (*u*<sup><em>T</em></sup> *v*)?
  <div class="eqn">
  <math display="block"><mrow><mi>u</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><msub><mi>u</mi><mn>1</mn></msub></mtd></mtr><mtr><mtd><msub><mi>u</mi><mn>2</mn></msub></mtd></mtr></mtable><mo>]</mo></mrow><mspace width="1.4em"/><mi>v</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><msub><mi>v</mi><mn>1</mn></msub></mtd></mtr><mtr><mtd><msub><mi>v</mi><mn>2</mn></msub></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
  </div>
  - Plot *u* on graph
    - i.e *u*<sub>1</sub> vs. *u*<sub>2<figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [19].png"/></figure></sub>
  - One property which is good to have is the <span class="term"><strong>norm</strong> </span>of a vector
    - Written as ||u||
      - This is the euclidean length of vector u
    - So ||u|| = SQRT(*u*<sub>1</sub><sup><em>2</em></sup> + *u*<sub>2</sub><sup><em>2</em></sup>) = real number
      - i.e. length of the arrow above
      - Can show via Pythagoras
  - For the inner product, take *v* and orthogonally project down onto u
    - First we can plot v on the same axis in the same way (*v*<sub>1 </sub>vs *v*<sub>2</sub>)
    - Measure the length/magnitude of the projection
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [20].png"/></figure>
    - So here, the green line is the projection
      - p = length along u to the intersection
      - p is the magnitude of the projection of vector *v* onto vector *u*
  - Possible to show that
    - *u*<sup><em>T</em></sup> *v* = p \* ||u||
      - So this is one way to compute the inner product
    - *u*<sup><em>T</em></sup> <em>v = </em>*u*<sub>1</sub>*v*<sub>1</sub>+ *u*<sub>2</sub>*v*<sub>2</sub>
    - So therefore
      - **<span class="hl-red">p * ||u|| = <em>u</em><sub>1</sub><em>v</em><sub>1</sub>+ <em>u</em><sub>2</sub><em>v</em><sub>2</sub></span>**
      - This is an important rule in linear algebra
    - We can reverse this too
      - So we could do
        - *v*<sup><em>T</em></sup> *u = v*<sub>1</sub>*u*<sub>1</sub>+ v<sub>2</sub>*u*<sub>2</sub>
        - Which would obviously give you the same number
  - p can be negative if the angle between them is more than 90 degrees
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [21].png"/></figure>
    - So here p is negative
- Use the vector inner product theory to try and understand SVMs a little better

### SVM decision boundary

<div class="eqn">
<math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>&#x3b8;</mi></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
<math display="block"><mtable><mtr><mtd columnalign="left"><mtext>s.t.</mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2265;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>if</mtext><mspace width="0.35em"/><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mn>1</mn></mtd></mtr><mtr><mtd columnalign="left"><mtext></mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2264;</mo><mo form="prefix">&#x2212;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>if</mtext><mspace width="0.35em"/><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mn>0</mn></mtd></mtr></mtable></math>
</div>

- For the following explanation - two simplification
  - Set θ<sub>0</sub>= 0 (i.e. ignore intercept terms)
  - Set n = 2 - (x<sub>1</sub>, x<sub>2</sub>)
    - i.e. each example has only 2 features
- Given we only have two parameters we can simplify our function to
  <div class="eqn">
  <math display="block"><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><mo form="prefix">(</mo><msubsup><mi>&#x3b8;</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>&#x3b8;</mi><mn>2</mn><mn>2</mn></msubsup><mo form="postfix">)</mo></mrow></math>
  </div>
- And, can be re-written as
  <div class="eqn">
  <math display="block"><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><msup><mrow><mo>(</mo><msqrt><msubsup><mi>&#x3b8;</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>&#x3b8;</mi><mn>2</mn><mn>2</mn></msubsup></msqrt><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
  </div>
  - Should give same thing
- We may notice that
  <div class="eqn">
  <math display="block"><mrow><msqrt><msubsup><mi>&#x3b8;</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>&#x3b8;</mi><mn>2</mn><mn>2</mn></msubsup></msqrt><mo>=</mo><mrow><mo stretchy="false">&#x2016;</mo><mi>&#x3b8;</mi><mo stretchy="false">&#x2016;</mo></mrow></mrow></math>
  <span class="eqn-note">The slide rings the square root on the line above and labels it &#x2016;&#x3b8;&#x2016; &#x2014; the length of the vector &#x3b8;. That is the step that turns the sum of squares into a norm.</span>
  </div>
  - The term in red is the norm of θ
    - If we take θ as a 2x1 vector
    - If we assume θ<sub>0</sub> = 0 its still true
- So, finally, this means our optimization function can be re-defined as
  <div class="eqn">
  <math display="block"><mrow><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><msup><mrow><mo stretchy="false">&#x2016;</mo><mi>&#x3b8;</mi><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></math>
  </div>
- So the SVM is minimizing the squared norm
- Given this, what are the (θ<sup><em>T</em></sup> x) parameters doing?
  - Given θ and given example x what is this equal to
    - We can look at this in a comparable manner to how we just looked at u and v
  - Say we have a single positive training example (red cross below)
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [27].png"/></figure>
  - Although we haven't been thinking about examples as vectors it can be described as such
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [28].png"/></figure>
  - Now, say we have our parameter vector θ and we plot that on the same axis
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [29].png"/></figure>
  - The next question is what is the inner product of these two vectors
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [30].png"/></figure>
    - p, is in fact p<sup>i</sup>, because it's the length of p for example i
      - Given our previous discussion we know <br/>(θ<sup><em>T</em></sup> x<sup><em>i</em> </sup>) = p<sup><em>i</em> </sup>\* ||θ||<br/> = θ<sub>1</sub>x<sup>i</sup><sub>1</sub> + θ<sub>2</sub>x<sup>i</sup><sub>2</sub>
      - So these are both equally valid ways of computing θ<sup><em>T</em></sup> x<sup><em>i</em></sup>
- What does this mean?
  - The constraints we defined earlier
    - (θ<sup><em>T</em></sup> x) >= 1 if y = 1
    - (θ<sup><em>T</em></sup> x) \<= -1 if y = 0
  - Can be replaced/substituted with the constraints
    - p<sup><em>i</em> </sup>\* ||θ|| >= 1 if y = 1
    - p<sup><em>i</em> </sup>\* ||θ|| \<= -1 if y = 0
  - Writing that into our optimization objective
    <div class="eqn">
    <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>&#x3b8;</mi></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><msup><mrow><mo stretchy="false">&#x2016;</mo><mi>&#x3b8;</mi><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></math>
    <math display="block"><mtable><mtr><mtd columnalign="left"><mtext>s.t.</mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>p</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x22c5;</mo><mrow><mo stretchy="false">&#x2016;</mo><mi>&#x3b8;</mi><mo stretchy="false">&#x2016;</mo></mrow><mo>&#x2265;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>if</mtext><mspace width="0.35em"/><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mn>1</mn></mtd></mtr><mtr><mtd columnalign="left"><mtext></mtext></mtd><mtd columnalign="left"><mspace width="0.6em"/><msup><mi>p</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x22c5;</mo><mrow><mo stretchy="false">&#x2016;</mo><mi>&#x3b8;</mi><mo stretchy="false">&#x2016;</mo></mrow><mo>&#x2264;</mo><mo form="prefix">&#x2212;</mo><mn>1</mn></mtd><mtd columnalign="left"><mspace width="1.2em"/><mtext>if</mtext><mspace width="0.35em"/><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mn>0</mn></mtd></mtr></mtable></math>
    <span class="eqn-note">The slide gives the condition on the second constraint as y = 1, which is the case the first constraint already covers. It must be y = 0. Corrected here.</span>
    </div>
- So, given we've redefined these functions let us now consider the training example below
  <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [32].png"/></figure>
  - Given this data, what boundary will the SVM choose? Note that we're still assuming θ<sub>0</sub> = 0, which means the boundary has to pass through the origin (0,0)
    - Green line - small margins
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [33].png"/></figure>
      - SVM would not choose this line
        - Decision boundary comes very close to examples
        - Lets discuss *why* the SVM would **not** choose this decision boundary
  - Looking at this line
    - We can show that θ is at 90 degrees to the decision boundary
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [34].png"/></figure>
      - **<span class="hl-red">θ is always at 90 degrees to the decision boundary</span>** (can show with linear algebra, although we're not going to!)
- So now lets look at what this implies for the optimization objective
  - Look at first example (x<sup>1</sup>)
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [35].png"/></figure>
  - Project a line from x<sup>1 </sup>on to the θ vector (so it hits at 90 degrees)
    - The distance between the intersection and the origin is (**<span class="hl-red">p<sup>1</sup></span>**)
  - Similarly, look at second example (x<sup>2</sup>)
    - Project a line from x<sup>2</sup> on to the θ vector
    - This is the magenta line, which will be **<span class="hl-purple">negative </span>**(**<span class="hl-purple">p<sup>2</sup></span>**)
  - If we overview these two lines below we see a graphical representation of what's going on;
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [36].png"/></figure>
  - We find that both these p values are going to be pretty small
  - If we look back at our optimization objective
    - We know we need p<sup>1</sup> \* ||θ|| to be bigger than or equal to 1 for positive examples
      - If p is small
        - <span class="hl-red">Means that ||θ|| must be pretty large</span>
    - Similarly, for negative examples we need p<sup>2</sup> \* ||θ|| to be smaller than or equal to -1
      - We saw in this example p<sup>2</sup> is a small negative number
        - <span class="hl-red">So ||θ|| must be a large number</span>
  - Why is this a problem?
    - The optimization objective is trying to find a set of parameters where the norm of theta is small
      - So this doesn't seem like a good direction for the parameter vector (because as p values get smaller ||θ|| must get larger to compensate)
        - So we should make p values larger which allows ||θ|| to become smaller
- So lets choose a different boundary
  <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [37].png"/></figure>
  - Now if you look at the projection of the examples to θ we find that p<sup>1 </sup>becomes large and ||θ|| can become small
  - So with some values drawn in
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [38].png"/></figure>
  - This means that by choosing this second decision boundary we can make ||θ|| smaller
    - Which is why the SVM chooses this hypothesis as better
    - This is how we generate the large margin effect
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [39].png"/></figure>
    - The magnitude of this margin is a function of the p values
      - So by maximizing these p values we minimize ||θ||
- Finally, we did this derivation assuming θ<sub>0</sub> = 0,
  - If this is the case we're entertaining only decision boundaries which pass through (0,0)
  - If you allow θ<sub>0</sub> to be other values then this simply means you can have decision boundaries which cross through the x and y values at points other than (0,0)
  - Can show with basically same logic that this works, and even when θ<sub>0</sub> is non-zero when you have optimization objective described above (when C is very large) that the SVM is looking for a large margin separator between the classes

## Kernels - 1: Adapting SVM to non-linear classifiers

- What are kernels and how do we use them
  - We have a training set
  - We want to find a non-linear boundary
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [40].png"/></figure>
  - Come up with a complex set of polynomial features to fit the data
    - Have h<sub>θ</sub>(x) which
      - Returns 1 if the combined weighted sum of vectors (weighted by the parameter vector) is greater than or equal to 0
      - Else return 0
    - Another way of writing this (new notation) is
      - That a hypothesis computes a decision boundary by taking the sum of the parameter vector multiplied by a **<span class="term">new feature vector f</span>**, which simply contains the various high order x terms
      - e.g.
        - h<sub>θ</sub>(x) = θ<sub>0</sub>+ θ<sub>1</sub>f<sub>1</sub>+ θ<sub>2</sub>f<sub>2 </sub>+ θ<sub>3</sub>f<sub>3</sub>
        - Where
          - f<sub>1</sub>= x<sub>1</sub>
          - f<sub>2 </sub>= x<sub>1</sub>x<sub>2</sub>
          - f<sub>3 </sub>= ...
          - i.e. not specific values, but each of the terms from your complex polynomial function
    - Is there a better choice of feature f than the high order polynomials?
      - As we saw with computer imaging, high order polynomials become computationally expensive
- New features
  - Define three features in this example (ignore x<sub>0</sub>)
  - Have a graph of x<sub>1 </sub>vs. x<sub>2 </sub>(don't plot the values, just define the space)
  - Pick three points in that space
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [41].png"/></figure>
  - These points l<sup>1</sup>, l<sup>2</sup>, and l<sup>3</sup>, were chosen manually and are called **<span class="term">landmarks</span>**
    - Given x, define f1 as the similarity between (x, l<sup>1</sup>)
      - \= exp(- (|| x - l<sup>1 </sup>||<sup>2</sup> ) / 2σ<sup>2</sup>)<br/> =
        <div class="eqn">
        <math display="block"><mrow><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><msup><mrow><mo stretchy="false">&#x2016;</mo><mi>x</mi><mo>&#x2212;</mo><msup><mi>l</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow><mrow><mn>2</mn><msup><mi>&#x3c3;</mi><mn>2</mn></msup></mrow></mfrac><mo form="postfix">)</mo></mrow></math>
        </div>
        ::: code-eg
        ```
        def kernel(x, l, sigma2=1.0):   # the Gaussian kernel
            return np.exp(-((x - l) ** 2).sum() / (2 * sigma2))

        l1 = np.array([3, 5])              # a landmark

        kernel(np.array([3, 5]), l1)   # 1.0     - x on the landmark
        kernel(np.array([6, 9]), l1)   # 3.7e-06 - x far away: effectively zero
        ```
        <p class="eqn-note">The similarity function computed: 1 on the landmark, and it dies off to nothing within a few &sigma; of it.</p>
        :::
      - **<span class="hl-red">|| x - </span>**<strong><span class="hl-red">l<sup>1 </sup></span></strong>**<span class="hl-red">||</span>** is the euclidean distance between the point x and the landmark l<sup>1 </sup>squared
        - Discussed more later
      - If we remember our statistics, we know that
        - σ is the **<span class="term">standard </span>**<strong><span class="term">deviation</span></strong>
        - σ<sup>2 </sup>is commonly called the **<span class="term">variance</span>**
    - Remember, that as discussed
      <div class="eqn">
      <math display="block"><mrow><msup><mrow><mo stretchy="false">&#x2016;</mo><mi>x</mi><mo>&#x2212;</mo><msup><mi>l</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup><mo>=</mo><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><mrow><mo>(</mo><msub><mi>x</mi><mi>j</mi></msub><mo>&#x2212;</mo><msup><msub><mi>l</mi><mi>j</mi></msub><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
      </div>
  - So, f2 is defined as
    - f2 = similarity(x, l<sup>2</sup>) = exp(- (|| x - l<sup>2 </sup>||<sup>2 </sup>) / 2σ<sup>2</sup>)
  - And similarly
    - f3 = similarity(x, l<sup>3</sup>) = exp(- (|| x - l<sup>3 </sup>||<sup>2 </sup>) / 2σ<sup>2</sup>)
  - This similarity function is called a **<span class="term">kernel</span>**
    - This function is a **<span class="term">Gaussian Kernel</span>**
  - So, instead of writing similarity between x and l we might write
    - f1 = k(x, l<sup>1</sup>)

### Diving deeper into the kernel

- So lets see what these kernels do and why the functions defined make sense
  - Say x is close to a landmark
    - Then the squared distance will be ~0
      - So
        <div class="eqn">
        <math display="block"><mrow><msub><mi>f</mi><mn>1</mn></msub><mo>&#x2248;</mo><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><msup><mn>0</mn><mn>2</mn></msup></mrow><mrow><mn>2</mn><msup><mi>&#x3c3;</mi><mn>2</mn></msup></mrow></mfrac><mo form="postfix">)</mo></mrow></math>
        <span class="eqn-note">When x sits on the landmark the distance is zero, so the exponent is zero and f&#x2081; &#x2248; 1.</span>
        </div>
        - Which is basically e<sup>-0</sup>
          - Which is close to 1
    - Say x is far from a landmark
      - Then the squared distance is big
        - Gives e<sup>-large number</sup>
          - Which is close to zero
    - Each landmark defines a new features
- If we plot f1 vs the kernel function we get a plot like this
  - Notice that when x = \[3,5\] then f1 = 1
  - As x moves away from \[3,5\] then the feature takes on values close to zero
  - So this measures how close x is to this landmark
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [45].png"/></figure>

### What does σ do?

- **<span class="term">σ<sup>2 </sup></span>**is a parameter of the Gaussian kernel
  - Defines the steepness of the rise around the landmark
- Above example σ<sup>2</sup> = 1
- Below σ<sup>2</sup> = 0.5
  <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [46].png"/></figure>
  - We see here that as you move away from 3,5 the feature f1 falls to zero much more rapidly
- The inverse can be seen if σ<sup>2</sup> = 3
  <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [47].png"/></figure>
- Given this definition, what kinds of hypotheses can we learn?
  - With training examples x we predict "1" when
  - θ<sub>0</sub>+ θ<sub>1</sub>f<sub>1</sub>+ θ<sub>2</sub>f<sub>2 </sub>+ θ<sub>3</sub>f<sub>3</sub> >= 0
    - For our example, lets say we've already run an algorithm and got the
      - θ<sub>0 </sub>= -0.5
      - θ<sub>1 </sub>= 1
      - θ<sub>2 </sub>= 1
      - θ<sub>3 </sub>= 0
    - Given our placement of three examples, what happens if we evaluate an example at the **<span class="hl-purple">magenta dot</span>** below?
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [48].png"/></figure>
    - Looking at our formula, we know f1 will be close to 1, but f2 and f3 will be close to 0
      - So if we look at the formula we have
        - θ<sub>0</sub>+ θ<sub>1</sub>f<sub>1</sub>+ θ<sub>2</sub>f<sub>2 </sub>+ θ<sub>3</sub>f<sub>3</sub> >= 0
        - \-0.5 + 1 + 0 + 0 = 0.5
          - 0.5 is greater than 0
            ::: code-eg
            ```
            theta = np.array([-0.5, 1, 1, 0])   # theta0..theta3 from above

            f_near = np.array([1, 1, 0, 0])     # near l1: f1 ~ 1, f2 and f3 ~ 0
            theta @ f_near                      # 0.5  >= 0 -> predict y = 1

            f_far = np.array([1, 0, 0, 0])      # far from all three landmarks
            theta @ f_far                       # -0.5 < 0 -> predict y = 0
            ```
            <p class="eqn-note">The worked example as arithmetic: near l<sup>1</sup> the hypothesis clears zero and predicts 1; far from every landmark only θ<sub>0</sub> is left, and it predicts 0.</p>
            :::
    - If we had **<span class="hl">another point</span>** far away from all three
      <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [49].png"/></figure>
      - This equates to -0.5
        - So we predict 0
  - Considering our parameter, for points near l<sup>1</sup> and l<sup>2</sup> you predict 1, but for points near l<sup>3 </sup>you predict 0
  - Which means we create a non-linear decision boundary that goes a lil' something like this;
    <figure><img alt="" loading="lazy" src="12_Support_Vector_Machines_files/Image [50].png"/></figure>
    - Inside we predict y = 1
    - Outside we predict y = 0
- So this show how we can create a non-linear boundary with landmarks and the kernel function in the support vector machine
  - But
    - How do we get/choose the landmarks
    - What other kernels can we use (other than the Gaussian kernel)

## Kernels II

- Filling in missing detail and practical implications regarding kernels
- Spoke about picking landmarks manually, defining the kernel, and building a hypothesis function
  - Where do we get the landmarks from?
  - For complex problems we probably want lots of them

### Choosing the landmarks

- Take the training data
- For each example place a landmark at exactly the same location
- So end up with m landmarks
  - One landmark per location per training example
  - Means our features measure how close to a training set example something is
- Given a new example, compute all the f values
  - Gives you a feature vector f (f<sub>0</sub> to f<sub>m</sub>)
    - f<sub>0</sub> = 1 always
- A more detailed look at generating the f vector
  - If we had a training example - features we compute would be using (x<sup>i</sup>, y<sup>i</sup>)
    - So we just cycle through each landmark, calculating how close to that landmark actually x<sup>i</sup> is
      - f<sub>1</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>1</sup>)
      - f<sub>2</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>2</sup>)
      - ...
      - f<sub>m</sub><sup>i</sup> = k(x<sup>i</sup>, l<sup>m</sup>)
    - Somewhere in the list we compare x to itself... (i.e. when we're at f<sub>i</sub><sup>i</sup>)
      - So because we're using the Gaussian Kernel this evaluates to 1
    - Take these m features (f<sub>1</sub>, f<sub>2</sub> ... f<sub>m</sub>) group them into an \[m +1 x 1\] dimensional vector called f
      - f<sup>i</sup> is the f feature vector for the ith example
      - And add a 0th term = 1
- Given these kernels, how do we use a support vector machine

### SVM hypothesis prediction with kernels

- Predict y = 1 if (θ<sup><em>T</em></sup> f) >= 0
  - Because θ = \[m+1 x 1\]
  - And f = \[m +1 x 1\]
- So, this is how you make a prediction assuming you already have θ
  - How do you get θ?

### SVM training with kernels

- Use the SVM learning algorithm
  <div class="eqn">
  <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>&#x3b8;</mi></munder><mspace width="0.35em"/><mi>C</mi><mspace width="0.2em"/><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mo form="prefix">[</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><msub><mtext>cost</mtext><mn>1</mn></msub><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>f</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><msub><mtext>cost</mtext><mn>0</mn></msub><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><msup><mi>f</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo form="postfix">]</mo><mo>+</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
  <span class="eqn-note">The slide omits the square brackets, leaving it unclear how far the sum runs; they are restored here to match the same cost function as it was written before kernels were introduced.</span>
  </div>
  - Now, we minimize using f as the feature vector instead of x
  - By solving this minimization problem you get the parameters for your SVM
- In this setup, m = n
  - Because number of features is the number of training data examples we have
- One final mathematic detail (not crucial to understand)
  - If we ignore θ<sub>0</sub> then the following is true
    <div class="eqn">
    <math display="block"><mrow><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3b8;</mi><mi>j</mi><mn>2</mn></msubsup><mo>=</mo><msup><mi>&#x3b8;</mi><mi>T</mi></msup><mi>&#x3b8;</mi></mrow></math>
    </div>
  - What many implementations do is
    <div class="eqn">
    <math display="block"><mrow><msup><mi>&#x3b8;</mi><mi>T</mi></msup><mi>M</mi><mi>&#x3b8;</mi></mrow></math>
    <span class="eqn-note">Most SVM implementations replace &#x3b8;<sup>T</sup>&#x3b8; with &#x3b8;<sup>T</sup>M&#x3b8;, where M depends on the kernel. It is a rescaled inner product, and it is what lets the SVM scale to much larger feature counts.</span>
    </div>
    - Where the matrix M depends on the kernel you use
    - Gives a slightly different minimization - means we determine a rescaled version of θ
    - Allows more efficient computation, and scale to much bigger training sets
    - If you have a training set with 10 000 values, means you get 10 000 features
      - Solving for all these parameters can become expensive
      - So by adding this in we avoid a for loop and use a matrix multiplication algorithm instead
- You can apply kernels to other algorithms
  - But they tend to be very computationally expensive
  - But the SVM is far more efficient - so more practical
- Lots of good off the shelf software to minimize this function
- <strong>SVM parameters (C)</strong>
  - Bias and variance trade off
  - Must choose C
    - C plays a role similar to 1/LAMBDA (where LAMBDA is the regularization parameter)
  - Large C gives a hypothesis of **<span class="term">low bias high variance</span>** --> overfitting
  - Small C gives a hypothesis of **<span class="term">high bias low variance</span>** --> underfitting
- <strong>SVM parameters (σ<sup>2</sup>)</strong>
  - Parameter for calculating f values
    - Large σ<sup>2</sup> - f features vary more smoothly - higher bias, lower variance
    - Small σ<sup>2</sup> - f features vary abruptly - low bias, high variance

## SVM - implementation and use

- So far spoken about SVM in a very abstract manner
- What do you need to do this
  - Use SVM software packages (e.g. scikit-learn's LinearSVC and SVC, built on the classic liblinear and libsvm libraries) to solve parameters θ
  - Need to specify
    - Choice of parameter C
    - Choice of kernel

### Choosing a kernel

- We've looked at the **<span class="term">Gaussian kernel</span>**
  - Need to define σ (σ<sup>2</sup>)
    - Discussed σ<sup>2</sup>
  - When would you choose a Gaussian?
    - If n is small and/or m is large
      - e.g. 2D training set that's large
  - If you're using a Gaussian kernel then you may need to implement the kernel function
    - e.g. a function<br/> f_i = kernel(x1, x2)
      - Returns a real number
    - Some SVM packages will expect you to define kernel
    - Although, some SVM implementations include the Gaussian and a few others
      - Gaussian is probably most popular kernel
  - NB - make sure you perform **<span class="term">feature scaling</span>** before using a Gaussian kernel
    - If you don't features with a large value will dominate the f value
- Could use no kernel - **<span class="term">linear kernel</span>**
  - Predict y = 1 if (θ<sup><em>T</em></sup> x) >= 0
    - So no f vector
    - Get a standard linear classifier
  - Why do this?
    - If n is large and m is small then
      - Lots of features, few examples
      - Not enough data - risk overfitting in a high dimensional feature-space
- Other choice of kernel
  - Linear and Gaussian are most common
  - Not all similarity functions you develop are valid kernels
    - Must satisfy **<span class="hl-red">Mercer's Theorem</span>**
    - SVM use numerical optimization tricks
      - Mean certain optimizations can be made, but they must follow the theorem
  - **<span class="term">Polynomial Kernel</span>**
    - We measure the similarity of x and l by doing one of
      - (x<sup><em>T</em></sup> l)<sup>2</sup>
      - (x<sup><em>T</em></sup> l)<sup>3</sup>
      - (x<sup><em>T</em></sup> l+1)<sup>3</sup>
    - General form is
      - (x<sup><em>T</em></sup> l+Con)<sup>D</sup>
    - If they're similar then the inner product tends to be large
    - Not used that often
    - Two parameters
      - Degree of polynomial (D)
      - Number you add to l (Con)
    - Usually performs worse than the Gaussian kernel
    - Used when x and l are both non-negative
  - **<span class="term">String kernel</span>**
    - Used if input is text strings
    - Use for text classification
  - **<span class="term">Chi-squared kernel</span>**
  - **<span class="term">Histogram intersection kernel</span>**

### Multi-class classification for SVM

- Many packages have built in multi-class classification packages
- Otherwise use one-vs all method
- Not a big issue

### Logistic regression vs. SVM

- When should you use SVM and when is logistic regression more applicable
- If n (features) is large vs. m (training set)
  - e.g. text classification problem
    - Feature vector dimension is 10 000
    - Training set is 10 - 1000
    - Then use logistic regression or SVM with a linear kernel
- If n is small and m is intermediate
  - n = 1 - 1000
  - m = 10 - 10 000
  - Gaussian kernel is good
- If n is small and m is large
  - n = 1 - 1000
  - m = 50 000+
    - SVM will be slow to run with Gaussian kernel
  - In that case
    - Manually create or add more features
    - Use logistic regression or SVM with a linear kernel
- Logistic regression and SVM with a linear kernel are pretty similar
  - Do similar things
  - Get similar performance
- A lot of SVM's power is using different kernels to learn complex non-linear functions
- For all these regimes a well designed NN should work
  - But, for some of these problems a NN might be slower - SVM well implemented would be faster
- SVM has a convex optimization problem - so you get a global minimum
- It's not always clear how to choose an algorithm
  - Often more important to get enough data
  - Designing new features
  - Debugging the algorithm
- SVM is widely perceived as a very powerful learning algorithm
