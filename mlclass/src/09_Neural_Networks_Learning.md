---
title: "09: Neural Networks - Learning"
nav_title: "09: Neural networks II"
index_title: "09: Neural Networks — Learning"
origin: "2011"
description: "Training a network: the cost function, backpropagation, gradient checking and random initialization."
---

## Neural network cost function

- NNs - one of the most powerful learning algorithms
  - Is a learning algorithm for fitting the derived parameters given a training set
  - Let's have a first look at a neural network cost function
- Focus on application of NNs for classification problems
- Here's the set up
  - Training set is {(x<sup>1</sup>, y<sup>1</sup>), (x<sup>2</sup>, y<sup>2</sup>), (x<sup>3</sup>, y<sup>3</sup>) ... (x<sup><em>m</em></sup>, y<sup><em>m</em></sup>)}
  - *L* = number of layers in the network
    - In our example below L = 4
  - s<sub>l</sub> = number of units (not counting bias unit) in layer l

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image.png"/></figure>

- So here
  - L = 4
  - s<sub>1</sub> = 3
  - s<sub>2</sub> = 5
  - s<sub>3</sub> = 5
  - s<sub>4</sub> = 4

### Types of classification problems with NNs

- Two types of classification, as we've previously seen
- **<span class="term">Binary classification</span>**
  - 1 output (0 or 1)
  - So single output node - value is going to be a real number
  - k = 1
    - NB k is number of units in output layer
  - s<sub><em>L</em></sub> = 1
- **<span class="term">Multi-class classification</span>**
  - k distinct classifications
  - Typically k is greater than or equal to three
  - If only two just go for binary
  - s<sub><em>L</em></sub> = k
  - So y is a k-dimensional vector of real numbers

<div class="eqn">
<math display="block"><mrow><mi>y</mi><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mi>K</mi></msup></mrow></math>
<math display="block"><mrow><mtext>e.g.</mtext><mspace width="0.6em"/><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>,</mo><mspace width="0.9em"/><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>,</mo><mspace width="0.9em"/><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>,</mo><mspace width="0.9em"/><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<span class="eqn-note">Left to right: pedestrian, car, motorcycle, truck.</span>
</div>

### Cost function for neural networks

- The (regularized) logistic regression cost function is as follows;

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>)</mo><mo>]</mo><mo>+</mo><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msubsup><mi>&#x3B8;</mi><mi>j</mi><mn>2</mn></msubsup></mrow></math>
</div>

- For neural networks our cost function is a generalization of this equation above, so instead of one output we generate *k* outputs

<div class="eqn">
<math display="block"><mrow><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mi>K</mi></msup><mspace width="1.6em"/><mrow><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>)</mo></mrow><msub><mo></mo><mi>i</mi></msub><mo>=</mo><msup><mi>i</mi><mtext>th</mtext></msup><mspace width="0.3em"/><mtext>output</mtext></mrow></math>
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x398;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mspace width="0.15em"/><mi>log</mi><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><msub><mo>)</mo><mi>k</mi></msub><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><msub><mo>)</mo><mi>k</mi></msub><mo>)</mo><mo>]</mo></mrow></math><math display="block"><mrow><mspace width="2.5em"/><mo>+</mo><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>l</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>L</mi><mo>&#x2212;</mo><mn>1</mn></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mi>l</mi></msub></munderover><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mrow><mi>l</mi><mo>+</mo><mn>1</mn></mrow></msub></munderover><msup><mrow><mo>(</mo><msubsup><mi>&#x398;</mi><mrow><mi>j</mi><mi>i</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- Our cost function now outputs a *k* dimensional vector
  - h<sub>Θ</sub>(x) is a k dimensional vector, so h<sub>Θ</sub>(x)<sub><em>i</em></sub> refers to the ith value in that vector
- Costfunction J(Θ) is
  - \[-1/m\] times a sum of a similar term to which we had for logistic regression
  - But now this is also a sum from k = 1 through to K (K is number of output nodes)
    - Summation is a sum over the k output units - i.e. for each of the possible classes
    - So if we had 4 output units then the sum is k = 1 to 4 of the logistic regression over each of the four output units in turn
  - This looks really complicated, but it's not so difficult
    - We don't sum over the bias terms (hence starting at 1 for the summation)
      - Even if you do and end up regularizing the bias term this is not a big problem
    - Is just summation over the terms

### Woah there - lets take a second to try and understand this!

- There are basically two halves to the neural network logistic regression cost function

### First half

<div class="eqn">
<math display="block"><mrow><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mspace width="0.15em"/><mi>log</mi><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><msub><mo>)</mo><mi>k</mi></msub><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><msub><mo>)</mo><mi>k</mi></msub><mo>)</mo><mo>]</mo></mrow></math>
</div>

- This is just saying
  - For each training data example (i.e. 1 to m - the first summation)
    - Sum for each position in the output vector
- This is an average sum of logistic regression

### Second half

<div class="eqn">
<math display="block"><mrow><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>l</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>L</mi><mo>&#x2212;</mo><mn>1</mn></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mi>l</mi></msub></munderover><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mrow><mi>l</mi><mo>+</mo><mn>1</mn></mrow></msub></munderover><msup><mrow><mo>(</mo><msubsup><mi>&#x398;</mi><mrow><mi>j</mi><mi>i</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- This is a massive regularization summation term, which I'm not going to walk through, but it's a fairly straightforward triple nested summation
- This is also called a <span class="term"><strong>weight decay</strong></span> term
- As before, the lambda value determines the importance of the two halves
- The regularization term is similar to that in logistic regression
- So, we have a cost function, but *how* do we minimize this bad boy?!

## Summary of what's about to go down

<p><em>The following section is, I think, the most complicated thing in the course, so I'm going to take a second to explain the general idea of what we're going to do;</em></p>

- We've already described **<span class="term">forward propagation</span>**
  - This is the algorithm which takes your neural network and the initial input into that network and pushes the input through the network
    - It leads to the generation of an output hypothesis, which may be a single real number, but can also be a vector
- We're now going to describe **<span class="term">back propagation</span>**
  - Back propagation basically takes the output you got from your network, compares it to the real value (y) and calculates how wrong the network was (i.e. how wrong the parameters were)
  - It then, using the error you've just calculated, back-calculates the error associated with each unit from the preceding layer (i.e. layer <em>L -</em> 1)
  - This goes on until you reach the input layer (where obviously there is no error, as the activation is the input)
  - These "error" measurements for each unit can be used to calculate the **partial derivatives**
    - Partial derivatives are the bomb, because gradient descent needs them to minimize the cost function
  - We use the partial derivatives with gradient descent to try minimize the cost function and update all the Θ values
  - This repeats until gradient descent reports convergence
- A few things which are good to realize from the get go
  - There is a Θ matrix for each layer in the network
    - This has each node in layer l as one dimension and each node in l+1 as the other dimension
  - Similarly, there is going to be a Δ matrix for each layer
    - This has each node as one dimension and each training data example as the other

## Back propagation algorithm

- We previously spoke about the neural network cost function
- Now we're going to deal with **<span class="term">back propagation</span>**
  - Algorithm used to minimize the cost function, as it **<span class="hl-red">allows us to calculate partial derivatives</span>**!

<div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x398;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mspace width="0.15em"/><mi>log</mi><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><msub><mo>)</mo><mi>k</mi></msub><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msubsup><mi>y</mi><mi>k</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msubsup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><msub><mo>)</mo><mi>k</mi></msub><mo>)</mo><mo>]</mo></mrow></math><math display="block"><mrow><mspace width="2.5em"/><mo>+</mo><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>l</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>L</mi><mo>&#x2212;</mo><mn>1</mn></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mi>l</mi></msub></munderover><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mrow><mi>l</mi><mo>+</mo><mn>1</mn></mrow></msub></munderover><msup><mrow><mo>(</mo><msubsup><mi>&#x398;</mi><mrow><mi>j</mi><mi>i</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div>

- The cost function used is shown above
  - We want to find parameters Θ which minimize J(Θ)
  - To do so we can use one of the algorithms already described such as
    - Gradient descent
    - Advanced optimization algorithms
- To minimize a cost function we just write code which computes the following
  - <strong>J(Θ)</strong>
    - i.e. the cost function itself!
    - Use the formula above to calculate this value, so we've done that
  - **Partial derivative terms**
    - So now we need some way to do that
      - This is not trivial! Θ is indexed in three dimensions because we have separate parameter values for each node in each layer going to each node in the following layer
      - i.e. each layer has a Θ matrix associated with it!
        - We want to calculate the partial derivative of J(Θ) with respect to a single parameter
          <div class="eqn">
          <math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msubsup><mi>&#x398;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x398;</mi><mo form="postfix">)</mo></mrow></math>
          </div>
    - Remember that the partial derivative term we calculate above is a REAL number (not a vector or a matrix)
      - Θ is the input parameters
        - Θ<sup>1</sup> is the matrix of weights which define the function mapping from layer 1 to layer 2
        - Θ<sub>10</sub><sup>1</sup> is the real number parameter which you multiply the bias unit (i.e. 1) with for the bias unit input into the first unit in the second layer
        - Θ<sub>11</sub><sup>1</sup> is the real number parameter which you multiply the first (real) unit with for the first input into the first unit in the second layer
        - Θ<sub>21</sub><sup>1</sup> is the real number parameter which you multiply the first (real) unit with for the first input into the second unit in the second layer
        - As discussed, for Θ<sub>ij</sub><sup>l</sup>
          - i here represents the unit in layer l+1 you're mapping to (destination node)
          - j is the unit in layer l you're mapping from (origin node)
          - l is the layer your mapping from (to layer l+1) (origin layer)
          - NB
            - *<span class="hl-red">The terms destination node, origin node and origin layer are terms I've made up!</span>*
    - So - this partial derivative term is
      - The partial derivative of a 3-way indexed dataset with respect to a real number (which is one of the values in that dataset)
  - **Gradient computation**
    - One training example
    - Imagine we just have a single pair (x,y) - entire training set
    - How would we deal with this example?
    - The forward propagation algorithm operates as follows
      - **Layer 1**
        - a<sup>1</sup> = x
        - z<sup>2</sup> = Θ<sup>1</sup>a<sup>1</sup>
      - **Layer 2**
        - a<sup>2</sup> = g(z<sup>2</sup>) (add a<sub>0</sub><sup>2</sup>)
        - z<sup>3</sup> = Θ<sup>2</sup>a<sup>2</sup>
      - **Layer 3**
        - a<sup>3</sup> = g(z<sup>3</sup>) (add a<sub>0</sub><sup>3</sup>)
        - z<sup>4</sup> = Θ<sup>3</sup>a<sup>3</sup>
      - **Output**
        - a<sup>4</sup> = h<sub>Θ</sub>(x) = g(z<sup>4</sup>)

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [8].png"/></figure>

- This is the vectorized implementation of forward propagation
  - Lets compute activation values sequentially (below just re-iterates what we had above!)

<div class="eqn">
<math display="block"><mrow><msup><mi>a</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>=</mo><mi>x</mi></mrow></math>
<math display="block"><mrow><msup><mi>z</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>=</mo><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><msup><mi>a</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup></mrow></math>
<math display="block"><mrow><msup><mi>a</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>=</mo><mi>g</mi><mo>(</mo><msup><mi>z</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow></math>
<math display="block"><mrow><msup><mi>z</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>=</mo><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><msup><mi>a</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup></mrow></math>
<math display="block"><mrow><msup><mi>a</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>=</mo><mi>g</mi><mo>(</mo><msup><mi>z</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow></math>
<math display="block"><mrow><msup><mi>z</mi><mrow><mo>(</mo><mn>4</mn><mo>)</mo></mrow></msup><mo>=</mo><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><msup><mi>a</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup></mrow></math>
<math display="block"><mrow><msup><mi>a</mi><mrow><mo>(</mo><mn>4</mn><mo>)</mo></mrow></msup><mo>=</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mi>g</mi><mo>(</mo><msup><mi>z</mi><mrow><mo>(</mo><mn>4</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow></math>
<span class="eqn-note">The bias unit a<sub>0</sub> is added to a<sup>(2)</sup> and a<sup>(3)</sup> after each activation.</span>
</div>

### What is back propagation?

- Use it to compute the partial derivatives
- Before we dive into the mechanics, let's get an idea regarding the intuition of the algorithm
  - For each node we can calculate (δ<sub>j</sub><sup>l</sup>) - this is **<span class="hl-red">the error of node j in layer l</span>**
    - If we remember, a<sub>j</sub><sup>l</sup> is the activation of node j in layer l
    - Remember the activation is a totally calculated value, so we'd expect there to be some error compared to the "real" value
      - The delta term captures this error
      - But the problem here is, "what is this 'real' value, and how do we calculate it?!"
        - The NN is a totally artificial construct
        - The only "real" value we have is our actual classification (our y value) - so that's where we start
- If we use our example and look at the fourth (output) layer, we can first calculate
  - δ<sub>j</sub><sup>4 </sup>= a<sub>j</sub><sup>4 </sup>- y<sub>j</sub>
    - \[Activation of the unit\] - \[the actual value observed in the training example\]
    - We could also write a<sub>j</sub><sup>4 </sup>as h<sub>Θ</sub>(x)<sub>j</sub>
      - Although I'm not sure why we would?
  - This is an individual example implementation
- Instead of focussing on each node, let's think about this as a vectorized problem
  - δ<sup>4 </sup>= a<sup>4 </sup>- y
    - So here δ<sup>4 </sup>is the vector of errors for the 4th layer
    - a<sup>4 </sup>is the vector of activation values for the 4th layer
- With δ<sup>4 </sup>calculated, we can determine the error terms for the other layers as follows;
  <div class="eqn">
  <math display="block"><mrow><msup><mi>&#x3B4;</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>=</mo><msup><mrow><mo>(</mo><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow><mi>T</mi></msup><msup><mi>&#x3B4;</mi><mrow><mo>(</mo><mn>4</mn><mo>)</mo></mrow></msup><mspace width="0.3em"/><mo>.</mo><mo>&#x2217;</mo><mspace width="0.15em"/><msup><mi>g</mi><mo>&#x2032;</mo></msup><mo>(</mo><msup><mi>z</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow></math>
  <math display="block"><mrow><msup><mi>&#x3B4;</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>=</mo><msup><mrow><mo>(</mo><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow><mi>T</mi></msup><msup><mi>&#x3B4;</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mspace width="0.3em"/><mo>.</mo><mo>&#x2217;</mo><mspace width="0.15em"/><msup><mi>g</mi><mo>&#x2032;</mo></msup><mo>(</mo><msup><mi>z</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>)</mo></mrow></math>
  <span class="eqn-note"><code>.*</code> is element-wise multiplication. There is no &#x3B4;<sup>(1)</sup> &mdash; the input layer has no error.</span>
  </div>
  ::: code-eg
  ```
  def g(z):
      return 1 / (1 + np.exp(-z))

  z = np.array([-2.0, 0.0, 3.0])
  a = g(z)

  h = 1e-6                                   # check the identity numerically
  numeric = (g(z + h) - g(z - h)) / (2 * h)
  a * (1 - a)   # [0.105, 0.25, 0.0452] - matches `numeric` to ~1e-10
  ```
  <p class="eqn-note">The derivative identity the equations above rely on: g&prime;(z) is exactly a .* (1 &minus; a), checked against a numerical derivative.</p>
  :::
- Taking a second to break this down
  - Θ<sup>3 </sup>is the vector of parameters for the 3->4 layer mapping
  - δ<sup>4 </sup>is (as calculated) the error vector for the 4th layer
  - g'(z<sup>3</sup>) is the first derivative of the activation function g evaluated by the input values given by z<sup>3 </sup>
    - You can do the calculus if you want (...), but when you calculate this derivative you get
    - g'(z<sup>3</sup>) = a<sup>3 </sup><span class="hl-red"><strong>. *</strong></span> (1 - a<sup>3</sup>)
  - So, more easily
    - δ<sup>3 </sup>= (Θ<sup>3</sup>)<sup>T </sup>δ<sup>4 </sup><span class="hl-red"><strong>. *</strong></span><strong>(a</strong><sup>3 </sup><span class="hl-red"><strong>. *</strong></span> <strong>(1 - </strong>a<sup>3</sup>))
  - **<span class="hl-red">. *</span>** is the element wise multiplication between the two vectors
    - Why element wise? Because this is essentially an extension of individual values in a vectorized implementation, so element wise multiplication gives that effect
    - We highlighted it just in case you think it's a typo!

### Analyzing the mathematics

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [11].png"/></figure>

- And if we take a second to consider the vector dimensionality (with our example above \[3-5-5-4\])
  - <span class="hl-purple">Θ<sup>3</sup></span><strong><span class="hl-purple"> </span>=</strong> is a matrix which is \[4 X 5\] (if we don't include the bias term, 4 X 6 if we do)
    - <strong> </strong><span class="hl-purple">(Θ<sup>3</sup>)<sup><strong>T</strong></sup></span> = therefore, is a \[5 X 4\] matrix
  - **<sup> </sup><span class="hl-purple">δ<sup>4</sup></span>** = is a 4x1 vector
  - So when we multiply a \[5 X 4\] matrix with a \[4 X 1\] vector we get a \[5 X 1\] vector
  - Which, lo and behold, is the same dimensionality as the **<span class="hl-purple">a</span><sup><span class="hl-purple">3</span> </sup>**vector, meaning we can run our pairwise multiplication
- For δ<sup>3 </sup>when you calculate the derivative terms you get<br/>a<sup>3 </sup><span class="hl-red"><strong>. *</strong></span> (1 - a<sup>3</sup>)
- Similarly For δ<sup>2</sup> when you calculate the derivative terms you get<br/>a<sup>2 </sup>**<span class="hl-red">. *</span>** (1 - a<sup>2</sup>)
  - So to calculate δ<sup>2</sup> we do<br/>δ<sup>2 </sup>= (Θ<sup>2</sup>)<sup>T </sup>δ<sup>3</sup><sup><span class="hl-red"> </span></sup><span class="hl-red"><strong>. *</strong></span><strong>(a</strong><sup>2 </sup><span class="hl-red"><strong>. *</strong></span><strong> (1 - </strong>a<sup>2</sup>))
- There's no δ<sup>1</sup> term
  - Because that was the input!

### Why do we do this?

- We do all this to get all the δ terms, and we want the δ terms because through a very complicated derivation you can use δ to get the partial derivative of J(Θ) with respect to individual parameters (if you ignore regularization, or regularization is 0, which we deal with later)
- <div class="eqn">
  <math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msubsup><mi>&#x398;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x398;</mi><mo form="postfix">)</mo></mrow></math>
  </div>
  \= a<sub>j</sub><sup>l </sup>δ<sub>i</sub><sup>(l+1) </sup>
- By doing back propagation and computing the delta terms you can then compute the **<span class="term">partial derivative terms</span>**
  - We need the partial derivatives to minimize the cost function!

### Putting it all together to get the partial derivatives!

- What is really happening - lets look at a more complex example
- Training set of m examples
  <div class="eqn">
  <math display="block"><mrow><mtext>Training set</mtext><mspace width="0.5em"/><mo>{</mo><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>y</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>)</mo><mo>,</mo><mi>&#x2026;</mi><mo>,</mo><mo>(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow></msup><mo>)</mo><mo>}</mo></mrow></math>
  </div>
- **First**, set the delta values
  <div class="eqn">
  <math display="block"><mrow><mtext>Set</mtext><mspace width="0.5em"/><msubsup><mi>&#x394;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>=</mo><mn>0</mn><mspace width="0.8em"/><mtext>(for all l, i, j)</mtext></mrow></math>
  </div>
  - Set equal to 0 for all values
  - Eventually these Δ values will be used to compute the partial derivative
    - Will be used as accumulators for computing the partial derivatives
- **Next**, loop through the training set
  <div class="eqn">
  <math display="block"><mrow><mtext>For</mtext><mspace width="0.5em"/><mi>i</mi><mo>=</mo><mn>1</mn><mspace width="0.4em"/><mtext>to</mtext><mspace width="0.4em"/><mi>m</mi></mrow></math>
  </div>
  - i.e. for each example in the training set (dealing with each example as (x,y)
  - Set a<sup>1 </sup>(activation of input layer) = x<sup>i</sup>
  - **Perform** **<span class="term">forward propagation</span>** to compute a<sup>l </sup>for each layer (l = 1,2, ... L)
    - i.e. run forward propagation
  - **Then**, use the output label for the specific example we're looking at to calculate δ<sup>L</sup> where δ<sup>L </sup>= a<sup>L </sup>- y<sup>i</sup>
    - So we initially calculate the delta value for the output layer
    - Then, using **<span class="term">back propagation</span>** we move back through the network from layer L-1 down to layer 2
  - Finally, use Δ to accumulate the partial derivative terms
    <div class="eqn">
    <math display="block"><mrow><msubsup><mi>&#x394;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>&#x2254;</mo><msubsup><mi>&#x394;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>+</mo><msubsup><mi>a</mi><mrow><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><msubsup><mi>&#x3B4;</mi><mrow><mi>i</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>+</mo><mn>1</mn><mo>)</mo></mrow></msubsup></mrow></math>
    </div>
  - Note here
    - l = layer
    - j = node in that layer
    - i = the error of the affected node in the target layer
  - You can vectorize the Δ expression too, as
    <div class="eqn">
    <math display="block"><mrow><msup><mi>&#x394;</mi><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msup><mo>&#x2254;</mo><msup><mi>&#x394;</mi><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msup><mo>+</mo><msup><mi>&#x3B4;</mi><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msup><msup><mrow><mo>(</mo><msup><mi>a</mi><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msup><mo>)</mo></mrow><mi>T</mi></msup></mrow></math>
    </div>
- **Finally**
  - After executing the body of the loop, exit the for loop and compute
    <div class="eqn">
    <math display="block"><mrow><msubsup><mi>D</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>&#x2254;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><msubsup><mi>&#x394;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>+</mo><mi>&#x3BB;</mi><msubsup><mi>&#x398;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mspace width="1em"/><mtext>if j &#x2260; 0</mtext></mrow></math>
    <math display="block"><mrow><msubsup><mi>D</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>&#x2254;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><msubsup><mi>&#x394;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mspace width="1em"/><mtext>if j = 0</mtext></mrow></math>
    <span class="eqn-note">j = 0 is the bias term, which is not regularised.</span>
    </div>
    - When j = 0 we have no regularization term
- At the end of ALL this
  - You've calculated all the *D* terms above using Δ
    - NB - each D term above is a real number!
  - We can show that each D is equal to the following
    - <div class="eqn">
      <math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msubsup><mi>&#x398;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x398;</mi><mo form="postfix">)</mo><mo>=</mo><msubsup><mi>D</mi><mrow><mi>i</mi><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup></mrow></math>
      </div>
  - We have calculated the partial derivative for each parameter
    - We can then use these in gradient descent or one of the advanced optimization algorithms
- Phew!
  - What a load of hassle!

## Back propagation intuition

- Some additionally back propagation notes
  - In case you found the preceding unclear, which it shouldn't be as it's fairly heavily modified with my own explanatory notes
- Back propagation is hard(ish...)
  - But don't let that discourage you
  - It's hard in as much as it's confusing - it's not difficult, just complex
- Looking at mechanical steps of back propagation

### Forward propagation with pictures!

<strong><figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [20].png"/></figure></strong>

- Feeding input into the input layer (x<sup>i</sup>, y<sup>i</sup>)
  - Note that x and y here are vectors from 1 to n where n is the number of features
    - So above, our data has two features (hence x<sub>1</sub> and x<sub>2</sub>)
- With our input data present we use **forward propagation**

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [21].png"/></figure>

- The sigmoid function applied to the z values gives the activation values
  - Below we show exactly how the z value is calculated for an example

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [22].png"/></figure>

### Back propagation

- With forwardprop done we move on to do back propagation
- Back propagation is doing something very similar to forward propagation, but backwards
  - Very similar though
- Let's look at the cost function again...
  - Below we have the cost function if there is a single output (i.e. binary classification)

<strong><div class="eqn">
<math display="block"><mrow><mi>J</mi><mo form="prefix">(</mo><mi>&#x398;</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mfrac><mn>1</mn><mi>m</mi></mfrac><mo>[</mo><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mspace width="0.15em"/><mi>log</mi><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>)</mo><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><mo>(</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>)</mo><mo>)</mo><mo>]</mo></mrow></math><math display="block"><mrow><mspace width="2.5em"/><mo>+</mo><mfrac><mi>&#x3BB;</mi><mrow><mn>2</mn><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>l</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>L</mi><mo>&#x2212;</mo><mn>1</mn></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mi>l</mi></msub></munderover><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>s</mi><mrow><mi>l</mi><mo>+</mo><mn>1</mn></mrow></msub></munderover><msup><mrow><mo>(</mo><msubsup><mi>&#x398;</mi><mrow><mi>j</mi><mi>i</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></math>
</div></strong>

- This function cycles over each example, so the cost for one example really boils down to this

<div class="eqn">
<math display="block"><mrow><mi>cost</mi><mo>(</mo><mi>i</mi><mo>)</mo><mo>=</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mspace width="0.15em"/><mi>log</mi><mspace width="0.15em"/><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>+</mo><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo>)</mo><mspace width="0.15em"/><mi>log</mi><mo>(</mo><mn>1</mn><mo>&#x2212;</mo><msub><mi>h</mi><mi>&#x398;</mi></msub><mo form="prefix">(</mo><msup><mi>x</mi><mrow><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msup><mo form="postfix">)</mo><mo>)</mo></mrow></math>
<span class="eqn-note">The original slide writes the second logarithm as log h(x<sup>(i)</sup>), dropping the 1 &#x2212; ; it is corrected here to match the same expression elsewhere in this chapter.</span>
</div>

- Which, we can think of as a sigmoidal version of the squared difference (check out the derivation if you don't believe me)
  - So, basically saying, "how well is the network doing on example <em>i </em>"?
- We can think about a δ term on a unit as the "error" of cost for the activation value associated with a unit
  - More formally (<em>don't worry about this...</em>), δ is
    <div class="eqn">
    <math display="block"><mrow><msubsup><mi>&#x3B4;</mi><mrow><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup><mo>=</mo><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msubsup><mi>z</mi><mrow><mi>j</mi></mrow><mrow><mo>(</mo><mi>l</mi><mo>)</mo></mrow></msubsup></mrow></mfrac><mspace width="0.2em"/><mi>cost</mi><mo>(</mo><mi>i</mi><mo>)</mo></mrow></math>
    </div>
    - Where cost is as defined above
    - Cost function is a function of y value and the hypothesis function
- So - for the output layer, back propagation sets the δ value as \[a - y\]
  - Difference between activation and actual value
- We then propagate these values backwards;

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [26].png"/></figure>

- Looking at another example to see *how* we actually calculate the delta value;

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [27].png"/></figure>

- So, in effect,
  - Back propagation calculates the δ, and those δ values are the weighted sum of the next layer's delta values, weighted by the parameter associated with the links
  - Forward propagation calculates the activation (a) values, which
- Depending on how you implement you may compute the delta values of the bias values
  - However, these aren't actually used, so it's a bit inefficient, but not a lot more!

## Implementation notes - unrolling parameters (matrices)

- Needed for using advanced optimization routines

```
def cost_function(theta):
    ...
    return jval, gradient

res = minimize(cost_function, initial_theta, jac=True)
opt_theta = res.x
```

- Is the Python code
  - But theta is going to be matrices
- minimize takes the cost function and initial theta values
  - These routines assume theta is a parameter vector
  - Also assumes the gradient created by costFunction is a vector
- For NNs, our parameters are matrices
  - e.g.

<div class="eqn">
<math display="block"><mrow><mtext>Neural Network (L = 4):</mtext></mrow></math>
<math display="block"><mrow><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mspace width="0.5em"/><mtext>&#x2014; matrices</mtext></mrow></math>
<math display="block"><mrow><msup><mi>D</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>D</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>,</mo><msup><mi>D</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mspace width="0.5em"/><mtext>&#x2014; matrices</mtext></mrow></math>
<span class="eqn-note">In the code these are <code>Theta1, Theta2, Theta3</code> and <code>D1, D2, D3</code>.</span>
</div>

### Example

<strong><figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [30].png"/></figure>
<div class="eqn">
<math display="block"><mrow><msub><mi>s</mi><mn>1</mn></msub><mo>=</mo><mn>10</mn><mo>,</mo><mspace width="0.4em"/><msub><mi>s</mi><mn>2</mn></msub><mo>=</mo><mn>10</mn><mo>,</mo><mspace width="0.4em"/><msub><mi>s</mi><mn>3</mn></msub><mo>=</mo><mn>1</mn></mrow></math>
<math display="block"><mrow><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mrow><mn>10</mn><mo>&#x00D7;</mo><mn>11</mn></mrow></msup><mo>,</mo><mspace width="0.5em"/><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mrow><mn>10</mn><mo>&#x00D7;</mo><mn>11</mn></mrow></msup><mo>,</mo><mspace width="0.5em"/><msup><mi>&#x398;</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mrow><mn>1</mn><mo>&#x00D7;</mo><mn>11</mn></mrow></msup></mrow></math>
<math display="block"><mrow><msup><mi>D</mi><mrow><mo>(</mo><mn>1</mn><mo>)</mo></mrow></msup><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mrow><mn>10</mn><mo>&#x00D7;</mo><mn>11</mn></mrow></msup><mo>,</mo><mspace width="0.5em"/><msup><mi>D</mi><mrow><mo>(</mo><mn>2</mn><mo>)</mo></mrow></msup><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mrow><mn>10</mn><mo>&#x00D7;</mo><mn>11</mn></mrow></msup><mo>,</mo><mspace width="0.5em"/><msup><mi>D</mi><mrow><mo>(</mo><mn>3</mn><mo>)</mo></mrow></msup><mo>&#x2208;</mo><msup><mi>&#x211D;</mi><mrow><mn>1</mn><mo>&#x00D7;</mo><mn>11</mn></mrow></msup></mrow></math>
</div>
<div class="code-eg">
<pre>Theta1 = np.ones((10, 11))
Theta2 = np.ones((10, 11))
Theta3 = np.ones((1, 11))

theta_vec = np.concatenate([Theta1.ravel(), Theta2.ravel(), Theta3.ravel()])
theta_vec.shape   <span class="comment"># (231,) = 110 + 110 + 11, one long vector</span>

np.array_equal(theta_vec[0:110].reshape(10, 11), Theta1)   <span class="comment"># True - the round trip</span></pre>
<p class="eqn-note">Unroll and reshape for the example network above: three matrices flatten into one 231-element vector and come back out unchanged.</p>
</div></strong>

- Use the <code class="hl-green"><strong>theta_vec = np.concatenate([Theta1.ravel(), Theta2.ravel(), Theta3.ravel()])</strong></code> to unroll the matrices into a long vector
- To go back you use
  - **<code class="hl-green">Theta1 = theta_vec[0:110].reshape(10, 11)</code>**

## Gradient checking

- Backpropagation has a lot of details, small bugs can be present and ruin it :-(
  - This may mean it looks like J(Θ) is decreasing, but in reality it may not be decreasing by as much as it should
- So using a numeric method to check the gradient can help diagnose a bug
  - Gradient checking helps make sure an implementation is working correctly
- **Example**
  - Have a function J(Θ)
  - Estimate derivative of function at point Θ (where Θ is a real number)
  - How?
    - Numerically
      - Compute Θ + ε
      - Compute Θ - ε
      - Join them by a straight line
      - Use the slope of that line as an approximation to the derivative

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [31].png"/></figure>

- Usually, epsilon is pretty small (0.0001)
  - If epsilon becomes REALLY small then the term BECOMES the slope's derivative
- This is the two sided difference (as opposed to one sided difference, which would be \[J(Θ + ε) - J(Θ)\] /ε)
- If Θ is a vector with n elements we can use a similar approach to look at the partial derivatives

<div class="eqn">
<math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>&#x2248;</mo><mfrac><mrow><mi>J</mi><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>+</mo><mi>&#x3F5;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>)</mo><mo>&#x2212;</mo><mi>J</mi><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>&#x2212;</mo><mi>&#x3F5;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>)</mo></mrow><mrow><mn>2</mn><mi>&#x3F5;</mi></mrow></mfrac></mrow></math>
<math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>&#x2248;</mo><mfrac><mrow><mi>J</mi><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>+</mo><mi>&#x3F5;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>)</mo><mo>&#x2212;</mo><mi>J</mi><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>&#x2212;</mo><mi>&#x3F5;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>)</mo></mrow><mrow><mn>2</mn><mi>&#x3F5;</mi></mrow></mfrac></mrow></math>
<math display="block"><mrow><mi>&#x22EE;</mi></mrow></math>
<math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><mi>&#x3B8;</mi><mo form="postfix">)</mo><mo>&#x2248;</mo><mfrac><mrow><mi>J</mi><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub><mo>+</mo><mi>&#x3F5;</mi><mo>)</mo><mo>&#x2212;</mo><mi>J</mi><mo>(</mo><msub><mi>&#x3B8;</mi><mn>1</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>2</mn></msub><mo>,</mo><msub><mi>&#x3B8;</mi><mn>3</mn></msub><mo>,</mo><mi>&#x2026;</mi><mo>,</mo><msub><mi>&#x3B8;</mi><mi>n</mi></msub><mo>&#x2212;</mo><mi>&#x3F5;</mi><mo>)</mo></mrow><mrow><mn>2</mn><mi>&#x3F5;</mi></mrow></mfrac></mrow></math>
<span class="eqn-note">Each partial derivative is checked by nudging one parameter and leaving the rest alone.</span>
</div>

::: code-eg
```
def J(theta):                  # a simple stand-in cost: J = sum of squares
    return (theta ** 2).sum()

theta = np.array([1.0, -2.0, 3.0])
EPSILON = 1e-4

grad_approx = np.zeros(3)
for i in range(3):
    theta_plus  = theta.copy(); theta_plus[i]  += EPSILON
    theta_minus = theta.copy(); theta_minus[i] -= EPSILON
    grad_approx[i] = (J(theta_plus) - J(theta_minus)) / (2 * EPSILON)

grad_approx   # [2., -4., 6.]  - the analytic gradient of sum-of-squares is 2*theta
```
<p class="eqn-note">Gradient checking on a cost simple enough to differentiate by hand: the two-sided differences land on 2θ to ten decimal places.</p>
:::

- So, in Python we use the following code to numerically compute the derivatives

```
for i in range(n):
    theta_plus = theta.copy()
    theta_plus[i] += EPSILON
    theta_minus = theta.copy()
    theta_minus[i] -= EPSILON
    grad_approx[i] = (J(theta_plus) - J(theta_minus)) \
                     / (2 * EPSILON)
```

- So on each loop theta_plus equals theta except in position i
  - The .copy() resets theta_plus on each loop - without it you would be nudging theta itself
- Create a vector of partial derivative approximations
- Using the vector of gradients from backprop (DVec)
  - Check that grad_approx is basically equal to DVec
  - Gives confidence that the Backprop implementation is correct
- Implementation note
  - Implement back propagation to compute DVec
  - Implement numerical gradient checking to compute grad_approx
  - Check they're basically the same (up to a few decimal places)
  - Before using the code for learning turn off gradient checking
    - Why?
      - The grad_approx computation is very computationally expensive
      - In contrast backprop is much more efficient (just more fiddly)

## Random initialization

- Pick random small initial values for all the theta values
  - If you start them on zero (which does work for linear regression) then the algorithm fails - all activation values for each layer are the same
- So choose random values!
  - Between 0 and 1, then scale by epsilon (where epsilon is a constant)

## Putting it all together

- **1) - pick a network architecture**
  - Number of
    - **<span class="term">Input units</span>** - number of dimensions x (dimensions of feature vector)
    - **<span class="term">Output units</span>** - number of classes in classification problem
    - **<span class="term">Hidden units</span>**
      - Default might be
        - 1 hidden layer
      - Should probably have
        - Same number of units in each layer
        - Or 1.5-2 x number of input features
      - Normally
    - More hidden units is better
    - But more is more computationally expensive
  - We'll discuss architecture more later

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [34].png"/></figure>

- **2) - Training a neural network**
  - <strong>2.1)</strong> Randomly initialize the weights
    - Small values near 0
  - <strong>2.2) </strong>Implement forward propagation to get h<sub>Θ</sub>(x)<sup>i</sup> for any x<sup>i</sup>
  - <strong>2.3) </strong>Implement code to compute the cost function J(Θ)
  - <strong>2.4) </strong>Implement back propagation to compute the partial derivatives
  - General implementation below

<pre>for i in range(m):
    forward propagation on (x<sup>i</sup>, y<sup>i</sup>)   <span class="comment"># get activation (a) terms</span>
    back propagation on (x<sup>i</sup>, y<sup>i</sup>)      <span class="comment"># get delta (δ) terms</span>
    compute Δ<sup>l</sup> := Δ<sup>l</sup> + δ<sup>l+1</sup>(a<sup>l</sup>)<sup>T</sup></pre>

### With this done compute the partial derivative terms

- Notes on implementation
  - Usually done with a for loop over training examples (for forward and back propagation)
  - *Can* be done without a for loop, but this is a much more complicated way of doing things
  - Be careful
- <strong>2.5) </strong>Use gradient checking to compare the partial derivatives computed using the above algorithm and numerical estimation of gradient of J(Θ)
  - Disable the gradient checking code for when you actually run it
- <strong>2.6)</strong> Use gradient descent or an advanced optimization method with back propagation to try to minimize J(Θ) as a function of parameters Θ
  - Here J(Θ) is non-convex
    - Can be susceptible to local minimum
    - In practice this is not usually a huge problem
    - Can't guarantee programs will find global optimum should find good local optimum at least

<figure><img alt="" loading="lazy" src="09_Neural_Networks_Learning_files/Image [35].png"/></figure>

- e.g. above pretending data only has two features to easily display what's going on
  - Our minimum here represents a hypothesis output which is pretty close to y
  - If you took one of the peaks hypothesis is far from y
- Gradient descent will start from some random point and move downhill
  - Back propagation calculates gradient down that hill
