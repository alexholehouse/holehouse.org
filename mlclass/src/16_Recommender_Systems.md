---
title: "16: Recommender Systems"
nav_title: "16: Recommender systems"
origin: "2011"
description: "Content-based and collaborative filtering, and low-rank matrix factorization for predicting ratings."
---

## Recommender systems - introduction

- Two motivations for talking about recommender systems
  - **<span class="hl-red">Important application of ML systems</span>**
    - Many technology companies find recommender systems to be absolutely key
    - Think about websites (amazon, Ebay, iTunes genius)
      - Try and recommend new content for you based on past purchases
      - Substantial part of Amazon's revenue generation
    - Improvement in recommender system performance can bring in more income
    - Kind of a funny problem
      - In academic learning, recommender systems receive a small amount of attention
      - But in industry it's an absolutely crucial tool
  - Talk about the big ideas in machine learning
    - Not so much a technique, but an idea
    - As seen, features are really important
      - There's a big idea in machine learning that for some problems you can learn what a good set of features are
      - So not select those features but learn them
    - Recommender systems do this - try and identify the crucial and relevant features

### Example - predict movie ratings

- You're a company who sells movies
  - You let users rate movies using a 1-5 star rating
    - To make the example nicer, allow 0-5 (makes math easier)
- You have five movies
- And you have four users
- Admittedly, business isn't going well, but you're optimistic about the future as a result of your truly outstanding (if limited) inventory
  <div class="table-wrap"><table>
  <caption>Five movies, four users. &#x201c;?&#x201d; means the user has not rated that film; ratings run 0 to 5.</caption>
  <tr><th scope="col">Movie</th><th scope="col">Alice (1)</th><th scope="col">Bob (2)</th><th scope="col">Carol (3)</th><th scope="col">Dave (4)</th></tr>
  <tr><th scope="row">Love at last</th><td>5</td><td>5</td><td>0</td><td>0</td></tr><tr><th scope="row">Romance forever</th><td>5</td><td>?</td><td>?</td><td>0</td></tr><tr><th scope="row">Cute puppies of love</th><td>?</td><td>4</td><td>0</td><td>?</td></tr><tr><th scope="row">Nonstop car chases</th><td>0</td><td>0</td><td>5</td><td>4</td></tr><tr><th scope="row">Swords vs. karate</th><td>0</td><td>0</td><td>5</td><td>0</td></tr>
  </table></div>
  <p class="eqn-note">The slide prints &#x201c;?&#x201d; for Dave&#x2019;s rating of <em>Swords vs. karate</em>, but the Y matrix encoding this same table gives it as 0, and the mean normalisation later in the chapter needs it to be 0 &#x2014; &#x3bc;<sub>5</sub> is stated as 1.25, the mean of (0,&nbsp;0,&nbsp;5,&nbsp;0). Corrected to 0 here.</p>
- To introduce some notation
  - n**<sub>u</sub>** - Number of users (called ?<sup>nu</sup> occasionally as we can't subscript in superscript)
  - n**<sub>m </sub>**- Number of movies
  - r(i, j) - 1 if user j has rated movie i (i.e. bitmap)
  - y<sup>(i,j)</sup> - rating given by user j to movie i (defined only if r(i,j) = 1)
- So for this example
  - n**<sub>u</sub>** = 4
  - n**<sub>m </sub>**= 5
  - Summary of scoring
    - Alice and Bob gave good ratings to rom coms, but low scores to action films
    - Carol and Dave gave good ratings for action films but low ratings for rom coms
  - We have the data given above
  - The problem is as follows
    - Given r(i,j) and y<sup>(i,j)</sup> - go through and try and predict missing values (?s)
    - Come up with a learning algorithm that can fill in these missing values

## Content based recommendation

- Using our example above, how do we predict?
  - For each movie we have a feature which measures degree to which each film is a
    - Romance (x**<sub>1</sub>**)
    - Action (x**<sub>2</sub>**)

<div class="table-wrap"><table>
<caption>The same ratings, with two features per film: how much romance and how much action it contains.</caption>
<tr><th scope="col">Movie</th><th scope="col">Alice (1)</th><th scope="col">Bob (2)</th><th scope="col">Carol (3)</th><th scope="col">Dave (4)</th><th scope="col">x<sub>1</sub><small>(romance)</small></th><th scope="col">x<sub>2</sub><small>(action)</small></th></tr>
<tr><th scope="row">Love at last</th><td>5</td><td>5</td><td>0</td><td>0</td><td>0.9</td><td>0</td></tr><tr><th scope="row">Romance forever</th><td>5</td><td>?</td><td>?</td><td>0</td><td>1.0</td><td>0.01</td></tr><tr><th scope="row">Cute puppies of love</th><td>?</td><td>4</td><td>0</td><td>?</td><td>0.99</td><td>0</td></tr><tr><th scope="row">Nonstop car chases</th><td>0</td><td>0</td><td>5</td><td>4</td><td>0.1</td><td>1.0</td></tr><tr><th scope="row">Swords vs. karate</th><td>0</td><td>0</td><td>5</td><td>0</td><td>0</td><td>0.9</td></tr>
</table></div>

- If we have features like these, each film can be recommended by a feature vector
  - Add an extra feature which is x**<sub>0</sub>** = 1 for each film
  - So for each film we have a \[3 x 1\] vector, which for film number 1 ("Love at Last") would be
    <div class="eqn">
    <math display="block"><mrow><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0.9</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">The leading 1 is the bias term; then romance 0.9, action 0.</span>
    </div>
  - i.e. for our dataset we have
    - {x<sup>1</sup>, x<sup>2</sup>, x<sup>3</sup>, x<sup>4</sup>, x<sup>5</sup>}
      - Where each of these is a \[3x1\] vector with an x**<sub>0</sub>** = 1 and then a romance and an action score
  - To be consistent with our notation, n is going to be the number of features NOT counting the x**<sub>0</sub>** term, so n = 2
- We could treat each rating for each user as a separate linear regression problem
  - For each user j we could learn a parameter vector
  - Then predict that user j will rate movie i with
    - (θ<sup>j</sup>)<sup><em>T </em></sup>x<sup>i </sup>= stars
    - inner product of parameter vector and features
  - So, lets take user 1 (Alice) and see what she makes of the modern classic Cute Puppies of Love (CPOL)
    - We have some parameter vector (θ<sup>1</sup>) associated with Alice
      - We'll explain later how we derived these values, but for now just take it that we have a vector
        <div class="eqn">
        <math display="block"><mrow><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
        </div>
    - CPOL has a parameter vector (x<sup>3</sup>) associated with it
      <div class="eqn">
      <math display="block"><mrow><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0.99</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      </div>
      ::: code-eg
      ```
      import numpy as np

      theta1 = np.array([0, 5, 0])    # Alice's parameters
      x3 = np.array([1, 0.99, 0])     # Cute Puppies of Love

      theta1 @ x3   # 4.95 - the predicted rating from the worked example
      ```
      <p class="eqn-note">The inner product computed: 0&times;1 + 5&times;0.99 + 0&times;0, which is the 4.95 stars derived above.</p>
      :::
    - Our prediction will be equal to
      - (θ<sup>1</sup>)<sup><em>T </em></sup>x<sup>3 </sup>= (0 \* 1) + (5 \* 0.99) + (0 \* 0)
      - \= 4.95
        - Which may seem like a reasonable value
  - All we're doing here is applying a linear regression method for each user
    - So we determine a future rating based on their interest in romance and action based on previous films
  - We should also add one final piece of notation
    - m<sup>j</sup>, - Number of movies rated by the user (j)

### How do we learn (θ<sup>j</sup>)

- Create some parameters which give values as close as those seen in the data when applied
  <div class="eqn">
  <math display="block"><mrow><munder><mo movablelimits="false">min</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn><msup><mi>m</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
  <span class="eqn-note">Ordinary least squares over just the films this user has actually rated; r(i,&nbsp;j) = 1 marks those.</span>
  </div>
- Sum over all values of i (all movies the user has used) when r(i,j) = 1 (i.e. all the films that the user has rated)
- This is just like linear regression with least-squared error
- We can also add a regularization term to make our equation look as follows<br/>
  <div class="eqn">
  <math display="block"><mrow><munder><mo movablelimits="false">min</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn><msup><mi>m</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn><msup><mi>m</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
  </div>
  - The regularization term goes from k=1 through to n, so (θ<sup>j</sup>) ends up being an n+1 feature vector
    - Don't regularize over the bias terms (0)
- If you do this you get a reasonable value
- We're rushing through this a bit, but it's just a linear regression problem
- To make this a little bit clearer you can get rid of the m<sup>j</sup> term (it's just a constant so shouldn't make any difference to minimization)
  - So to learn (θ<sup>j</sup>)
    <div class="eqn">
    <math display="block"><mrow><munder><mo movablelimits="false">min</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
    <span class="eqn-note">m<sup>(j)</sup> is a constant, so dropping it does not change which &#x3b8; minimises the expression.</span>
    </div>
  - But for our recommender system we want to learn parameters for *all* users, so we add an extra summation term to this which means we determine the minimum (θ<sup>j</sup>) value for every user
    <div class="eqn">
    <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mrow><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup></mrow></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>u</mi></msub></mrow></munderover><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>u</mi></msub></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
    <span class="eqn-note">The same objective as above, summed over all users.</span>
    </div>
  - When you do this as a function of each (θ<sup>j</sup>) parameter vector you get the parameters for each user
    - So this is our optimization objective -> J(θ<sup>1</sup>, ..., θ<sup>nu</sup>)
- In order to do the minimization we have the following gradient descent
  <div class="eqn pseudocode">
  <math display="block"><mrow><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>:=</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>&#x2212;</mo><mi>&#x3b1;</mi><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mspace width="1em"/><mtext>(for</mtext><mspace width="0.3em"/><mi>k</mi><mo>=</mo><mn>0</mn><mtext>)</mtext></mrow></math>
  <math display="block"><mrow><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>:=</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>&#x2212;</mo><mi>&#x3b1;</mi><mo form="prefix">(</mo><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>+</mo><mi>&#x3bb;</mi><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo form="postfix">)</mo><mspace width="1em"/><mtext>(for</mtext><mspace width="0.3em"/><mi>k</mi><mo>&#x2260;</mo><mn>0</mn><mtext>)</mtext></mrow></math>
  <span class="eqn-note">The bias term k = 0 is not regularised, exactly as in linear and logistic regression.</span>
  </div>
  - Slightly different to our previous gradient descent implementations
    - k = 0 and k != 0 versions
    - We can define the middle term above as
      <div class="eqn">
      <math display="block"><mrow><mfrac><mrow><mo>&#x2202;</mo></mrow><mrow><mo>&#x2202;</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mfrac><mi>J</mi><mo form="prefix">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">(</mo><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>+</mo><mi>&#x3bb;</mi><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo form="postfix">)</mo></mrow></math>
      </div>
    - Difference from linear regression
      - No 1/m terms (got rid of the 1/m term)
      - Otherwise very similar
- This approach is called content-based approach because we assume we have features regarding the content which will help us identify things that make them appealing to a user
  - However, often such features are not available - next we discuss a non-content based approach!

## Collaborative filtering - overview

- The collaborative filtering algorithm has a very interesting property - does feature learning
  - i.e. it can learn for itself what features it needs to learn
- Recall our original data set above for our five films and four raters
  - Here we assume someone had calculated the "romance" and "action" amounts of the films
    - This can be very hard to do in reality
    - Often want more features than just two
- So - let's change the problem and pretend we have a data set where we don't know any of the features associated with the films
  <div class="table-wrap"><table>
  <caption>Collaborative filtering turns the problem round: the feature values are now what we are trying to learn.</caption>
  <tr><th scope="col">Movie</th><th scope="col">Alice (1)</th><th scope="col">Bob (2)</th><th scope="col">Carol (3)</th><th scope="col">Dave (4)</th><th scope="col">x<sub>1</sub><small>(romance)</small></th><th scope="col">x<sub>2</sub><small>(action)</small></th></tr>
  <tr><th scope="row">Love at last</th><td>5</td><td>5</td><td>0</td><td>0</td><td>?</td><td>?</td></tr><tr><th scope="row">Romance forever</th><td>5</td><td>?</td><td>?</td><td>0</td><td>?</td><td>?</td></tr><tr><th scope="row">Cute puppies of love</th><td>?</td><td>4</td><td>0</td><td>?</td><td>?</td><td>?</td></tr><tr><th scope="row">Nonstop car chases</th><td>0</td><td>0</td><td>5</td><td>4</td><td>?</td><td>?</td></tr><tr><th scope="row">Swords vs. karate</th><td>0</td><td>0</td><td>5</td><td>0</td><td>?</td><td>?</td></tr>
  </table></div>
  - Now let's make a different assumption
    - We've polled each user and found out how much each user likes
      - Romantic films
      - Action films
    - Which has generated the following parameter set
      <div class="eqn">
      <math display="block"><mrow><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>,</mo><mspace width='0.8em'/><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>,</mo><mspace width='0.8em'/><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>,</mo><mspace width='0.8em'/><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      <span class="eqn-note">Alice and Bob like romance, Carol and Dave like action.</span>
      </div>
    - Alice and Bob like romance but hate action
    - Carol and Dave like action but hate romance
- If we can get these parameters from the users we can infer the missing values from our table
  - Lets look at "Love at Last"
    - Alice and Bob loved it
    - Carol and Dave hated it
  - We know from the feature vectors Alice and Bob love romantic films, while Carol and Dave hate them
    - Based on the factor Alice and Bob liked "Love at Last" and Carol and Dave hated it we may be able to (correctly) conclude that "Love at Last" is a romantic film
- This is a bit of a simplification in terms of the maths, but what we're really asking is
  - "What feature vector should x<sup>1</sup> be so that
    - (θ<sup>1</sup>)<sup><em>T </em></sup>x<sup>1 </sup>is about 5
    - (θ<sup>2</sup>)<sup><em>T </em></sup>x<sup>1 </sup>is about 5
    - (θ<sup>3</sup>)<sup><em>T </em></sup>x<sup>1 </sup>is about 0
    - (θ<sup>4</sup>)<sup><em>T </em></sup>x<sup>1 </sup>is about 0
  - From this we can guess that x<sup>1 </sup>may be
    <div class="eqn">
    <math display="block"><mrow><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>1.0</mn></mtd></mtr><mtr><mtd><mn>0.0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">Learned from the ratings rather than supplied &#x2014; and it comes out close to the 0.9 / 0 that was given by hand earlier.</span>
    </div>
  - Using that same approach we should then be able to determine the remaining feature vectors for the other films

### Formalizing the collaborative filtering problem

- We can more formally describe the approach as follows
  - Given (θ<sup>1</sup>, ..., θ<sup>nu</sup>) (i.e. given the parameter vectors for each user's preferences)
  - We must minimize an optimization function which tries to identify the best parameter vector associated with a film<br/>
    <div class="eqn">
    <math display="block"><mrow><munder><mo movablelimits="false">min</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mi>j</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
    <span class="eqn-note">The mirror image of fitting a user: now the parameters are known and the features are what we solve for.</span>
    </div>
    - So we're summing over all the indices j for where we have data for movie i
    - We're minimizing this squared error
  - Like before, the above equation gives us a way to learn the features for one film
    - We want to learn all the features for *all* the films - so we need an additional summation term

### How does this work with the previous recommendation system

- Content based recommendation systems
  - Saw that if we have a set of features for movie rating you can learn a user's preferences
- Now
  - If you have your users preferences you can therefore determine a film's features
- This is a bit of a chicken & egg problem
- What you can do is
  - Randomly guess values for θ
  - Then use collaborative filtering to generate x
  - Then use content based recommendation to improve θ
  - Use that to improve x
  - And so on
- This actually works
  - Causes your algorithm to converge on a reasonable set of parameters
  - This is collaborative filtering
- We call it collaborative filtering because in this example the users are collaborating together to help the algorithm learn better features and help the system and the other users

## Collaborative filtering Algorithm

- Here we combine the ideas from before to build a collaborative filtering algorithm
- Our starting point is as follows
  - If we're given the film's features we can use that to work out the users' preference
    <div class="eqn pseudocode">
    <math display="block"><mrow><mtext>Given</mtext><mspace width="0.4em"/><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mspace width="0.4em"/><mtext>estimate</mtext><mspace width="0.4em"/><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>:</mo></mrow></math>
    <math display="block"><mrow><mspace width="1.4em"/><munder><mo movablelimits="false">min</mo><mrow><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup></mrow></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>u</mi></msub></mrow></munderover><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>u</mi></msub></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
    </div>
  - If we're given the users' preferences we can use them to work out the film's features
    <div class="eqn pseudocode">
    <math display="block"><mrow><mtext>Given</mtext><mspace width="0.4em"/><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mspace width="0.4em"/><mtext>estimate</mtext><mspace width="0.4em"/><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>:</mo></mrow></math>
    <math display="block"><mrow><mspace width="1.4em"/><munder><mo movablelimits="false">min</mo><mrow><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup></mrow></munder><mspace width="0.35em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>m</mi></msub></mrow></munderover><munder><mo>&#x2211;</mo><mrow><mi>j</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>m</mi></msub></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
    </div>
- One thing you could do is
  - Randomly initialize parameter
  - Go back and forward
- But there's a more efficient algorithm which can solve θ and x simultaneously
  - Define a new optimization objective which is a function of x and θ

<div class="eqn pseudocode">
<math display="block"><mrow><mtext>Minimising</mtext><mspace width="0.4em"/><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mspace width="0.4em"/><mtext>and</mtext><mspace width="0.4em"/><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mspace width="0.4em"/><mtext>simultaneously:</mtext></mrow></math>
<math display="block"><mrow><mi>J</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>m</mi></msub></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>u</mi></msub></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
<math display="block"><mrow><munder><mo movablelimits="false">min</mo><mrow><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup></mrow></munder><mspace width="0.5em"/><mi>J</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow></math>
<span class="eqn-note">Solving for both at once. The single sum over (i,&nbsp;j) with r(i,&nbsp;j)&nbsp;=&nbsp;1 replaces the nested sums above &#x2014; it runs over every rating that exists, once each.</span>
</div>

- Understanding this optimization objective
  - The squared error term is the same as the squared error term in the two individual objectives above
    <div class="eqn">
    <math display="block"><mrow><munder><mo>&#x2211;</mo><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
    </div>
    - So it's summing over every movie rated by every user
    - Note the ":" means, "for which"
      - Sum over all pairs (i,j) for which r(i,j) is equal to 1
  - The regularization terms
    - Are simply added to the end from the original two optimization functions
- This newly defined function has the property that
  - If you held x constant and only solved θ then you solve the, "Given x, solve θ" objective above
  - Similarly, if you held θ constant you could solve x
- In order to come up with just one optimization function we treat this function as a function of both film features x and user parameters θ
  - Only difference between this in the back-and-forward approach is that we minimize with respect to both x and θ simultaneously
- When we're learning the features this way
  - Previously had a convention that we have an x**<sub>0</sub>** = 1 term
  - When we're using this kind of approach we have no x**<sub>0</sub>**,
    - So now our vectors (both x and θ) are n-dimensional (not n+1)
  - We do this because we are now learning all the features so if the system needs a feature always = 1 then the algorithm can learn one

### Algorithm Structure

- <strong>1)</strong> Initialize θ<sup>1</sup>, ..., θ<sup>nu </sup>and x<sup>1</sup>, ..., x<sup>nm</sup> to small random values
  - A bit like neural networks - initialize all parameters to small random numbers
- <strong>2)</strong> Minimize cost function J(x<sup>1</sup>, ..., x<sup>nm</sup>, θ<sup>1</sup>, ...,θ<sup>nu</sup>) using gradient descent
  - We find that the update rules look like this
    <div class="eqn">
    <math display="block"><mrow><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>:=</mo><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>&#x2212;</mo><mi>&#x3b1;</mi><mo form="prefix">(</mo><munder><mo>&#x2211;</mo><mrow><mi>j</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>+</mo><mi>&#x3bb;</mi><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo form="postfix">)</mo></mrow></math>
    </div>
    <div class="eqn">
    <math display="block"><mrow><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>:=</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>&#x2212;</mo><mi>&#x3b1;</mi><mo form="prefix">(</mo><munder><mo>&#x2211;</mo><mrow><mi>i</mi><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>+</mo><mi>&#x3bb;</mi><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo form="postfix">)</mo></mrow></math>
    <span class="eqn-note">In collaborative filtering there is no bias term, so unlike the update above there is no separate case for k = 0. Every parameter is regularised.</span>
    </div>
  - Where the top term is the partial derivative of the cost function with respect to x**<sub>k</sub>**<sup>i</sup> while the bottom is the partial derivative of the cost function with respect to θ**<sub>k</sub>**<sup>i</sup>
  - So here we regularize EVERY parameters (no longer x**<sub>0 </sub>**parameter) so no special case update rule
- <strong>3)</strong> Having minimized the values, given a user (user j) with parameters θ and movie (movie i) with learned features x, we predict a star rating of (θ<sup>j</sup>)<sup><em>T </em></sup>x<sup>i </sup>
  - This is the collaborative filtering algorithm, which should give pretty good predictions for how users like new movies

## Vectorization: Low rank matrix factorization

- Having looked at collaborative filtering algorithm, how can we improve this?
  - Given one product, can we determine other relevant products?
- We start by working out another way of writing out our predictions
  - So take all ratings by all users in our example above and group into a matrix Y
    <div class="eqn">
    <math display="block"><mrow><mi>Y</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>5</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd><mtd><mo>?</mo></mtd><mtd><mo>?</mo></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mo>?</mo></mtd><mtd><mn>4</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>4</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">One row per film, one column per user.</span>
    </div>
    - 5 movies
    - 4 users
    - Get a \[5 x 4\] matrix
  - Given \[Y\] there's another way of writing out all the predicted ratings
    <div class="eqn">
    <math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd><mtd><mo>&#x2026;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd><mtd><mo>&#x2026;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd></mtr><mtr><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd><mtd><mo>&#x2026;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">Every user&#x2019;s predicted rating for every film. This whole matrix is X&#x398;<sup>T</sup>, which is why the method is also called low-rank matrix factorisation.</span>
    </div>
    - With this matrix of predictive ratings
    - We determine the (i,j) entry for EVERY movie
- We can define another matrix X
  - Just like matrix we had for linear regression
  - Take all the features for each movie and stack them in rows
    <div class="eqn">
    <math display="block"><mrow><mi>X</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mo>&#x2212;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mtd><mtd><mo>&#x2212;</mo></mtd></mtr><mtr><mtd><mo>&#x2212;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mtd><mtd><mo>&#x2212;</mo></mtd></mtr><mtr><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd></mtr><mtr><mtd><mo>&#x2212;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mtd><mtd><mo>&#x2212;</mo></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">One film per row.</span>
    </div>
    - Think of each movie as one example
  - Also define a matrix Θ<br/>
    <div class="eqn">
    <math display="block"><mrow><mi>&#x398;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mo>&#x2212;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mtd><mtd><mo>&#x2212;</mo></mtd></mtr><mtr><mtd><mo>&#x2212;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mtd><mtd><mo>&#x2212;</mo></mtd></mtr><mtr><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd></mtr><mtr><mtd><mo>&#x2212;</mo></mtd><mtd><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mtd><mtd><mo>&#x2212;</mo></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">One user per row. Capital &#x398; is the matrix; lower-case &#x3b8;<sup>(j)</sup> is one user's parameters.</span>
    </div>
    - Take each per user parameter vector and stack in rows
- Given our new matrices X and Θ
  - We can have a vectorized way of computing the prediction range matrix by doing X \* Θ<sup><em>T</em></sup>
- We can given this algorithm another name - **<span class="hl-red">low rank matrix factorization</span>**
  - This comes from the property that the X \* Θ<sup><em>T </em></sup>calculation has a property in linear algebra that we create a **<span class="hl-red">low rank</span>** matrix
    - Don't worry about what a low rank matrix is

### Recommending new movies to a user

- Finally, having run the collaborative filtering algorithm, we can use the learned features to find related films
  - When you learn a set of features you don't know what the features will be - lets you identify the features which define a film
  - Say we learn the following features
    - x**<sub>1</sub>** - romance
    - x**<sub>2</sub>** - action
    - x**<sub>3</sub>** - comedy
    - x**<sub>4</sub>** - ...
  - So we have n features all together
  - After you've learned features it's often very hard to come in and apply a human understandable metric to what those features are
    - Usually learn features which are very meaningful for understanding what users like
- Say you have movie i
  - Find a movie j which is similar to i, which you can recommend
  - Our features allow a good way to measure movie similarity
  - If we have two movies x<sup>i</sup> and x<sup>j</sup>
    - We want to minimize ||x<sup>i</sup> - x<sup>j</sup>||
      - i.e. the distance between those two movies
  - Provides a good indicator of how similar two films are in the sense of user perception
    - NB - Maybe ONLY in terms of user perception

## Implementation detail: Mean Normalization

- Here we have one final implementation detail - make algorithm work a bit better
- To show why we might need mean normalization let's consider an example where there's a user who hasn't rated *any* movies
  <div class="table-wrap"><table>
  <caption>Eve has rated nothing at all.</caption>
  <tr><th scope="col">Movie</th><th scope="col">Alice (1)</th><th scope="col">Bob (2)</th><th scope="col">Carol (3)</th><th scope="col">Dave (4)</th><th scope="col">Eve (5)</th></tr>
  <tr><th scope="row">Love at last</th><td>5</td><td>5</td><td>0</td><td>0</td><td>?</td></tr><tr><th scope="row">Romance forever</th><td>5</td><td>?</td><td>?</td><td>0</td><td>?</td></tr><tr><th scope="row">Cute puppies of love</th><td>?</td><td>4</td><td>0</td><td>?</td><td>?</td></tr><tr><th scope="row">Nonstop car chases</th><td>0</td><td>0</td><td>5</td><td>4</td><td>?</td></tr><tr><th scope="row">Swords vs. karate</th><td>0</td><td>0</td><td>5</td><td>0</td><td>?</td></tr>
  </table></div>
  <div class="eqn">
  <math display="block"><mrow><mi>Y</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>5</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>5</mn></mtd><mtd><mo>?</mo></mtd><mtd><mo>?</mo></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mo>?</mo></mtd><mtd><mn>4</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>4</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
  <span class="eqn-note">The same table as a matrix, with a column of unrated entries for Eve. Without mean normalisation every one of her predictions comes out at zero.</span>
  </div>
  - Lets see what the algorithm does for this user
    - Say n = 2
    - We now have to learn θ<sup>5</sup> (which is an n-dimensional vector)
  - Looking in the first term of the optimization objective
    - There are *no* films for which r(i,j) = 1
    - So this term plays no role in determining θ<sup>5</sup>
    - So we're just minimizing the final regularization term

<div class="eqn">
<math display="block"><mrow><munder><mo movablelimits="false">min</mo><mrow><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>m</mi></msub><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><msub><mi>n</mi><mi>u</mi></msub><mo stretchy="false">)</mo></mrow></msup></mrow></munder><mspace width="0.5em"/><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><munder><mo>&#x2211;</mo><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>:</mo><mi>r</mi><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></munder><msup><mrow><mo stretchy="false">(</mo><msup><mrow><mo stretchy="false">(</mo><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msup><mi>y</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo>,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>m</mi></msub></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>x</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mrow><msub><mi>n</mi><mi>u</mi></msub></mrow></munderover><munderover><mo>&#x2211;</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mrow><mi>n</mi></mrow></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mi>k</mi></mrow><mrow><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
<math display="block"><mrow><mtext>for Eve, this reduces to</mtext><mspace width="0.6em"/><mfrac><mrow><mi>&#x3bb;</mi></mrow><mrow><mn>2</mn></mrow></mfrac><mo form="prefix">[</mo><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mn>1</mn></mrow><mrow><mo stretchy="false">(</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo>+</mo><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>&#x3b8;</mi><mrow><mn>2</mn></mrow><mrow><mo stretchy="false">(</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow><mn>2</mn></msup><mo form="postfix">]</mo></mrow></math>
<span class="eqn-note">Eve has rated nothing, so no term of the first sum involves her and the middle term does not involve &#x3b8; at all. Only the regularisation on her own parameters is left &#x2014; and that is minimised by setting them to zero, which predicts zero for every film.</span>
</div>

- Of course, if the goal is to minimize this term then
  <div class="eqn">
  <math display="block"><mrow><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
  </div>
  - Why - If there's no data to pull the values away from 0 this gives the min value
- So this means we predict ANY movie to be zero
  - Presumably Eve doesn't hate all movies...
  - So if we're doing this we can't recommend any movies to her either
- Mean normalization should let us fix this problem

### How does mean normalization work?

- Group all our ratings into matrix Y as before
  - We now have a column of ?s which corresponds to Eve's rating
    <div class="eqn">
    <math display="block"><mrow><mi>Y</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>5</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>5</mn></mtd><mtd><mo>?</mo></mtd><mtd><mo>?</mo></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mo>?</mo></mtd><mtd><mn>4</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>4</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>?</mo></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    </div>
  - Now we compute the average rating each movie obtained and store it in an n**<sub>m</sub>** - dimensional column vector
    <div class="eqn">
    <math display="block"><mrow><mi>&#x3bc;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>2.5</mn></mtd></mtr><mtr><mtd><mn>2.5</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>2.25</mn></mtd></mtr><mtr><mtd><mn>1.25</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">Each film&#x2019;s average over the users who did rate it. Verified against the matrix above.</span>
    </div>
  - If we look at all the movie ratings in \[Y\] we can subtract off the mean rating
    <div class="eqn">
    <math display="block"><mrow><mi>Y</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>2.5</mn></mtd><mtd><mn>2.5</mn></mtd><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>2.5</mn></mrow></mtd><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>2.5</mn></mrow></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mn>2.5</mn></mtd><mtd><mo>?</mo></mtd><mtd><mo>?</mo></mtd><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>2.5</mn></mrow></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mo>?</mo></mtd><mtd><mn>2</mn></mtd><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>2</mn></mrow></mtd><mtd><mo>?</mo></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>2.25</mn></mrow></mtd><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>2.25</mn></mrow></mtd><mtd><mn>2.75</mn></mtd><mtd><mn>1.75</mn></mtd><mtd><mo>?</mo></mtd></mtr><mtr><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>1.25</mn></mrow></mtd><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>1.25</mn></mrow></mtd><mtd><mn>3.75</mn></mtd><mtd><mrow><mo form="prefix">&#x2212;</mo><mn>1.25</mn></mrow></mtd><mtd><mo>?</mo></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">Every row now averages to zero. Predictions are made on this matrix and the film&#x2019;s mean added back, so Eve &#x2014; whose parameters are still zero &#x2014; is predicted to rate each film at its average rather than at zero.</span>
    </div>
    ::: code-eg
    ```
    nan = np.nan
    Y = np.array([[5,   5,   0, 0, nan],
                  [5, nan, nan, 0, nan],
                  [nan, 4,   0, nan, nan],
                  [0,   0,   5, 4, nan],
                  [0,   0,   5, 0, nan]])

    mu = np.nanmean(Y, axis=1)   # [2.5, 2.5, 2., 2.25, 1.25] - the vector above
    Y_norm = Y - mu[:, None]     # the normalised matrix above; ?s stay ?
    np.nanmean(Y_norm, axis=1)   # [0., 0., 0., 0., 0.] - every film now averages 0
    ```
    <p class="eqn-note">Mean normalization computed with nan for the unrated cells: nanmean averages over only the ratings that exist, and broadcasting subtracts each film's mean from its whole row.</p>
    :::
    - Means we normalize each film to have an average rating of 0
  - Now, we take the new set of ratings and use it with the collaborative filtering algorithm
    - Learn θ<sup>j</sup> and x<sup>i</sup> from the mean normalized ratings
- For our prediction of user j on movie i, predict
  - (θ<sup>j</sup>)<sup><em>T </em></sup>x<sup>i</sup> + μ**<sub>i</sub>**
    - Where these vectors are the mean normalized values
    - We have to add μ because we removed it from our θ values
  - So for user 5 the same argument applies, so
    <div class="eqn">
    <math display="block"><mrow><msup><mi>&#x3b8;</mi><mrow><mo stretchy="false">(</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">Normalising does not give Eve parameters; it changes what predicting zero means.</span>
    </div>
  - So on any movie i we're going to predict
    - (θ<sup>5</sup>)<sup><em>T </em></sup>x<sup>i </sup>+ μ**<sub>i</sub>**
      - Where (θ<sup>5</sup>)<sup><em>T </em></sup>x<sup>i </sup>= to 0 (still)
      - But we then add the mean (μ**<sub>i</sub>**) which means Eve has an average rating assigned to each movie for her
- This makes sense
  - If Eve hasn't rated any films, predict the average rating of the films based on everyone
    - This is the best we can do
- As an aside - we spoke here about mean normalization for users with no ratings
  - If you have some movies with no ratings you can also play with versions of the algorithm where you normalize the columns
  - BUT this is probably less relevant - probably shouldn't recommend an unrated movie
- To summarize, this shows how you do mean normalization preprocessing to allow your system to deal with users who have not yet made any ratings
  - Means we recommend the user we know little about the best average rated products
