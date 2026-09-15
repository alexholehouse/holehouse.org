---
title: "14: Dimensionality Reduction (PCA)"
nav_title: "14: Dimensionality reduction"
index_title: "14: Dimensionality Reduction"
origin: "2011"
description: "Principal component analysis for compressing and visualising data, and how many components to keep."
---

## Motivation 1: Data compression

- Start talking about a second type of unsupervised learning problem - **<span class="term">dimensionality reduction</span>**
  - Why should we look at dimensionality reduction?

### Compression

- Speeds up algorithms
- Reduces space used by data for them
- What is dimensionality reduction?
  - So you've collected many features - maybe more than you need
    - Can you "simplify" your data set in a rational and useful way?
  - Example
    - Redundant data set - different units for same attribute
    - Reduce data to 1D (2D->1D)
      <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image.png"/></figure>
      - Example above isn't a perfect straight line because of round-off error
  - Data redundancy can happen when different teams are working independently
    - Often generates redundant data (especially if you don't control data collection)
  - Another example
    - Helicopter flying - do a survey of pilots (x1 = skill, x2 = pilot enjoyment)
      - These features may be highly correlated
      - This correlation can be combined into a single attribute called aptitude (for example)
- What does dimensionality reduction mean?
  - In our example we plot a line
  - Take exact example and record position on that line
    <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [1].png"/></figure>
  - So before x<sup>1 </sup>was a 2D feature vector (X and Y dimensions)
  - Now we can represent x<sup>1</sup> as a 1D number (Z dimension)
- So we approximate original examples
  - Allows us to halve the amount of storage
  - Gives lossy compression, but an acceptable loss (probably)
    - The loss above comes from the rounding error in the measurement, however
- Another example 3D -> 2D
  - So here's our data
    <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [2].png"/></figure>
  - Maybe all the data lies in one plane
    - This is sort of hard to explain in 2D graphics, but that plane may be aligned with one of the axes
      - Or may not...
      - Either way, the plane is a small, constant 3D space
    - In the diagram below, imagine all our data points are sitting "inside" the blue tray (has a dark blue exterior face and a light blue inside)
      <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [3].png"/></figure>
    - Because they're all in this relative shallow area, we can basically ignore one of the dimensions, so we draw two new lines (z1 and z2) along the x and y planes of the box, and plot the locations in that box
    - i.e. we lose the data in the z-dimension of our "shallow box" (NB "z-dimensions" here refers to the dimension relative to the box (i.e. its depth) and NOT the z dimension of the axis we've got drawn above) but because the box is shallow it's OK to lose this. Probably....
  - Plot values along those projections
    <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [4].png"/></figure>
  - So we've now reduced our 3D vector to a 2D vector
- In reality we'd normally try and do 1000D -> 100D

## Motivation 2: Visualization

- It's hard to visualize highly dimensional data
  - Dimensionality reduction can improve how we display information in a tractable manner for human consumption
  - Why do we care?
    - Often helps to develop algorithms if we can understand our data better
    - Dimensionality reduction helps us do this, see data in a helpful way
    - Good for explaining something to someone if you can "show" it in the data
- Example;
  - Collect a large data set about many facts of a country around the world
    <div class="table-wrap"><table>
      <caption>The first six of about fifty features per country. x<sub>1</sub> is GDP
      through to x<sub>6</sub>, mean household income; the rest continue off the right.</caption>
      <tr><th scope="col">Country</th><th scope="col">GDP<br/><small>(trillions of US$)</small></th><th scope="col">Per capita GDP<br/><small>(thousands of intl. $)</small></th><th scope="col">Human Development Index</th><th scope="col">Life expectancy</th><th scope="col">Poverty Index<br/><small>(Gini as percentage)</small></th><th scope="col">Mean household income<br/><small>(thousands of US$)</small></th><th scope="col">&#x2026;</th></tr>
      <tr><th scope="row">Canada</th><td>1.577</td><td>39.17</td><td>0.908</td><td>80.7</td><td>32.6</td><td>67.293</td><td>&#x2026;</td></tr><tr><th scope="row">China</th><td>5.878</td><td>7.54</td><td>0.687</td><td>73</td><td>46.9</td><td>10.22</td><td>&#x2026;</td></tr><tr><th scope="row">India</th><td>1.632</td><td>3.41</td><td>0.547</td><td>64.7</td><td>36.8</td><td>0.735</td><td>&#x2026;</td></tr><tr><th scope="row">Russia</th><td>1.48</td><td>19.84</td><td>0.755</td><td>65.5</td><td>39.9</td><td>0.72</td><td>&#x2026;</td></tr><tr><th scope="row">Singapore</th><td>0.223</td><td>56.69</td><td>0.866</td><td>80</td><td>42.5</td><td>67.1</td><td>&#x2026;</td></tr><tr><th scope="row">USA</th><td>14.527</td><td>46.86</td><td>0.91</td><td>78.3</td><td>40.8</td><td>84.3</td><td>&#x2026;</td></tr>
      <tr><th scope="row">&#x2026;</th><td>&#x2026;</td><td>&#x2026;</td><td>&#x2026;</td><td>&#x2026;</td><td>&#x2026;</td><td>&#x2026;</td><td>&#x2026;</td></tr>
    </table></div>
    - So
      - x<sub>1</sub> = GDP
      - ...
      - x<sub>6</sub> = mean household
    - Say we have 50 features per country
    - How can we understand this data better?
      - Very hard to plot 50 dimensional data
  - Using dimensionality reduction, instead of each country being represented by a 50-dimensional feature vector
    - Come up with a different feature representation (z values) which summarize these features
      <table>
        <caption>The same countries after reducing 50 dimensions to 2. Each row is now
        a pair of numbers that can be plotted.</caption>
        <tr><th scope="col">Country</th>
            <th scope="col">z<sub>1</sub></th><th scope="col">z<sub>2</sub></th></tr>
        <tr><th scope="row">Canada</th><td>1.6</td><td>1.2</td></tr><tr><th scope="row">China</th><td>1.7</td><td>0.3</td></tr><tr><th scope="row">India</th><td>1.6</td><td>0.2</td></tr><tr><th scope="row">Russia</th><td>1.4</td><td>0.5</td></tr><tr><th scope="row">Singapore</th><td>0.5</td><td>1.7</td></tr><tr><th scope="row">USA</th><td>2</td><td>1.5</td></tr>
        <tr><th scope="row">&#x2026;</th><td>&#x2026;</td><td>&#x2026;</td></tr>
      </table>
  - This gives us a 2-dimensional vector
    - Reduce 50D -> 2D
    - Plot as a 2D plot
  - Typically you don't generally ascribe meaning to the new features (so we have to determine what these summary values mean)
    - e.g. may find horizontal axis corresponds to overall country size/economic activity
    - and y axis may be the per-person well being/economic activity
  - So despite having 50 features, there may be two "dimensions" of information, with features associated with each of those dimensions
    - It's up to you to assess which of the features can be grouped to form summary features, and how best to do that (feature scaling is probably important)
  - Helps show the two main dimensions of variation in a way that's easy to understand

## Principal Component Analysis (PCA): Problem Formulation

- For the problem of dimensionality reduction the most commonly used algorithm is **<span class="term">PCA</span>**
  - Here, we'll start talking about how we formulate precisely what we want PCA to do
- So
  - Say we have a 2D data set which we wish to reduce to 1D
    <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [7].png"/></figure>
  - In other words, find a single line onto which to project this data
    - How do we determine this line?
      - The distance between each point and the projected version should be small (blue lines below are short)
      - PCA tries to find a lower dimensional surface so the sum of squares onto that surface is minimized
      - The blue lines are sometimes called the **<span class="hl-red">projection error</span>**
        - PCA tries to find the surface (a straight line in this case) which has the minimum projection error
          <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [8].png"/></figure>
      - As an aside, you should normally do **<span class="term">mean normalization</span>** and **<span class="term">feature scaling</span>** on your data before PCA
- A more formal description is
  - For 2D-1D, we must find a vector u<sup>(1)</sup>, which is of some dimensionality
  - Onto which you can project the data so as to minimize the projection error
    <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [9].png"/></figure>
  - u<sup>(1)</sup> can be positive or negative (-u<sup>(1)</sup>) which makes no difference
    - Each of the vectors define the same red line
- In the more general case
  - To reduce from nD to kD we
    - Find *k* vectors (u<sup>(1)</sup>, u<sup>(2)</sup>, ... u<sup>(k)</sup>) onto which to project the data to minimize the projection error
    - So lots of vectors onto which we project the data
    - Find a set of vectors which we project the data onto the linear subspace spanned by that set of vectors
      - We can define a point in a plane with k vectors
  - e.g. 3D->2D
    - Find pair of vectors which define a 2D plane (surface) onto which you're going to project your data
    - Much like the "shallow box" example in compression, we're trying to create the shallowest box possible (by defining two of it's three dimensions, so the box's depth is minimized)
      <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [10].png"/></figure>
- How does PCA relate to linear regression?
  - PCA is **not** linear regression
    - Despite cosmetic similarities, very different
  - For linear regression, fitting a straight line to minimize the **<span class="hl-red">squared distance</span>** between a point and the line
    - NB - **<span class="hl-red">VERTICAL distance</span>** between point
  - For PCA minimizing the magnitude of the shortest **<span class="hl-red">orthogonal distance</span>**
    - Gives very different effects
  - More generally
    - With linear regression we're trying to predict "y"
    - With PCA there is no "y" - instead we have a list of features and all features are treated equally
      - If we have 3D dimensional data 3D->2D
        - Have 3 features treated symmetrically

## PCA Algorithm

- Before applying PCA must do data preprocessing
  - Given a set of m unlabeled examples we must do
    - **Mean normalization**
      - Replace each x<sub>j</sub><sup>i</sup> with x<sub>j</sub> - μ<sub>j</sub>,
        - In other words, determine the mean of each feature set, and then for each feature subtract the mean from the value, so we re-scale the mean to be 0
    - <strong>Feature scaling (depending on data)</strong>
      - If features have very different scales then scale so they all have a comparable range of values
        - e.g. x<sub>j</sub><sup>i</sup> is set to (x<sub>j</sub> - μ<sub>j</sub>) / s<sub>j</sub>
          - Where s<sub>j </sub>is some measure of the range, so could be
            - Biggest - smallest
            - Standard deviation (more commonly)
- With preprocessing done, PCA finds the lower dimensional sub-space which minimizes the sum of the squares
  - In summary, for 2D->1D we'd be doing something like this;
    <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [11].png"/></figure>
  - Need to compute two things;
    - Compute the **u vectors**
      - The new planes
    - Need to compute the **z vectors**
      - z vectors are the new, lower dimensionality feature vectors
- A mathematical derivation for the u vectors is very complicated
  - But once you've done it, the procedure to find each u vector is not that hard

### Algorithm description

- Reducing data from *n*-dimensional to k-dimensional
  - Compute the covariance matrix
    <div class="eqn">
    <math display="block"><mrow><mi>&#x3a3;</mi><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><msup><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mrow></math>
    <span class="eqn-note">The slide runs this sum to n, the number of features, where it has to run to m, the number of training examples &#x2014; the 1/m in front of it is averaging over examples. Corrected here. Each term is an [n&#x00d7;1] column times a [1&#x00d7;n] row, so every one of them is [n&#x00d7;n], and so is &#x3a3;.</span>
    </div>
    ::: code-eg
    ```
    import numpy as np

    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, 100)
    X = np.c_[x1, 2 * x1 + rng.normal(0, 0.1, 100)]   # x2 is nearly 2*x1
    X = X - X.mean(axis=0)                            # mean normalize first

    Sigma = (X.T @ X) / len(X)          # the covariance matrix
    U, S, Vt = np.linalg.svd(Sigma)
    U[:, 0]   # [-0.446, -0.895] - the y = 2x direction (the sign is arbitrary),
              # i.e. [1, 2]/sqrt(5): the single direction the data really varies in
    ```
    <p class="eqn-note">PCA end to end on correlated data: the first column of U recovers the y = 2x line the data was generated along.</p>
    :::
    - This is commonly denoted as Σ (greek upper case sigma) - NOT summation symbol
    - Σ = sigma
      - This is an \[n x n\] matrix
        - Remember that x<sup>i </sup>is a \[n x 1\] matrix
    - In Python we can implement this as follows;
      <pre><code>Sigma = (X.T @ X) / m</code></pre>
      <p class="eqn-note">X holds the examples as rows, so X.T @ X sums the outer products in one matrix multiply &#x2014; no loop over the m examples needed.</p>
  - Compute eigenvectors of matrix Σ
    - <code class="hl-green"><strong>U, S, Vt = np.linalg.svd(Sigma)</strong></code>
      - svd = singular value decomposition
        - More numerically stable than <code class="hl-green"><strong>np.linalg.eig</strong></code>
      - <span class="hl-green"><strong><code>np.linalg.eig</code></strong></span> = also gives eigenvectors
  - U,S and V are matrices
    - U matrix is also an \[n x n\] matrix
    - Turns out the columns of U are the u vectors we want!
    - So to reduce a system from n-dimensions to k-dimensions
      - Just take the first *k-vectors* from U (first k columns)<br/>
        <div class="eqn">
        <math display="block"><mrow><mi>U</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mo stretchy="false">|</mo></mtd><mtd><mo stretchy="false">|</mo></mtd><mtd></mtd><mtd><mo stretchy="false">|</mo></mtd></mtr><mtr><mtd><msup><mi>u</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup></mtd><mtd><msup><mi>u</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup></mtd><mtd><mo>&#x2026;</mo></mtd><mtd><msup><mi>u</mi><mrow><mo stretchy="false">(</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></msup></mtd></mtr><mtr><mtd><mo stretchy="false">|</mo></mtd><mtd><mo stretchy="false">|</mo></mtd><mtd></mtd><mtd><mo stretchy="false">|</mo></mtd></mtr></mtable><mo>]</mo></mrow><mspace width="0.5em"/><mo>&#x2208;</mo><msup><mi>&#x211d;</mi><mrow><mi>n</mi><mo>&#x00d7;</mo><mi>n</mi></mrow></msup></mrow></math>
        <span class="eqn-note">One column per principal component, each a unit vector in the original n-dimensional space. Taking the first k columns gives U<sub>reduce</sub>.</span>
        </div>
- Next we need to find some way to change x (which is n dimensional) to z (which is k dimensional)
  - (reduce the dimensionality)
  - Take first k columns of the u matrix and stack in columns
    - n x k matrix - call this U<sub>reduce</sub>
  - We calculate z as follows
    - z = (U<sub>reduce</sub>)<sup><em>T</em></sup> \* x
      - So \[k x n\] \* \[n x 1\]
      - Generates a matrix which is
        - k \* 1
      - If that's not witchcraft I don't know what is!
- Exactly the same as with supervised learning except we're now doing it with unlabeled data
- So in summary
  - Preprocessing
  - Calculate sigma (covariance matrix)
  - Calculate eigenvectors with <span class="hl-green"><strong>svd</strong></span>
  - Take k vectors from U (U_reduce = U\[:, :k\])
  - Calculate z (z = U_reduce.T @ x)
- No mathematical derivation
  - Very complicated
  - But it works

## Reconstruction from Compressed Representation

- Earlier spoke about PCA as a compression algorithm
  - If this is the case, is there a way to **<span class="term">decompress</span>** the data from low dimensionality back to a higher dimensionality format?
- Reconstruction
  - Say we have an example as follows
    <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [15].png"/></figure>
    <div class="eqn">
    <math display="block"><mrow><mi>z</mi><mo>=</mo><msubsup><mi>U</mi><mtext>reduce</mtext><mi>T</mi></msubsup><mspace width="0.15em"/><mi>x</mi></mrow></math>
    <span class="eqn-note">U<sub>reduce</sub> is [n&#x00d7;k], so its transpose times an [n&#x00d7;1] example gives the [k&#x00d7;1] vector z.</span>
    </div>
  - We have our examples (x<sup>1</sup>, x<sup>2</sup> etc.)
  - Project onto z-surface
  - Given a point z<sup>1</sup>, how can we go back to the 2D space?
- Considering
  - z (vector) = (U<sub>reduce</sub>)<sup><em>T</em></sup> \* x
- To go in the opposite direction we must do
  - x<sub>approx</sub> = U<sub>reduce</sub><em> </em>\* z
    - To consider dimensions (and prove this really works)
      - U<sub>reduce</sub><em> </em>= \[n x k\]
      - z \[k \* 1\]
    - So
      - x<sub>approx</sub> = \[n x 1\]
- So this creates the following representation
  <figure><img alt="" loading="lazy" src="14_Dimensionality_Reduction_files/Image [16].png"/></figure>
- We lose some of the information (i.e. everything is now perfectly on that line) but it is now projected into 2D space

## Choosing the number of Principal Components

- How do we choose <em>k </em>?
  - k = number of **<span class="term">principal components</span>**
  - Guidelines about how to choose k for PCA
- To choose k think about how PCA works
  - PCA tries to minimize averaged squared projection error
    <div class="eqn">
    <math display="block"><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msubsup><mi>x</mi><mtext>approx</mtext><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></math>
    </div>
  - Total variation in data can be defined as the average over data saying how far are the training examples from the origin
    <div class="eqn">
    <math display="block"><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></math>
    </div>
- When we're choosing k typical to use something like this
  <div class="eqn">
  <math display="block"><mrow><mfrac><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msubsup><mi>x</mi><mtext>approx</mtext><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></mfrac><mo>&#x2264;</mo><mn>0.01</mn><mspace width="1.6em"/><mtext>(1%)</mtext></mrow></math>
  <span class="eqn-note">Projection error over total variation. At or below 0.01, 99% of the variance is retained.</span>
  </div>
  ::: code-eg
  ```
  k = 1
  U_reduce = U[:, :k]              # keep one principal component
  z = X @ U_reduce                 # project every example to 1-D
  X_approx = z @ U_reduce.T        # reconstruct

  num = ((X - X_approx) ** 2).sum(axis=1).mean()
  den = (X ** 2).sum(axis=1).mean()
  num / den   # 0.00039 -> 99.96% of the variance retained by one component
  ```
  <p class="eqn-note">The variance-retained test on the data above: squashing 2-D to 1-D loses 0.04% of the variation, because the second dimension was almost entirely redundant.</p>
  :::
  - Ratio between averaged squared projection error with total variation in data
    - Want ratio to be small - means we retain 99% of the variance
  - If it's small (0) then this is because the numerator is small
    - The numerator is small when x<sup>i</sup> = x<sub>approx</sub><sup>i</sup>
      - i.e. we lose very little information in the dimensionality reduction, so when we decompress we regenerate the same data
- So we choose k in terms of this ratio
- Often can significantly reduce data dimensionality while retaining the variance
- How do you do this
  <div class="eqn pseudocode">
  <math display="block"><mrow><mtext>Algorithm:</mtext></mrow></math>
  <math display="block"><mrow><mtext>Try PCA with</mtext><mspace width="0.35em"/><mi>k</mi><mo>=</mo><mn>1</mn></mrow></math>
  <math display="block"><mrow><mtext>Compute</mtext><mspace width="0.4em"/><msub><mi>U</mi><mtext>reduce</mtext></msub><mo>,</mo><msup><mi>z</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msup><mi>z</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>z</mi><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msubsup><mi>x</mi><mtext>approx</mtext><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msubsup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msubsup><mi>x</mi><mtext>approx</mtext><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></math>
  <math display="block"><mrow><mtext>Check if</mtext></mrow></math>
  <math display="block"><mrow><mspace width="1.4em"/><mfrac><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msubsup><mi>x</mi><mtext>approx</mtext><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></mfrac><mo>&#x2264;</mo><mn>0.01</mn><mo>?</mo></mrow></math>
  <span class="eqn-note">If it fails, try k = 2, then k = 3, and so on. The chapter notes that the singular values from svd give the same answer far more cheaply, without recomputing PCA at every k.</span>
  </div>

## Advice for Applying PCA

- Can use PCA to speed up algorithm running time
  - Explain how
  - And give general advice

### Speeding up supervised learning algorithms

- Say you have a supervised learning problem
  - Input x and y
    - x is a 10 000 dimensional feature vector
    - e.g. 100 x 100 images = 10 000 pixels
    - Such a huge feature vector will make the algorithm slow
  - With PCA we can reduce the dimensionality and make it tractable
  - How
    - 1\) Extract xs
      - So we now have an unlabeled training set
    - 2\) Apply PCA to x vectors
      - So we now have a reduced dimensional feature vector z
    - 3\) This gives you a new training set
      - Each vector can be re-associated with the label
    - 4\) Take the reduced dimensionality data set and feed to a learning algorithm
      - Use y as labels and z as feature vector
    - 5\) If you have a new example map from higher dimensionality vector to lower dimensionality vector, then feed into learning algorithm
- PCA maps one vector to a lower dimensionality vector
  - x -> z
  - Defined by PCA **only** on the training set
  - The mapping computes a set of parameters
    - Feature scaling values
    - U<sub>reduce</sub>
      - Parameter learned by PCA
      - Should be obtained only by determining PCA on your training set
  - So we use those learned parameters for our
    - Cross validation data
    - Test set
- Typically you can reduce data dimensionality by 5-10x without a major hit to algorithm

## Applications of PCA

- **<span class="term">Compression</span>**
  - Why
    - Reduce memory/disk needed to store data
    - Speed up learning algorithm
  - How do we choose k?
    - % of variance retained
- **<span class="term">Visualization</span>**
  - Typically choose k = 2 or k = 3
  - Because we can plot these values!
- One thing often done wrong regarding PCA
  - A bad use of PCA: Use it to prevent over-fitting
    - Reasoning
      - If we have x<sup>i</sup> we have n features, z<sup>i</sup> has k features which can be lower
      - If we *only* have k features then maybe we're less likely to over fit...
    - This doesn't work
      - BAD APPLICATION
      - Might work OK, but not a good way to address over fitting
      - Better to use regularization
    - PCA throws away some data without knowing what values it's losing
      - Probably OK if you're keeping most of the data
      - But if you're throwing away some crucial data bad
      - So you have to go to like 95-99% variance retained
        - So here regularization will give you AT LEAST as good a way to solve over fitting
- A second PCA myth
  - Used for compression or visualization - good
  - Sometimes used
    - Design ML system with PCA from the outset
      - But, what if you did the whole thing without PCA
    - See how a system performs without PCA
      - ONLY if you have a reason to believe PCA will help should you then add PCA
    - PCA is easy enough to add on as a processing step
      - Try without first!
