---
title: "13: Clustering"
nav_title: "13: Clustering"
origin: "2011"
description: "Unsupervised learning with K-means: the algorithm, its objective, and choosing the number of clusters."
---

## Unsupervised learning - introduction

- Talk about **<span class="term">clustering</span>**
  - **<span class="hl-red">Learning from unlabeled data</span>**
- Unsupervised learning
  - Useful to contrast with supervised learning
- Compare and contrast
  - Supervised learning
    - Given a set of labels, fit a hypothesis to it
  - Unsupervised learning
    - Try and determine structure in the data
    - Clustering algorithm groups data together based on data features
- What is clustering good for
  - **Market segmentation** - group customers into different market segments
  - **Social network analysis** - Facebook "smartlists"
  - **Organizing computer clusters** and data centers for network layout and location
  - **Astronomical data analysis** - Understanding galaxy formation

## K-means algorithm

- Want an algorithm to automatically group the data into coherent clusters
- K-means is **<span class="term">by far</span>** the most widely used clustering algorithm

### Overview

- Take unlabeled data and group into two clusters
  <figure><img alt="" loading="lazy" src="13_Clustering_files/Image.png"/></figure>
- Algorithm overview
  - 1\) Randomly allocate two points as the **<span class="term">cluster centroids</span>**
    - Have as many cluster centroids as clusters you want to do (*K* cluster centroids, in fact)
    - In our example we just have two clusters
  - 2\) Cluster assignment step
    - Go through each example and depending on if it's closer to the red or blue centroid assign each point to one of the two clusters
    - To demonstrate this, we've gone through the data and "colour" each point red or blue
      <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [1].png"/></figure>
  - 3\) Move centroid step
    - Take each centroid and move to the average of the correspondingly assigned data-points
      <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [2].png"/></figure>
    - Repeat 2) and 3) until convergence
- More formal definition
  - <strong>Input:</strong>
    - K (number of clusters in the data)
    - Training set {x<sup>1</sup>, x<sup>2</sup><em>, </em>x<sup>3</sup><em> </em>..., x<sup>n</sup>)
  - <strong>Algorithm:</strong>
    - Randomly initialize K cluster centroids as {μ<sub>1</sub>, μ<sub>2</sub>, μ<sub>3</sub> ... μ<sub>K</sub>}
      <div class="eqn pseudocode">
      <math display="block"><mrow><mtext>Repeat {</mtext></mrow></math>
      <math display="block"><mrow><mspace width="1.6em"/><mtext>for</mtext><mspace width="0.35em"/><mi>i</mi><mo>=</mo><mn>1</mn><mspace width="0.35em"/><mtext>to</mtext><mspace width="0.35em"/><mi>m</mi></mrow></math>
      <math display="block"><mrow><mspace width="3.2em"/><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mspace width="0.4em"/><mo>:=</mo><mspace width="0.4em"/><mtext>index (from 1 to</mtext><mspace width="0.3em"/><mi>K</mi><mspace width="0.3em"/><mtext>) of cluster centroid</mtext></mrow></math>
      <math display="block"><mrow><mspace width="5.4em"/><mtext>closest to</mtext><mspace width="0.4em"/><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup></mrow></math>
      <math display="block"><mrow><mspace width="1.6em"/><mtext>for</mtext><mspace width="0.35em"/><mi>k</mi><mo>=</mo><mn>1</mn><mspace width="0.35em"/><mtext>to</mtext><mspace width="0.35em"/><mi>K</mi></mrow></math>
      <math display="block"><mrow><mspace width="3.2em"/><msub><mi>&#x3bc;</mi><mi>k</mi></msub><mspace width="0.4em"/><mo>:=</mo><mspace width="0.4em"/><mtext>average (mean) of points assigned to cluster</mtext><mspace width="0.3em"/><mi>k</mi></mrow></math>
      <math display="block"><mrow><mtext>}</mtext></mrow></math>
      <span class="eqn-note">The two halves are the whole algorithm: the first loop assigns every example to its nearest centroid, the second moves every centroid to the mean of the examples just assigned to it. Repeat until nothing moves.</span>
      </div>
      ::: code-eg
      ```
      import numpy as np

      X = np.array([[1.0, 1], [1.5, 2], [1, 1.5],   # three points near (1, 1.5)
                    [8, 8], [8.5, 9], [8, 8.5]])    # three points near (8, 8.5)
      mu = X[[0, 3]].copy()                          # init on two random examples

      for _ in range(10):
          c  = np.array([((x - mu) ** 2).sum(1).argmin() for x in X])   # assignment step
          mu = np.array([X[c == k].mean(0) for k in range(2)])          # move step

      c    # [0, 0, 0, 1, 1, 1]           - the two clusters, found
      mu   # [[1.17, 1.5], [8.17, 8.5]]  - each centroid at its cluster's mean
      ```
      <p class="eqn-note">The whole algorithm on six points: the assignment step is the argmin from the box above, the move step is the mean, and two clusters fall out.</p>
      :::
      - Loop 1
        - This inner loop repeatedly sets the c<sup>(i) </sup>variable to be the index of the cluster centroid closest to x<sup>i </sup>
        - i.e. take i<sup>th</sup> example, measure squared distance to each cluster centroid, assign c<sup>(i)</sup>to the cluster closest
          <div class="eqn">
          <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mi>k</mi></munder><mspace width="0.35em"/><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msub><mi>&#x3bc;</mi><mi>k</mi></msub><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></math>
          <span class="eqn-note">The slide draws an arrow from the minimisation to c<sup>(i)</sup>: what gets stored is not the distance but the k that attains it, so this is strictly an argmin. Squaring makes no difference to which k wins, and it is the form that matches the cost function below.</span>
          </div>
      - Loop 2
        - Loops over each centroid calculating the average mean based on all the points associated with each centroid from c<sup>(i)</sup>
    - What if there's a centroid with no data
      - Remove that centroid, so end up with K-1 classes
      - Or, randomly reinitialize it
        - Not sure when though...

### K-means for non-separated clusters

- So far looking at K-means where we have well defined clusters
- But often K-means is applied to datasets where there aren't well defined clusters
  - e.g. T-shirt sizing
    <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [5].png"/></figure>
- Not obvious discrete groups
- Say you want to have three sizes (S,M,L) how big do you make these?
  - One way would be to run K-means on this data
  - May do the following
    <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [6].png"/></figure>
  - So creates three clusters, even though they aren't really there
  - Look at first population of people
    - Try and design a small T-shirt which fits the 1st population
    - And so on for the other two
  - This is an example of market segmentation
    - Build products which suit the needs of your subpopulations

## K means optimization objective

- Supervised learning algorithms have an optimization objective (cost function)
  - K-means does too
- K-means has an optimization objective like the supervised learning functions we've seen
  - Why is this good?
  - Knowing this is useful because it helps for debugging
  - Helps find better clusters
- While K-means is running we keep track of two sets of variables
  - c<sup>i</sup> is the index of clusters {1,2, ..., K} to which x<sup>i</sup> is currently assigned
    - i.e. there are *m* c<sup>i</sup> values, as each example has a c<sup>i</sup> value, and that value is one of the clusters (i.e. can only be one of K different values)
  - μ<sub>k</sub>, is the cluster associated with centroid *k*
    - Locations of cluster centroid k
    - So there are K
    - So these the centroids which exist in the training data space
  - μ<sub>c</sub><sup>i</sup>, is the cluster centroid of the cluster to which example x<sup>i</sup> has been assigned
    - This is more for convenience than anything else
      - You could look up that example i is indexed to cluster j (using the c vector), where j is between 1 and K
      - Then look up the value associated with cluster j in the μ vector (i.e. what are the features associated with μ<sub>j</sub>)
      - But instead, for easy description, we have this variable which gets exactly the same value
    - Lets say x<sup>i </sup>has been assigned to cluster 5
      - Means that
        - c<sup>i</sup> = 5
        - μ<sub>c</sub><sup>i</sup>, = μ<sub>5</sub>
- Using this notation we can write the optimization objective;
  <div class="eqn">
  <math display="block"><mrow><mi>J</mi><mo stretchy="false">(</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>1</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msub><mi>&#x3bc;</mi><mn>1</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>&#x3bc;</mi><mi>K</mi></msub><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">&#x2016;</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><msub><mi>&#x3bc;</mi><mrow><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup></mrow></msub><mo stretchy="false">&#x2016;</mo></mrow><mn>2</mn></msup></mrow></math>
  <span class="eqn-note">The mean squared distance from each example to the centroid it was assigned to. Both steps of the algorithm above reduce it, which is why K-means converges.</span>
  </div>
  ::: code-eg
  ```
  ((X - mu[c]) ** 2).sum(1).mean()   # J = 0.222 - the distortion at convergence
  ```
  <p class="eqn-note">The distortion of the clustering just found &mdash; mu[c] lines each example up against its own centroid, so the whole cost is one line.</p>
  :::
  - i.e. squared distances between training example x<sup>i </sup>and the cluster centroid to which x<sup>i </sup>has been assigned
    - This is just what we've been doing, as the visual description below shows;
      <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [8].png"/></figure>
    - The red line here shows the distances between the example x<sup>i </sup>and the cluster to which that example has been assigned
      - Means that when the example is very close to the cluster, this value is small
      - When the cluster is very far away from the example, the value is large
  - This is sometimes called the **<span class="term">distortion</span>** (or **<span class="term">distortion cost function</span>**)
  - So we are finding the values which minimizes this function;
    <div class="eqn">
    <math display="block"><mrow><munder><mo movablelimits="false">min</mo><mtable><mtr><mtd><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>1</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo></mtd></mtr><mtr><mtd><msub><mi>&#x3bc;</mi><mn>1</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>&#x3bc;</mi><mi>K</mi></msub></mtd></mtr></mtable></munder><mspace width="0.5em"/><mi>J</mi><mo stretchy="false">(</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>1</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msub><mi>&#x3bc;</mi><mn>1</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>&#x3bc;</mi><mi>K</mi></msub><mo stretchy="false">)</mo></mrow></math>
    <span class="eqn-note">Minimised over both sets of variables at once &#x2014; the assignments and the centroid positions.</span>
    </div>
- If we consider the k-means algorithm
  - The **<span class="hl-red">cluster assigned step</span>** is minimizing J(...) with respect to c<sup>1</sup>, c<sup>2</sup> ... c<sup>i</sup>
    - i.e. find the centroid closest to each example
    - Doesn't change the centroids themselves
  - The **<span class="hl-red">move centroid step</span>**
    - We can show this step is choosing the values of μ which minimizes J(...) with respect to μ
  - So, we're partitioning the algorithm into two parts
    - First part minimizes the c variables
    - Second part minimizes the μ variables
- We can use this knowledge to help debug our K-means algorithm

### Random initialization

- How we initialize K-means
  - And how to avoid local optimum
- Consider clustering algorithm
  - Never spoke about how we initialize the centroids
    - A few ways - one method is most recommended
- Have number of centroids set to less than number of examples (K \< m) (if K > m we have a problem)
  - Randomly pick K training examples
  - Set μ<sub>1</sub> up to μ<sub>K</sub> to these example's values
- K means can converge to different solutions depending on the initialization setup
  - Risk of local optimum
    <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [10].png"/></figure>
  - The local optimum are valid convergence, but local optimum not global ones
- If this is a concern
  - We can do multiple random initializations
    - See if we get the same result - many same results are likely to indicate a global optimum
- Algorithmically we can do this as follows;
  <div class="eqn pseudocode">
  <math display="block"><mrow><mtext>For</mtext><mspace width="0.35em"/><mi>i</mi><mo>=</mo><mn>1</mn><mspace width="0.35em"/><mtext>to</mtext><mspace width="0.35em"/><mn>100</mn><mspace width="0.4em"/><mtext>{</mtext></mrow></math>
  <math display="block"><mrow><mspace width="1.6em"/><mtext>Randomly initialize K-means.</mtext></mrow></math>
  <math display="block"><mrow><mspace width="1.6em"/><mtext>Run K-means. Get</mtext><mspace width="0.5em"/><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>1</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msub><mi>&#x3bc;</mi><mn>1</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>&#x3bc;</mi><mi>K</mi></msub></mrow></math>
  <math display="block"><mrow><mspace width="1.6em"/><mtext>Compute cost function (distortion)</mtext></mrow></math>
  <math display="block"><mrow><mspace width="3.2em"/><mi>J</mi><mo stretchy="false">(</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>1</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msup><mi>c</mi><mrow><mo stretchy="false">(</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></msup><mo>,</mo><msub><mi>&#x3bc;</mi><mn>1</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>&#x3bc;</mi><mi>K</mi></msub><mo stretchy="false">)</mo></mrow></math>
  <math display="block"><mrow><mtext>}</mtext></mrow></math>
  <span class="eqn-note">K-means only finds a local optimum, and which one depends on where the centroids start. Running it many times and keeping the clustering with the lowest distortion is the standard fix.</span>
  </div>
  - A typical number of times to initialize K-means is 50-1000
  - Randomly initialize K-means
    - For each of the 100 random initializations run K-means
    - Then compute the distortion on the set of cluster assignments and centroids at convergence
    - End with 100 ways of clustering the data
    - Pick the clustering which gave the lowest distortion
- If you're running K means with 2-10 clusters this can help find better global optimum
  - If K is larger than 10, then multiple random initializations are less likely to be necessary
  - First solution is probably good enough (better granularity of clustering)

## How do we choose the number of clusters?

- Choosing K?
  - Not a great way to do this automatically
  - Normally use visualizations to do it manually
- What are the intuitions regarding the data?
- Why is this hard
  - Sometimes very ambiguous
    - e.g. two clusters or four clusters
    - Not necessarily a correct answer
  - This is why doing it automatic this is hard

### Elbow method

- Vary K and compute cost function at a range of K values
- As K increases J(...) minimum value should decrease (i.e. you decrease the granularity so centroids can better optimize)
  - Plot this (K vs J())
- Look for the "elbow" on the graph
  <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [12].png"/></figure>
- Chose the "elbow" number of clusters
- If you get a nice plot this is a reasonable way of choosing K
- Risks
  - Normally you don't get a nice line -> no clear elbow on curve
  - Not really that helpful

### Another method for choosing K

- Using K-means for market segmentation
- Running K-means for a later/downstream purpose
  - See how well different number of clusters serve your later needs
- e.g.
  - T-shirt size example
    - If you have three sizes (S,M,L)
    - Or five sizes (XS, S, M, L, XL)
    - Run K means where K = 3 and K = 5
  - How does this look
    <figure><img alt="" loading="lazy" src="13_Clustering_files/Image [13].png"/></figure>
  - This gives a way to choose the number of clusters
    - Could consider the cost of making extra sizes vs. how well distributed the products are
    - How important are those sizes though? (e.g. more sizes might make the customers happier)
    - So applied problem may help guide the number of clusters
