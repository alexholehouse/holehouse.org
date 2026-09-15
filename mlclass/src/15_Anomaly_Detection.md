---
title: "15: Anomaly Detection"
nav_title: "15: Anomaly detection"
origin: "2011"
description: "Modelling normal data with a Gaussian and flagging what does not fit, including the multivariate case."
---

## Anomaly detection - problem motivation

- Anomaly detection is a reasonably commonly used type of machine learning application
  - Can be thought of as a solution to an unsupervised learning problem
  - But, has aspects of supervised learning
- What is anomaly detection?
  - Imagine you're an aircraft engine manufacturer
  - As engines roll off your assembly line you're doing QA
    - Measure some features from engines (e.g. heat generated and vibration)
  - You now have a dataset of <strong>x<sup>1</sup> to </strong><strong>x<sup>m</sup> (i.e. <em>m </em>engines were tested)</strong>
  - **Say we plot that dataset <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image.png"/></figure>**
  - **Next day you have a new engine**
    - <strong>An anomaly detection method is used to see if the new engine is </strong>anomalous (when compared to the previous engines)
  - **If the new engine looks like this;<figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [1].png"/></figure>**
    - **Probably OK - looks like the ones we've seen before**
  - **But if the engine looks like this<figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [2].png"/></figure>**
    - Uh oh! - this looks like an **<span class="hl-red">anomalous </span>**<span class="hl-red"><strong>data-point</strong></span>
- More formally
  - We have a dataset which contains <span class="term"><strong>normal</strong> </span>(data)
    - How we ensure they're normal is up to us
    - In reality it's OK if there are a few which aren't actually normal
  - Using that dataset as a reference point we can see if other examples are **<span class="term">anomalous</span>**
- How do we do this?
  - First, using our training dataset we build a model
    - We can access this model using **<span class="hl-red">p(x)</span>**
      - This asks, "What is the probability that example x is normal"
  - Having built a model
    - if p(**x<sub>test</sub>**) \< ε --> flag this as an anomaly
    - if p(**x<sub>test</sub>**) >= ε --> this is OK
    - ε is some threshold probability value which we define, depending on how sure we need/want to be
  - We expect our model to (graphically) look something like this;
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [3].png"/></figure>
    - i.e. this would be our model if we had 2D data

### Applications

- <u>Fraud detection</u>
  - Users have activity associated with them, such as
    - Length of time on-line
    - Location of login
    - Spending frequency
  - Using this data we can build a model of what normal users' activity is like
  - What is the probability of "normal" behavior?
  - Identify unusual users by sending their data through the model
    - Flag up anything that looks a bit weird
    - Automatically block cards/transactions
- <u>Manufacturing</u>
  - Already spoke about aircraft engine example
- <u>Monitoring computers in data center</u>
  - If you have many machines in a cluster
  - Compute features of machine
    - **x<sub>1</sub>** = memory use
    - **x<sub>2</sub>** = number of disk accesses/sec
    - **x<sub>3</sub>** = CPU load
  - In addition to the measurable features you can also define your own complex features
    - **x<sub>4</sub>** = CPU load/network traffic
  - If you see an anomalous machine
    - Maybe about to fail
    - Look at replacing bits from it

## The Gaussian distribution (optional)

- Also called the **<span class="term">normal distribution</span>**
- Example
  - Say x (data set) is made up of real numbers
    - Mean is μ
    - Variance is σ<sup>2</sup>
      - σ is also called the **<span class="term">standard </span>**<strong><span class="term">deviation</span></strong> - specifies the width of the Gaussian probability
    - The data has a Gaussian distribution
  - Then we can write this ~ <em>N(</em>μ,σ<sup>2</sup> )
    - \~ means "is distributed as"
    - *N* (should really be "script" N (even curlier!) -> means normal distribution
    - <p>μ, σ<sup>2</sup> represent the mean and variance, respectively</p>
      - These are the two parameters a Gaussian needs
  - Looks like this;
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [4].png"/></figure>
  - This specifies the probability of x taking a value
    - As you move away from μ
- Gaussian equation is
  - P(x : μ , σ<sup>2</sup>) (probability of x, parameterized by the mean and variance)<br/>
    <div class="eqn">
    <math display="block"><mrow><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><msqrt><mn>2</mn><mi>&#x3c0;</mi></msqrt><mi>&#x3c3;</mi></mrow></mfrac><mspace width="0.25em"/><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow><mrow><mn>2</mn><msup><mi>&#x3c3;</mi><mn>2</mn></msup></mrow></mfrac><mo form="postfix">)</mo></mrow></math>
    </div>
- Some examples of Gaussians below
  - Area is always the same (must = 1)
  - But width changes as standard deviation changes

<figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [6].png"/></figure>

### Parameter estimation problem

- What is it?
  - Say we have a data set of m examples
  - Given each example is a real number - we can plot the data on the x axis as shown below
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [7].png"/></figure>
  - Problem is - say you suspect these examples come from a Gaussian
    - Given the dataset can you estimate the distribution?
  - Could be something like this
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [8].png"/></figure>
  - Seems like a reasonable fit - data seems like a higher probability of being in the central region, lower probability of being further away
- Estimating μ and σ<sup>2</sup>
  - μ = average of examples
  - σ<sup>2</sup> = standard deviation squared <br/>
    <div class="eqn">
    <math display="block"><mrow><msup><mi>&#x3c3;</mi><mn>2</mn></msup><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
    <span class="eqn-note">Dividing by m rather than m&#x2212;1. The chapter notes this is the maximum-likelihood estimate, and that with a reasonable number of examples the difference does not matter in practice.</span>
    </div>
    ::: code-eg
    ```
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 1000)        # data with true mu = 5, sigma = 2

    mu = x.mean()                     # 4.904
    sigma2 = ((x - mu) ** 2).mean()   # 3.816  (true value 4)

    def p(v, mu, sigma2):             # the Gaussian density above
        return np.exp(-(v - mu) ** 2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

    p(mu, mu, sigma2)       # 0.204  - highest exactly at the mean
    p(mu + 6, mu, sigma2)   # 0.0018 - three sigmas out, nearly nothing
    ```
    <p class="eqn-note">Parameter estimation and the density itself: fit &mu; and &sigma;&sup2; from a thousand samples, then read probabilities off the curve.</p>
    :::
  - As a side comment
    - These parameters are the maximum likelihood estimation values for μ and σ<sup>2</sup>
    - You can also do 1/(m) or 1/(m-1) doesn't make too much difference
      - Slightly different mathematical problems, but in practice it makes little difference

## Anomaly detection algorithm

- Unlabeled training set of m examples
  - Data = {x<sup>1</sup>, x<sup>2</sup>, ..., x<sup>m </sup>}
    - Each example is an n-dimensional vector (i.e. a feature vector)
    - We have n features!
  - Model P(x) from the data set
    - What are high probability features and low probability features
    - x is a vector
    - So model p(x) as
      - \= p(x**<sub>1</sub>**; μ**<sub>1</sub>** , σ**<sub>1</sub>**<sup>2</sup>) \* p(x**<sub>2</sub>**; μ**<sub>2</sub>** , σ**<sub>2</sub>**<sup>2</sup>) \* ... p(x**<sub>n </sub>**; μ**<sub>n</sub>** , σ**<sub>n</sub>**<sup>2</sup>)
    - Multiply the probabilities of each feature together
      - We model each of the features by assuming each feature is distributed according to a Gaussian distribution
      - p(x**<sub>i</sub>**; μ**<sub>i</sub>** , σ**<sub>i</sub>**<sup>2</sup>)
        - The probability of feature x**<sub>i</sub>** given μ**<sub>i</sub>** and σ**<sub>i</sub>**<sup>2</sup>, using a Gaussian distribution
  - As a side comment
    - Turns out this equation makes an **<span class="term">independence assumption</span>** for the features, although algorithm works if features are independent or not
      - Don't worry too much about this, although if your features are tightly linked you should be able to do some dimensionality reduction anyway!
  - We can write this chain of multiplication more compactly as follows;
    <div class="eqn">
    <math display="block"><mrow><mo>=</mo><munderover><mo>&#x220f;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>p</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>j</mi></msub><mo>;</mo><msub><mi>&#x3bc;</mi><mi>j</mi></msub><mo>,</mo><msubsup><mi>&#x3c3;</mi><mi>j</mi><mn>2</mn></msubsup><mo stretchy="false">)</mo></mrow></math>
    <span class="eqn-note">One Gaussian per feature, multiplied together &#x2014; which assumes the features are independent. This is the Naive Bayes assumption of chapter 05, used here without labels.</span>
    </div>
    - Capital PI (Π) is the product of a set of values
  - The problem of estimating this distribution is sometimes called the problem of **<span class="term">density estimation</span>**

### Algorithm

<div class="eqn pseudocode">
<math display="block"><mrow><mn>1.</mn><mspace width="0.5em"/><mtext>Choose features</mtext><mspace width="0.35em"/><msub><mi>x</mi><mi>i</mi></msub><mspace width="0.35em"/><mtext>that might be indicative of anomalous examples.</mtext></mrow></math>
<math display="block"><mrow><mn>2.</mn><mspace width="0.5em"/><mtext>Fit parameters</mtext><mspace width="0.4em"/><msub><mi>&#x3bc;</mi><mn>1</mn></msub><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msub><mi>&#x3bc;</mi><mi>n</mi></msub><mo>,</mo><msubsup><mi>&#x3c3;</mi><mn>1</mn><mn>2</mn></msubsup><mo>,</mo><mo>&#x2026;</mo><mo>,</mo><msubsup><mi>&#x3c3;</mi><mi>n</mi><mn>2</mn></msubsup></mrow></math>
<math display="block"><mrow><mspace width="1.8em"/><msub><mi>&#x3bc;</mi><mi>j</mi></msub><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msubsup><mi>x</mi><mi>j</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></math>
<math display="block"><mrow><mspace width="1.8em"/><msubsup><mi>&#x3c3;</mi><mi>j</mi><mn>2</mn></msubsup><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mrow><mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>j</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>&#x2212;</mo><msub><mi>&#x3bc;</mi><mi>j</mi></msub><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></math>
<math display="block"><mrow><mn>3.</mn><mspace width="0.5em"/><mtext>Given a new example</mtext><mspace width="0.35em"/><mi>x</mi><mo>,</mo><mspace width="0.35em"/><mtext>compute</mtext><mspace width="0.35em"/><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>:</mo></mrow></math>
<math display="block"><mrow><mspace width="1.8em"/><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><munderover><mo>&#x220f;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>p</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>j</mi></msub><mo>;</mo><msub><mi>&#x3bc;</mi><mi>j</mi></msub><mo>,</mo><msubsup><mi>&#x3c3;</mi><mi>j</mi><mn>2</mn></msubsup><mo stretchy="false">)</mo><mo>=</mo><munderover><mo>&#x220f;</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mfrac><mrow><mn>1</mn></mrow><mrow><msqrt><mn>2</mn><mi>&#x3c0;</mi></msqrt><msub><mi>&#x3c3;</mi><mi>j</mi></msub></mrow></mfrac><mspace width="0.25em"/><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><msup><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>j</mi></msub><mo>&#x2212;</mo><msub><mi>&#x3bc;</mi><mi>j</mi></msub><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow><mrow><mn>2</mn><msup><msub><mi>&#x3c3;</mi><mi>j</mi></msub><mn>2</mn></msup></mrow></mfrac><mo form="postfix">)</mo></mrow></math>
<math display="block"><mrow><mspace width="1.8em"/><mtext>Anomaly if</mtext><mspace width="0.4em"/><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>&lt;</mo><mi>&#x3b5;</mi></mrow></math>
</div>

- **1 - Choose features**
  - Try to come up with features which might help identify something anomalous - may be unusually large or small values
  - More generally, choose features which describe the general properties
  - This is nothing unique to anomaly detection - it's just the idea of building a sensible feature vector
- **2 - Fit parameters**
  - Determine parameters for each of your features μ**<sub>i</sub>** and σ**<sub>i</sub>**<sup>2</sup>
    - Fit is a bit misleading, really should just be "Calculate parameters for 1 to n"
  - So you're calculating standard deviation and mean for each feature
  - You should of course use some vectorized implementation rather than a loop probably
- <strong>3 - compute p(x)</strong>
  - You compute the formula shown (i.e. the formula for the Gaussian probability)
  - If the number is very small, very low chance of it being "normal"

### Anomaly detection example

- **x<sub>1</sub>**
  - Mean is about 5
  - Standard deviation looks to be about 2
- **x<sub>2</sub>**
  - Mean is about 3
  - Standard deviation about 1
- So we have the following system
  <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [12].png"/></figure>
- If we plot the Gaussian for **x<sub>1</sub>** and **x<sub>2</sub>** we get something like this
  <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [13].png"/></figure>
- If you plot the product of these things you get a surface plot like this
  <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [14].png"/></figure>
  - With this surface plot, the height of the surface is the probability - p(x)
  - We can't always do surface plots, but for this example it's quite a nice way to show the probability of a 2D feature vector
- Check if a value is anomalous
  - Set epsilon as some value
  - Say we have two new data points with the values
    - x<sup>1</sup>**<sub>test</sub>**
    - x<sup>2</sup>**<sub>test</sub>**
  - We compute
    - p(x<sup>1</sup>**<sub>test</sub>**) = 0.436 >= epsilon (~40% chance it's normal)
      - Normal
    - p(x<sup>2</sup>**<sub>test</sub>**) = 0.0021 \< epsilon (~0.2% chance it's normal)
      - Anomalous
  - What this is saying is if you look at the surface plot, all values above a certain height are normal, all the values below that threshold are probably anomalous

## Developing and evaluating an anomaly detection system

- Here talk about developing a system for anomaly detection
  - How to evaluate an algorithm
- Previously we spoke about the importance of real-number evaluation
  - Often need to make a lot of choices (e.g. features to use)
    - Easier to evaluate your algorithm if it returns a **<span class="hl-red">single number</span>** to show if changes you made improved or worsened an algorithm's performance
  - To develop an anomaly detection system quickly, would be helpful to have a way to evaluate your algorithm
- Assume we have some labeled data
  - So far we've been treating anomalous detection with unlabeled data
  - If you have labeled data allows evaluation
    - i.e. if you think something is anomalous you can be sure if it is or not
- So, taking our engine example
  - You have some labeled data
    - Data for engines which were non-anomalous -> y = 0
    - Data for engines which were anomalous -> y = 1
  - Training set is the collection of normal examples
    - OK even if we have a few anomalous data examples
  - Next define
    - Cross validation set
    - Test set
    - For both assume you can include a few examples which have anomalous examples
  - Specific example
    - Engines
      - Have 10 000 good engines
        - OK even if a few bad ones are here...
        - LOTS of y = 0
      - 20 flawed engines
        - Typically when y = 1 have 2-50
    - Split into
      - Training set: 6000 good engines (y = 0)
      - CV set: 2000 good engines, 10 anomalous
      - Test set: 2000 good engines, 10 anomalous
      - Ratio is 3:1:1
    - Sometimes we see a different way of splitting
      - Take 6000 good in training
      - Same CV and test set (4000 good in each) different 10 anomalous,
      - Or even 20 anomalous (same ones)
      - This is bad practice - should use different data in CV and test set
  - Algorithm evaluation
    - Take training set { x<sup>1</sup>, x<sup>2</sup>, ..., x<sup>m </sup>}
      - Fit model p(x)
    - On cross validation and test set, test the example x
      - y = 1 if p(x) \< epsilon (anomalous)
      - y = 0 if p(x) >= epsilon (normal)
    - Think of algorithm as trying to predict if something is anomalous
      - But you have a label so can check!
      - Makes it look like a supervised learning algorithm
- What's a good metric to use for evaluation
  - y = 0 is very common
    - So classification accuracy would be bad
  - Compute fraction of true positives/false positive/false negative/true negative
  - Compute precision/recall
  - Compute F1-score
- Earlier, also had **<span class="term">epsilon</span>** (the threshold value)
  - Threshold to show when something is anomalous
  - If you have CV set you can see how varying epsilon affects various evaluation metrics
    - Then pick the value of epsilon which maximizes the score on your CV set
  - Evaluate algorithm using cross validation
  - Do final algorithm evaluation on the test set

## Anomaly detection vs. supervised learning

- If we have labeled data, why not use a supervised learning algorithm?
  - Here we'll try and understand when you should use supervised learning and when anomaly detection would be better

### Anomaly detection

- **<span class="hl-red">Very small number of positive </span>**<strong><span class="hl-red">examples</span></strong>
  - Save positive examples just for CV and test set
  - Consider using an anomaly detection algorithm
  - Not enough data to "learn" positive examples
- <span class="hl-red"><strong>Have a very large number of negative examples</strong></span>
  - Use these negative examples for p(x) fitting
  - Only need negative examples for this
- **<span class="hl-red">Many "types" of anomalies</span>**
  - Hard for an algorithm to learn from positive examples when anomalies may look nothing like one another
    - So anomaly detection doesn't know what they look like, but knows what they *don't* look like
  - When we looked at SPAM email,
    - Many types of SPAM
    - For the spam problem, usually enough positive examples
    - So this is why we usually think of SPAM as supervised learning
- Application and why they're anomaly detection
  - **Fraud detection**
    - Many ways you may do fraud
    - If you're a major on line retailer/very subject to attacks, sometimes might shift to supervised learning
  - **Manufacturing**
    - If you make HUGE volumes maybe have enough positive data -> make supervised
      - Means you make an assumption about the kinds of errors you're going to see
      - It's the unknown unknowns we don't like!
  - **Monitoring machines in data centers**

### Supervised learning

- **<span class="hl-red">Reasonably large number of positive and negative examples</span>**
- Have enough positive examples to give your algorithm the opportunity to see what they look like
  - If you expect anomalies to look anomalous in the same way
- Application
  - Email/SPAM classification
  - Weather prediction
  - Cancer classification

## Choosing features to use

- One of the things which has a huge effect is which features are used
- **Non-Gaussian features**
  - Plot a histogram of data to check it has a Gaussian description - nice sanity check
    - Often still works if data is non-Gaussian
    - Use <span class="hl-green"><strong>plt.hist</strong></span> (matplotlib) to plot histogram
  - Non-Gaussian data might look like this
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [15].png"/></figure>
  - Can play with different transformations of the data to make it look more Gaussian
  - Might take a log transformation of the data
    - i.e. if you have some feature **x<sub>1</sub>**, replace it with log(**x<sub>1</sub>**)
      <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [16].png"/></figure>
      - This looks much more Gaussian
    - Or do log(**x<sub>1</sub>**+c)
      - Play with c to make it look as Gaussian as possible
    - Or do x<sup>1/2</sup>
    - Or do x<sup>1/3</sup>

### Error analysis for anomaly detection

- Good way of coming up with features
- Like supervised learning error analysis procedure
  - Run algorithm on CV set
  - See which one it got wrong
  - Develop new features based on trying to understand *why* the algorithm got those examples wrong
- Example
  - p(x) large for normal, p(x) small for abnormal
  - e.g.
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [17].png"/></figure>
  - Here we have one dimension, and our anomalous value is sort of buried in it (in green - Gaussian superimposed in blue)
    - Look at data - see what went wrong
    - Can looking at that example help develop a new feature (x2) which can help distinguish further anomalous examples
- Example - data center monitoring
  - Features
    - **x<sub>1</sub>** = memory use
    - **x<sub>2</sub>** = number of disk access/sec
    - **x<sub>3</sub>** = CPU load
    - **x<sub>4</sub>** = network traffic
  - We suspect CPU load and network traffic grow linearly with one another
    - If server is serving many users, CPU is high and network is high
    - Fail case is infinite loop, so CPU load grows but network traffic is low
      - New feature - CPU load/network traffic
      - May need to do feature scaling

## Multivariate Gaussian distribution

- Is a slightly different technique which can sometimes catch some anomalies which non-multivariate Gaussian distribution anomaly detection fails to
  - Unlabeled data looks like this
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [18].png"/></figure>
  - Say you can fit a Gaussian distribution to CPU load and memory use
  - Lets say in the test set we have an example which looks like an anomaly (e.g. **x<sub>1</sub>** = 0.4, **x<sub>2</sub>** = 1.5)
    - Looks like most of data lies in a region far away from this example
      - Here memory use is high and CPU load is low (if we plot **x<sub>1 </sub>**vs. **x<sub>2</sub>** our green example looks miles away from the others)
  - Problem is, if we look at each feature individually they may fall within acceptable limits - the issue is we know we shouldn't get those kinds of values **together**
    - But individually, they're both acceptable
      <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [19].png"/></figure>
  - This is because our function makes probability prediction in concentric circles around the means of both
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [20].png"/></figure>
    - Probability of the two red circled examples is basically the same, even though we can clearly see the green one as an outlier
  - Doesn't understand the meaning

### Multivariate Gaussian distribution model

- To get around this we develop the **<span class="term">multivariate Gaussian distribution</span>**
  - Model p(x) all in one go, instead of each feature separately
    - What are the parameters for this new model?
      - μ - which is an *n* dimensional vector (where n is number of features)
      - Σ - which is an \[n x n\] matrix - the **covariance matrix**
- For the sake of completeness, the formula for the multivariate Gaussian distribution is as follows<br/>
  <div class="eqn">
  <math display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo>;</mo><mi>&#x3bc;</mi><mo>,</mo><mi>&#x3a3;</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><msup><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>&#x3c0;</mi><mo stretchy="false">)</mo></mrow><mrow><mfrac><mrow><mi>n</mi></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup><mspace width="0.2em"/><msup><mrow><mo stretchy="false">|</mo><mi>&#x3a3;</mi><mo stretchy="false">|</mo></mrow><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup></mrow></mfrac><mspace width="0.3em"/><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>&#x3a3;</mi><mrow><mo form="prefix">&#x2212;</mo><mn>1</mn></mrow></msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mo form="postfix">)</mo></mrow></math>
  <span class="eqn-note">&#x3a3; is the covariance matrix, an [n&#x00d7;n] matrix, and &#x3bc; is an [n&#x00d7;1] vector. Unlike the product of independent Gaussians, this one can model features that vary together.</span>
  </div>
  - NB don't memorize this - you can always look it up
  - What does this mean?
    - <div class="eqn">
      <math display="block"><mrow><mrow><mo stretchy="false">|</mo><mi>&#x3a3;</mi><mo stretchy="false">|</mo></mrow></mrow></math>
      <span class="eqn-note">The determinant of the covariance matrix, not an absolute value. In NumPy it is <code>np.linalg.det(Sigma)</code>.</span>
      </div>
      \= absolute value of Σ (determinant of sigma)
      - This is a mathematic function of a matrix
      - You can compute it in Python using **<code class="hl-green">np.linalg.det(Sigma)</code>**
- More importantly, what does this p(x) look like?
  - 2D example
    <div class="eqn">
    <math display="block"><mrow><mi>&#x3bc;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow><mspace width="1.4em"/><mi>&#x3a3;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    <span class="eqn-note">With the identity as the covariance matrix the contours are circles, the same spread in both directions and no correlation between them.</span>
    </div>
    - Sigma here is the identity matrix
      <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [24].png"/></figure>
      - p(x) looks like this
        - For inputs of **x<sub>1</sub>** and **x<sub>2</sub>** the height of the surface gives the value of p(x)
  - What happens if we change Sigma?
    <div class="eqn">
    <math display="block"><mrow><mi>&#x3bc;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr></mtable><mo>]</mo></mrow><mspace width="1.4em"/><mi>&#x3a3;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0.6</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0.6</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    </div>
  - So now we change the plot to
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [26].png"/></figure>
    - Now the width of the bump decreases and the height increases
  - If we set sigma to be different values we change the shape of our graph<br/>
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [27].png"/></figure>
  - Using these values we can, therefore, define the shape of this to better fit the data, rather than assuming symmetry in every dimension
- One of the cool things is you can use it to model correlation between data
  - If you start to change the off-diagonal values in the covariance matrix you can control how well the various dimensions correlate<br/>
    <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [28].png"/></figure>
    - So we see here the final example gives a very tall thin distribution, shows a strong positive correlation
    - We can also make the off-diagonal values negative to show a negative correlation
- Hopefully this shows an example of the kinds of distribution you can get by varying sigma
  - We can, of course, also move the mean (μ) which varies the peak of the distribution

## Applying multivariate Gaussian distribution to anomaly detection

- Saw some examples of the kinds of distributions you can model
  - Now let's take those ideas and look at applying them to different anomaly detection algorithms
- As mentioned, multivariate Gaussian modeling uses the following equation;
  <div class="eqn">
  <math display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo>;</mo><mi>&#x3bc;</mi><mo>,</mo><mi>&#x3a3;</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><msup><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>&#x3c0;</mi><mo stretchy="false">)</mo></mrow><mrow><mfrac><mrow><mi>n</mi></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup><mspace width="0.2em"/><msup><mrow><mo stretchy="false">|</mo><mi>&#x3a3;</mi><mo stretchy="false">|</mo></mrow><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup></mrow></mfrac><mspace width="0.3em"/><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>&#x3a3;</mi><mrow><mo form="prefix">&#x2212;</mo><mn>1</mn></mrow></msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mo form="postfix">)</mo></mrow></math>
  </div>
- Which comes with the parameters μ and Σ
  - Where
    - μ - the mean (n-dimensional vector)
    - Σ - covariance matrix (\[nxn\] matrix)
- Parameter fitting/estimation problem
  - If you have a set of examples
    - {x<sup>1</sup>, x<sup>2</sup>, ..., x<sup>m </sup>}
  - The formula for estimating the parameters is
    <div class="eqn">
    <math display="block"><mrow><mi>&#x3bc;</mi><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup></mrow></math>
    </div>
    <div class="eqn">
    <math display="block"><mrow><mi>&#x3a3;</mi><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mi>m</mi></mrow></mfrac><munderover><mo>&#x2211;</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>m</mi></munderover><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><msup><mrow><mo stretchy="false">(</mo><msup><mi>x</mi><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></msup><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mrow></math>
    <span class="eqn-note">The same covariance estimate as in the PCA chapter, but with the mean subtracted first rather than assumed to be zero.</span>
    </div>
  - Using these two formulas you get the parameters

### Anomaly detection algorithm with multivariate Gaussian distribution

- <strong>1)</strong> Fit model - take data set and calculate μ and Σ using the formula above
- <strong>2)</strong> We're next given a new example (**x<sub>test</sub>**) - see below
  <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [32].png"/></figure>
  - For it compute p(x) using the following formula for multivariate distribution
    <div class="eqn">
    <math display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><msup><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>&#x3c0;</mi><mo stretchy="false">)</mo></mrow><mrow><mfrac><mrow><mi>n</mi></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup><mspace width="0.2em"/><msup><mrow><mo stretchy="false">|</mo><mi>&#x3a3;</mi><mo stretchy="false">|</mo></mrow><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup></mrow></mfrac><mspace width="0.3em"/><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>&#x3a3;</mi><mrow><mo form="prefix">&#x2212;</mo><mn>1</mn></mrow></msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mo form="postfix">)</mo></mrow></math>
    </div>
- <strong>3)</strong> Compare the value with ε (threshold probability value)
  - if p(**x<sub>test</sub>**) \< ε --> flag this as an anomaly
  - if p(**x<sub>test</sub>**) >= ε --> this is OK
- If you fit a multivariate Gaussian model to our data we build something like this
  <figure><img alt="" loading="lazy" src="15_Anomaly_Detection_files/Image [34].png"/></figure>
- Which means it's likely to identify the green value as anomalous
- Finally, we should mention how multivariate Gaussian relates to our original simple Gaussian model (where each feature is looked at individually)
  - Original model corresponds to multivariate Gaussian where the Gaussians' contours are axis aligned
  - i.e. the normal Gaussian model is a special case of multivariate Gaussian distribution
    - This can be shown mathematically
    - Has this constraint that the covariance matrix sigma as ZEROs on the non-diagonal values
      <div class="eqn">
      <math display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo>;</mo><mi>&#x3bc;</mi><mo>,</mo><mi>&#x3a3;</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mn>1</mn></mrow><mrow><msup><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>&#x3c0;</mi><mo stretchy="false">)</mo></mrow><mrow><mfrac><mrow><mi>n</mi></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup><mspace width="0.2em"/><msup><mrow><mo stretchy="false">|</mo><mi>&#x3a3;</mi><mo stretchy="false">|</mo></mrow><mrow><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac></mrow></msup></mrow></mfrac><mspace width="0.3em"/><mi>exp</mi><mo form="prefix">(</mo><mo form="prefix">&#x2212;</mo><mfrac><mrow><mn>1</mn></mrow><mrow><mn>2</mn></mrow></mfrac><msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><msup><mi>&#x3a3;</mi><mrow><mo form="prefix">&#x2212;</mo><mn>1</mn></mrow></msup><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>&#x2212;</mo><mi>&#x3bc;</mi><mo stretchy="false">)</mo></mrow><mo form="postfix">)</mo></mrow></math>
      <math display="block"><mrow><mtext>where</mtext><mspace width="0.6em"/><mi>&#x3a3;</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><msubsup><mi>&#x3c3;</mi><mn>1</mn><mn>2</mn></msubsup></mtd><mtd><mn>0</mn></mtd><mtd><mo>&#x2026;</mo></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><msubsup><mi>&#x3c3;</mi><mn>2</mn><mn>2</mn></msubsup></mtd><mtd><mo>&#x2026;</mo></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd><mtd><mo>&#x22f1;</mo></mtd><mtd><mo>&#x22ee;</mo></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mo>&#x2026;</mo></mtd><mtd><msubsup><mi>&#x3c3;</mi><mi>n</mi><mn>2</mn></msubsup></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      <span class="eqn-note">The original model is the multivariate Gaussian with every off-diagonal entry forced to zero &#x2014; which is exactly the assumption that the features are independent. Put the per-feature variances on the diagonal and the two models are identical.</span>
      </div>
      ::: code-eg
      ```
      mu = np.array([2.0, 5.0])
      sigma2 = np.array([1.0, 3.0])     # per-feature variances
      x = np.array([1.5, 4.0])

      p_product = np.prod(p(x, mu, sigma2))   # the original model: 0.0686

      Sigma = np.diag(sigma2)                 # same variances, zero off-diagonals
      d = x - mu
      p_multi = np.exp(-d @ np.linalg.inv(Sigma) @ d / 2) \
                / ((2 * np.pi) ** (2 / 2) * np.sqrt(np.linalg.det(Sigma)))
      p_multi                                 # 0.0686 - identical, to machine precision
      ```
      <p class="eqn-note">The equivalence above, checked numerically: with a diagonal &Sigma; the multivariate Gaussian and the product of per-feature Gaussians give the same number.</p>
      :::
    - If you plug your variance values into the covariance matrix the models are actually identical

### Original model vs. Multivariate Gaussian

### Original Gaussian model

- Probably used more often
- There is a need to manually create features to capture anomalies where **x<sub>1</sub>** and **x<sub>2 </sub>**take unusual combinations of values
  - So **<span class="hl-red">need to make extra features</span>**
  - Might not be obvious what they should be
    - This is always a risk - where you're using your own expectation of a problem to "predict" future anomalies
    - Typically, the things that catch you out aren't going to be the things you thought of
      - If you thought of them they'd probably be avoided in the first place
    - Obviously this is a bigger issue, and one which may or may not be relevant depending on your problem space
- Much **<span class="hl-red">cheaper computationally</span>**
- <span class="hl-red"><strong>Scales much better</strong></span> to very large feature vectors
  - Even if n = 100 000 the original model works fine
- **<span class="hl-red">Works well even with a small training set</span>**
  - e.g. 50, 100
- Because of these factors it's used more often because it really represents an optimized but axis-symmetric specialization of the general model

### Multivariate Gaussian model

- Used less frequently
- **<span class="hl-red">Can capture feature correlation</span>**
  - So no need to create extra values
- **<span class="hl-red">Less computationally efficient</span>**
  - Must compute inverse of matrix which is \[n x n\]
  - So lots of features is bad - makes this calculation very expensive
  - So if n = 100 000 not very good
- **<span class="hl-red">Needs m &gt; n </span>**
  - i.e. number of examples must be greater than number of features
  - If this is not true then we have a singular matrix (non-invertible)
  - So should be used only if m >> n
- If you find the matrix is non-invertible, could be for one of two main reasons
  - m \< n
    - So use original simple model
  - Redundant features (i.e. linearly dependent)
    - i.e. two features that are the same
    - If this is the case you could use PCA or sanity check your data
