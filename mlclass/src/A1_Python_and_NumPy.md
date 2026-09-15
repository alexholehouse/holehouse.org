---
title: "Appendix 1: Python and NumPy"
nav_title: "Appendix 1: Python and NumPy"
origin: "2026"
description: "The NumPy primer behind the code examples: arrays, shapes, broadcasting and vectorization."
---

## A note on this chapter

- The original course had a programming-environment tutorial in the chapter 05 slot, and the original notes never covered it
  - This page stood empty for years, pointing readers at an online tutorial instead
- Every code example in this rewrite is Python with NumPy, so this appendix is a short primer on exactly the idioms those examples use - chapter 05 itself now covers probability and Bayes' rule
  - Nothing here is specific to machine learning - it is the vocabulary the rest of the notes' code is written in
  - If you already use NumPy, skip this appendix entirely
- Python is very easy to pick up, and the [official NumPy beginner's guide](https://numpy.org/doc/stable/user/absolute_beginners.html) is a good fuller tutorial

## Arrays and shapes

- The one data structure that matters is the NumPy array
  - The matrices and vectors of chapter 03, as objects
- Every array has a `.shape` - the \[rows x columns\] dimensions the maths notation writes in brackets

```
import numpy as np

A = np.array([[1402,  191],
              [1371,  821],
              [ 949, 1437],
              [ 147, 1448]])

A.shape       # (4, 2)  -> a [4 x 2] matrix

v = np.array([460, 232, 315, 178])
v.shape       # (4,)    -> a 1-D array of 4 numbers
v.reshape(4, 1).shape  # (4, 1)  -> an explicit column vector
```

<p class="eqn-note">A 1-D array has no orientation &mdash; it is neither a row nor a column until you reshape it. For most of the code in these notes the 1-D form is fine.</p>

- Indexing starts at 0, not 1
  - The maths in these notes indexes from 1 (x<sub>1</sub> is the first feature), so code is always one off from the notation: x<sub>1</sub> lives at `x[0]`
  - Chapter 03 flags this every time it matters

## The operations the notes lean on

- Three symbols do nearly all the work
  - `@` — matrix multiplication (chapter 03's matrix-matrix and matrix-vector products)
  - `.T` — transpose
  - `*` — element-wise multiplication, which is *not* matrix multiplication

```
X = np.array([[1, 2104],
              [1, 1416],
              [1, 1534],
              [1,  852]])          # data with a column of 1s, as in chapter 03
theta = np.array([-40, 0.25])

X @ theta     # [486., 314., 343.5, 173.]  -> one prediction per row
X.T.shape     # (2, 4)   -> the transpose
```

- Broadcasting
  - Operations between an array and a scalar (or a smaller array) apply element-wise, stretching the smaller one to fit
  - This is how "subtract the mean from every example" is one line, with no loop

```
X[:, 1] - X[:, 1].mean()   # mean normalization of a feature, in one line
(X[:, 1] > 1500)            # [ True, False,  True, False]  -> boolean mask
```

## Vectorization

- The habit the rest of these notes assume: replace loops over examples with one array expression
  - `X @ theta` above computes four hypotheses at once - chapter 03 builds this idea up properly
  - NumPy's operations run in compiled code, so the vectorized form is both shorter and much faster than a Python loop
- Rule of thumb: if the maths is written as a sum over examples, the code is usually a matrix multiply

## The toolbox used in later chapters

- `np.linalg` - the linear algebra routines
  - `np.linalg.pinv` - pseudo-inverse, used for the normal equation (chapter 04)
  - `np.linalg.svd` - singular value decomposition, used for PCA (chapter 14)
  - `np.linalg.det` - determinant, used for the multivariate Gaussian (chapter 15)
  - `np.linalg.solve` - preferred over inverting a matrix when solving a linear system
- `scipy.optimize.minimize` - the general-purpose cost function minimizer
  - Stands in wherever the course reaches for an advanced optimization routine (chapters 06, 07 and 09)
  - Give it a function returning the cost and gradient, an initial θ, and it does the rest
- `matplotlib` - plotting, e.g. `plt.hist` for the sanity-check histograms of chapter 15

```
from scipy.optimize import minimize

def cost_function(theta):        # returns (cost, gradient)
    jval = ((theta - 5) ** 2).sum()
    gradient = 2 * (theta - 5)
    return jval, gradient

res = minimize(cost_function, np.zeros(2), jac=True)
res.x         # [5., 5.]  -> the minimum, found for us
```

<p class="eqn-note">The same worked example chapter 06 uses to introduce advanced optimization &mdash; J(θ) = (θ<sub>1</sub> &minus; 5)&sup2; + (θ<sub>2</sub> &minus; 5)&sup2;, minimised at (5, 5).</p>
