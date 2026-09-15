---
title: "03: Linear Algebra - Review"
nav_title: "03: Linear algebra"
index_title: "03: Linear Algebra — Review"
origin: "2011"
description: "Matrices, vectors, multiplication, inverse and transpose: the notation the rest of the course leans on."
---

## Matrices - overview

- Rectangular array of numbers written between square brackets
  - 2D array
  - Named as capital letters (A,B,X,Y)
- Dimension of a matrix are \[Rows x Columns\]
  - Start at top left
  - To bottom left
  - To bottom right
  - R<sup>[r x c]</sup> means a matrix which has r rows and c columns
    <div class="eqn">
    <math display="block"><mrow><mi>A</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1402</mn></mtd><mtd><mn>191</mn></mtd></mtr><mtr><mtd><mn>1371</mn></mtd><mtd><mn>821</mn></mtd></mtr><mtr><mtd><mn>949</mn></mtd><mtd><mn>1437</mn></mtd></mtr><mtr><mtd><mn>147</mn></mtd><mtd><mn>1448</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
    </div>
    ::: code-eg
    ```
    import numpy as np

    A = np.array([[1402,  191],
                  [1371,  821],
                  [ 949, 1437],
                  [ 147, 1448]])

    A.shape          # (4, 2)  -> rows, columns
    ```
    :::
    - Is a \[4 x 2\] matrix
- Matrix elements
  - A<sub>(i,j)</sub> = entry in i<sup>th</sup> row and jth column

<div class="eqn">
<math display="block"><mrow><mi>A</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1402</mn></mtd><mtd><mn>191</mn></mtd></mtr><mtr><mtd><mn>1371</mn></mtd><mtd><mn>821</mn></mtd></mtr><mtr><mtd><mn>949</mn></mtd><mtd><mn>1437</mn></mtd></mtr><mtr><mtd><mn>147</mn></mtd><mtd><mn>1448</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<math display="block"><mrow><msub><mi>A</mi><mrow><mn>11</mn></mrow></msub><mo>=</mo><mn>1402</mn><mspace width="1.5em"/><msub><mi>A</mi><mrow><mn>12</mn></mrow></msub><mo>=</mo><mn>191</mn></mrow></math>
<math display="block"><mrow><msub><mi>A</mi><mrow><mn>32</mn></mrow></msub><mo>=</mo><mn>1437</mn><mspace width="1.5em"/><msub><mi>A</mi><mrow><mn>41</mn></mrow></msub><mo>=</mo><mn>147</mn></mrow></math>
<span class="eqn-note">A<sub>ij</sub> is the entry in the i<sup>th</sup> row and j<sup>th</sup> column.</span>
</div>

::: code-eg
```
A[0, 0]          # 1402   the notes' A11
A[0, 1]          # 191    the notes' A12
A[2, 1]          # 1437   the notes' A32
A[3, 0]          # 147    the notes' A41
```
<span class="eqn-note">NumPy indexes from <strong>0</strong>, so the notes' A<sub>32</sub> is <code>A[2, 1]</code> &mdash; subtract one from each index.</span>
:::

- Provides a way to organize, index and access a lot of data

## Vectors - overview

- Is an n by 1 matrix
  - Usually referred to as a lower case letter
  - n rows
  - 1 column
  - e.g.

<div class="eqn">
<math display="block"><mrow><mi>y</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>460</mn></mtd></mtr><mtr><mtd><mn>232</mn></mtd></mtr><mtr><mtd><mn>315</mn></mtd></mtr><mtr><mtd><mn>178</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
</div>

::: code-eg
```
y = np.array([460, 232, 315, 178])
y.shape          # (4,)    a 1-D array, not a column

y_col = y.reshape(4, 1)
y_col.shape      # (4, 1)  an explicit column vector
```
<span class="eqn-note">A 1-D array has no orientation &mdash; it is neither a row nor a column. Reshape when the distinction matters.</span>
:::

- Is a 4 dimensional vector
  - Refer to this as a vector R4
- Vector elements
  - v<sub>i</sub> = i<sup>th </sup>element of the vector
  - Vectors can be 0-indexed (Python, C++) or 1-indexed (maths notation)
  - In math 1-indexed is most common
    - But in machine learning 0-index is useful
  - Normally assume using 1-index vectors, but be aware sometimes these will (explicitly) be 0 index ones

## Matrix manipulation

- <strong><em>Addition</em></strong>
  - Add up elements one at a time
  - Can only add matrices of the *same dimensions*
    - Creates a new matrix of the same dimensions of the ones added

<div class="eqn">
<math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>+</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>4</mn></mtd><mtd><mn>0.5</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>5</mn></mtd><mtd><mn>0.5</mn></mtd></mtr><mtr><mtd><mn>4</mn></mtd><mtd><mn>10</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
</div>

::: code-eg
```
M = np.array([[1, 0], [2, 5], [3, 1]])
N = np.array([[4, 0.5], [2, 5], [0, 1]])

M + N            # [[5, 0.5], [4, 10], [3, 2]]
```
:::

- <strong><em>Multiplication by scalar</em></strong>
  - Scalar = real number
  - Multiply each element by the scalar
  - Generates a matrix of the same size as the original matrix

<div class="eqn">
<math display="block"><mrow><mn>3</mn><mo>&#x00D7;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>3</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>6</mn></mtd><mtd><mn>15</mn></mtd></mtr><mtr><mtd><mn>9</mn></mtd><mtd><mn>3</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
</div>

::: code-eg
```
3 * M            # [[3, 0], [6, 15], [9, 3]]
```
:::

- <strong><em>Division by a scalar</em></strong>
  - Same as multiplying a matrix by 1/4
  - Each element is divided by the scalar
- <strong><em>Combination of operands</em></strong>
  - Evaluate multiplications first

<div class="eqn">
<math display="block"><mrow><mn>3</mn><mo>&#x00D7;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>4</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>+</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>&#x2212;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>3</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>/</mo><mn>3</mn></mrow></math>
</div>

::: code-eg
```
a = np.array([1, 4, 2])
b = np.array([0, 0, 5])
c = np.array([3, 0, 2])

3 * a + b - c / 3    # [2.,  12.,  10.33333333]
```
<span class="eqn-note">Multiplication and division bind tighter than addition, so this evaluates in the order the maths above shows.</span>
:::

- **Matrix by vector multiplication**
  - \[3 x 2\] matrix \* \[2 x 1\] vector
    - New matrix is \[3 x 1\]
      - More generally if \[a x b\] \* \[b x c\]
        - Then new matrix is \[a x c\]
    - How do you do it?
      - Take the two vector numbers and multiply them with the first row of the matrix
        - Then add results together - this number is the first number in the new vector
      - Then multiply second row by vector and add the results together
      - Then multiply final row by vector and add them together

<div class="eqn">
<math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>3</mn></mtd></mtr><mtr><mtd><mn>4</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>&#x00D7;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>16</mn></mtd></mtr><mtr><mtd><mn>4</mn></mtd></mtr><mtr><mtd><mn>7</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<math display="block"><mrow><mn>1</mn><mo>&#x00D7;</mo><mn>1</mn><mo>+</mo><mn>3</mn><mo>&#x00D7;</mo><mn>5</mn><mo>=</mo><mn>16</mn></mrow></math>
<math display="block"><mrow><mn>4</mn><mo>&#x00D7;</mo><mn>1</mn><mo>+</mo><mn>0</mn><mo>&#x00D7;</mo><mn>5</mn><mo>=</mo><mn>4</mn></mrow></math>
<math display="block"><mrow><mn>2</mn><mo>&#x00D7;</mo><mn>1</mn><mo>+</mo><mn>1</mn><mo>&#x00D7;</mo><mn>5</mn><mo>=</mo><mn>7</mn></mrow></math>
<span class="eqn-note">A [3 x 2] matrix times a [2 x 1] vector gives a [3 x 1] vector.</span>
</div>

::: code-eg
```
P = np.array([[1, 3], [4, 0], [2, 1]])   # [3 x 2]
v = np.array([[1], [5]])                 # [2 x 1]

P @ v            # [[16], [4], [7]]
```
<span class="eqn-note"><code>@</code> is matrix multiplication. <code>*</code> multiplies element by element and is <em>not</em> the same thing.</span>
:::

- Detailed explanation
  - A \* x = y
    - A is m x n matrix
    - x is n x 1 matrix
    - n must match between vector and matrix
      - i.e. inner dimensions must match
    - Result is an m-dimensional vector
  - To get y<sub>i</sub> - multiply A's i<sup>th </sup>row with all the elements of vector x and add them up
- Neat trick
  - Say we have a data set with four values
  - Say we also have a hypothesis h<sub>θ</sub>(x) = -40 + 0.25x
    - Create your data as a matrix which can be multiplied by a vector
    - Have the parameters in a vector which your matrix can be multiplied by
  - Means we can do
    - Prediction = Data Matrix \* Parameters
      <div class="eqn">
      <math display="block"><mrow><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mn>40</mn><mo>+</mo><mn>0.25</mn><mi>x</mi></mrow></math>
      <math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2104</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1416</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1534</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>852</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>&#x00D7;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mrow><mo>&#x2212;</mo><mn>40</mn></mrow></mtd></mtr><mtr><mtd><mn>0.25</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mrow><mo>&#x2212;</mo><mn>40</mn><mo>&#x00D7;</mo><mn>1</mn><mo>+</mo><mn>0.25</mn><mo>&#x00D7;</mo><mn>2104</mn></mrow></mtd></mtr><mtr><mtd><mrow><mo>&#x2212;</mo><mn>40</mn><mo>&#x00D7;</mo><mn>1</mn><mo>+</mo><mn>0.25</mn><mo>&#x00D7;</mo><mn>1416</mn></mrow></mtd></mtr><mtr><mtd><mrow><mo>&#x2212;</mo><mn>40</mn><mo>&#x00D7;</mo><mn>1</mn><mo>+</mo><mn>0.25</mn><mo>&#x00D7;</mo><mn>1534</mn></mrow></mtd></mtr><mtr><mtd><mrow><mo>&#x2212;</mo><mn>40</mn><mo>&#x00D7;</mo><mn>1</mn><mo>+</mo><mn>0.25</mn><mo>&#x00D7;</mo><mn>852</mn></mrow></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      <span class="eqn-note">Prediction = data matrix &#x00D7; parameters. A [4 x 2] matrix times a [2 x 1] vector gives the four predictions as a [4 x 1] vector.</span>
      </div>
      ::: code-eg
      ```
      X = np.array([[1, 2104],
                    [1, 1416],
                    [1, 1534],
                    [1,  852]])          # data, with a column of 1s for theta_0
      theta = np.array([[-40], [0.25]])  # parameters

      X @ theta        # [[486.], [314.], [343.5], [173.]]
      ```
      <span class="eqn-note">The column of 1s is what lets &#x3B8;<sub>0</sub> be carried through the multiplication.</span>
      :::
    - Here we add an extra column to the data with 1s - this means our θ<sub>0 </sub>values can be calculated and expressed
- The diagram above shows how this works
  - This can be far more efficient computationally than lots of for loops
  - This is also easier and cleaner to code (assuming you have appropriate libraries to do matrix multiplication)
- <strong><em>Matrix-matrix multiplication</em></strong>
  - General idea
    - Step through the second matrix one column at a time
    - Multiply each column vector from second matrix by the entire first matrix, each time generating a vector
    - The final product is these vectors combined (not added or summed, but literally just put together)
  - Details
    - A x B = C
      - A = \[m x n\]
      - B = \[n x o\]
      - C = \[m x o\]
        - With vector multiplications o = 1
    - Can only multiply matrix where columns in A match rows in B
  - Mechanism
    - Take column 1 of B, treat as a vector
    - Multiply A by that column - generates an \[m x 1\] vector
    - Repeat for each column in B
      - There are o columns in B, so we get o columns in C
  - Summary
    - *<span class="hl-red">The</span><span class="hl-red"> i <sup>th </sup>column of matrix C is obtained by multiplying A with the <em><span class="hl-red">i <sup>th </sup></span></em>column of B</span>*
  - Start with an example
  - A x B

<div class="eqn">
<math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>4</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>3</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd><mtd><mn>2</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
</div>

::: code-eg
```
A2 = np.array([[1, 3, 2], [4, 0, 1]])    # [2 x 3]
B2 = np.array([[1, 3], [0, 1], [5, 2]])  # [3 x 2]

A2 @ B2          # [[11, 10], [9, 14]]   a [2 x 2]
```
:::

- Initially
  - Take matrix A and multiply by the first column vector from B
  - Take the matrix A and multiply by the second column vector from B

<div class="eqn">
<math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>4</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>&#x00D7;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>5</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>11</mn></mtd></mtr><mtr><mtd><mn>9</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>3</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>4</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>&#x00D7;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>3</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>10</mn></mtd></mtr><mtr><mtd><mn>14</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<span class="eqn-note">Each column of the result comes from multiplying A by the corresponding column of B.</span>
</div>

::: code-eg
```
A2 @ B2[:, [0]]  # [[11], [9]]   first column of B
A2 @ B2[:, [1]]  # [[10], [14]]  second column of B
```
<span class="eqn-note">Each column of the product is A times the corresponding column of B, exactly as above.</span>
:::

- 2 x 3 times 3 x 2 gives you a 2 x 2 matrix

## Implementation/use

- House prices, but now we have three hypotheses and the same data set
- To apply all three hypotheses to all data we can do this efficiently using matrix-matrix multiplication
  - Have
    - Data matrix
    - Parameter matrix
  - Example
    - Four houses, where we want to predict the price
    - Three competing hypotheses
    - Because our hypotheses are one variable, to make the matrices match up we make our data (house sizes) vector into a 4x2 matrix by adding an extra column of 1s

<div class="eqn">
<math display="block"><mrow><mn>1.</mn><mspace width="0.4em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mn>40</mn><mo>+</mo><mn>0.25</mn><mi>x</mi></mrow></math>
<math display="block"><mrow><mn>2.</mn><mspace width="0.4em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mn>200</mn><mo>+</mo><mn>0.1</mn><mi>x</mi></mrow></math>
<math display="block"><mrow><mn>3.</mn><mspace width="0.4em"/><msub><mi>h</mi><mi>&#x3B8;</mi></msub><mo form="prefix">(</mo><mi>x</mi><mo form="postfix">)</mo><mo>=</mo><mo form="prefix">&#x2212;</mo><mn>150</mn><mo>+</mo><mn>0.4</mn><mi>x</mi></mrow></math>
<math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2104</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1416</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>1534</mn></mtd></mtr><mtr><mtd><mn>1</mn></mtd><mtd><mn>852</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>&#x00D7;</mo><mrow><mo>[</mo><mtable><mtr><mtd><mrow><mo>&#x2212;</mo><mn>40</mn></mrow></mtd><mtd><mn>200</mn></mtd><mtd><mrow><mo>&#x2212;</mo><mn>150</mn></mrow></mtd></mtr><mtr><mtd><mn>0.25</mn></mtd><mtd><mn>0.1</mn></mtd><mtd><mn>0.4</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>486</mn></mtd><mtd><mn>410</mn></mtd><mtd><mn>692</mn></mtd></mtr><mtr><mtd><mn>314</mn></mtd><mtd><mn>342</mn></mtd><mtd><mn>416</mn></mtd></mtr><mtr><mtd><mn>344</mn></mtd><mtd><mn>353</mn></mtd><mtd><mn>464</mn></mtd></mtr><mtr><mtd><mn>173</mn></mtd><mtd><mn>285</mn></mtd><mtd><mn>191</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<span class="eqn-note">One column of predictions per hypothesis &mdash; twelve predictions from a single matrix multiplication. Values are rounded to whole numbers, as on the original slide.</span>
</div>

::: code-eg
```
thetas = np.array([[-40, 200, -150],
                   [0.25, 0.1,  0.4]])   # one column per hypothesis

np.round(X @ thetas)
# [[486., 410., 692.],
#  [314., 342., 416.],
#  [344., 353., 464.],
#  [173., 285., 191.]]
```
<span class="eqn-note">Twelve predictions from one call &mdash; this is why the vectorized form matters.</span>
:::

- What does this mean
  - Can quickly apply three hypotheses at once, making 12 predictions
  - Lots of good linear algebra libraries to do this kind of thing very efficiently

## Matrix multiplication properties

- Can pack a lot into one operation
  - However, should be careful of how you use those operations
  - Some interesting properties
- **Commutativity**
  - When working with raw numbers/scalars multiplication is commutative
    - 3 \* 5 == 5 \* 3
  - This is not true for matrices
    - A x B != B x A
    - **<span class="hl-red">Matrix multiplication is not commutative</span>**
- **Associativity**
  - 3 x 5 x 2 == 3 x 10 = 15 x 2
    - Associative property
  - **<span class="hl-red">Matrix multiplications is associative</span>**
    - A x (B x C) == (A x B) x C
- **Identity matrix**
  - 1 is the identity for any scalar
    - i.e. 1 x z = z
      - for any real number
  - In matrices we have an identity matrix called *I*
    - Sometimes called *I<sub>{n x n}</sub>*
      <div class="eqn">
      <math display="block"><mrow><msub><mi>I</mi><mrow><mn>2</mn><mo>&#x00D7;</mo><mn>2</mn></mrow></msub><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      <math display="block"><mrow><msub><mi>I</mi><mrow><mn>3</mn><mo>&#x00D7;</mo><mn>3</mn></mrow></msub><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      <math display="block"><mrow><msub><mi>I</mi><mrow><mn>4</mn><mo>&#x00D7;</mo><mn>4</mn></mrow></msub><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
      <span class="eqn-note">A different identity matrix for each set of dimensions: 1s down the diagonal, 0s everywhere else.</span>
      </div>
      ::: code-eg
      ```
      np.eye(2)        # [[1., 0.], [0., 1.]]
      np.eye(3)        # 3 x 3 identity
      np.eye(4)        # 4 x 4 identity

      A2 @ np.eye(3)   # unchanged: [[1., 3., 2.], [4., 0., 1.]]
      ```
      :::
- See some identity matrices above
  - Different identity matrix for each set of dimensions
  - Has
    - 1s along the diagonals
    - 0s everywhere else
  - 1x1 matrix is just "1"
- Has the property that any matrix A which can be multiplied by an identity matrix gives you matrix A back
  - So if A is \[m x n\] then
    - A \* I
      - I = n x n
    - I \* A
      - I = m x m
    - (To make inside dimensions match to allow multiplication)
- Identity matrix dimensions are implicit
- Remember that matrices are not commutative AB != BA
  - Except when B is the identity matrix
  - Then AB == BA

## Inverse and transpose operations

- **Matrix inverse**
  - How does the concept of "the inverse" relate to real numbers?
    - 1 = "identity element" (as mentioned above)
      - Each number has an inverse
        - This is the number you multiply a number by to get the identity element
        - i.e. if you have x, x \* 1/x = 1
    - e.g. given the number 3
      - 3 \* 3<sup>-1</sup> = 1 (the identity number/matrix)
    - In the space of real numbers **<span class="hl-red">not everything has an inverse</span>**
      - e.g. 0 does not have an inverse
  - What is the inverse of a matrix
    - If A is an m x m matrix, then A inverse = A<sup>-1</sup>
    - So A\*A<sup>-1</sup> = *I*
    - Only matrices which are m x m have inverses
      - Square matrices only!
  - Example
    - 2 x 2 matrix
      <div class="eqn">
      <math display="block"><mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd><mtd><mn>16</mn></mtd></mtr></mtable><mo>]</mo></mrow><mrow><mo>[</mo><mtable><mtr><mtd><mn>0.4</mn></mtd><mtd><mrow><mo>&#x2212;</mo><mn>0.1</mn></mrow></mtd></mtr><mtr><mtd><mrow><mo>&#x2212;</mo><mn>0.05</mn></mrow></mtd><mtd><mn>0.075</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>1</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>=</mo><msub><mi>I</mi><mrow><mn>2</mn><mo>&#x00D7;</mo><mn>2</mn></mrow></msub></mrow></math>
      <span class="eqn-note">A &#x00D7; A<sup>&#x2212;1</sup> = I</span>
      </div>
      ::: code-eg
      ```
      C = np.array([[3, 4], [2, 16]])
      C_inv = np.linalg.inv(C)

      np.round(C_inv, 3)     # [[ 0.4  , -0.1  ], [-0.05 ,  0.075]]
      np.round(C @ C_inv)    # [[1., 0.], [0., 1.]]  -> the identity
      ```
      <span class="eqn-note">In practice prefer <code>np.linalg.solve(C, b)</code> to <code>inv(C) @ b</code> &mdash; it is faster and numerically better behaved. A singular matrix raises <code>LinAlgError</code>.</span>
      :::
    - How did you find the inverse
      - Turns out that you can sometimes do it by hand, although this is very hard
      - Numerical software for computing a matrix's inverse
        - Lots of open source libraries
  - If A is all zeros then there is no inverse matrix
    - Some others don't, intuition should be matrices that don't have an inverse are a singular matrix or a degenerate matrix (i.e. when it's too close to 0)
    - So if all the values of a matrix reach zero, this can be described as reaching singularity
- **Matrix transpose**
  - Have matrix A (which is \[n x m\]) how do you change it to become \[m x n\] while keeping the same values
    - i.e. swap rows and columns!
  - How you do it;
    - Take first row of A - becomes 1st column of A<sup><em>T</em></sup>
    - Second row of A - becomes 2nd column...
  - A is an m x n matrix
    - B is a transpose of A
    - Then B is an n x m matrix
    - A<sub>(i,j)</sub> = B<sub>(j,i)</sub>

<div class="eqn">
<math display="block"><mrow><mi>A</mi><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd><mtd><mn>0</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>5</mn></mtd><mtd><mn>9</mn></mtd></mtr></mtable><mo>]</mo></mrow><mspace width="2em"/><msup><mi>A</mi><mi>T</mi></msup><mo>=</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>3</mn></mtd></mtr><mtr><mtd><mn>2</mn></mtd><mtd><mn>5</mn></mtd></mtr><mtr><mtd><mn>0</mn></mtd><mtd><mn>9</mn></mtd></mtr></mtable><mo>]</mo></mrow></mrow></math>
<span class="eqn-note">The first row of A becomes the first column of A<sup>T</sup>, and so on.</span>
</div>

::: code-eg
```
D = np.array([[1, 2, 0], [3, 5, 9]])

D.T              # [[1, 3], [2, 5], [0, 9]]
D.shape, D.T.shape   # ((2, 3), (3, 2))
```
:::
