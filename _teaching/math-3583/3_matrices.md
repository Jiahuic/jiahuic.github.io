---
layout: page
permalink: /teaching/math-3583/3_matrices/
title: Matrices
---

### **3.1 Matrices**
#### Polynomial Fitting
The least square problem often arises in data fitting scenarios, like when you're trying to find a polynomial that closely approximates a set of data points $$(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$$. The idea is to find coefficients $$a_0, a_1, \ldots, a_m$$ such that the polynomial $$P(x) = a_0 + a_1 x + a_2 x^2 + \ldots + a_m x^m$$ best fits the given data points.

#### Euclidean Distance
The notion of "best fit" is quantified using Euclidean distance. Specifically, you minimize the sum of the squared differences between the predicted $y$ values from your model $P(x)$ and the actual $y$ values:

$$
E = \sum_{i=1}^n \left(P(x_i) - y_i\right)^2
$$

#### Minimizing the Distance
To find the coefficients that minimize $$E$$, you differentiate $$E$$ with respect to each coefficient $$a_j$$ and set it equal to zero. This results in a system of linear equations known as the normal equations.

$$|| Ax - b ||_2^2 = (Ax - b)^T (Ax - b)$$
Expanding this out, we get:
$$(Ax - b)^T (Ax - b) = x^T A^T A x - 2b^T A x + b^T b$$
Now we take the derivative with respect to \( x \):

$$\frac{d}{dx} \left( x^T A^T A x - 2b^T A x + b^T b \right) = 0$$

Differentiating each term, we get:

$$2A^T A x - 2 A^T b = 0$$

Simplifying, we get:
$$A^T A x = A^T b$$

Here, $$A$$ is the matrix formed by evaluating the basis functions (e.g., $$1, x, x^2, \ldots$$) at the given $$x$$ values, $$x$$ is the vector of coefficients $$[a_0, a_1, \ldots, a_m]$$, and $$b$$ is the vector of $$y$$ values.

#### Explicit Formula
The least squares solution can be found explicitly by solving:

$$
x = (A^T A)^{-1} A^T b
$$

### **3.1.1 Echelon Form (Row Echelon Form, REF)**

A matrix is said to be in **echelon form** (or **row echelon form**), if it satisfies the following properties:

1. All zero rows (rows where every entry is zero) are at the bottom of the matrix.
2. The leading entry (or the leftmost nonzero entry) of a nonzero row is strictly to the right of the leading entry of the row just above.
3. The leading entry in any nonzero row is 1.
4. Below and above the leading 1, all entries in that column are zero.

#### Example of a matrix in echelon form:
$$ 
\begin{bmatrix}
1 & 2 & 0 & 5 \\
0 & 0 & 1 & 3 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

### Row Reduced Echelon Form (RREF)

A matrix is said to be in **row reduced echelon form** if it satisfies all the properties of echelon form, plus:

1. The leading entry in any nonzero row is the only nonzero entry in its column.

#### Example of a matrix in row reduced echelon form:
$$ 
\begin{bmatrix}
1 & 0 & 0 & 5 \\
0 & 0 & 1 & 3 \\
0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

Note the difference between the two forms: while the echelon form only requires zeros below the leading 1, the RREF needs zeros both above and below each leading 1.

### Comparison with Examples:

Let's consider the system of linear equations:

1) $x + 2y + z = 9$
2) $2x + 4y - z = 1$
3) $3x + 6y + 2z = 3$

The augmented matrix for this system is:
$$ 
\begin{bmatrix}
1 & 2 & 1 & | & 9 \\
2 & 4 & -1 & | & 1 \\
3 & 6 & 2 & | & 3 \\
\end{bmatrix}
$$

Using elementary row operations, we can transform this matrix into echelon form:
$$ 
\begin{bmatrix}
1 & 2 & 1 & | & 9 \\
0 & 0 & -3 & | & -17 \\
0 & 0 & 0 & | & 0 \\
\end{bmatrix}
$$

Continuing with the operations, we can transform it into row reduced echelon form:
$$ 
\begin{bmatrix}
1 & 2 & 0 & | & 6 \\
0 & 0 & 1 & | & 17/3 \\
0 & 0 & 0 & | & 0 \\
\end{bmatrix}
$$

This tells us that $$x = 6 - 2y$$, $$y$$ is free (can be any real number), and $$z = 17/3$$.

### **3.2.1 Introduction to Gauss Jordan Elimination**

**Gauss-Jordan elimination** is a method used to bring a matrix to its RREF. It involves a series of elementary row operations:

1. **Swapping** two rows.
2. **Multiplying** a row by a nonzero scalar.
3. **Adding** or **subtracting** a multiple of one row to/from another row.

#### Steps to Perform Gauss-Jordan Elimination:

1. Begin with the left-most column of the matrix.
2. Pivot the matrix, if necessary, so that the entry in the top-left corner is non-zero.
3. Scale the top row, if necessary, so that the entry in the top-left corner is 1.
4. Use row operations to ensure that all other entries in the left-most column are 0.
5. Repeat the process for the next row and the next column, working from top to bottom and left to right, until the matrix is in RREF.

#### Example
Given system of equations:

$$
\begin{align*}
2x + 3y &= 1 \\
4x + 9y &= 7
\end{align*}
$$

This can be represented in matrix form $$Ax = b$$ as:

$$
A = \begin{pmatrix}
2 & 3 \\
4 & 9
\end{pmatrix},
x = \begin{pmatrix}
x \\
y
\end{pmatrix},
b = \begin{pmatrix}
1 \\
7
\end{pmatrix}
$$

Step 1: Eliminate $$y$$ from the second equation.

Perform row operations to zero out the $$y$$-term in the second equation. For example, multiply the first equation by $$3$$ and the second equation by $$-1$$ and add them.

$$
\begin{pmatrix}
2 & 3 \\
0 & 3
\end{pmatrix}
$$

Step 2: Solve for $y$ and back-substitute to find $$x$$.

### **3.2.2 LU Decomposition**

LU decomposition represents a matrix $A$ as the product of a lower triangular matrix $$L$$ and an upper triangular matrix $$U$$.

### Algorithm

1. Start with $A$ as your input matrix.
2. Initialize $L$ to be the identity matrix and $U$ to be $$A$$.
3. For $$j$$ from 1 to $$n$$:
   1. For $$i$$ from $$j+1$$ to $$n$$:
      1. $$L_{ij} = U_{ij} / U_{jj}$$
      2. $$U_{i,:} = U_{i,:} - L_{ij} \times U_{j,:}$$


```python
import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.identity(n)
    U = np.copy(A)
    
    for j in range(n):
        for i in range(j+1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, :] -= L[i, j] * U[j, :]
            
    return L, U

# Example usage:
A = np.array([[2, 3], [4, 9]])
L, U = lu_decomposition(A)
print("L:", L)
print("U:", U)
```

### **3.2.3 Gauss-Seidel method**

The Gauss-Seidel method is an iterative numerical technique used for solving a system of linear equations. The method is suitable for large systems of equations, including sparse matrices where most of the elements are zero.

#### Algorithmic Steps

1. **Initialize**: Choose an initial approximation to the true solution.
  
2. **Iterate**: For each $$ i $$ from $$ 1, 2, \ldots, n $$, update $$ x_i $$ using the equation:

$$
x_i^{(k+1)} = \frac{b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)}}{a_{ii}}
$$

3. **Check Convergence**: Compute the error as $$||x^{(k+1)} - x^{(k)}||$$. If this is smaller than a predetermined tolerance $$\epsilon$$, stop the iteration.

4. **Repeat**: If the error is still too large, go back to step 2.

#### Notes on Gauss-Seidel Method

1. **Convergence**: The Gauss-Seidel method is guaranteed to converge if the matrix $$ A $$ is either diagonally dominant or symmetric and positive definite.

2. **Relaxation**: The method can be extended to the Successive Over-Relaxation (SOR) method by introducing a relaxation factor $$ \omega $$.

3. **In-Place Update**: One of the advantages of the Gauss-Seidel method is that it updates the solution in-place, reducing memory requirements.

#### Time Complexity

1. **Single Iteration**: Each iteration involves $$ O(n^2) $$ operations because we need to compute the sums for each $$ x_i $$, and each sum involves $$ n $$ terms.

2. **Convergence**: The number of iterations $$ k $$ required for convergence is problem-dependent. For some problems, $$ k $$ can be quite large, affecting the overall time complexity. Therefore, the worst-case time complexity can be $$ O(kn^2) $$.

3. **Sparse Matrices**: The time complexity for sparse matrices can be reduced significantly. If each row contains $$ c $$ non-zero elements on average, then the time complexity becomes $$ O(kcn) $$.

4. **Stopping Criteria**: The choice of stopping criteria also influences the number of iterations and thus the time complexity.

#### Python Example for Gauss-Seidel Time Complexity

Here's a Python function that calculates the time complexity based on the number of non-zero elements in the matrix and the number of iterations:


```python
import time
import numpy as np

def gauss_seidel_time_complexity(A, b, tol=1.0e-12, max_iter=500):
    start_time = time.time()
    x = np.zeros(len(b))
    iter_count = 0
    for iter_count in range(max_iter):
        x_new = np.copy(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        if np.allclose(x, x_new, atol=tol):
            break
        x = x_new
    end_time = time.time()
    elapsed_time = end_time - start_time
    non_zero_elements = np.count_nonzero(A)
    time_complexity = f"O({iter_count} * {non_zero_elements})"
    
    return x, iter_count, elapsed_time, time_complexity

# Example usage
A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]], dtype=float)
b = np.array([15, 10, 10, 10], dtype=float)
x, iterations, elapsed_time, time_complexity = gauss_seidel_time_complexity(A, b)

print(f"Solution: {x}")
print(f"Iterations: {iterations}")
print(f"Elapsed Time: {elapsed_time} seconds")
print(f"Time Complexity: {time_complexity}")
```

    Solution: [4.99997445 4.99998168 4.99999144 4.99999715]
    Iterations: 8
    Elapsed Time: 0.0017039775848388672 seconds
    Time Complexity: O(8 * 10)

