---
layout: page
permalink: /teaching/math-3583/project_1/
title: "Project 1: Analysis and Comparison of Gauss-Seidel and SOR Methods for Solving Finite Difference Problems"
---
To understand, implement, and analyze the Gauss-Seidel and SOR iterative methods for solving the discretized 1D Poisson equation using finite difference methods using `python`. The project will involve studying the role of the relaxation parameter $ \omega $ in the SOR method.

1. Generate a Matrix for 1D Finite Difference
For the 1D Poisson equation $$-u''(x) = f(x)$$ over an interval with $$ N+1 $$ points, 
the matrix $$ A $$ and the vector $$ b $$ can be described generally as:
$$
A = \begin{pmatrix}
  2 & -1 & 0 & \cdots & 0 \\
  -1 & 2 & -1 & \cdots & 0 \\
  0 & -1 & 2 & \cdots & 0 \\
  \vdots & \vdots & \vdots & \ddots & \vdots \\
  0 & 0 & 0 & -1 & 2
\end{pmatrix}_{N \times N},
$$
$$
b = \begin{pmatrix}
  h^2 f(x_1) + \alpha \\
  h^2 f(x_2) \\
  h^2 f(x_3) \\
  \vdots \\
  h^2 f(x_{N-1}) \\
  h^2 f(x_N) + \beta
\end{pmatrix}_{N \times 1}
$$
Here, $$ h = \frac{b-a}{N} $$ is the step size, 
$$ \alpha $$ and $$ \beta $$ are the boundary conditions at $$ x = a $$ and $$ x = b $$ respectively, 
and $$ f(x) $$ is the given function for the right-hand side, 
which could be $$ f(x) = \sin (x) $$ or $$ f(x) = \cos (x) $$ for instance.

2. Code the Gauss-Seidel Method
Write code to implement the Gauss-Seidel method to solve $$ Ax = b $$.

3. Modify the Gauss-Seidel code to implement the SOR method. The update formula for SOR is:
$$
x^{(k+1)}_i = (1-\omega) x^{(k)}_i + \omega \left( \frac{b_i - \sum_{j>i} A_{ij} x^{(k)}_j - \sum_{j < i} A_{ij} x^{(k+1)}_j}{A_{ii}} \right)
$$
where $$ \omega $$ is the relaxation parameter.

4. Investigate the effects of different $$ \omega $$ values on the rate of convergence. Typically, $$ \omega $$ is in the range $$ (0, 2) $$.
Perform a convergence analysis for both the Gauss-Seidel and SOR methods on a set of test matrices to understand their rates and conditions for convergence. In this context, investigate the impact of matrix properties like diagonally dominance and symmetry on the convergence behavior. Second, conduct an optimization study to determine the optimal value of $$ \omega $$ in the SOR method that results in the fastest convergence for different types of matrices. Utilize techniques like grid search or golden section search to find this value and validate it with theoretical expectations. Third, compare the computational time and the number of iterations required for both methods to reach a specified error tolerance, considering matrices of varying sizes and complexities.
