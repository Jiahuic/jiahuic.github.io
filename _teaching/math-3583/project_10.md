---
layout: page
permalink: /teaching/math-3583/project_10/
title: "Project 10: Numerical Solutions of the 2D Laplacian Equation"
---


#### Objective

To numerically solve the 2D Laplacian equation using finite difference methods and analyze the results.

#### Background

The 2D Laplacian equation, often appearing in contexts like heat conduction, electrostatics, and fluid dynamics, is given by:

$$
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0
$$

#### Project Tasks

##### Step 1: Mathematical Formulation

- Review the mathematical foundation and physical interpretations of the 2D Laplacian equation.
- Discuss various boundary conditions, such as Dirichlet and Neumann, and their physical significance.

##### Step 2: Introduction to Finite Difference Methods

- Study the basics of finite difference methods for solving PDEs.
- Describe how these methods can be applied to the 2D Laplacian equation.

##### Step 3: Numerical Implementation

- Implement the finite difference method to solve the 2D Laplacian equation on a grid.
- Ensure the program can handle different types of boundary conditions.

Languages and tools you may use:
- Python with NumPy and Matplotlib for computational tasks and visualization
- MATLAB for both computation and plotting

##### Step 4: Validation and Testing

- Validate the numerical solution against analytical solutions for simple boundary conditions.
- Test the program for various boundary conditions and grid sizes.

##### Step 5: Error Analysis

- Perform an error analysis comparing the numerical and analytical solutions.
- Study how the error changes with grid spacing and discuss the implications.

##### Step 6: Case Study

- **Domain**: Let's consider a rectangular domain $$\Omega$$ defined by the coordinates $$(x, y)$$ where $$0 \le x \le 2$$ and $$0\le y \le 1$$.

- **Boundary condition**: we use Dirichlet boundary condition for this problem
    1. **Left Edge (x = 0)**: Let's set $$ f_1(y) $$ to be a linear function of $$ y $$.
       $$ f_1(y) = 0 $$
    
    2. **Right Edge (x = 2)**: For $$ f_2(y) $$, let's use a constant function.
       $$ f_2(y) = 0 $$
    
    3. **Bottom Edge (y = 0)**: Let $$ g_1(x) $$ be a sinusoidal function.
       $$ g_1(x) = 0 $$
    
    4. **Top Edge (y = 1)**: Finally, for $$ g_2(x) $$, we can choose a quadratic function.
       $$ g_2(x) = 0 $$

##### Step 7: Reporting and Documentation

Compile all the steps, methodologies, results, and discussions into a structured report.

#### Expected Outcomes

- Proficiency in using finite difference methods for solving PDEs.
- Understanding of the 2D Laplacian equation and its applications.
- Ability to perform error analysis in numerical solutions.

#### Tools Required

- Programming language for numerical simulation (Python, MATLAB)
- Data visualization tools (Matplotlib for Python, built-in functions in MATLAB)

#### Evaluation Criteria

- Quality and efficiency of the numerical implementation
- Rigor in validation and error analysis
- Depth of understanding of the 2D Laplacian equation and finite difference methods
