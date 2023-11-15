---
layout: page
permalink: /teaching/math-3583/projects/
title: Projects
---

### Project 1: Analysis and Comparison of Gauss-Seidel and SOR Methods for Solving Finite Difference Problems
To understand, implement, and analyze the Gauss-Seidel and SOR iterative methods for solving the discretized 1D Poisson equation using finite difference methods using `python`. The project will involve studying the role of the relaxation parameter $ \omega $ in the SOR method.

1. Generate a Matrix for 1D Finite Difference
For the 1D Poisson equation $-u''(x) = f(x)$ over an interval with $ N+1 $ points, 
the matrix $ A $ and the vector $ b $ can be described generally as:
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
Here, $ h = \frac{b-a}{N} $ is the step size, 
$ \alpha $ and $ \beta $ are the boundary conditions at $ x = a $ and $ x = b $ respectively, 
and $ f(x) $ is the given function for the right-hand side, 
which could be $ f(x) = \sin (x) $ or $ f(x) = \cos (x) $ for instance.

2. Code the Gauss-Seidel Method
Write code to implement the Gauss-Seidel method to solve $ Ax = b $.

3. Modify the Gauss-Seidel code to implement the SOR method. The update formula for SOR is:
$$
x^{(k+1)}_i = (1-\omega) x^{(k)}_i + \omega \left( \frac{b_i - \sum_{j>i} A_{ij} x^{(k)}_j - \sum_{j < i} A_{ij} x^{(k+1)}_j}{A_{ii}} \right)
$$
where $ \omega $ is the relaxation parameter.

4. Investigate the effects of different $ \omega $ values on the rate of convergence. Typically, $ \omega $ is in the range $ (0, 2) $.
Perform a convergence analysis for both the Gauss-Seidel and SOR methods on a set of test matrices to understand their rates and conditions for convergence. In this context, investigate the impact of matrix properties like diagonally dominance and symmetry on the convergence behavior. Second, conduct an optimization study to determine the optimal value of $\omega$ in the SOR method that results in the fastest convergence for different types of matrices. Utilize techniques like grid search or golden section search to find this value and validate it with theoretical expectations. Third, compare the computational time and the number of iterations required for both methods to reach a specified error tolerance, considering matrices of varying sizes and complexities. 


### Project 2: Dimensional Analysis of the Linearized KdV Equation
The problem involves finding the velocity $u (x, t) $ of the waves in shallow water.
It is assumed to satisfy the linearized KdV equation,
which is 
$$
\frac{\partial u}{\partial t} + K \frac{\partial^3 u}{\partial x^3} = 0 \text{,   for } -\infty < x < \infty \text{, } 0< t,
$$
where $K$ is a positive constant.
The boundary conditions are $u=0$ as $ x\rightarrow \pm \infty $, and instead of an initial condition, assume the solution satisfies
$$
\int_{-\infty}^{\infty} u d x = \gamma, \forall t > 0.
$$
1. What are the dimensions of $K$ and $ \gamma $?
2. What four physical quantities does $u(x,t)$ depend on? With this, write down the equivalent version of $ [[ u ]] = [[ x^at^bD^c (u_0)^d ]] $ for this problem, and use the exponent $d$ for $ \gamma $.
3. The key observation is that if you increase $\gamma$ by a factor $\alpha$, then the solution $u$ increases by this same factor. To prove this, let $U(x,t)$ by the solution for a given $\gamma$. Setting $u=\alpha U(x,t)$, what problem does $u$ satisfy? Assuming the solution is unique, explain why this proves the key observation.
4. Explain why the key observation made in part 3. means that $d=1$. With this, find the general dimensionally reduced form of the solution. As usual, let $\eta$ denote the similarity variable.
5. Using the result from part 4., transform the KdV equation into an ordinary differential equation. How do the boundary conditions transform?
6. Show that the third order differential equation in part 5. can be simplified to one that is second order. You can assume $F^{\prime\prime}\rightarrow 0$ and $\eta F \rightarrow 0$ as $\eta \rightarrow \pm \infty$. As a hint, you might want to look for the expression $(\eta F)^\prime$ in your equation.
7. Rewrite the initial condition in terms of $F$.
8. The solution of the problem is $F(\eta) = \text{Ai} (\eta)$, where Ai is known as the Airy function. This function plays a central role in wave problems, both in fluids, electromagnetics, and quantum mechanics. Read [the history of Airy function](https://www.worldscientific.com/doi/suppl/10.1142/p345/suppl_file/p345_chap01.pdf).

### Project 3: Introduction to Singular Perturbations
To introduce, understand, and apply perturbation methods in problems with singular perturbations, thereby gaining insights into the behavior of solutions in the presence of small parameters.
Based on **the section 2.4 Introduction to Singular Perturbations** of the text book, to study this problem.

Singular perturbation problems often arise in various fields of science and engineering where the behavior of a system changes dramatically as a small parameter approaches zero.

1. Introduction to Singular Perturbations
Understand the concept of singular perturbations and identify scenarios where singular perturbations are applicable. Common examples include boundary layer problems in fluid mechanics and the behavior of electrical circuits with small inductances or capacitances.

2. Mathematical Formulation
Identify a simple singular perturbation problem that can be analytically solved. Typical problems may be of the form:
$$
\epsilon y'' + A y' + By = f(x), \quad y(a) = \alpha, \, y(b) = \beta
$$
where $ \epsilon $ is a small parameter, and $ A, B, \alpha, \beta $ are constants. 

3. In the third step of the project, students will focus on learning and applying two advanced mathematical techniques, namely regular perturbation and boundary layer methods, to address the problem identified in Step 2. For the regular perturbation aspect, students will derive the series expansion of the solution $ y(x) $ as a power series in $ \epsilon $. This involves obtaining approximations for both leading-order and next-order terms to understand how the solution behaves as $ \epsilon $ changes. On the other hand, for the boundary layer method, the first task is to identify the boundary layer in the given problem, followed by deriving inner and outer solutions within this layer. Subsequently, these inner and outer solutions are matched to synthesize an approximate global solution for the problem. Through these exercises, students will gain hands-on experience in employing regular perturbation and boundary layer methods to solve complex mathematical problems, thereby deepening their understanding of these techniques.

4. In the fourth step of the project, students will conduct an in-depth error analysis to compare the perturbative solution with the numerical solution. A focal point of this analysis will be to discuss the range of $ \epsilon $ values for which the perturbative approach yields accurate results. The expected outcomes of this step include gaining a nuanced understanding of the concept of singular perturbations and their relevance in the problem at hand. Students will also acquire practical experience in applying perturbation methods to tackle a singular perturbation problem, along with insights into the accuracy and limitations of these methods. The evaluation criteria will focus on the quality of mathematical derivations, the accuracy and efficiency of the numerical implementations, the thoroughness of the error analysis, and an overarching understanding of the role and utility of perturbation methods in singular perturbations. Through this comprehensive task, students will cultivate both theoretical and practical skills in dealing with singular perturbation problems.

### Project 4: Investigation of Perturbation Methods for Boundary Layer Problems
To introduce and explore perturbation methods specifically tailored for boundary layer problems commonly encountered in fluid mechanics, heat transfer, and other areas of applied mathematics.
Boundary layer problems typically involve phenomena concentrated in a thin layer near a boundary, requiring specialized methods for accurate approximation.
Read **the section 2.5 Introduction to Boundary Layers** to study this problem.

1. Understanding Boundary Layers
Study the general features of boundary layer problems. 
Understand the concept of boundary layers and identify how and why they form in specific equations or systems, such as the Navier-Stokes equations for fluid flow.

2. Mathematical Formulation
Identify a well-defined boundary layer problem. It could be in the context of fluid flow, heat transfer, or any other suitable domain. Formulate the governing equations and boundary conditions.
Example problem:
$$
\epsilon y'' + 2y' + 2y = 0, \quad \text{for } 0 < x < 1, \quad y(0)=0, \quad y(1) = 1
$$
where $ \epsilon $ is a small parameter representing the thickness of the boundary layer.

3. In the third step of the project, you are tasked with delving into perturbation methods, specifically focusing on regular perturbation and boundary layer methods, to solve the problem outlined in Step 2. For the regular perturbation component, you will formulate a series expansion for $ y(x) $ in terms of $ \epsilon $, followed by deriving the leading and next-order terms to understand the solution's behavior as $ \epsilon $ varies. Concurrently, using boundary layer methods, you will identify the location and width of the problem's boundary layer. By applying suitable variable transformations, they will derive both inner and outer solutions. These solutions will then be matched to generate a uniform approximation for $ y(x) $. This step aims to provide a hands-on understanding of how to apply these specialized techniques to solve complex problems effectively.
Should go through **the section 2.5 Introduction to Boundary Layers**.

4. In the fourth phase of the project, you will execute an in-depth error analysis to scrutinize the accuracy of the perturbative solutions, 
with a particular focus on studying how this error evolves with changes in $ \epsilon $. 
This analytical endeavor aims to achieve several educational outcomes: 
a comprehensive grasp of boundary layer phenomena and their mathematical repercussions, 
proficiency in applying perturbation techniques to boundary layer challenges, and insights into the limitations and effective applications of these methods. 
The evaluation of this work will hinge on multiple criteria, 
including the rigor and quality of mathematical derivations, 
the accuracy of numerical simulations, and the depth of the conducted error analysis. 
Through this multifaceted exercise, you are expected to deepen both their theoretical understanding and practical skills in perturbation methods, particularly within the context of boundary layer problems.


### Project 5: Analysis of Kinetic Equations with Null-Cline and Phase Plane Methods
To study a system of kinetic equations, understand its dynamics, and analyze the behavior of its solutions in terms of sinks, sources, and spirals in the phase plane.

1. Consider a system of first-order ordinary differential equations that describe a kinetic system. A typical example could be a simple predator-prey model, which can be represented by:
$$
dx/dt = \alpha x - \beta xy
dy/dt = \detal xy - \gamma y
$$
where $x$ and $y$ represent the populations of prey and predator, respectively, and $\alpha$, $\beta$, $\gamma$, $\delta$ are constants.

2. Engage in null-cline and phase plane analysis to explore the dynamics of the system in question. Initially, you will plot the null-clines of the system, identifying their intersections as these represent the steady-state solutions. Following this, phase plane analysis will be employed to visualize the trajectories of these solutions. Through these exercises, you will gain both a qualitative and quantitative understanding of the system's behavior, specifically focusing on its steady states and their stability.

3. Focus on the stability analysis of the identified steady states. This involves computing the Jacobian matrix at each of these steady states and subsequently using it to categorize the nature and stability of each state—whether it is a node, saddle, spiral, or some other type. By accomplishing this, you will deepen their understanding of stability concepts and will be better equipped to predict the long-term behavior of the system under study.

4. Classifying each steady state as either a sink, a source, or a spiral based on the eigenvalues of the Jacobian matrix.
**Here, sink, source, and spiral are new concepts.** You should do research about the definition of each one.
The expected outcomes for this task include honing the ability to apply null-cline and phase plane methods to kinetic equations and deepening the understanding of how to classify and interpret the stability and behavior of steady states. 
Evaluation will focus on the rigor and correctness of the mathematical analysis, 
the quality and accuracy of numerical simulations, 
and the ability to meaningfully interpret and explain the roles of sinks, sources, and spirals within the kinetic system. 
Through this step, students will enhance both their theoretical and practical skills in understanding complex dynamical systems.

### Project 6: Fourier Transform for Solving Diffusion Equations
Go through **the section 4.5 Fourier Transform** of the test book, have a report of the introduction of Fourier Transform for solving diffusion equations.
$$
u_t = Du_{xx}, \quad \text{for } -\infty < x < \infty, \quad 0 < t,
$$
with the initial condition
$$
u(x, 0) = f(x).
$$
It is assumed that $f(x)$ is piecewise continuous with $\lim_{x\rightarrow \pm\infty}f(x) = 0$.
Students choose this project should be able to understand the convolution theorem and how to solve the diffusion equation.

### Project 7: 2D Random Walk Simulation and Its Continuous Approximation
To implement a 2D random walk simulation and study its behavior for large time steps, comparing the results to the continuous form of diffusion.
Random walks are elementary stochastic processes that serve as discrete analogs to continuous diffusion phenomena. In two dimensions, a random walk can be represented as a sequence of steps in the $x$ and $y$ directions.

1. Introduction to Random Walks
Study the basic principles behind random walks, focusing on the 2D case. Understand the rules governing the steps and how they relate to diffusion phenomena.

2. Program the 2D Random Walk
- Implement a simulation for a 2D random walk where a "walker" can move one step in either the $x$ or $y$ direction at each time step.
- Initialize the walker at the origin and run the simulation for $ N $ time steps, where $ N $ is large (e.g., $ N = 10^5 $).

Languages and libraries you can use:
- Python with libraries like NumPy and Matplotlib for visualization

3. Data Analysis

- Compute statistical measures like mean squared displacement (MSD) to quantify how far the walker typically moves from the origin.
- Generate plots of the walker's position over time.

4. Compare to Continuous Form
- Compare the results of the 2D random walk simulation to the 2D diffusion equation, which is the continuous form of a random walk.

The 2D diffusion equation is given by:
$$
\frac{\partial u}{\partial t} = D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

- Discuss how well the random walk approximates continuous diffusion for large $ N $.

5. Report Writing
Compile all the steps, results, and discussions into a well-organized report.

#### Expected Outcomes
- Hands-on experience with simulating stochastic processes
- Understanding of how a discrete process like a random walk can approximate a continuous process like diffusion
- Skill in programming and data analysis

#### Evaluation
- Correctness and efficiency of the random walk simulation
- Quality of data analysis and comparison to continuous form
- Overall understanding of random walks and their relationship to diffusion phenomena

8. [Project 8: Numerical Solutions of the 1D Diffusion Equation](/teaching/math-3583/project_8/)
9. [Project 9: Numerical Solution of the 1D Hyperbolic Equation](/teaching/math-3583/project_9/)
10. [Project 10: Numerical Solutions of the 2D Elliptic Equation](/teaching/math-3583/project_10/)
