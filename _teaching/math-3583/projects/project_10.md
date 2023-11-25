---
layout: page
permalink: /teaching/math-3583/project_10/
title: "Project 10: Numerical Solutions of the 2D Elliptic Equation"
---
#### Objective
To numerically solve the 2D Piosson equation using finite difference methods and analyze the results.

#### Problem Description
The 2D Laplacian equation, often appearing in contexts like heat conduction, electrostatics, and fluid dynamics, is given by:

$$
-\Delta u = f(x,y)
$$

where $$\Delta u = \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$$, $$f(x,y)$$ is a given function and when $$f(x,y) = 0$$, it is the Laplacian equation.

- **Domain**: Let's consider a rectangular domain $$\Omega$$ defined by the coordinates $$(x, y)$$ where $$0 \le x \le 1$$ and $$0\le y \le 1$$.

- **Boundary condition**: we use Dirichlet boundary condition for this problem
    1. **Left Edge (x = 0)**: Let's set $$ f_1(y) $$ to be a linear function of $$ y $$.
       $$ f_1(y) = 2\sin(2\pi y) $$
    
    2. **Right Edge (x = 1)**: For $$ f_2(y) $$, let's use a constant function.
       $$ f_2(y) = 2\sin(2\pi y) $$
    
    3. **Bottom Edge (y = 0)**: Let $$ g_1(x) $$ be a sinusoidal function.
       $$ g_1(x) = \sin(2\pi x) $$
    
    4. **Top Edge (y = 1)**: Finally, for $$ g_2(x) $$, we can choose a quadratic function.
       $$ g_2(x) = \sin(2\pi x) $$

##### Step 1: Discretization and initialization
Discretize the domain $$\Omega$$ into a grid of $$(N+1) \times (N+1)$$ points, where $$N_x$$ and $$N_y$$ are the number of grid points in the $$x$$ and $$y$$ directions, respectively. The grid spacing in the $$x$$ and $$y$$ directions are $$\Delta x = 1/N$$ and $$\Delta y = 1/N$$, respectively. The grid points are indexed by $$i$$ and $$j$$ in the $$x$$ and $$y$$ directions, respectively, as shown in the figure below.
```python
import numpy as np
N = 10
x = np.linspace(0,1,N+1) # N+1 points in x direction
y = np.linspace(0,1,N+1)
dx = x[1]-x[0]
dy = y[1]-y[0]
```

##### Step 3: Boundary Condition Function
Define the boundary condition functions $$f_1(y)$$, $$f_2(y)$$, $$g_1(x)$$, and $$g_2(x)$$.
```python
def f1(y):
    return 
def f2(y):
    return 
def g1(x):
    return 
def g2(x):
    return 
```

##### Step 4: Numerical Solution
Use the finite difference method to discretize the 2D Laplacian equation and obtain the numerical solution $$u(x,y)$$ at $$x=x_i$$ and $$y=y_j$$ for $$i=1,2,\cdots,N-1$$ and $$j=1,2,\cdots,N-1$$.

The Poisson Equation is discretized using 
$$\delta_x^2$$ is the central difference approximation of the second derivative in the $$x$$ direction
\begin{equation}\delta_x^2=\frac{1}{h^2}(u_{i+1j}-2u_{ij}+u_{i-1j}), \end{equation}
and $$\delta_y^2$$ is the central difference approximation of the second derivative in the $$y$$ direction
\begin{equation}\delta_y^2=\frac{1}{h^2}(u_{ij+1}-2u_{ij}+u_{ij-1}). \end{equation}
This gives the Poisson Difference Equation,
\begin{equation}-(\delta_x^2u_{ij}+\delta_y^2u_{ij})=f_{ij} \ \ (x_i,y_j) \in \Omega_h, \end{equation}
\begin{equation}u_{ij}=g_{ij} \ \ (x_i,y_j) \in \partial\Omega_h, \end{equation}
where $$u_ij$$ is the numerical approximation of $$U$$ at $$x_i$$ and $$y_j$$.
Expanding the Poisson Difference Equation gives the five-point method,
\begin{equation}-(u_{i-1j}+u_{ij-1}-4u_{ij}+u_{ij+1}+u_{i+1j})=h^2f_{ij} \end{equation}
for $$i=1,...,N-1$$ and $$j=1,...,N-1$$ which can be written
$$\nabla^2_h u_{ij}=f_{ij}$$

The understanding of this five-point method is important to construct the matrix $$A$$ and the right-hand side vector $$b$$ in the linear system 
\begin{equation}
A\textbf{u}=\textbf{b}
\label{eq:linear_system}
\end{equation}
$$\textbf{u}$$ is the vector of unknowns $$u_{ij}$$, $$A$$ is the matrix of coefficients of the unknowns, and $$\textbf{b}$$ is the right-hand side vector.

This can be written as a system of $$(N-1)\times(N-1)$$ equations can be arranged in matrix form
\begin{equation} A\mathbf{u}=\mathbf{r},\end{equation},
where $$A$$ is an $$(N-1)^2\times(N-1)^2$$  matrix made up of the following block tridiagonal structure

$$\left(\begin{array}{ccccccc}
T&I&0&0&.&.&.\\
I&T&I&0&0&.&.\\
.&.&.&.&.&.&.\\
.&.&.&0&I&T&I\\
.&.&.&.&0&I&T\\
\end{array}\right),
$$

where $$I$$ denotes an $$(N-1) \times (N-1)$$ identity matrix and $$T$$ is the tridiagonal matrix of the form:

$$
T=\left(\begin{array}{ccccccc}
-4&1&0&0&.&.&.\\
1&-4&1&0&0&.&.\\
.&.&.&.&.&.&.\\
.&.&.&0&1&-4&1\\
.&.&.&.&0&1&-4\\
\end{array}\right).
$$

```python
N2=(N-1)*(N-1)
A=np.zeros((N2,N2))
## Diagonal            
for i in range (0,N-1):
    for j in range (0,N-1):           
        A[i+(N-1)*j,i+(N-1)*j]=-4

# LOWER DIAGONAL        
for i in range (1,N-1):
    for j in range (0,N-1):           
        A[i+(N-1)*j,i+(N-1)*j-1]=1   
# UPPPER DIAGONAL        
for i in range (0,N-2):
    for j in range (0,N-1):           
        A[i+(N-1)*j,i+(N-1)*j+1]=1   

# LOWER IDENTITY MATRIX
for i in range (0,N-1):
    for j in range (1,N-1):           
        A[i+(N-1)*j,i+(N-1)*(j-1)]=1        
        
        
# UPPER IDENTITY MATRIX
for i in range (0,N-1):
    for j in range (0,N-2):           
        A[i+(N-1)*j,i+(N-1)*(j+1)]=1
```

After we assemble the matrix $$A$$ in Eq. (\ref{eq:linear_system}), we need to construct the right-hand side vector $$b$$ in Eq. (\ref{eq:linear_system}). The right-hand side vector $$b$$ is a vector of length $$(N-1)^2$$, where $$b_i$$ is the right-hand side of the $$i$$-th equation in Eq. (\ref{eq:linear_system}). The right-hand side vector $$b$$ is constructed by evaluating the function $$f(x,y)$$ at the grid points $$(x_i,y_j)$$ for $$i=1,2,\cdots,N-1$$ and $$j=1,2,\cdots,N-1$$.

```python
# Construct the right-hand side vector b 
b = np.zeros(N2)

```

##### Step 5: Plot the Numerical Solution
To solve the system for $$\mathbf{u}$$ invert the matrix $$A$$
\begin{equation} A\mathbf{u}=\mathbf{r},\end{equation}
such that
\begin{equation} \mathbf{u}=A^{-1}\mathbf{r}.\end{equation}
Lastly, as $$\mathbf{u}$$ is in vector it has to be reshaped into grid form to plot.

```python
C=np.dot(np.linalg.inv(A),r-b)
w[1:N,1:N]=C.reshape((N-1,N-1))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d');
# Plot a basic wireframe.
ax.plot_wireframe(X, Y, w,color='r');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('w');
plt.title(r'Numerical Approximation of the Poisson Equation',fontsize=24,y=1.08);
plt.show();
```

##### Step 6: Analyze the Numerical Solution
We only study a simple boundary case, where $$f_1(y) = f_2(y) = g_1(x) = g_2(x) = 0$$. In this case, the exact solution is $$u(x,y) = 0$$. We can compute the error between the numerical solution and the exact solution.
```python
error = np.linalg.norm(u)
print('Error = ', error)
```

#### Deliverables
Have a jupyter-notebook to show your code and results.
Based on the results, prepare a 4-minute presentation to show your results and answer the question.

**Reference**: [Finite Difference Methods for the Poisson Equation](https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2009%20-%20Elliptic%20Equations/903_Poisson%20Equation-Boundary.html)
