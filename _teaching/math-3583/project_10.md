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
Discretize the domain $$\Omega$$ into a grid of $$N_x \times N_y$$ points, where $$N_x$$ and $$N_y$$ are the number of grid points in the $$x$$ and $$y$$ directions, respectively. The grid spacing in the $$x$$ and $$y$$ directions are $$\Delta x = 2/(N_x-1)$$ and $$\Delta y = 1/(N_y-1)$$, respectively. The grid points are indexed by $$i$$ and $$j$$ in the $$x$$ and $$y$$ directions, respectively, as shown in the figure below.
```python
import numpy as np
Nx = 100
Ny = 100
x = np.linspace(0,2,Nx)
y = np.linspace(0,1,Ny)
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
Use the finite difference method to discretize the 2D Laplacian equation and obtain the numerical solution $$u(x,y)$$ at $$x=x_i$$ and $$y=y_j$$ for $$i=1,2,\cdots,N_x-2$$ and $$j=1,2,\cdots,N_y-2$$.
```python
u = np.zeros((Nx,Ny))
u[0,:] = f1(y)
u[Nx-1,:] = f2(y)
u[:,0] = g1(x)
u[:,Ny-1] = g2(x)
for j in range(1,Ny-1):
    for i in range(1,Nx-1):
        # TODO: fill in the finite difference scheme
```

##### Step 5: Plot the Numerical Solution
Plot the numerical solution $$u(x,y)$$.
```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Numerical Solution')
ax.contourf(x,y,u.T)
plt.show()
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

**Reference**: https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2009%20-%20Elliptic%20Equations/903_Poisson%20Equation-Boundary.html
