---
layout: page
permalink: /teaching/math-3583/project_8/
title: "Project 8: Numerical Solutions of the 1D Diffusion Equation"
---
#### Objective
To numerically solve the 1D diffusion equation using different numerical methods and to analyze the accuracy and stability of these methods.

#### Problem Description
The 1D diffusion equation is a fundamental equation in various scientific domains:

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}
$$

where $$u(x, t)$$ is the dependent variable, and $$D=1$$ is the diffusion coefficient.

To make this equation well-posed, we need to specify initial and boundary conditions. For instance, we can consider the following initial and boundary conditions:

$$
u(x, 0) = \sin(\pi x), x\in[0,1], \quad u(0, t) = u(1, t) = 0, t>0
$$

##### Step 1: Discretization
Discretize the domain $$0\le x \le1$$ into $$N$$ equally spaced points, i.e., $$x_i = i\Delta x$$, where $$\Delta x = 1/N$$. Denote $$u_i(t) = u(x_i, t)$$.
The time domain $$t\ge0$$ is also discretized into $$M$$ equally spaced points, i.e., $$t_j = j\Delta t$$, where $$\Delta t = T/M$$ and $$T$$ is the final time.
```python
import numpy as np
N = 10
M = 1000
x = np.linspace(0,1,N+1)
t = np.linspace(0,10,M+1)
# initialize u 
u = np.zeros((M+1,N+1))
```

##### Step 2: Initial Condition
Compute the initial condition $$u(x,0)=\sin(\pi x)$$ at $$t=0$$ in Python.
```python
u0 = np.sin(np.pi*x)
u[0,:] = u0
```

##### Step 3: Boundary Condition
Compute the boundary condition $$u(0,t)=u(1,t)=0$$ at $$t=t_j$$ for $$j=1,2,\cdots,M$$ in Python.
```python
u[:,0] = 0
u[:,-1] = 0
```

##### Step 4: Numerical Solution
Use the forward Euler scheme to discretize the differentiaon equation and obtain the numerical solution $$u(x,t)$$ at $$t=t_j$$ for $$j=1,2,\cdots,M$$.
Written out:

$$
\frac{u_i^{j+1}-u_i^j}{\Delta t} = D \frac{u_{i+1}^j-2u_i^j+u_{i-1}^j}{\Delta x^2}
$$

```python
for j in range(1,M+1):
    for i in range(1,N):
        u[j,i] = u[j-1,i] + (u[j-1,i+1]-2*u[j-1,i]+u[j-1,i-1])*dt/dx**2
```

##### Step 5: Plot the Numerical Solution
Plot the numerical solution $$u(x,t)$$ at $$t=t_j$$ for $$j=1,2,\cdots,M$$.
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(x,u[0,:],label='t=0')
plt.plot(x,u[10,:],label='t=1')
plt.plot(x,u[20,:],label='t=2')
plt.plot(x,u[30,:],label='t=3')
plt.plot(x,u[40,:],label='t=4')
plt.plot(x,u[50,:],label='t=5')
plt.plot(x,u[60,:],label='t=6')
plt.plot(x,u[70,:],label='t=7')
plt.plot(x,u[80,:],label='t=8')
plt.plot(x,u[90,:],label='t=9')
plt.plot(x,u[100,:],label='t=10')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Numerical Solution')
plt.legend()
plt.show()
```

#### Deliverables
Have a jupyter-notebook to show your code and results.
Based on the results, prepare a 4-minute presentation to show your results and answer the question.
