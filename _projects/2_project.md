---
layout: page
title: Cyclically parallelized Treecode for Fast Computations of Electrostatic In teractions on Molecular Surfaces
description: treecode, electrostatics, parallel computing, MPI, load balancing
img: assets/img/project2/cyclic.png
importance: 2
category: Computational Math
---

We study the parallelization of a flexible order Cartesian treecode algorithm
for evaluating electrostatic potentials of charged particle systems in which
$$N$$ particles are located on the molecular surfaces of biomolecules such as proteins.
When the well-separated condition is satisfied,
the treecode algorithm uses a far-field Taylor expansion
to compute $${O}(N\log{N})$$ particle-cluster interactions
to replace the $${O}(N^2)$$ particle-particle interactions.
The algorithm is implemented using the Message Passing Interface (MPI) standard
by creating identical tree structures in the memory of each task for concurrent computing.
We design a cyclic order scheme to uniformly distribute spatially-closed target particles to all available tasks,
which significantly improves parallel load balancing.
We also investigate the parallel efficiency
subject to treecode parameters such as
Taylor expansion order $$p$$,
maximum particles per leaf $$N_0$$, and
maximum acceptance criterion $$\theta$$.
This cyclically parallelized treecode can solve interactions among up to tens of millions of particles.
However, if the problem size exceeds the memory limit of each task,
a scalable domain decomposition (DD) parallelized treecode using an orthogonal recursive bisection (ORB) tree can be used instead.
In addition to efficiently computing the $$N$$-body problem of charged particles,
our approach can potentially accelerate GMRES iterations
for solving the boundary integral Poisson-Boltzmann equation.

## Methods
### Treecode for electrostatic interactions
For a system of $$N$$ particles located at $$\textbf{x}_i$$ with partial charges $$q_{i}, i=1,\ldots,N$$,
we denote the induced potential at $$\textbf{x}_{i}$$ by
\begin{equation}
\label{particle-particle}
V_{i}=\sum_{j=1, j\neq i}^N q_{i}G(\textbf x_i, \textbf x_j)
\end{equation}
where $$G(\textbf x,\textbf y)$$ is the Coulomb or the screened Coulomb potential,
defined respectively by
\begin{equation}
\label{eq_CP}
G_0(\textbf x, \textbf y)=\frac{1}{4\pi|\textbf x-\textbf y|}
\end{equation}
and
\begin{equation}
\label{eq_sCP}
G_\kappa(\textbf x, \textbf y)=\frac{e^{-\kappa
|\textbf x-\textbf y|}}{4\pi|\textbf x-\textbf y|}.
\end{equation}
Note we attempted to use CGI units here but 
supply the additional $$4\pi$$ coefficient 
in the denominator to represent electrostatic potential generated 
from partial charges with units of fundamental charges, 
as from most force field generators such as [CHARMM](https://www.charmm.org/) and [AMBER](https://ambermd.org/).

### Particle-cluster interaction

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project1/p1_f1.png" title="treecode" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Details of treecode; (a) tree structure of particle clusters; 
    (b) particle-cluster interaction between particle \(\textbf{x}_i\) and cluster \(c=\{\textbf{x}_j\}\); 
    \(\textbf{x}_c\): cluster center, \(R\): particle-cluster distance, \(r_c\): cluster radius.
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project2/p2_f1.png" title="cyclic-sequencial" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (a) methods for assigning target particles to tasks: sequential order (top) vs cyclic order (bottom);
    (b) an illustration of an ORB tree using tasks 0-15 in four subdivisions. The binary code in color shows the partner of each task at different level. For example: task 0 \(\sim(0000)_2\) has  task 8 \(\sim(1000)_2\), task 4 \(\sim(0100)_2\), task 2 \(\sim(0010)_2\), and task 1 \(\sim(0001)_2\) as its 0-1 partner at level 1 (red), level 2 (green), level 3 (purple), and level 4 (orange) respectively.
</div>
