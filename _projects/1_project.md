---
layout: page
title: On preconditioning the treecode-accelerated boundary integral Poisson-Boltzmann solver
description: Preconditioning, GMRES, treecode, electrostatics, boundary integral, Poisson-Boltzmann equation
img: assets/img/project1/FMM_treecode.png
importance: 1
category: Computational Math
---

This project works on a preconditioning scheme using preconditioner matrix $$M$$
such that $$M^{-1}A$$ has much improved condition 
while $$M^{-1}z$$ can be rapidly computed for any vector $$z$$.
In this scheme,
the matrix $$M$$ carries the interactions between boundary elements
on the same leaf only in the tree structure
thus is block diagonal with many computational advantages.
The sizes of the blocks in $$M$$ are conveniently controlled by treecode parameter $$N_0$$,
the maximum number of particles per leaf.
The numerical results show that this new preconditioning scheme improves
the treecode-accelerated boundary integral (TABI) solver with significantly
reduced iteration numbers and better accuracy,
particularly for protein sets on which TABI solver previously converges slowly.

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
