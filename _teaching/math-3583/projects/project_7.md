---
layout: page
permalink: /teaching/math-3583/project_7/
title: "Project 7: 2D Random Walk Simulation and Its Continuous Approximation"
---
To implement a 2D random walk simulation and study its behavior for large time steps, comparing the results to the continuous form of diffusion.
Random walks are elementary stochastic processes that serve as discrete analogs to continuous diffusion phenomena. In two dimensions, a random walk can be represented as a sequence of steps in the $$x$$ and $$y$$ directions.

1. Introduction to Random Walks
Study the basic principles behind random walks, focusing on the 2D case. Understand the rules governing the steps and how they relate to diffusion phenomena.

2. Program the 2D Random Walk
    - Implement a simulation for a 2D random walk where a "walker" can move one step in either the $$x$$ or $$y$$ direction at each time step.
    - Initialize the walker at the origin and run the simulation for $$N$$ time steps, where $$N$$ is large (e.g., $$N = 10^5$$).
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

    - Discuss how well the random walk approximates continuous diffusion for large $$N$$.

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
