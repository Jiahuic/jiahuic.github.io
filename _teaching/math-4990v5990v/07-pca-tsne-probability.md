---
layout: page
permalink: /teaching/math-4990v5990v/07-pca-tsne-probability/
title: "Lecture 7 — PCA, t-SNE, and a Probability Review"
---


*Part I · Week 7*

## Learning Goals

- Derive PCA **two ways**: variance maximization (Lagrangian) and minimum reconstruction error.
- Connect PCA to the **SVD** and the **Eckart–Young** optimality theorem.
- Interpret explained variance; explain why standardization matters.
- State the actual **t-SNE objective** (KL of neighbor distributions) and contrast it with PCA.
- Review **Bayes' rule**, joint/marginal/conditional distributions, and KL divergence — the probability toolkit the rest of the course (and Part II's generative models) draws on.

## 1. Why Reduce Dimensions?

High-dimensional data is hard to visualize, store, and model (curse of dimensionality, Lecture 6). Dimensionality reduction finds a faithful low-dimensional representation. PCA is the linear, variance-optimal answer.

## 2. PCA as Variance Maximization (exam material)

Center the data ($\tfrac1n\sum_i x_i = 0$) and form the covariance $\Sigma = \tfrac{1}{n}X^\top X \in \mathbb{R}^{p\times p}$. The first principal direction maximizes the projected variance:

$$
u_1 = \arg\max_{\lVert u\rVert=1}\ \frac{1}{n}\sum_i (u^\top x_i)^2 = \arg\max_{\lVert u\rVert=1} u^\top \Sigma\, u .
$$

**Derivation (Lagrange multipliers).** Maximize $u^\top\Sigma u$ subject to $u^\top u = 1$. The Lagrangian $u^\top\Sigma u - \lambda(u^\top u - 1)$ has stationarity condition

$$
\Sigma u = \lambda u,
$$

so $u$ must be an **eigenvector** of $\Sigma$, and the objective at that point is $u^\top\Sigma u = \lambda$. The maximum is the **top eigenvector**, with variance $\lambda_1$ = largest eigenvalue. Subsequent components maximize variance subject to orthogonality → the remaining eigenvectors in decreasing $\lambda$ order.

- **Projection:** $z_i = U_k^\top x_i$ maps $x_i\in\mathbb R^p$ to $\mathbb R^k$ using the top-$k$ eigenvectors $U_k$.
- **Explained-variance ratio:** $\lambda_j/\sum_l \lambda_l$; the cumulative sum picks $k$.

## 3. PCA as Minimum Reconstruction Error (dual view)

Equivalently, PCA finds the rank-$k$ orthonormal basis minimizing squared reconstruction error:

$$
\min_{U_k^\top U_k = I}\ \sum_{i=1}^n \big\lVert x_i - U_k U_k^\top x_i \big\rVert^2 .
$$

Expanding, minimizing reconstruction error equals maximizing retained variance — the two objectives coincide. So PCA is simultaneously the max-variance projection *and* the best linear reconstruction. This encoder ($x\mapsto z=U_k^\top x$) / decoder ($z \mapsto U_k z$) pairing is the linear special case of the autoencoders in Lecture 15.

## 4. The SVD and Eckart–Young

Let the centered data have SVD $X = U\Sigma_{\!s} V^\top$ (singular values $d_1\ge d_2 \ge \dots$). Then:

- Principal **directions** = right singular vectors $v_j$ (eigenvectors of $X^\top X$).
- Principal **scores** = $XV = U\Sigma_{\!s}$.
- Eigenvalues $\lambda_j = d_j^2/n$.

The **Eckart–Young theorem** states the rank-$k$ truncation $X_k = \sum_{j\le k} d_j u_j v_j^\top$ is the **best rank-$k$ approximation** of $X$ in both Frobenius and spectral norm. PCA is therefore optimal low-rank compression, not just a heuristic. Computing PCA via the SVD of $X$ is more stable than eigendecomposing $X^\top X$ (the conditioning argument of Lecture 2 §9).

**Standardize first:** PCA maximizes variance, so unscaled large-variance features dominate the components regardless of importance. (Whitening — Lecture 1 §10 — is PCA followed by scaling each score to unit variance.)

## 5. t-SNE: the Objective (not just the vibe)

**t-Distributed Stochastic Neighbor Embedding** is a **non-linear** visualization method (usually to 2D). It builds neighbor probabilities in both spaces and matches them:

- **High-dim affinities:** symmetrized Gaussians $p_{ij}\propto \exp(-\lVert x_i-x_j\rVert^2/2\sigma_i^2)$, where each $\sigma_i$ is set so the local **perplexity** (effective number of neighbors) matches a target.
- **Low-dim affinities:** a heavy-tailed **Student-$t$** (one d.o.f.), $q_{ij}\propto (1+\lVert z_i-z_j\rVert^2)^{-1}$.
- **Objective:** minimize $\mathrm{KL}(P\Vert Q) = \sum_{i\ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}$ by gradient descent.

The **asymmetry** of KL penalizes putting nearby points far apart much more than the reverse → **local neighborhoods** are preserved. The heavy Student-$t$ tail fixes the **crowding problem** (in 2D there isn't room for all moderate-distance neighbors) by allowing dissimilar points extra room. Consequences: results depend on `perplexity` and the random seed; **cluster sizes and inter-cluster distances are not meaningful**; it has no out-of-sample transform.

| | PCA | t-SNE |
|---|---|---|
| Mapping | linear | non-linear |
| Preserves | global variance/structure | local neighborhoods |
| Deterministic | yes | no (seed, perplexity) |
| Use | compression, features, fast viz | 2D visualization |
| Far-cluster distance | meaningful | **not** meaningful |

## 6. Probability Review: Bayes' Rule, Distributions, and KL Divergence

The rest of the course leans on probability more heavily from here on — this section collects the pieces used later, most of them already used once already (Lecture 2's Gaussian-MLE derivation of least squares; the t-SNE $\mathrm{KL}$ objective just above).

**Joint, marginal, conditional.** For random variables $x,y$: the joint $p(x,y)$ determines the marginal $p(x) = \int p(x,y)\,dy$ and the conditional $p(y\mid x) = p(x,y)/p(x)$.

**Bayes' rule.** Rearranging the conditional both ways gives

$$
p(y\mid x) = \frac{p(x\mid y)\,p(y)}{p(x)}, \qquad p(x) = \int p(x\mid y)\,p(y)\,dy .
$$

Read as **posterior** $\propto$ **likelihood** $\times$ **prior**. This is exactly the MAP argument behind ridge/lasso (Lecture 5 §3: a Gaussian or Laplace *prior* on $\theta$ combined with the Gaussian *likelihood* of Lecture 2 §2b) and it is the organizing idea behind the latent-variable generative models later in the course (Lecture 15's $p(x) = \int p(x\mid z)p(z)\,dz$ is literally this marginalization with $y\to z$, a hidden/latent variable instead of an observed one).

**Common distributions used later:** the **Gaussian** $\mathcal N(\mu,\sigma^2)$ (noise models, weight priors, VAE latents); the **Bernoulli/categorical** (binary/multiclass labels, Lectures 4 and 11); the **standard normal prior** $z\sim\mathcal N(0,I)$ used to seed a latent space (Lectures 15–16).

**KL divergence.** For distributions $p,q$ on the same space,

$$
\mathrm{KL}(p\Vert q) = \mathbb{E}_{x\sim p}\Big[\log\frac{p(x)}{q(x)}\Big] \ \ge 0,
$$

with equality iff $p=q$ (Jensen's inequality applied to $-\log$, a convex function). $\mathrm{KL}$ is **not symmetric** ($\mathrm{KL}(p\Vert q)\ne \mathrm{KL}(q\Vert p)$ in general) — already visible in t-SNE's asymmetric neighbor-preservation behavior above. $\mathrm{KL}$ reappears as the regularizer in the VAE's ELBO (Lecture 15) and in the cross-entropy loss itself: minimizing cross-entropy $H(p,q) = H(p) + \mathrm{KL}(p\Vert q)$ against a fixed data distribution $p$ is the same as minimizing $\mathrm{KL}(p\Vert q)$ (Lecture 4's loss, seen through this lens).

## 7. Worked Example (PCA intuition)

Two nearly perfectly correlated features lie along a line: $\lambda_1\gg\lambda_2$, so PC1 aligns with the line and captures almost all variance; projecting onto it halves the dimension with negligible reconstruction error — the Eckart–Young statement in miniature.

## 8. Implementation Example

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)      # 64-dim, 10 classes
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=2).fit(Xs)
print("explained variance ratio:", pca.explained_variance_ratio_)
Z_pca  = pca.transform(Xs)
Z_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Xs)
# scatter Z_pca and Z_tsne colored by y to compare
```

> **Graduate depth.** PRML Ch. 12.1 derives PCA both ways and Ch. 12.2 the probabilistic (latent-variable)
> PCA; ESL Ch. 14.5 covers the SVD view; PRML Ch. 1.6 and Ch. 2 cover the probability/KL background
> in Section 6. Be able to derive $\Sigma u=\lambda u$ via Lagrange multipliers, state Eckart–Young,
> explain the KL objective and crowding problem of t-SNE (van der Maaten & Hinton, 2008), and prove
> $\mathrm{KL}(p\Vert q)\ge 0$ via Jensen's inequality.

## Connection to This Week

- **Lab 5** — PCA and t-SNE on image digits; compare and discuss.
- **Quiz 5** — standardizing before PCA, explained variance, t-SNE pitfalls.
- **Homework 4 due / Homework 5 assigned** — due Mon Oct 5.

## References

- Weekly reading map, Week 7.
- ESL Ch. 14.5 and Ch. 14.9; PRML Ch. 12.1.
- van der Maaten and Hinton (2008), *Visualizing Data using t-SNE*.
- PRML Ch. 1.6 and Ch. 2 for the probability review.
- scikit-learn Decomposition and Manifold learning guides.
