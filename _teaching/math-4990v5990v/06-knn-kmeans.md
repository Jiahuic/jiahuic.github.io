---
layout: page
permalink: /teaching/math-4990v5990v/06-knn-kmeans/
title: "Lecture 6 — KNN, Kernel Functions, and k-Means Clustering"
---


*Part I · Week 6*

## Learning Goals

- Describe KNN for classification/regression and its link to the **Bayes classifier** (consistency).
- State the **Cover–Hart** bound and the **curse of dimensionality**.
- Explain **Mahalanobis** distance and why scaling/whitening matters.
- See KNN's hard $k$-neighbor vote as a special case of a **kernel-weighted** local estimate — the bridge to SVM kernels.
- Derive k-means as **block-coordinate descent** on its objective; explain convergence, k-means++, and the EM connection.
- Contrast **supervised** KNN with **unsupervised** k-means.

## 1. K-Nearest Neighbors (supervised)

KNN is **non-parametric** and **lazy**: it stores the data and predicts from the $k$ closest training points to a query $x$.

- **Classification:** majority vote among the $k$ neighbors — an estimate of $P(y=c\mid x)$ by the local frequency $\tfrac{1}{k}\sum_{i\in N_k(x)}\mathbb{1}[y_i=c]$.
- **Regression:** average of the neighbors' targets — a local estimate of $\mathbb{E}[y\mid x]$ (Lecture 1).

**Consistency.** KNN is a direct approximation to the optimal predictors of Lecture 1. As $n\to\infty$ with $k\to\infty$ and $k/n\to 0$, the KNN classifier converges to the **Bayes classifier** (universal consistency). Even $1$-NN is strong: the **Cover–Hart** theorem bounds its asymptotic error by twice the Bayes error, $R_{\text{1NN}} \le 2R_{\text{Bayes}}$.

**Choosing $k$ (bias–variance).** Small $k$ (e.g. 1) → flexible, low bias, high variance (noisy boundary); large $k$ → smoother, higher bias, lower variance. Effective complexity scales like $n/k$. Choose $k$ by cross-validation (Lecture 5).

## 2. Distance, Scaling, and the Curse of Dimensionality

Default distance is Euclidean, $d(x,x') = \lVert x-x'\rVert_2$. Since $d^2 = \sum_j (x_j-x'_j)^2$ sums over features, a large-scale feature (dollars, $10^4$) swamps a small one ($10^{-1}$) unless **standardized** (Lecture 1). The scale-and-correlation-invariant choice is the **Mahalanobis** distance $d_\Sigma(x,x') = \sqrt{(x-x')^\top\Sigma^{-1}(x-x')}$ — Euclidean distance in whitened coordinates.

**Curse of dimensionality (ESL 2.5).** In high $p$, "nearest" loses meaning: to capture a fraction $r$ of the data in a $p$-dimensional cube you need an edge of length $r^{1/p}$, which $\to 1$ (the whole range) as $p$ grows — neighborhoods are no longer local. Pairwise distances also **concentrate** (min and max distance become comparable), so KNN degrades. This motivates dimensionality reduction before distance-based methods (Lecture 7).

## 3. Kernel Functions: A Bridge to SVM

Plain KNN gives every one of the $k$ neighbors an equal vote and every point outside the ball zero vote — a hard cutoff. A smoother alternative is **kernel-weighted** regression/classification: replace the 0/1 neighbor indicator with a **kernel function** $K(x,x')$ that decays with distance, e.g. the Gaussian/RBF kernel $K(x,x') = \exp(-\gamma\lVert x-x'\rVert^2)$, and predict a *weighted* average $\hat y(x) = \sum_i K(x,x_i)\,y_i \big/ \sum_i K(x,x_i)$ instead of a hard vote over the $k$ nearest points. This is the same RBF kernel that reappears as the default nonlinear kernel for SVM (Lecture 8) — both methods measure similarity through a function of distance, KNN with a hard $k$-cutoff and SVM with a smooth, optimization-derived weighting. Recognizing "distance → similarity → kernel" here previews the more general **kernel trick** (Mercer's condition, feature-space inner products) formalized in Lecture 8.

## 4. k-Means Clustering (unsupervised)

Given only $X$, partition points into $k$ clusters minimizing within-cluster spread (**inertia**):

$$
J(\{c_i\}, \{\mu_c\}) = \sum_{i=1}^n \big\lVert x_i - \mu_{c_i} \big\rVert_2^2 .
$$

**Lloyd's algorithm as block-coordinate descent.** $J$ is minimized by alternating over its two argument blocks:

1. **Assign** (fix centers, optimize labels): $c_i = \arg\min_c \lVert x_i-\mu_c\rVert^2$ — nearest center.
2. **Update** (fix labels, optimize centers): $\mu_c = \tfrac{1}{|C_c|}\sum_{i\in C_c} x_i$ — the mean, because $\nabla_{\mu_c}\sum_{i\in C_c}\lVert x_i-\mu_c\rVert^2 = 0$ gives exactly the mean.

Each step **weakly decreases** $J$, and there are finitely many partitions, so Lloyd's algorithm **converges in finite steps** — but only to a **local** minimum (the global problem is NP-hard). Run several random initializations (`n_init`) and keep the best.

**k-means++** seeds centers spread out (probability $\propto$ squared distance to the nearest chosen center), giving an $O(\log k)$-approximation guarantee in expectation and far better solutions in practice.

**EM connection (PRML 9.1–9.2).** k-means is the hard-assignment limit of fitting a Gaussian mixture by EM: replace soft posterior responsibilities with a nearest-center hard assignment and fix isotropic unit covariances. This foreshadows probabilistic clustering (Lecture 7's probability review, and the latent-variable models of Lecture 15).

**Choosing $k$.** The **elbow method** plots $J$ vs. $k$ ($J$ always decreases; pick the elbow). The **silhouette score** (cohesion vs. separation, in $[-1,1]$) is a complementary, less subjective criterion.

## 5. KNN vs. k-Means (a common confusion)

| | KNN | k-means |
|---|---|---|
| Task | supervised (labels) | unsupervised (no labels) |
| "$k$" means | # neighbors for a prediction | # clusters |
| Output | class/value for a query | partition + centers |

## 6. Worked Example (KNN by hand)

Test point at origin; training points/labels:
$(0,3,0)$R, $(2,0,0)$R, $(0,1,3)$R, $(0,1,2)$G, $(-1,0,1)$G, $(1,1,1)$R.
Distances to origin: $3,\ 2,\ 3.16,\ 2.24,\ \mathbf{1.41},\ 1.73$.
- $k=1$: nearest $(-1,0,1)$ at $1.41$ → **Green**.
- $k=3$: $1.41$(G), $1.73$(R), $2$(R) → majority **Red**.

## 7. Implementation Example

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)                 # scale for distances

knn = KNeighborsClassifier(n_neighbors=5).fit(Xs, y)             # supervised
km  = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)   # unsupervised
print("KNN train acc:", knn.score(Xs, y))
print("k-means inertia J:", km.inertia_)
```

> **Graduate depth.** ESL Ch. 13.3 proves KNN consistency and the curse of dimensionality;
> PRML Ch. 9.1 derives k-means and Ch. 9.2 the EM/GMM generalization. Be able to show the mean is
> the optimal center, that $J$ decreases each Lloyd step, and to state the Cover–Hart bound.

## Connection to This Week

- **Lab 4** — KNN classification and k-means from scratch (elbow method).
- **Quiz 4** — choosing $k$, why scaling matters for distances.
- **Homework 4** — KNN by hand and KNN classifier vs. regression; due Mon Sep 28.

## References

- Weekly reading map, Week 6.
- ESL Ch. 2.3.2 and Ch. 13.3 for nearest-neighbor methods.
- ESL Ch. 14.3.6 and PRML Ch. 9.1 for k-means.
- scikit-learn Nearest Neighbors and Clustering guides.
