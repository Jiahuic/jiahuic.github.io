---
layout: page
permalink: /teaching/math-4990v5990v/08-svm-kernels/
title: "Lecture 8 — Support Vector Machines and Kernels"
---


*Part I · Week 8*

## Learning Goals

- Derive the **margin** $=1/\lVert w\rVert$ and the max-margin primal problem.
- Derive the **Lagrangian dual** and read off support vectors from the **KKT** conditions.
- Explain the soft margin, its box-constrained dual, and the **hinge-loss + L2** view.
- State **Mercer's condition** and the kernel trick; name common kernels.

## 1. Geometry: Margin $= 1/\lVert w\rVert$ (exam material)

For labels $y_i\in\{-1,+1\}$, a linear classifier is $f(x)=w^\top x + b$, predicting $\operatorname{sign} f(x)$. The signed distance from a point $x$ to the hyperplane $\{f=0\}$ is $f(x)/\lVert w\rVert$. Fixing the scale of $(w,b)$ by the canonical normalization $\min_i y_i(w^\top x_i + b) = 1$, the distance to the nearest point (the **margin**) is $1/\lVert w\rVert$. Maximizing the margin is minimizing $\lVert w\rVert$:

$$
\min_{w,b}\ \tfrac{1}{2}\lVert w\rVert^2
\quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 \ \ \forall i .
$$

This is a convex quadratic program with a unique solution.

## 2. The Lagrangian Dual and KKT (exam material)

Introduce multipliers $\alpha_i \ge 0$ for the constraints:

$$
\mathcal L(w,b,\alpha) = \tfrac12\lVert w\rVert^2 - \sum_i \alpha_i\big[y_i(w^\top x_i + b) - 1\big].
$$

Stationarity ($\nabla_w\mathcal L = 0$, $\partial_b\mathcal L = 0$) gives

$$
w = \sum_i \alpha_i y_i x_i, \qquad \sum_i \alpha_i y_i = 0 .
$$

The optimal $w$ is a **linear combination of the training points**. Substituting back yields the **dual**:

$$
\max_{\alpha \ge 0}\ \sum_i \alpha_i - \tfrac12 \sum_{i,j}\alpha_i\alpha_j\, y_i y_j\, x_i^\top x_j
\quad\text{s.t.}\quad \sum_i \alpha_i y_i = 0 .
$$

**KKT complementary slackness** $\alpha_i[y_i f(x_i)-1]=0$ means $\alpha_i>0$ **only** for points exactly on the margin — the **support vectors**. All other points have $\alpha_i=0$ and can be deleted without changing the solution. Crucially, the dual depends on the data **only through inner products** $x_i^\top x_j$ — the hook for kernels.

## 3. Soft Margin and $C$

Real data is not separable. Add slack $\xi_i\ge 0$:

$$
\min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i
\quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1-\xi_i,\ \ \xi_i\ge 0 .
$$

The only change to the dual is a **box constraint** $0 \le \alpha_i \le C$. Interpretation:

- **$C$ small:** wide margin, more violations tolerated → more bias, less variance (smoother).
- **$C$ large:** narrow margin, few violations → less bias, more variance (can overfit).

**Regularized-ERM view.** Eliminating the slack ($\xi_i = \max(0, 1-y_i f(x_i))$) shows the soft-margin SVM minimizes **hinge loss + L2 regularization**:

$$
\min_{w,b}\ \sum_i \max\!\big(0,\ 1 - y_i f(x_i)\big) + \frac{1}{2C}\lVert w\rVert^2 .
$$

So SVM is the same "loss + penalty" template as ridge/logistic (Lectures 4–5), with the margin-maximizing hinge loss.

## 4. The Kernel Trick and Mercer's Condition

Map features to a higher- (possibly infinite-) dimensional space $\phi(x)$. Since both the dual and the prediction $f(x) = \sum_i \alpha_i y_i\, \phi(x_i)^\top\phi(x) + b$ use only inner products, replace them with a **kernel**

$$
K(x_i,x_j) = \phi(x_i)^\top \phi(x_j),
$$

never forming $\phi$ explicitly. **Mercer's condition:** any symmetric function whose Gram matrix $K_{ij}$ is positive semidefinite for every sample corresponds to an inner product in *some* feature space — so it is a valid kernel. (This is the representer-theorem/RKHS view: the solution lies in the span of $\{K(x_i,\cdot)\}$.) Common kernels:

- **Linear:** $K = x_i^\top x_j$.
- **Polynomial:** $K = (x_i^\top x_j + c)^d$ — feature space of degree-$\le d$ monomials.
- **RBF (Gaussian):** $K = \exp(-\gamma\lVert x_i - x_j\rVert^2)$ — an **infinite-dimensional** feature space; the default nonlinear kernel.

**RBF $\gamma$:** large $\gamma$ → each point's influence is very local → wiggly boundary, overfits; small $\gamma$ → smooth. Tune $C$ and $\gamma$ **together** by cross-validation. **Scaling matters:** RBF uses distances, so standardize first (Lecture 1).

## 5. Worked Example

Two 1D classes: negatives at $x=-2,-1$; positives at $x=1,2$. Max-margin boundary $x=0$; support vectors $x=-1$ and $x=+1$; from $y_i(wx_i+b)=1$ with $b=0$ we get $w=1$, so margin $1/\lVert w\rVert = 1$.

## 6. Implementation Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10], "svc__gamma": [0.001, 0.01, 0.1]}, cv=5)
grid.fit(X, y)
print("best:", grid.best_params_, "CV acc:", grid.best_score_)
# grid.best_estimator_[-1].support_vectors_  -> the alpha_i > 0 points
```

> **Graduate depth.** PRML Ch. 7.1 derives the SVM dual, KKT conditions, and $\nu$-SVM; ESL Ch. 12.1–12.3
> covers the hinge-loss/regularization view; MFDL Ch. 4.2–4.5 bridges perceptron → kernels → networks.
> Be able to derive the dual from the Lagrangian, identify support vectors via complementary slackness,
> and state Mercer's condition.

## Connection to This Week

- **Lab 6** — SVM with linear vs. RBF kernels; tune $C$ and $\gamma$; visualize the boundary.
- **Quiz 6** — effect of $C$ and $\gamma$ on margin/boundary.
- **Homework 5 due / Homework 6 assigned** — due Wed Oct 14 (before Exam 2).

## References

- Weekly reading map, Week 8.
- ESL Ch. 12.1-12.3; PRML Ch. 7.1.
- MFDL Ch. 4.2-4.5 for the perceptron-to-kernels-to-neural-networks bridge.
- scikit-learn SVM user guide.
