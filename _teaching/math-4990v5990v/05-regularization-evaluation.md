---
layout: page
permalink: /teaching/math-4990v5990v/05-regularization-evaluation/
title: "Lecture 5 — Regularization, Cross-Validation, and Model Evaluation"
---


*Part I · Week 4*

## Learning Goals

- Explain overfitting as excess variance and regularization as a bias-for-variance trade.
- Derive the **ridge** closed form; compute its **bias, variance, and effective degrees of freedom**.
- Give the **Bayesian (MAP)** view: ridge = Gaussian prior, lasso = Laplace prior.
- Explain **lasso** sparsity via the L1 geometry and the **soft-thresholding** solution.
- Explain **k-fold CV**, the LOOCV **hat-matrix shortcut**, and nested CV for honest selection.

## 1. Overfitting and the Regularization Idea

A flexible model can drive training error to zero by fitting noise → high **variance**, poor risk (Lecture 2). **Regularization** adds a complexity penalty to the loss, deliberately introducing bias to cut variance by more (Lecture 2 §6, and the regularization preview in Lecture 2 §7). Add a penalty on coefficient size:

$$
\textbf{Ridge (L2):}\ \ L(w) = \lVert y - Xw \rVert_2^2 + \lambda \lVert w \rVert_2^2,
\qquad
\textbf{Lasso (L1):}\ \ L(w) = \lVert y - Xw \rVert_2^2 + \lambda \lVert w \rVert_1 .
$$

$\lambda \ge 0$ is the **regularization strength**: $\lambda=0$ is OLS; $\lambda\to\infty$ shrinks $w\to 0$. Penalize the coefficients but **not** the intercept, and always **standardize** features first, or the penalty unfairly targets large-scale features.

## 2. Ridge: Closed Form, Bias, Variance, Degrees of Freedom (exam material)

Setting $\nabla_w L = -2X^\top(y-Xw) + 2\lambda w = 0$ gives

$$
\boxed{\ \hat w_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y\ }.
$$

Because $X^\top X \succeq 0$, every eigenvalue of $X^\top X + \lambda I$ is $\ge \lambda > 0$: the matrix is **always invertible**, so ridge fixes the rank-deficiency of OLS (Lecture 2 §3) and improves conditioning ($\kappa$ drops from $\lambda_{\max}/\lambda_{\min}$ to $(\lambda_{\max}+\lambda)/(\lambda_{\min}+\lambda)$).

Diagonalize with the SVD $X = U\Sigma V^\top$ ($d_j$ = singular values). Then ridge shrinks each principal-direction coefficient by a factor $\tfrac{d_j^2}{d_j^2+\lambda}$:

- **Bias:** $\mathbb{E}[\hat w_{\text{ridge}}] \ne w^\star$ — biased toward 0 (bias grows with $\lambda$).
- **Variance:** strictly **smaller** than OLS; shrinkage damps the high-variance small-$d_j$ directions.
- **Effective degrees of freedom:** $\displaystyle \operatorname{df}(\lambda) = \sum_j \frac{d_j^2}{d_j^2+\lambda}$, decreasing from $p$ (at $\lambda=0$) toward $0$. This is the continuous analogue of "number of parameters" and quantifies model complexity.

## 3. Bayesian View (MAP)

Regularization is a **prior**. With Gaussian likelihood (Lecture 3) and a prior on $w$, the MAP estimate $\arg\max_w \log p(y\mid w) + \log p(w)$ becomes penalized least squares:

- **Gaussian prior** $w_j \sim \mathcal N(0, \tau^2)$ ⇒ $\log p(w) \propto -\lVert w\rVert_2^2/(2\tau^2)$ ⇒ **ridge**, with $\lambda = \sigma^2/\tau^2$.
- **Laplace prior** $w_j \sim \text{Laplace}(0,b)$ ⇒ $\log p(w) \propto -\lVert w\rVert_1/b$ ⇒ **lasso**.

So the choice of penalty is a choice of belief about the coefficients (small vs. sparse).

## 4. Lasso: Sparsity and Soft-Thresholding

The L1 ball has **corners** on the axes; the first contour of the loss to touch it typically hits a corner, setting some coefficients **exactly to zero** — lasso does automatic **feature selection**, unlike ridge (whose spherical penalty shrinks but never zeroes). There is no closed form in general (subgradient at 0; solved by coordinate descent), but for an **orthonormal** design the solution is explicit **soft-thresholding**:

$$
\hat w_j = \operatorname{sign}(\hat w_j^{\text{OLS}})\,\big(|\hat w_j^{\text{OLS}}| - \lambda\big)_+ ,
$$

which literally clips small coefficients to zero — the analytic origin of sparsity. (Ridge for the orthonormal case is proportional shrinkage $\hat w_j^{\text{OLS}}/(1+\lambda)$: never exactly zero.)

## 5. Cross-Validation

A single split wastes data and gives a high-variance estimate. **k-fold CV**: partition the training data into $k$ folds; for each, train on $k-1$ and validate on the held-out fold; average the $k$ scores. Bias–variance of the *estimator itself*: small $k$ (e.g. 2) → high bias (each model trained on little data); large $k$ → low bias but higher variance and cost. **$k=5$ or $10$** is the standard compromise.

**LOOCV shortcut (linear models).** For least squares, leave-one-out CV needs *no refitting* thanks to the hat matrix $H$ (Lecture 2 §4):

$$
\text{CV}_{\text{LOO}} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat y_i}{1 - H_{ii}}\right)^2 .
$$

A single fit gives all $n$ leave-one-out errors — a clean payoff of the projection view.

## 6. The Right Protocol (and Nested CV)

Touch the test set **once**, at the end: split off test → use CV *on the training set* to select hyperparameters → refit on all training data → report on test. Fit preprocessing **inside** each fold (use a `Pipeline`), or scaling leaks (Lecture 1 §3). When you also want an **unbiased estimate of the selected model's error**, use **nested CV**: an inner CV loop selects $\lambda$, an outer loop estimates risk — otherwise the CV score that guided selection is optimistic.

## 7. Worked Example

Choosing $\lambda$ by 5-fold CV: mean validation MSE for $\lambda\in\{0.01,0.1,1,10\}$ is $0.61, 0.55, 0.53, 0.58$ → choose $\lambda=1$ (lowest). Reading the curve: the left end (small $\lambda$) is high-variance overfitting; the right end (large $\lambda$) is high-bias underfitting.

## 8. Implementation Example

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
pipe = make_pipeline(StandardScaler(), Ridge())      # scaling inside CV -> no leakage
grid = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-2, 3, 12)},
                    cv=5, scoring="neg_mean_squared_error").fit(X, y)
print("best alpha:", grid.best_params_, " best CV MSE:", -grid.best_score_)
```

> **Graduate depth.** PRML Ch. 3.1.4 and Ch. 3.2 give the MAP/evidence view of regularization;
> ESL Ch. 3.4 covers ridge/lasso shrinkage and the SVD analysis; ESL Ch. 7.10 covers CV pitfalls.
> Be able to derive the ridge closed form and $\operatorname{df}(\lambda)$, the soft-thresholding
> solution, and the LOOCV hat-matrix identity.

## Connection to This Week

- **Lab 5** — ridge/lasso and cross-validation; a $\lambda$ sweep.
- **Quiz 5** — CV mechanics, over/underfitting from a validation curve.
- **Homework 2** — cross-validation and gradient descent.

## References

- Weekly reading map, Week 7.
- MFDL Ch. 9 and Ch. 11; ESL Ch. 3.4 and Ch. 7.10.
- PRML Ch. 3.1.4 and Ch. 3.2 for regularized regression and bias-variance.
- scikit-learn Pipeline, cross-validation, Ridge, and Lasso guides.
