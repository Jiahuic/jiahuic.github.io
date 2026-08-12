---
layout: page
permalink: /teaching/math-4990v5990v/02-linear-regression/
title: "Lecture 2 — Linear Regression, Least Squares, and a First Look at Networks"
---


*Part I · Week 2*

## Learning Goals

- Derive least squares two ways: as **loss minimization** and as **maximum likelihood** under Gaussian noise.
- Derive the **normal equations** by matrix calculus and characterize when the solution is unique.
- Interpret least squares geometrically via the **hat (projection) matrix**.
- State the **statistical properties** of the estimator (unbiasedness, covariance, Gauss–Markov).
- Derive the **bias–variance decomposition** and connect it to conditioning and regularization.
- Preview the **regularized** objective and see linear regression as a **single-layer network**.

## 1. Recap: The Model

Lecture 1 §11 set up the model $\hat y = X\theta$ for design matrix $X\in\mathbb R^{n\times(p+1)}$ (intercept absorbed) and parameter vector $\theta\in\mathbb R^{p+1}$, with hypothesis class $\mathcal F=\{x\mapsto\theta^\top\tilde x\}$ — linear in $\theta$, which is what makes everything below convex. This lecture fits $\theta$, analyzes the fit, and previews where it's going next.

## 2. Two Derivations of Least Squares

### 2a. Loss minimization

We measure fit by the **residual sum of squares** (RSS), equivalently the mean squared error:

$$
L(\theta) = \frac{1}{n}\sum_{i=1}^n (y_i - x_i^\top \theta)^2 = \frac{1}{n}\lVert y - X\theta \rVert_2^2 .
$$

Squared error is smooth and convex (Section 3), which is why it dominates in practice.

### 2b. Maximum likelihood (the probabilistic view)

Assume the data are generated as $y_i = x_i^\top\theta + \varepsilon_i$ with i.i.d. Gaussian noise $\varepsilon_i \sim \mathcal N(0,\sigma^2)$. Then $y_i \mid x_i \sim \mathcal N(x_i^\top\theta,\ \sigma^2)$, and the log-likelihood is

$$
\ell(\theta) = \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(\!-\frac{(y_i - x_i^\top\theta)^2}{2\sigma^2}\Big)
= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - x_i^\top\theta)^2 .
$$

Only the last term depends on $\theta$, and it is $-\tfrac{1}{2\sigma^2}\cdot(\text{RSS})$. Therefore

$$
\hat\theta_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\min_\theta \lVert y - X\theta\rVert_2^2 = \hat\theta_{\text{LS}} .
$$

**Least squares is maximum likelihood under Gaussian noise.** This is why the squared loss is not arbitrary: it encodes an assumption about the noise. (Heavy-tailed noise → the L1/absolute loss is the MLE instead; that is why MAE is more robust to outliers — Lab 1, Quiz 1.)

## 3. The Normal Equations (exam material)

Expand the objective as a quadratic in $\theta$:

$$
L(\theta) = \tfrac{1}{n}\big(y - X\theta\big)^\top\!\big(y - X\theta\big)
= \tfrac{1}{n}\big(y^\top y - 2\,\theta^\top X^\top y + \theta^\top X^\top X\,\theta\big).
$$

Using $\nabla_\theta(\theta^\top a) = a$ and $\nabla_\theta(\theta^\top A \theta) = 2A\theta$ for symmetric $A = X^\top X$,

$$
\nabla_\theta L(\theta) = \tfrac{1}{n}\big(-2X^\top y + 2X^\top X\,\theta\big) = -\tfrac{2}{n}X^\top(y - X\theta).
$$

Setting the gradient to zero yields the **normal equations**

$$
X^\top X\,\theta = X^\top y, \qquad\text{and if } X^\top X \text{ is invertible,}\qquad
\boxed{\ \hat\theta = (X^\top X)^{-1} X^\top y\ }.
$$

**Why this is a minimum, and when it is unique.** The Hessian is $\nabla^2_\theta L = \tfrac{2}{n}X^\top X \succeq 0$, so $L$ is convex and every stationary point is a global minimizer. It is **strictly** convex — hence the minimizer is *unique* — iff $X^\top X \succ 0$, i.e. iff $X$ has **full column rank** ($\operatorname{rank} X = p+1$, which needs $n \ge p+1$ and no exactly collinear features).

**When $X^\top X$ is singular** (collinear features or $p+1 > n$): infinitely many minimizers exist. Remedies — drop/combine features, take the minimum-norm solution via the **pseudoinverse** $\hat\theta = X^+ y$, or add **regularization** (ridge, Section 7 below and Lecture 5), which restores strict convexity.

## 4. Geometry: The Hat Matrix

The fitted values are a linear image of $y$:

$$
\hat y = X\hat\theta = \underbrace{X(X^\top X)^{-1}X^\top}_{\textstyle H}\, y = Hy .
$$

$H$ is the **hat matrix**. It is symmetric ($H^\top = H$) and idempotent ($H^2 = H$), so it is the **orthogonal projection** onto the column space $\operatorname{col}(X)$. Consequences:

- The residual $r = y - \hat y = (I - H)y$ lies in the orthogonal complement: $X^\top r = 0$ — the normal equations *are* the statement "residual $\perp$ every feature."
- $\operatorname{trace}(H) = \operatorname{rank}(X) = p+1$ = the model's **degrees of freedom** (this reappears in the unbiased noise estimate and in AIC-type criteria).
- Fitting is Pythagoras: $\lVert y\rVert^2 = \lVert \hat y\rVert^2 + \lVert r\rVert^2$, which underlies the $R^2 = 1 - \text{RSS}/\text{TSS}$ decomposition.

## 5. Statistical Properties of $\hat\theta$

Treat $X$ as fixed and $y = X\theta^\star + \varepsilon$ with $\mathbb{E}[\varepsilon]=0$, $\operatorname{Cov}(\varepsilon)=\sigma^2 I$. Substituting into $\hat\theta = (X^\top X)^{-1}X^\top y$:

$$
\hat\theta = \theta^\star + (X^\top X)^{-1}X^\top \varepsilon .
$$

- **Unbiased:** $\mathbb{E}[\hat\theta] = \theta^\star$.
- **Covariance:** $\operatorname{Cov}(\hat\theta) = (X^\top X)^{-1}X^\top(\sigma^2 I)X(X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$.
- **Unbiased noise estimate:** $\hat\sigma^2 = \dfrac{\lVert r\rVert^2}{n - (p+1)}$ (the denominator is $n - \operatorname{trace}H$).
- **Gauss–Markov theorem:** among all *linear unbiased* estimators of $\theta^\star$, OLS has the smallest variance — it is **BLUE** (Best Linear Unbiased Estimator). Under the additional Gaussian assumption it is also the minimum-variance unbiased estimator outright.

The covariance $\sigma^2(X^\top X)^{-1}$ is the analytical bridge to Section 6: when features are nearly collinear, $(X^\top X)^{-1}$ has huge entries, so coefficients have enormous variance — the very instability regularization fixes.

## 6. Bias–Variance Decomposition (exam material)

Fix a test point $x_0$ with target $y = f(x_0) + \varepsilon$, $\mathbb{E}[\varepsilon]=0$, $\operatorname{Var}(\varepsilon)=\sigma^2$. Let $\hat f(x_0)$ be the prediction of a model trained on a random dataset. The expected squared prediction error decomposes as

$$
\mathbb{E}\big[(y - \hat f(x_0))^2\big] = \underbrace{\sigma^2}_{\text{irreducible}} + \underbrace{\big(\mathbb{E}[\hat f(x_0)] - f(x_0)\big)^2}_{\text{Bias}^2} + \underbrace{\operatorname{Var}\big(\hat f(x_0)\big)}_{\text{Variance}} .
$$

**Derivation.** Write $\mathbb{E}[(y-\hat f)^2] = \mathbb{E}[(y - f)^2] + \mathbb{E}[(f - \hat f)^2] + 2\,\mathbb{E}[(y-f)(f-\hat f)]$. The first term is $\sigma^2$; the cross term vanishes because $\varepsilon = y-f$ is independent of the training-set–dependent $\hat f$ and has mean 0. Then add and subtract $\mathbb{E}[\hat f]$ inside the middle term:
$\mathbb{E}[(f - \hat f)^2] = (f - \mathbb{E}\hat f)^2 + \mathbb{E}[(\hat f - \mathbb{E}\hat f)^2] = \text{Bias}^2 + \text{Variance}$. $\qquad\blacksquare$

- **High bias / underfitting:** model too rigid (a line for a curved trend). Train and test error both high.
- **High variance / overfitting:** model too flexible, fits noise. Low train error, high test error.
- Flexible methods help when $n$ is large and the signal is nonlinear; inflexible (or regularized) methods help when $n$ is small or $p$ is large relative to $n$. Regularization deliberately **adds bias to remove more variance**.

## 7. Regularization Preview

Section 5 showed $\operatorname{Cov}(\hat\theta) = \sigma^2(X^\top X)^{-1}$ blows up when $X^\top X$ is nearly singular. The fix previewed here (full treatment in Lecture 5) is to penalize large coefficients:

$$
L_{\text{ridge}}(\theta) = \lVert y - X\theta\rVert_2^2 + \lambda\lVert\theta\rVert_2^2, \qquad \lambda \ge 0 .
$$

$\lambda=0$ recovers OLS; larger $\lambda$ shrinks $\theta$ toward $0$, trading a little bias for a lot less variance — exactly the Section 6 tradeoff, now made *tunable* instead of fixed by the model class. Lecture 5 derives the closed form, its bias/variance, and the L1 (lasso) alternative.

## 8. From Linear Regression to a Single-Layer Network

Linear regression $\hat y = \theta_0 + w^\top x$ is, in neural-network language, a single neuron with **no activation function** (or the identity activation) and squared-error loss. Two changes turn it into the building block of Part II:

- **Add a nonlinearity.** $\hat y = g(\theta_0 + w^\top x)$ for an **activation** $g$ (Lecture 3 introduces this to bridge into logistic regression, where $g=\sigma$).
- **Stack neurons into layers.** A layer of $m$ neurons is $m$ independent copies of this affine map, i.e. $\hat y = g(W x + b)$ for a matrix $W\in\mathbb R^{m\times p}$; stacking layers composes these maps (Lecture 11).

Everything you derived above — a loss, a gradient, a convexity argument, a bias–variance tradeoff — reappears for networks; the difference is that composing nonlinear layers makes the loss **non-convex** (Lecture 11 §4), so the closed-form normal equations of Section 3 no longer apply and we fall back on the iterative method of Lecture 3: gradient descent.

## 9. Numerical Note: Don't Invert $X^\top X$

Forming and inverting $X^\top X$ **squares the conditioning** of the problem: $\kappa_2(X^\top X) = \kappa_2(X)^2$. With ill-conditioned or nearly collinear features this destroys precision. Production solvers instead factor $X$ directly — **QR** ($X = QR$, solve $R\theta = Q^\top y$) or the **SVD** ($X = U\Sigma V^\top$, $\hat\theta = V\Sigma^+ U^\top y$). scikit-learn's `LinearRegression` uses the SVD path, which also returns the sensible minimum-norm solution when $X$ is rank-deficient. Solve the normal equations by hand for insight; use a factorization in code.

## 10. Worked Example (by hand)

Fit $\hat y = \theta_0 + \theta_1 x$ to $(1,1),(2,2),(3,2)$:

$$
X = \begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\quad y=\begin{bmatrix}1\\2\\2\end{bmatrix},\qquad
X^\top X = \begin{bmatrix}3&6\\6&14\end{bmatrix},\quad X^\top y=\begin{bmatrix}5\\11\end{bmatrix}.
$$

Solving $\begin{bmatrix}3&6\\6&14\end{bmatrix}\theta = \begin{bmatrix}5\\11\end{bmatrix}$ gives $\theta_1 = 0.5,\ \theta_0 = \tfrac{2}{3}$, so $\hat y = 0.667 + 0.5x$. The fitted values are $\hat y = (1.17, 1.67, 2.17)$ and the residuals $r = (-0.17, 0.33, -0.17)$ satisfy $\sum_i r_i = 0$ and $\sum_i x_i r_i = 0$ — exactly $X^\top r = 0$.

## 11. Implementation Example

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

# Normal equations by hand reproduce sklearn (add an intercept column):
Xtil = np.c_[np.ones(len(X_train)), X_train]
theta = np.linalg.solve(Xtil.T @ Xtil, Xtil.T @ y_train)   # [intercept, coefs...]
print("match:", np.allclose(theta[0], reg.intercept_) and np.allclose(theta[1:], reg.coef_))
```

`LinearRegression` solves the least-squares problem via a stable SVD-based solver, not by literally inverting $X^\top X$ (Section 9).

> **Graduate depth.** Read the MLE/decision-theory framing in PRML Ch. 3.1–3.2 and the bias–variance
> treatment in ESL Ch. 7.2–7.3. Be able to (i) derive $\operatorname{Cov}(\hat\theta)=\sigma^2(X^\top X)^{-1}$,
> (ii) prove Gauss–Markov, and (iii) show ridge trades the unbiasedness of OLS for lower variance
> (Lecture 5). These are the 5990V-level expectations for regression.

## Connection to This Week

- **Lab 1** — fit linear regression, train/test split, and read **learning curves**.
- **Homework 1** — regression concepts + a fit on real data; due Mon Aug 31.
- **Quiz 1** — MSE vs MAE, reading learning curves.

## References

- Weekly reading map, Week 2.
- MFDL Ch. 2; ESL Ch. 3.1-3.2.
- PRML Ch. 3.1-3.2 and ESL Ch. 7.2-7.3 for deeper bias-variance treatment.
- scikit-learn Linear Models user guide.
