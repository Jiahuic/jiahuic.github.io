---
layout: page
permalink: /teaching/math-4990v5990v/03-gradient-descent/
title: "Lecture 3 — Gradient Descent, Convergence, PyTorch Tensors, and Activation Functions"
---


*Part I · Week 3*

## Learning Goals

- State the gradient descent update and prove it decreases a smooth loss (**descent lemma**).
- Define **convexity**, **$L$-smoothness**, and **$\mu$-strong convexity**, and state the resulting convergence **rates**.
- Derive the exact **stability threshold** and convergence rate of GD on a quadratic, and connect it to the condition number.
- Distinguish batch / stochastic / mini-batch GD and state the SGD step-size conditions.
- Compute gradients automatically with **PyTorch tensors and autograd**.
- Explain why a linear model needs a nonlinear **activation function** to become a classifier — the bridge to Lecture 4.

## 1. Why Gradient Descent?

The closed form $\hat\theta = (X^\top X)^{-1}X^\top y$ costs $O(np^2 + p^3)$ and needs the inverse — expensive for large $p$ and **impossible** for models with no closed form (logistic regression, neural nets). Gradient descent is the general first-order method and the workhorse of all of Part II.

## 2. The Update and the Descent Lemma

To minimize $L(\theta)$, step opposite the gradient:

$$
\theta_{t+1} = \theta_t - \eta\, \nabla L(\theta_t), \qquad \eta > 0 \ (\text{learning rate}).
$$

Call $L$ **$L$-smooth** if $\nabla L$ is $L$-Lipschitz: $\lVert \nabla L(a) - \nabla L(b)\rVert \le L\lVert a-b\rVert$. Smoothness gives the quadratic upper bound $L(\theta') \le L(\theta) + \nabla L(\theta)^\top(\theta'-\theta) + \tfrac{L}{2}\lVert\theta'-\theta\rVert^2$. Substituting the GD step $\theta' = \theta - \eta\nabla L(\theta)$:

$$
L(\theta_{t+1}) \le L(\theta_t) - \eta\Big(1 - \tfrac{L\eta}{2}\Big)\lVert \nabla L(\theta_t)\rVert^2 .
$$

**Descent lemma:** for any $\eta \le 1/L$ the bracket is $\ge \tfrac12$, so the loss **strictly decreases** until $\nabla L = 0$. This is the precise version of "a small enough step reduces $L$." For least squares, $\nabla L(\theta) = -\tfrac{2}{n}X^\top(y - X\theta)$.

## 3. Convergence Rates (MFDL Ch. 17)

Two extra assumptions sharpen the guarantee. $L$ is **convex** if it lies above its tangents; it is **$\mu$-strongly convex** if $L(\theta) - \tfrac{\mu}{2}\lVert\theta\rVert^2$ is still convex ($\nabla^2 L \succeq \mu I$).

| Assumptions | Step size | Rate to reach $L(\theta_t)-L^\star \le \epsilon$ |
|---|---|---|
| convex, $L$-smooth | $\eta = 1/L$ | $O(1/t)$ — sublinear |
| $\mu$-strongly convex, $L$-smooth | $\eta = 1/L$ | $\big(1-\tfrac{\mu}{L}\big)^{t}$ — **linear** (geometric) |

The strongly convex rate depends on the **condition number** $\kappa = L/\mu = \lambda_{\max}/\lambda_{\min}$ of the Hessian: large $\kappa$ ⇒ contraction factor $\approx 1 - 1/\kappa \to 1$ ⇒ slow. This is the theoretical reason scaling matters (Section 5).

## 4. The Quadratic Case, Exactly (exam material)

Least squares has constant Hessian $H = \tfrac{2}{n}X^\top X$ with eigenvalues $0 < \lambda_{\min} \le \dots \le \lambda_{\max}$. GD becomes the linear recursion $\theta_{t+1} - \theta^\star = (I - \eta H)(\theta_t - \theta^\star)$, so along eigenvector $j$ the error scales by $(1 - \eta\lambda_j)$ each step. Convergence needs $|1-\eta\lambda_j|<1$ for all $j$, i.e.

$$
\boxed{\ 0 < \eta < \frac{2}{\lambda_{\max}}\ } \qquad(\text{diverges for } \eta > 2/\lambda_{\max}).
$$

The optimal step $\eta^\star = \tfrac{2}{\lambda_{\min}+\lambda_{\max}}$ gives contraction factor $\tfrac{\kappa-1}{\kappa+1}$. This is not hand-waving: in Lab 2 the standardized California design has $\lambda_{\max}\approx 4.05$, so the threshold is $2/\lambda_{\max}\approx 0.49$ — which is exactly why `lr=0.1` converges and `lr=1.0` diverges.

## 5. Why Scaling Helps

Unscaled features give $X^\top X$ a large eigenvalue spread → large $\kappa$ → a stretched, elongated bowl. The largest safe step $2/\lambda_{\max}$ is tiny relative to the flat directions, so GD zig-zags and crawls. Standardizing (Lecture 1) equalizes the eigenvalues, shrinking $\kappa$ toward 1 so one $\eta$ works everywhere. Full whitening gives $\kappa=1$ and one-step convergence for a quadratic.

## 6. Batch, Stochastic, Mini-batch (MFDL Ch. 7)

Computing the full-batch gradient costs $O(np)$ per step. **Stochastic gradient descent (SGD)** uses one example (or a mini-batch $B$): $\nabla L_{i}(\theta)$ is a **random, unbiased** estimate, $\mathbb{E}_i[\nabla L_i(\theta)] = \nabla L(\theta)$, at cost $O(p)$.

- **Batch:** stable, expensive; exact gradient.
- **SGD (one point):** cheap, scalable, but noisy — the iterate rattles around the optimum.
- **Mini-batch ($B\approx 32$):** the standard compromise; variance $\propto 1/B$.

With a **constant** step, SGD converges only to a noise ball of radius $\propto \eta$. Convergence to the exact optimum needs a **decreasing** schedule satisfying the Robbins–Monro conditions $\sum_t \eta_t = \infty,\ \sum_t \eta_t^2 < \infty$ (e.g. $\eta_t \propto 1/t$). One **epoch** = one full pass over the data. Momentum and Adam (Lecture 12) accelerate by reusing past gradients.

## 7. Worked Example (one step by hand)

Minimize $L(\theta)=\theta^2$ ($\nabla L = 2\theta$, so $L=2$, threshold $\eta<1$) from $\theta_0=5$:
- $\eta=0.1$: $\theta_1 = 5-0.1(10)=4$, $\theta_2=3.2$, $\theta_3=2.56,\dots\to 0$ (factor $1-2\eta=0.8$).
- $\eta=1.1 > 1$: $\theta_1 = 5-1.1(10)=-6$, $\theta_2 = 7.2,\dots$ — magnitude **grows**: divergence, exactly as $|1-2\eta|=1.2>1$ predicts.

## 8. Implementation: Least Squares by Gradient Descent (NumPy)

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearRegressionGD:
    def __init__(self, n_features):
        self.coef_ = np.zeros(n_features); self.intercept_ = 0.0
        self.scaler = StandardScaler()

    def fit(self, X, y, lr=0.1, epochs=1000):
        X = self.scaler.fit_transform(X); n = X.shape[0]
        for _ in range(epochs):
            error = X @ self.coef_ + self.intercept_ - y          # (n,)
            self.coef_     -= lr * (2/n) * (X.T @ error)
            self.intercept_ -= lr * (2/n) * error.sum()
        return self

    def predict(self, X):
        return self.scaler.transform(X) @ self.coef_ + self.intercept_
```

## 9. PyTorch Tensors and Autograd

A `torch.Tensor` is NumPy's array plus two things: it can live on a GPU, and it can **record operations** for automatic differentiation. Setting `requires_grad=True` tells PyTorch to build a computation graph as you compute; calling `.backward()` walks that graph in reverse (reverse-mode automatic differentiation) and fills in `.grad` on every leaf tensor — no hand-derived formulas. This is the *same* gradient as Section 2, computed mechanically; it is the engine that trains every network in Part II.

```python
import torch
w = torch.zeros(X.shape[1], 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(1000):
    loss = ((X @ w + b - y) ** 2).mean()   # MSE, built from graph-tracked ops
    loss.backward()                          # autograd fills w.grad, b.grad
    with torch.no_grad():                    # updates themselves must NOT be tracked
        w -= 0.1 * w.grad; b -= 0.1 * b.grad
        w.grad.zero_(); b.grad.zero_()       # gradients accumulate -> must reset
```

`with torch.no_grad()` matters: the update step `w -= ...` is arithmetic we do *to* the parameters, not part of the loss's computation graph — tracking it would be wrong (and wasteful). `.grad.zero_()` matters because `.backward()` **accumulates** into `.grad` by design (useful for gradient accumulation over multiple mini-batches); forgetting it silently sums gradients across steps.

> If `X` is not standardized, $\lambda_{\max}$ blows up, the safe step $2/\lambda_{\max}$ shrinks, and training destabilizes — a live demonstration of Sections 4–5, whether you compute the gradient by hand or via autograd.

## 10. Activation Functions: A Bridge to Classification

Everything so far predicts a real number, $\hat y = \theta_0 + w^\top x \in \mathbb{R}$ — fine for regression, wrong for classification, where we want a *probability* in $[0,1]$. The fix is to pass the linear score $z = \theta_0+w^\top x$ through a nonlinear **activation function** $g$: $\hat y = g(z)$.

- **Sigmoid** $\sigma(z) = 1/(1+e^{-z})$ squashes $\mathbb{R}\to(0,1)$ — a valid probability. This is exactly what Lecture 4 uses to build logistic regression.
- **ReLU** $g(z)=\max(0,z)$ is the default *hidden-layer* activation in networks (Lecture 11) — cheap and avoids the vanishing-gradient problem that deep sigmoid stacks suffer.
- **Why nonlinearity matters at all:** composing two *linear* maps is still linear ($W_2(W_1x)=(W_2W_1)x$), so without a nonlinear $g$, stacking layers would buy nothing. The activation is what lets a model represent curved decision boundaries and, eventually (stacked across layers), arbitrarily complex functions (universal approximation, Lecture 11).

Swapping the identity activation for $\sigma$ and the squared-error loss for cross-entropy is the entire conceptual jump from this lecture to the next.

> **Graduate depth.** MFDL Ch. 17.1–17.3 proves the $O(1/t)$ (convex) and linear (strongly convex)
> rates via the descent lemma; MFDL Ch. 7 covers SGD and the Robbins–Monro conditions. Be able to
> derive the quadratic stability threshold $\eta < 2/\lambda_{\max}$ and the optimal-rate $\tfrac{\kappa-1}{\kappa+1}$.

## Connection to This Week

- **Lab 2** — implement gradient descent from scratch and with PyTorch tensors/autograd; sweep the learning rate.
- **Quiz 2** — learning rate, epochs, scaling, convergence vs. divergence.
- **Homework 1 due / Homework 2 assigned** — due Wed Sep 9.

## References

- Weekly reading map, Week 3.
- MFDL Ch. 7.1-7.3 for stochastic gradient descent basics.
- MFDL Ch. 17.1-17.3 for graduate-level gradient-descent convergence.
- PyTorch tensors and autograd tutorials.
