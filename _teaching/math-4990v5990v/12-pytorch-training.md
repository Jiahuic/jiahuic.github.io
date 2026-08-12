---
layout: page
permalink: /teaching/math-4990v5990v/12-pytorch-training/
title: "Lecture 12 — PyTorch Modules, Optimizers, and Training Dynamics"
---


*Part II · Week 11*

## Learning Goals

- Build models with `torch.nn.Module` and write the standard training loop.
- State the update rules for **SGD, momentum, and Adam** and what each fixes.
- Explain neural-net **regularization** mathematically: weight decay, dropout, early stopping, batch norm.
- Diagnose training from the loss curve (the engineering view).

## 1. The Five Pieces of Any Training Job

1. **Data** — tensors, served in mini-batches by a `DataLoader`.
2. **Model** — an `nn.Module` mapping inputs to outputs.
3. **Loss** — `nn.MSELoss`, `nn.CrossEntropyLoss`, …
4. **Optimizer** — `SGD`, `Adam`, … holding the learning rate.
5. **Loop** — forward → loss → `backward()` → `step()` → `zero_grad()`.

Same gradient-descent recipe as Lecture 3, now automated (autograd) and batched (SGD).

## 2. Implementation: Model and Training Loop

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
model  = MLP(X_train.shape[1], 64, n_classes)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()          # clear old gradients
        loss = loss_fn(model(xb), yb)
        loss.backward()          # autograd: gradients
        opt.step()               # update parameters
```

**Why `zero_grad()`?** `backward()` *accumulates* into `.grad`; without zeroing, gradients from previous batches sum and corrupt the step (Quiz 11).

## 3. Optimizers: SGD, Momentum, Adam

All descend the loss but differ in how they use the (stochastic) gradient $g_t = \nabla L_{\text{batch}}(\theta_t)$.

**SGD.** $\theta_{t+1} = \theta_t - \eta\, g_t$. Unbiased but noisy (Lecture 3 §6); struggles in ill-conditioned "ravines," oscillating across the steep direction.

**Momentum (heavy ball).** Accumulate an exponentially-weighted velocity:

$$
v_{t} = \beta v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t \quad (\beta\approx 0.9).
$$

Averaging successive gradients cancels oscillation and accelerates along consistent directions — an $O(1/t^2)$ accelerated rate (Nesterov) versus SGD's $O(1/t)$ on convex problems.

**Adam (adaptive moments).** Track first and second gradient moments with bias correction:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$
$$
\hat m_t = \tfrac{m_t}{1-\beta_1^t}, \quad \hat v_t = \tfrac{v_t}{1-\beta_2^t}, \qquad
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Dividing by $\sqrt{\hat v_t}$ gives each parameter its **own effective step**, so Adam is robust to bad scaling and the common default. Learning-rate **schedules** (step decay, cosine, warmup) further help.

## 4. Regularization in Networks (the math)

Networks are over-parameterized, so controlling variance (Lecture 2) is central:

- **Weight decay** = L2 penalty $\tfrac{\lambda}{2}\lVert\theta\rVert^2$; its gradient adds $\lambda\theta$, shrinking weights each step (ridge for nets, Lecture 5).
- **Dropout** zeroes each unit independently with probability $p$ during training; it trains an *exponential ensemble* of sub-networks that share weights, and at test time scaling by $(1-p)$ uses the **expected** activation. A stochastic, cheap analogue of bagging (Lecture 10).
- **Early stopping** halts when validation loss rises; for convex quadratics it is provably equivalent to L2 regularization with $\lambda$ tied to the number of steps — *implicit* regularization.
- **Batch normalization** standardizes each layer's pre-activations per mini-batch (Lecture 1's standardization idea, inside the network), stabilizing and speeding training.

## 5. Diagnostics

```python
model.eval()
with torch.no_grad():                          # no gradient tracking at eval
    val_acc = (model(X_val).argmax(1) == y_val).float().mean()
```

Track **train and validation loss per epoch**: training loss flat → LR too small, bad init, or a bug; validation loss rising while training falls → **overfitting** (add weight decay / dropout / early stopping / data). `CrossEntropyLoss` expects **raw logits** (it applies log-softmax internally) — do not add a softmax layer before it (Quiz 11).

| Knob | Effect |
|---|---|
| learning rate | too high → diverge; too low → slow |
| batch size | larger → smoother, fewer steps/epoch; smaller → noisier, more regularizing |
| epochs | too few → underfit; too many → overfit |
| optimizer | Adam adapts step sizes; SGD+momentum a robust baseline |
| width / depth | capacity vs. overfitting |

## 6. Worked Example (sanity check)

A one-layer `nn.Linear(p, 1)` with `MSELoss` and Adam on standardized California housing converges to essentially scikit-learn's `LinearRegression` MSE — the network machinery reduces to the least-squares gradient descent of Lecture 3 when the model is linear.

> **Graduate depth.** MFDL Ch. 7–8 cover SGD, momentum/adaptive methods, and multilayer training
> dynamics. Be able to write the momentum and Adam updates, explain bias correction, and justify
> dropout as an expectation over sub-networks and early stopping as implicit L2.

## Connection to This Week

- **Lab 8** — train an MLP classifier in PyTorch; tune the loop.
- **Quiz 8** — training loop mechanics (zero_grad, batch size, loss not decreasing).
- **Homework 8** — networks, softmax, and training-loop diagnostics (implemented in PyTorch); due Mon Nov 2.

## References

- Weekly reading map, Week 11.
- MFDL Ch. 7-8 on SGD and multilayer training.
- PyTorch Quickstart, Autograd, `nn`, and Optimization tutorials.
