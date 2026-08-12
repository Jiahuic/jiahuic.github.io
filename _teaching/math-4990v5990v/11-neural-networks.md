---
layout: page
permalink: /teaching/math-4990v5990v/11-neural-networks/
title: "Lecture 11 — Neural Networks and Backpropagation"
---


*Part II · Week 11*

## Learning Goals

- Write a feedforward network as a composition of affine maps and nonlinear activations.
- State the **universal approximation** theorem and the role of **depth**.
- **Derive** the backpropagation recursion from the chain rule.
- Explain the **non-convex** loss surface, activation-derivative issues, and initialization.
- Count parameters and reason about compute cost.

## 1. From Linear Models to Networks

Logistic regression is a single neuron $\hat y = \sigma(w^\top x + b)$. A network **composes** layers of such units. With $a^{(0)}=x$, layer $l$ computes

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = g\big(z^{(l)}\big),
$$

and $\hat y = a^{(L)}$. Everything from Part I carries over: this is a parameterized $f_\theta$ trained by minimizing a loss with gradient descent (Lecture 3) — MSE for regression, cross-entropy for classification (Lecture 4).

## 2. Why Non-Linearity, and How Much a Network Can Represent

Without $g$, layers collapse: $W_2(W_1 x) = (W_2 W_1)x$ — still linear. The activation is what buys nonlinearity.

- **ReLU** $g(z)=\max(0,z)$, $g'(z)=\mathbb 1[z>0]$ — the default; cheap, mitigates vanishing gradients.
- **sigmoid/tanh** — squashing; $\sigma'=\sigma(1-\sigma)$ is $\le \tfrac14$, so deep sigmoid stacks suffer **vanishing gradients**.
- **softmax** at the output for class probabilities (Lecture 4).

**Universal approximation (Cybenko/Hornik).** A network with a *single* hidden layer and enough units can approximate any continuous function on a compact set to arbitrary accuracy. Existence is not efficiency, though: some functions need exponentially many units in one layer but only polynomially many with **depth**, which is the practical case for depth.

## 3. Backpropagation: Derivation (exam material)

We need $\partial L/\partial W^{(l)}$ and $\partial L/\partial b^{(l)}$ for every layer. Define the **error signal** $\delta^{(l)} = \partial L/\partial z^{(l)}$. Two chain-rule steps give everything.

**Output layer.** $L$ depends on $z^{(L)}$ through $a^{(L)}=g(z^{(L)})$, so

$$
\delta^{(L)} = \frac{\partial L}{\partial a^{(L)}}\odot g'\big(z^{(L)}\big) = \nabla_{\hat y}L \odot g'\big(z^{(L)}\big).
$$

**Recursion.** $z^{(l)}$ influences $L$ only through $z^{(l+1)} = W^{(l+1)}g(z^{(l)}) + b^{(l+1)}$. By the chain rule,

$$
\delta^{(l)} = \Big(\frac{\partial z^{(l+1)}}{\partial z^{(l)}}\Big)^{\!\top}\delta^{(l+1)}
= \big(W^{(l+1)\top}\delta^{(l+1)}\big)\odot g'\big(z^{(l)}\big).
$$

**Parameter gradients.** Since $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$,

$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \big(a^{(l-1)}\big)^\top, \qquad
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)} .
$$

One **forward pass** stores the $a^{(l)}$; one **backward pass** propagates $\delta^{(l)}$ from $L$ to $1$. Both cost $O(\#\text{params})$ — this is exactly reverse-mode automatic differentiation (PyTorch autograd, Lecture 3 §9). You rarely write these by hand, but must be able to.

## 4. The Loss Surface Is Non-Convex

Unlike Parts I's convex objectives, a network's loss is **non-convex** in $\theta$: many equivalent minima (from permuting/rescaling hidden units) and abundant **saddle points**. In high dimensions saddles, not bad local minima, dominate, and SGD's noise (Lecture 3 §6) helps escape them; over-parameterized networks empirically reach near-global minima. **Initialization** sets the gradient scale: Xavier/Glorot (variance $\propto 1/n_{\text{in}}$) for tanh, He (variance $\propto 2/n_{\text{in}}$) for ReLU, keeps signal and gradient magnitudes stable across layers.

## 5. Counting Parameters

A dense layer $m\to n$ has $n(m+1)$ parameters (weights + biases). Network $4\to 2\to 3\to 1$: $2(4{+}1)+3(2{+}1)+1(3{+}1)=10+9+4=\mathbf{23}$.

## 6. Worked Example (tiny backprop)

One neuron, MSE, single example: $\hat y = \sigma(wx+b)$, $L=\tfrac12(\hat y-y)^2$. Then $\delta = (\hat y - y)\sigma'(wx+b)$ and

$$
\frac{\partial L}{\partial w} = \delta\, x = (\hat y - y)\,\sigma'(wx+b)\,x, \qquad \sigma'=\sigma(1-\sigma).
$$

This is exactly the logistic-regression gradient (Lecture 4): a network is a composition of these, and backprop is the bookkeeping that reuses shared subexpressions.

## 7. Implementation Preview (PyTorch)

```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(4, 16), nn.ReLU(),
    nn.Linear(16, 3))          # 3-class output; use CrossEntropyLoss (applies softmax to logits)
```

> **Graduate depth.** MFDL Ch. 5–6 develop feedforward networks, backprop, and approximation
> theory; PRML Ch. 5.1–5.3 give the same with a probabilistic slant; ESL Ch. 11 the statistical view.
> Be able to derive the $\delta$ recursion, state universal approximation, and explain why depth,
> initialization, and SGD noise matter on a non-convex surface.

## Connection to This Week

- **Lab 8** — build and train a neural network in PyTorch.
- **Homework 8** — network expressions, parameter counts, softmax properties; due Mon Nov 2.
- **Exam 3 (Week 16)** — forward pass, backprop by hand, parameter counting.

## References

- Weekly reading map, Week 11.
- MFDL Ch. 5-6; PRML Ch. 5.1-5.3.
- ESL Ch. 11 for a statistical-learning view of neural networks.
- PyTorch `nn` module documentation.
