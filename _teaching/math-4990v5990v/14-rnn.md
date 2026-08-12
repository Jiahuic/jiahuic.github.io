---
layout: page
permalink: /teaching/math-4990v5990v/14-rnn/
title: "Lecture 14 — Recurrent Neural Networks"
---


*Part II · Week 13*

## Learning Goals

- Explain the RNN recurrence and **backprop through time (BPTT)**.
- Derive the vanishing/exploding-gradient math as a **product of Jacobians**.
- Explain how **LSTM** gating creates a gradient highway that fixes it.
- Describe, at a conceptual level, how **attention** lets a model bypass recurrence entirely.

## 1. RNNs and Backprop Through Time

**Recurrent networks** carry a hidden state across a sequence $x_1,\dots,x_T$ with **shared weights over time**:

$$
h_t = g\big(W_h h_{t-1} + W_x x_t + b\big).
$$

Training uses **backpropagation through time**: unroll the recurrence into a $T$-layer feedforward network (each "layer" reusing the same $W_h, W_x$), then backprop as usual (Lecture 11 §3). The gradient of a late loss w.r.t. an early state is a **product of Jacobians**:

$$
\frac{\partial h_t}{\partial h_s} = \prod_{r=s+1}^{t} \operatorname{diag}\big(g'(\cdot)\big)\, W_h^\top .
$$

If the relevant singular values of $W_h$ are $<1$ this product shrinks geometrically → **vanishing gradients** (no long-range learning); if $>1$ it blows up → **exploding gradients** (clipped in practice). This is the same product-of-many-terms fragility as deep feedforward nets (Lecture 11 §4), except here "depth" is the sequence length $T$, so even a modest network becomes very deep along the time axis.

## 2. Worked Example: Vanishing and Exploding Gradients (exam material)

Take the scalar case ($h_t = W_h h_{t-1}$, linear activation $g'=1$, so there's nothing to obscure the Jacobian product) and a 5-step sequence, $s=0,\ t=5$:

$$
\frac{\partial h_5}{\partial h_0} = W_h^5 .
$$

- $W_h = 0.5$: $\;0.5^5 = 0.03125$ — the gradient from step 5 back to step 0 is scaled down by **97%**; a signal that should influence early weights has essentially vanished after 5 steps.
- $W_h = 1.5$: $\;1.5^5 \approx 7.59$ — the gradient **grows** by a factor of ~7.6 over the same 5 steps; over 20 steps this reaches $1.5^{20}\approx 3325$, numerically unstable.
- $W_h = 1$: $\;1^5=1$ — gradients neither vanish nor explode, the knife-edge case gating (Section 3) tries to approximate.

Real (matrix, nonlinear) RNNs behave the same way through the largest singular value of $W_h$ combined with $g'(\cdot)\in[0,1]$ for saturating activations — which only makes the shrinkage worse, since $g'\le 1$ multiplies in on every step alongside $W_h$.

## 3. LSTMs: A Gradient Highway

**LSTMs** add a **cell state** $c_t$ updated *additively* through gates (input $i_t$, forget $f_t$, output $o_t$):

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t, \qquad h_t = o_t \odot \tanh(c_t).
$$

Because $c_t$ depends on $c_{t-1}$ through a near-identity (gated) path rather than a repeated matrix multiply, $\partial c_t/\partial c_{t-1}\approx f_t$ — gradients flow across many steps without vanishing when $f_t\approx 1$, sidestepping the $W_h^t$ product of Section 2 entirely. GRUs are a lighter variant. This is what let recurrent models capture long-range dependencies before transformers.

## 4. Attention: Bypassing Recurrence

**Attention** lets each output position read from *all* input positions by relevance, instead of squeezing history into one recurrent state $h_t$. With queries $Q$, keys $K$, values $V$ (rows = positions):

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V .
$$

- **Why $\sqrt{d_k}$:** for unit-variance entries the dot product $q^\top k$ has variance $d_k$; dividing by $\sqrt{d_k}$ keeps the softmax logits $O(1)$ so it does not saturate into vanishing gradients (the same saturation concern as Section 2, now in the softmax rather than in $g'$).
- **Cost:** the $QK^\top$ matrix is $T\times T$, so attention is $O(T^2 d)$ — quadratic in sequence length but **fully parallel** across positions (unlike an RNN's inherently sequential $O(T)$, which cannot start step $t$ before step $t-1$ finishes).
- The **transformer** ("Attention Is All You Need," Vaswani et al. 2017) stacks self-attention + feedforward blocks, dropping recurrence — and its Jacobian-product problem — entirely; it underlies modern large language models.

| Data modality | Typical choice | Inductive bias |
|---|---|---|
| Short sequences | RNN/LSTM or 1D CNN | temporal locality |
| Long sequences / language | Transformer | all-pairs relations |

## 5. Implementation Sketch (RNN in PyTorch)

```python
import torch.nn as nn
rnn = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
head = nn.Linear(32, 1)
# x: (batch, seq_len, 10) -> out: (batch, seq_len, 32), (h_T, c_T)
out, (h_T, c_T) = rnn(x)
y_hat = head(h_T.squeeze(0))          # use the final hidden state for a sequence-level prediction
```

`nn.LSTM` implements exactly the gated recursion of Section 3; `h_T` is $h_t$ at the last time step, the RNN analogue of the "final layer output" in a feedforward net.

> **Graduate depth.** MFDL Ch. 13 develops RNNs, LSTMs, and attention/transformers; PRML Ch. 13
> covers sequential-data background. Be able to derive the BPTT Jacobian product, explain why
> $|W_h|<1$ vanishes and $|W_h|>1$ explodes, explain the LSTM's near-identity gradient path, and
> derive the $\sqrt{d_k}$ scaling in scaled dot-product attention (Vaswani et al., 2017).

## Connection to This Week

- **Lab 10** — train a small RNN/LSTM on a sequence task in PyTorch.
- **Quiz 10** — BPTT, vanishing/exploding gradients, training diagnostics for sequence models.
- **Homework 9 due / Homework 10 assigned** — due Mon Nov 16.

## References

- Weekly reading map, Week 13.
- MFDL Ch. 13 for RNNs, LSTMs, and attention/transformers.
- PRML Ch. 13 for sequential-data background.
- Vaswani et al. (2017), *Attention Is All You Need*; PyTorch sequence-model tutorials.
