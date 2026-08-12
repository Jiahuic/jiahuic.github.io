---
layout: page
permalink: /teaching/math-4990v5990v/16-generative-models/
title: "Lecture 16 — Generative Models: GANs, Diffusion, and Autoregressive Models"
---


*Part II · Weeks 15–16*

## Learning Goals

- Classify a generative model as **likelihood-based**, **implicit**, or **score/diffusion-based**, and name one example of each.
- State the **GAN minimax objective** and explain mode collapse and generator gradient vanishing.
- **Derive** the diffusion forward process's closed form for $x_t$ given $x_0$.
- Describe the reverse (denoising) process at a conceptual level and its connection to score matching.
- Sketch a minimal training step for a diffusion model in PyTorch.

## 1. Three Families of Generative Models

Lecture 15 built one generative model — a VAE — from the marginalization identity $p(x)=\int p(x\mid z)p(z)\,dz$ (Lecture 7 §6's Bayes' rule review). That identity is one of three broad strategies for building $p_\theta(x)$ or a sampler from it:

| Family | Idea | Examples |
|---|---|---|
| **Likelihood-based** | model $p_\theta(x)$ (or a tractable piece of it) directly, train by maximum likelihood / ELBO | autoregressive models, VAE (Lecture 15) |
| **Implicit** | never write down $p_\theta(x)$; train a sampler via an adversarial game | GAN |
| **Score/diffusion-based** | learn to reverse a fixed noising process | DDPM-style diffusion models |

All three ultimately answer the same question as every other model in this course — "what distribution best explains the data, and how do I fit its parameters by gradient descent?" — but differ sharply in *what* is tractable and *what* the loss looks like.

## 2. Likelihood-Based: Autoregressive Models and the VAE Recap

**Autoregressive models** factor the joint distribution using the chain rule of probability (Lecture 7 §6) with no approximation:

$$
p_\theta(x) = \prod_{i=1}^{n} p_\theta\big(x_i \mid x_{<i}\big),
$$

e.g. pixel-by-pixel or token-by-token, each conditional modeled by a network (this is the mechanism behind autoregressive language models). The likelihood is **exact** and directly optimizable by maximum likelihood — no lower bound needed — at the cost of strictly sequential, $O(n)$ generation.

The **VAE** (Lecture 15) is likelihood-based too, but the marginal $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$ is intractable, so training optimizes the ELBO instead of the exact likelihood. Recall the two-term decomposition (Lecture 15 §3): reconstruction quality traded against how close the encoder's posterior stays to the prior.

## 3. Implicit Models: GANs and the Minimax Objective (exam material)

A **generative adversarial network (GAN)** never evaluates $p_\theta(x)$ at all. Instead, a **generator** $G_\theta(z)$ maps noise $z\sim p(z)$ (same simple prior idea as Lecture 15 §2) to a sample, and a **discriminator** $D_\phi(x)$ tries to distinguish real data from $G_\theta(z)$'s output. They are trained against each other:

$$
\min_\theta \max_\phi\ \mathbb E_{x\sim p_{\text{data}}}\big[\log D_\phi(x)\big] + \mathbb E_{z\sim p(z)}\big[\log\big(1 - D_\phi(G_\theta(z))\big)\big].
$$

**Reading the objective.** For a *fixed* generator, the optimal discriminator is $D^\star(x) = \dfrac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_\theta(x)}$ (a ratio of densities — a Bayes-optimal classifier between the two distributions). Substituting $D^\star$ back in shows the generator's objective reduces to minimizing $2\cdot\mathrm{JS}(p_{\text{data}}\Vert p_\theta) - \log 4$, the **Jensen–Shannon divergence** between the real and generated distributions — a symmetrized relative of the KL divergence from Lecture 7 §6.

**Why training is unstable.** Early in training, $D_\phi$ easily tells real from fake ($D_\phi(G_\theta(z))\approx 0$), so $\log(1-D_\phi(G_\theta(z)))$ sits in a region where its gradient with respect to $\theta$ is nearly flat — the generator's learning signal **vanishes** exactly when it needs it most (the practical fix is to instead maximize $\log D_\phi(G_\theta(z))$, a non-saturating alternative with the same fixed point). **Mode collapse** is the complementary failure: $G_\theta$ discovers a handful of outputs that reliably fool $D_\phi$ and stops exploring the rest of $p_{\text{data}}$'s support, so the minimax game has reached a poor equilibrium rather than the intended $p_\theta = p_{\text{data}}$ global solution.

## 4. Score/Diffusion-Based Models: Forward and Reverse Process

**Diffusion models** define a fixed **forward process** that gradually destroys structure by adding Gaussian noise over $T$ steps,

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \qquad \epsilon_t \sim \mathcal N(0,I),\ \ t=1,\dots,T,
$$

for a small noise schedule $\beta_t\in(0,1)$, until $x_T$ is essentially pure noise. Generation then learns the **reverse process** — a network that predicts (or removes) the noise added at each step, turning noise back into data one small denoising step at a time. This is a direct descendant of **score matching**: the reverse step is, up to a reparameterization, an estimate of the score function $\nabla_x \log p_t(x)$ of the noised data distribution at each noise level, i.e. "which direction increases the density."

## 5. The Forward Process in Closed Form (exam material)

Training would be impossibly slow if computing $x_t$ required simulating all $t$ forward steps for every training example. It doesn't have to: define $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. **Claim:**

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \qquad \epsilon\sim\mathcal N(0,I) .
$$

**Derivation by induction.** True at $t=1$ by definition ($\bar\alpha_1=\alpha_1$). Assume it holds at $t-1$: $x_{t-1} = \sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon'$ for $\epsilon'\sim\mathcal N(0,I)$. Substitute into the forward step:

$$
x_t = \sqrt{\alpha_t}\Big(\sqrt{\bar\alpha_{t-1}}x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon'\Big) + \sqrt{1-\alpha_t}\,\epsilon_t
= \sqrt{\alpha_t\bar\alpha_{t-1}}\,x_0 + \Big(\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\epsilon' + \sqrt{1-\alpha_t}\,\epsilon_t\Big).
$$

The bracketed term sums two independent Gaussians, so it is itself Gaussian with mean $0$ and variance $\alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t) = 1-\alpha_t\bar\alpha_{t-1} = 1-\bar\alpha_t$ — i.e. the bracket equals $\sqrt{1-\bar\alpha_t}\,\epsilon$ for a single $\epsilon\sim\mathcal N(0,I)$ (the reparameterization trick of Lecture 15 §5 again, used here to *collapse* $t$ noise draws into one). And $\sqrt{\alpha_t\bar\alpha_{t-1}} = \sqrt{\bar\alpha_t}$ by the definition of $\bar\alpha_t$. This proves the claim, and it means **any** $x_t$ can be sampled in one shot from $x_0$ — the fact that makes diffusion training tractable: pick a random $t$, add the corresponding closed-form noise, and train a network to predict $\epsilon$ from $x_t$.

## 6. Worked Example (closed-form noise level)

Take a 3-step schedule $\beta_1=0.1,\ \beta_2=0.2,\ \beta_3=0.3$, so $\alpha_1=0.9,\ \alpha_2=0.8,\ \alpha_3=0.7$. Then $\bar\alpha_3 = 0.9\times0.8\times0.7 = 0.504$, so

$$
x_3 = \sqrt{0.504}\,x_0 + \sqrt{1-0.504}\,\epsilon \approx 0.710\, x_0 + 0.704\,\epsilon .
$$

At $t=3$ the signal ($x_0$) and noise ($\epsilon$) contribute almost equally — already close to a 50/50 mix after just 3 steps, illustrating how quickly a modest noise schedule erases the original signal.

## 7. Implementation Sketch (diffusion training step, PyTorch)

```python
import torch

def diffusion_loss(eps_model, x0, alpha_bar, T):
    B = x0.shape[0]
    t = torch.randint(1, T + 1, (B,))                       # random step per example
    eps = torch.randn_like(x0)
    abar_t = alpha_bar[t].view(B, *([1] * (x0.dim() - 1)))  # broadcast
    x_t = abar_t.sqrt() * x0 + (1 - abar_t).sqrt() * eps     # Section 5's closed form
    eps_hat = eps_model(x_t, t)                              # network predicts the noise
    return torch.nn.functional.mse_loss(eps_hat, eps)
```

One training step needs no simulation loop over $t$ steps — a direct payoff of Section 5's closed form. Sampling (generation) does need the sequential reverse process: start from $x_T\sim\mathcal N(0,I)$ and repeatedly apply the trained denoiser.

## 8. Comparing the Three Families

| | Likelihood (autoregressive) | Likelihood (VAE) | Implicit (GAN) | Score/diffusion |
|---|---|---|---|---|
| Trains by | exact log-likelihood | ELBO (lower bound) | minimax game | denoising regression |
| Sampling cost | sequential, $O(n)$ | one decoder pass | one generator pass | sequential, $O(T)$ |
| Common failure | slow generation | blurry samples (Section 3's JS-divergence-adjacent intuition applies loosely here too) | mode collapse, unstable training | slow sampling, schedule tuning |

> **Graduate depth.** Goodfellow et al. (2014), *Generative Adversarial Networks*, derives the
> optimal-discriminator and JS-divergence results of Section 3; Ho et al. (2020), *Denoising
> Diffusion Probabilistic Models*, develops the forward/reverse process of Sections 4–5 and its
> score-matching connection; MFDL Ch. 15 surveys generative models in the course's functional
> language. Be able to derive the closed-form forward process by induction and explain precisely
> why the naive GAN generator loss saturates early in training.

## Connection to This Week

- No new lab or quiz these weeks (Weeks 15–16; Week 15 meets only Monday for Thanksgiving).
- **Homework 11** (assigned with Lecture 15, due Mon Nov 30) extends into this lecture's material —
  the VAE-to-generative-models capstone assignment.
- **Exam 3 (Wed Dec 2)** covers this material conceptually, alongside Lectures 9–15.

## References

- Weekly reading map, Weeks 15–16.
- Goodfellow et al. (2014), *Generative Adversarial Networks*.
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models*.
- MFDL Ch. 15 for a survey of generative models.
