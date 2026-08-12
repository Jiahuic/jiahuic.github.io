---
layout: page
permalink: /teaching/math-4990v5990v/15-vae/
title: "Lecture 15 — Variational Autoencoders"
---


*Part II · Week 14*

## Learning Goals

- Set up a latent-variable generative model $p_\theta(x) = \int p_\theta(x\mid z)\,p(z)\,dz$ and explain why it is intractable.
- **Derive** the evidence lower bound (ELBO) from Jensen's inequality and from the KL-to-true-posterior decomposition.
- Derive the closed-form KL divergence between two Gaussians used in the VAE loss.
- Explain the **reparameterization trick** and why it is necessary for backpropagation through a sampling step.
- Implement a VAE encoder/decoder pair in PyTorch.

## 1. From Autoencoders to a Generative Model

Lecture 7 §3 showed PCA is a *linear* encoder/decoder pair: $z = U_k^\top x$ (encode), $\hat x = U_k z$ (decode), trained by minimizing reconstruction error. A **variational autoencoder (VAE)** generalizes this in two ways: the encoder and decoder become nonlinear neural networks, and the model is made **generative** — a full probability distribution over $x$, from which we can *sample new data*, not just compress existing data.

The generative story: draw a latent code from a fixed, simple prior, then decode it into data,

$$
z \sim p(z) = \mathcal N(0, I), \qquad x \mid z \sim p_\theta(x\mid z),
$$

so the marginal likelihood of the data is $p_\theta(x) = \int p_\theta(x\mid z)\,p(z)\,dz$ — Bayes' rule's marginalization identity from Lecture 7 §6, with the observed/hidden roles reversed: $z$ is never observed.

## 2. The Model and the Intractable Posterior

$p_\theta(x\mid z)$ (the **decoder**) is a neural network mapping a latent code to (the parameters of) a distribution over $x$, e.g. a Gaussian with mean $g_\theta(z)$ for continuous data. Training by maximum likelihood needs $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$, but this integral has no closed form once $g_\theta$ is a nonlinear network, and the true posterior $p_\theta(z\mid x) = p_\theta(x\mid z)p(z)/p_\theta(x)$ is equally intractable (it needs the same integral in its denominator).

The **VAE's fix**: introduce a second network $q_\phi(z\mid x)$ (the **encoder**), a tractable approximation to the true posterior — typically $q_\phi(z\mid x) = \mathcal N\big(\mu_\phi(x),\ \operatorname{diag}(\sigma_\phi^2(x))\big)$ — and optimize both networks jointly against a tractable lower bound on $\log p_\theta(x)$ instead of the likelihood itself.

## 3. The ELBO Derivation (exam material)

**Via Jensen's inequality.** For any distribution $q_\phi(z\mid x)$,

$$
\log p_\theta(x) = \log\!\int p_\theta(x,z)\,dz = \log \mathbb E_{q_\phi(z\mid x)}\!\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\ \ge\ \mathbb E_{q_\phi(z\mid x)}\!\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] =: \mathcal L(\theta,\phi; x),
$$

using $\log \mathbb E[\cdot] \ge \mathbb E[\log(\cdot)]$ (Jensen, since $\log$ is concave). $\mathcal L(\theta,\phi;x)$ is the **evidence lower bound (ELBO)**.

**Via the KL decomposition (why maximizing it is the right goal).** Expand the KL divergence to the *true* posterior:

$$
\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\, p_\theta(z\mid x)\big)
= \mathbb E_q\!\left[\log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\right]
= \log p_\theta(x) - \mathbb E_q\!\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
= \log p_\theta(x) - \mathcal L(\theta,\phi;x).
$$

Rearranged, $\log p_\theta(x) = \mathcal L(\theta,\phi;x) + \mathrm{KL}\big(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\big)$. Since $\mathrm{KL}\ge 0$ (Lecture 7 §6), $\mathcal L \le \log p_\theta(x)$ — confirming the bound — and **maximizing the ELBO over $\phi$ simultaneously maximizes a lower bound on the likelihood and drives $q_\phi(z\mid x)$ toward the true (intractable) posterior**, since $\log p_\theta(x)$ doesn't depend on $\phi$.

**The trainable form.** Splitting $p_\theta(x,z) = p_\theta(x\mid z)\,p(z)$ inside the ELBO:

$$
\mathcal L(\theta,\phi;x) = \underbrace{\mathbb E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction term}} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\, p(z)\big)}_{\text{regularization term}}.
$$

The first term rewards decoding $z$ back into $x$ accurately (for a Gaussian decoder this is, up to constants, negative squared reconstruction error — the autoencoder loss of Section 1); the second penalizes the encoder's posterior for drifting from the prior $\mathcal N(0,I)$, which is what makes the latent space usable for *sampling* new $z$ at generation time (Lecture 16).

## 4. Closed-Form KL for Gaussians

With $q_\phi(z\mid x) = \mathcal N(\mu,\operatorname{diag}(\sigma^2))$ ($\mu,\sigma\in\mathbb R^d$, coordinatewise) and prior $p(z)=\mathcal N(0,I)$, the KL term has a closed form (no Monte Carlo needed):

$$
\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\, p(z)\big) = \frac{1}{2}\sum_{j=1}^{d}\Big(\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2\Big).
$$

*(Derivation sketch: for 1-D Gaussians $\mathcal N(\mu,\sigma^2)$ vs. $\mathcal N(0,1)$, $\mathrm{KL} = \tfrac12(\sigma^2+\mu^2-1-\log\sigma^2)$ by direct integration of $\mathbb E_q[\log q - \log p]$, using $\mathbb E_q[(z-0)^2]=\sigma^2+\mu^2$; independence across the $d$ coordinates sums the terms.)* Each term is $\ge 0$ and $=0$ exactly at $\mu=0,\sigma=1$ — the KL penalty is zero only when the encoder outputs the prior itself.

## 5. The Reparameterization Trick

Training needs $\nabla_\phi \mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]$, but $z$ is *sampled* from a distribution whose parameters $\phi$ we are differentiating through — sampling is not a differentiable operation, so autograd (Lecture 3 §9) cannot backpropagate through it directly.

**Fix:** rewrite the sample as a deterministic, differentiable function of $\phi$ and an independent noise source:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal N(0, I).
$$

Now the randomness lives entirely in $\epsilon$, which does **not** depend on $\phi$; the path from $\phi$ to $z$ is an ordinary differentiable computation (multiply, add), so `.backward()` (Lecture 3 §9) can flow gradients through $\mu_\phi$ and $\sigma_\phi$ exactly as it would through any other layer. This single algebraic move is what makes VAEs trainable end-to-end with standard gradient descent.

## 6. Worked Example (KL by hand)

Suppose for a single latent dimension the encoder outputs $\mu=1,\ \sigma=0.5$ for a given $x$ (so $\sigma^2=0.25$). Plugging into Section 4's formula:

$$
\mathrm{KL} = \tfrac12\big(0.25 + 1^2 - 1 - \log 0.25\big) = \tfrac12\big(0.25 + 1 - 1 + 1.386\big) = \tfrac12(1.636) \approx 0.818 .
$$

Compare $\mu=0,\sigma=1$ (matching the prior exactly): $\mathrm{KL}=\tfrac12(1+0-1-0)=0$, confirming the "zero penalty only at the prior" claim of Section 4. The $\mu=1,\sigma=0.5$ encoder is penalized for being both off-center and overconfident (too small a $\sigma$) relative to the prior.

## 7. Implementation Sketch (PyTorch)

```python
import torch, torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)          # predict log(sigma^2) for numerical stability
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, in_dim))

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)                     # the "outside randomness" of Section 5
        return mu + sigma * eps

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum")           # Section 3 reconstruction term
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())        # Section 4 closed form
    return recon + kl
```

`logvar = log(sigma^2)` is predicted (not $\sigma$ directly) so the network can output any real number and still exponentiate to a valid positive variance — a standard numerical-stability trick.

> **Graduate depth.** Kingma & Welling (2013), *Auto-Encoding Variational Bayes*, is the original
> derivation of the ELBO and reparameterization trick above; MFDL Ch. 15 covers latent-variable
> generative models in the same functional language as this course. Be able to derive the ELBO
> both ways (Jensen and the KL decomposition), derive the Gaussian-Gaussian KL closed form, and
> explain precisely why the reparameterization trick is necessary for gradient-based training.

## Connection to This Week

- **Lab 11** — build and train a VAE in PyTorch on a small image dataset; inspect the latent space.
- **Quiz 11** — ELBO terms, reconstruction vs. regularization tradeoff, training diagnostics.
- **Homework 10 due / Homework 11 assigned** — VAE derivation and implementation, extending into
  Lecture 16's generative survey; due Mon Nov 30.

## References

- Weekly reading map, Week 14.
- Kingma & Welling (2013), *Auto-Encoding Variational Bayes*.
- MFDL Ch. 15 for latent-variable generative models.
- PyTorch VAE tutorial examples.
