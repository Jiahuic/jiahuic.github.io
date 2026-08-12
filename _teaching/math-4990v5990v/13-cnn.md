---
layout: page
permalink: /teaching/math-4990v5990v/13-cnn/
title: "Lecture 13 — Convolutional Neural Networks"
---


*Part II · Week 12*

## Learning Goals

- Define convolution, compute output sizes and parameter counts, and explain **weight sharing / equivariance**.
- Explain why CNNs suit image (and other spatially/locally structured) data.
- Count parameters of a conv layer and contrast with a dense layer of the same input size.
- Sketch a small CNN in PyTorch and match layers to the math.

## 1. Convolutional Neural Networks

A **convolution** slides a small filter over the input, computing a dot product at each location:

$$
(\,\text{feature map})_{i,j} = \sum_{c=1}^{C_{\text{in}}}\sum_{a=1}^{k}\sum_{b=1}^{k} W_{c,a,b}\, x_{c,\, i+a,\, j+b} + \beta .
$$

Two structural priors make this suit images:

- **Local connectivity:** each output sees only a $k\times k$ patch (local structure of images).
- **Weight sharing:** the *same* filter is reused at every location — a dense layer whose weight matrix is constrained to be block-Toeplitz with tied entries. This gives **translation equivariance** (shifting the input shifts the feature map); pooling then adds approximate **invariance**.

**Output size** for input $H$, filter $k$, padding $P$, stride $S$: $\big\lfloor (H - k + 2P)/S \big\rfloor + 1$. **Parameters** for a conv layer: $C_{\text{out}}\,(k^2 C_{\text{in}} + 1)$ — *independent of image size*, unlike a dense layer, whose parameter count scales with $H\times W$. Stacking layers grows the **receptive field**; a CNN interleaves conv + ReLU + pooling and ends in dense layers.

## 2. Why Not Just Use a Dense (Fully Connected) Layer?

A dense layer from a $32\times32\times3$ image to even a modest 100 hidden units needs $32\cdot32\cdot3\cdot100 + 100 \approx 307{,}300$ parameters — and none of them are shared, so the network must re-learn "edge detector" from scratch in every spatial location. The conv layer in Section 3's worked example needs 78 parameters for the *entire image*, because weight sharing forces the same local pattern-detector to be reused everywhere it's relevant — the correct inductive bias (Lecture 1 §4) for data with translation structure (images, and 1D signals like audio).

## 3. Worked Example: Convolution Size and Parameters

Take a $32\times32$ grayscale image, three $5\times5$ filters, stride $S=1$, and no padding.
The spatial output size is $\lfloor(32-5)/1\rfloor+1=28$, so the output tensor has shape
$3\times28\times28$. Each filter has $25$ weights and one bias; the layer therefore has
$3(25+1)=\mathbf{78}$ trainable parameters. The count stays 78 even if the input image is larger:
weight sharing changes the number of filter applications, not the number of parameters.

## 4. Implementation Sketch (CNN in PyTorch)

```python
import torch.nn as nn
cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),   # 28->26->13
    nn.Flatten(), nn.Linear(8 * 13 * 13, 10))                    # 28x28 input
```

Each `Conv2d(in_channels, out_channels, kernel_size)` follows exactly the parameter formula of
Section 1 with $C_{\text{in}}=1,\ C_{\text{out}}=8,\ k=3$: $8(3^2\cdot1+1) = 80$ parameters,
regardless of whether the input image is $28\times28$ or $280\times280$.

> **Graduate depth.** MFDL Ch. 14 develops CNNs in the same functional-analysis language as the
> rest of the course; PRML Ch. 5.5.6 covers convolutional networks. Be able to compute conv output
> sizes and parameter counts and explain weight sharing / translation equivariance precisely
> (not just "it uses fewer parameters").

## Connection to This Week

- **Lab 9** — build and train a small CNN in PyTorch on image data.
- **Quiz 9** — conv output-size and parameter-count computations; training diagnostics.
- **Homework 8 due / Homework 9 assigned** — CNN parameter counting and the weight-sharing constraint; due Mon Nov 9.
- **Exam 3 (Week 16)** covers CNN parameter counting and the ideas above (conceptual, not code).

## References

- Weekly reading map, Week 12.
- MFDL Ch. 14 for CNNs.
- PRML Ch. 5.5.6 for convolutional networks.
- PyTorch `torchvision` and convolution tutorials.
