---
layout: page
permalink: /teaching/math-4990v5990v/10-ensembles/
title: "Lecture 10 — Random Forests, Bagging, and Boosting"
---


*Part I · Week 10*

## Learning Goals

- Derive the bootstrap out-of-bag fraction and the **variance of a correlated average**.
- Explain why random forests **decorrelate** trees and how that beats plain bagging.
- Derive **gradient boosting** as forward stagewise **functional gradient descent**.
- Contrast bagging (variance) with boosting (bias) precisely.

## 1. The Bootstrap

A **bootstrap sample** draws $n$ observations *with replacement*. The probability a given point is left out of one sample is

$$
\Big(1 - \frac1n\Big)^n \xrightarrow{n\to\infty} \frac1e \approx 0.368,
$$

so each bootstrap sample contains $\approx 63.2\%$ of the unique data; the omitted **out-of-bag (OOB)** points form a free validation set.

## 2. Bagging and the Variance of a Correlated Average (exam material)

Train $B$ models on $B$ bootstrap samples and average:

$$
\hat f_{\text{bag}}(x) = \frac1B \sum_{b=1}^B \hat f^{(b)}(x).
$$

Suppose each $\hat f^{(b)}$ has variance $\sigma^2$ and **pairwise correlation** $\rho$. The variance of the average is

$$
\operatorname{Var}\big(\hat f_{\text{bag}}\big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .
$$

*(From $\operatorname{Var}(\tfrac1B\sum_b Z_b) = \tfrac1{B^2}[\,B\sigma^2 + B(B-1)\rho\sigma^2\,]$.)* Two lessons: (i) the second term $\to 0$ as $B$ grows, so more trees help **for free** and never overfit — the variance **plateaus** at $\rho\sigma^2$; (ii) that plateau is set by the **correlation** $\rho$. Bagging reduces variance without changing bias, so it shines on high-variance, low-bias learners like deep trees (Lecture 9).

## 3. Random Forests

Bagged trees are highly correlated because a few strong features dominate the top splits, so $\rho$ (and thus the plateau $\rho\sigma^2$) stays large. **Random forests** attack $\rho$ directly: at **each split** consider only a random subset of $m$ features (default $m\approx\sqrt p$ for classification, $p/3$ for regression). This **decorrelates** the trees, lowering $\rho$ and hence the variance floor — the whole reason a random forest beats plain bagged trees. Bonuses: **OOB error** as a free validation estimate, and impurity-decrease **feature importances**.

## 4. Boosting as Forward Stagewise Additive Modeling

Boosting builds an additive model **sequentially**, each term correcting the current ensemble:

$$
F_m(x) = F_{m-1}(x) + \nu\, h_m(x).
$$

**Functional gradient view (gradient boosting).** Think of minimizing the empirical loss $\sum_i \ell(y_i, F(x_i))$ over the *function* $F$. The steepest-descent direction in function space is the negative gradient evaluated at the training points, the **pseudo-residuals**

$$
r_{im} = -\left.\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}} .
$$

Each stage fits a shallow tree $h_m$ to these pseudo-residuals and takes a step of size $\nu$ (shrinkage). For **squared loss**, $r_{im} = y_i - F_{m-1}(x_i)$ — literally the residuals; for other losses it is their gradient generalization. **AdaBoost** is the special case of this with the exponential loss $\ell = e^{-yF}$. Key hyperparameters: `n_estimators` $M$, learning rate $\nu$, and tree depth. Small $\nu$ with large $M$ generalizes best but costs compute; because each stage reduces **bias** while adding a little variance, too many stages (or large $\nu$) can **overfit**.

## 5. Bagging vs. Boosting

| | Bagging / Random Forest | Boosting |
|---|---|---|
| Trees | parallel, independent | sequential, dependent |
| Base learner | deep (low bias, high variance) | shallow (high bias) |
| Reduces | variance ($\rho\sigma^2 + \tfrac{1-\rho}{B}\sigma^2$) | bias (stagewise) |
| More trees overfit? | no (plateaus) | can |
| Tuning | little | learning rate + #trees matter |

## 6. Worked Example (combining predictions)

Ten bootstrapped trees estimate $P(\text{Green}\mid x)$: $0.1,0.15,0.2,0.2,0.55,0.6,0.6,0.65,0.7,0.75$.
- **Majority vote** (threshold 0.5): 6/10 exceed 0.5 → **Green**.
- **Average probability:** mean $=0.45<0.5$ → **Red**. The two rules can disagree.

## 7. Implementation Example

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42).fit(Xtr, ytr)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42).fit(Xtr, ytr)
print("RF test:", rf.score(Xte, yte), "OOB:", round(rf.oob_score_, 3), " GB test:", gb.score(Xte, yte))
```

> **Graduate depth.** ESL Ch. 15 derives the correlated-average variance and random forests; ESL Ch. 10
> and Ch. 16 develop boosting as forward stagewise additive modeling and AdaBoost as exponential-loss
> minimization; PRML Ch. 14.2–14.3 covers committees and boosting. Be able to derive
> $\rho\sigma^2 + \tfrac{1-\rho}{B}\sigma^2$ and the pseudo-residual step.

## Connection to This Week

- **Lab 7** — random forest and gradient boosting; feature importances and `n_estimators` sweep.
- **Quiz 7** — bias/variance of ensembles, learning rate in boosting.
- **Homework 7** — bootstrap probability, majority vs. average voting; due Mon Oct 26.
- **Exam 3 (Week 16)** covers Lectures 9–16 (trees and ensembles alongside Part II).

## References

- Weekly reading map, Week 10.
- ESL Ch. 10, Ch. 15, and Ch. 16 on boosting, random forests, and ensemble learning.
- PRML Ch. 14.2-14.3 on committees and boosting.
- scikit-learn Ensemble methods user guide.
