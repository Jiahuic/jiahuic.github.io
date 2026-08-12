---
layout: page
permalink: /teaching/math-4990v5990v/04-logistic-regression/
title: "Lecture 4 — Logistic Regression and Classification Metrics"
---


*Part I · Week 2*

## Learning Goals

- Write the logistic model, the sigmoid, and the linear **log-odds** interpretation.
- Derive the **cross-entropy** objective from maximum likelihood and derive its **gradient** and **Hessian**.
- Prove the loss is **convex**; describe Newton's method / **IRLS**.
- Generalize to **softmax** and derive its gradient.
- Read a confusion matrix; compute accuracy, precision, recall, F1, ROC-AUC; choose a metric for imbalanced data.

## 1. Model, Sigmoid, and Log-Odds

For binary labels $y \in \{0,1\}$, the linear score $z = w^\top x + b$ is unbounded, but we need a probability. Pass it through the **sigmoid**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \hat{p} = P(y=1\mid x) = \sigma(w^\top x + b).
$$

Inverting shows the model is **linear in the log-odds**:

$$
\log\frac{P(y=1\mid x)}{P(y=0\mid x)} = w^\top x + b .
$$

Predict class 1 when $\hat p \ge 0.5$, i.e. $z \ge 0$; the decision boundary $w^\top x + b = 0$ is a **hyperplane**. A useful identity for everything below: $\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)$.

## 2. Cross-Entropy from Maximum Likelihood (exam material)

Model each label as Bernoulli with parameter $\hat p_i = \sigma(z_i)$, $z_i = w^\top x_i + b$. The likelihood is $\prod_i \hat p_i^{\,y_i}(1-\hat p_i)^{1-y_i}$; the negative log-likelihood (averaged) is the **cross-entropy loss**:

$$
L(w,b) = -\frac{1}{n}\sum_{i=1}^n \Big[\, y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\,\Big].
$$

**Gradient (full derivation).** For a single term, use $\partial_z[-y\log\sigma(z) - (1-y)\log(1-\sigma(z))]$. With $\sigma' = \sigma(1-\sigma)$,

$$
\frac{\partial}{\partial z}\big[-y\log\sigma - (1-y)\log(1-\sigma)\big]
= -y(1-\sigma) + (1-y)\sigma = \sigma(z) - y = \hat p - y .
$$

By the chain rule $\partial_z/\partial w = x$, so

$$
\nabla_w L = \frac{1}{n}\sum_i (\hat p_i - y_i)\, x_i = \frac{1}{n} X^\top(\hat p - y), \qquad
\frac{\partial L}{\partial b} = \frac{1}{n}\sum_i (\hat p_i - y_i).
$$

Strikingly, this is the **same form** as the least-squares gradient (Lecture 2), with $\hat p$ replacing the linear prediction — a consequence of pairing each output nonlinearity with its matching (canonical) loss.

## 3. Convexity, Hessian, and IRLS

There is **no closed form**, so we optimize numerically. The Hessian is

$$
\nabla^2_w L = \frac{1}{n}\sum_i \hat p_i(1-\hat p_i)\, x_i x_i^\top = \frac{1}{n} X^\top D X, \qquad D = \operatorname{diag}\big(\hat p_i(1-\hat p_i)\big) \succeq 0 .
$$

Since $D \succeq 0$, the Hessian is positive semidefinite, so **$L$ is convex** — gradient descent (Lecture 3) reaches the global minimum, and it is strictly convex (unique minimizer) when $X$ has full column rank and the classes are not perfectly separable. **Newton's method** uses the Hessian:

$$
\theta \leftarrow \theta - (X^\top D X)^{-1} X^\top(\hat p - y),
$$

which is exactly a **weighted least-squares** solve at each step — hence the classic name **IRLS** (iteratively reweighted least squares, PRML 4.3.3). It converges in far fewer iterations than plain GD but costs a $p\times p$ solve per step.

> **Separable data caveat:** if the classes are linearly separable, the MLE pushes $\lVert w\rVert \to \infty$ (probabilities saturate to 0/1). Regularization (an L2 penalty, Lecture 5) keeps $w$ finite — which is why scikit-learn regularizes by default.

## 4. Multiclass: Softmax

For $K$ classes with scores $z_k = w_k^\top x + b_k$, the **softmax** gives

$$
P(y=k\mid x) = \frac{e^{z_k}}{\sum_{l=1}^K e^{z_l}} =: s_k .
$$

Paired with categorical cross-entropy $L = -\sum_k \mathbb{1}[y=k]\log s_k$, the gradient collapses to the same clean residual form $\partial L/\partial z_k = s_k - \mathbb{1}[y=k]$. Softmax is **shift-invariant** ($s(z+c\mathbf 1) = s(z)$), which is used for numerically stable implementations (subtract $\max_k z_k$) and reappears throughout Part II (Lectures 11–16).

## 5. Classification Metrics

From the confusion matrix (TP, FP, TN, FN):

| Metric | Formula | Use when |
|---|---|---|
| Accuracy | $(TP+TN)/\text{all}$ | classes balanced |
| Precision | $TP/(TP+FP)$ | false positives costly |
| Recall (sensitivity) | $TP/(TP+FN)$ | false negatives costly |
| F1 | harmonic mean of P & R | balance P and R |
| ROC-AUC | area under TPR–FPR curve | threshold-independent ranking |

**Threshold matters.** The 0.5 cutoff is a decision-theoretic choice: the Bayes-optimal threshold under costs $(c_{\text{FP}}, c_{\text{FN}})$ is $\hat p^\star = c_{\text{FP}}/(c_{\text{FP}}+c_{\text{FN}})$. Lowering the threshold raises recall and lowers precision. **AUC** equals the probability the model ranks a random positive above a random negative — a threshold-free measure of ranking quality. On **imbalanced** data (e.g. 99% negatives) accuracy is misleading — "always negative" scores 99% — so prefer precision/recall/F1 or AUC.

## 6. Worked Example

Spam filter on 100 emails: TP=30, FP=10, FN=5, TN=55.
Accuracy $=0.85$; Precision $=\tfrac{30}{40}=0.75$; Recall $=\tfrac{30}{35}=0.857$; F1 $=2\cdot\tfrac{0.75\cdot0.857}{0.75+0.857}=0.80$.

## 7. Implementation Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr)                  # fit on train, transform both
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(confusion_matrix(yte, clf.predict(Xte)))
print("AUC:", roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
```

> **Graduate depth.** MFDL Ch. 3 and PRML Ch. 4.3.2–4.3.4 develop logistic regression as a
> generalized linear model with the canonical link, IRLS, and Laplace-approximation Bayesian
> extensions. Be able to derive the gradient and Hessian, prove convexity, and explain the
> separable-data divergence and its regularization fix.

## Connection to This Week

- **Lab 3** — logistic regression from scratch (gradient descent) + metrics.
- **Quiz 3** — thresholds, confusion matrix, metric choice.
- **Homework 3** — logistic regression, decision boundaries, and classification metrics; due Mon Sep 14.
- **Exam 1 (Week 5)** covers Lectures 1–4.

## References

- Weekly reading map, Week 4.
- MFDL Ch. 3; ESL Ch. 4.1-4.4.
- PRML Ch. 4.1 and 4.3.2-4.3.4 for probabilistic classification context.
- scikit-learn LogisticRegression and model evaluation guides.
