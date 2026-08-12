---
layout: page
permalink: /teaching/math-4990v5990v/09-decision-trees/
title: "Lecture 9 — Decision Trees (CART)"
---


*Part I · Week 10*

## Learning Goals

- Write the tree as a **recursive partition** and derive the optimal leaf prediction.
- Define **Gini**, **entropy**, and misclassification impurity; compute **information gain**.
- Prove splits never increase (weighted) impurity via **concavity** (Jensen).
- Explain the greedy algorithm, its NP-hardness, and **cost-complexity pruning**.

## 1. The Model: A Recursive Partition

CART partitions the feature space into $M$ axis-aligned regions $R_1,\dots,R_M$ (leaves) by a sequence of binary questions "is $x_j \le s$?", and predicts a constant per region:

$$
f(x) = \sum_{m=1}^M c_m\, \mathbb{1}[x \in R_m].
$$

**Optimal leaf constant.** For squared loss, minimizing $\sum_{i\in R_m}(y_i - c_m)^2$ gives $c_m = \bar y_{R_m}$, the region **mean** (Lecture 1: $\mathbb{E}[y\mid x]$ estimated locally). For classification with 0–1 loss it is the **majority class**. Trees are interpretable, need no feature scaling (splits are threshold comparisons — invariant to monotonic rescaling), and handle mixed feature types.

## 2. Impurity and Information Gain (exam material)

Finding the globally optimal tree is **NP-hard**, so CART grows **greedily**: at each node pick the split that most reduces impurity. At a node with class proportions $\hat p_{mk}$:

- **Gini:** $G = \sum_k \hat p_{mk}(1-\hat p_{mk}) = 1 - \sum_k \hat p_{mk}^2$.
- **Entropy:** $H = -\sum_k \hat p_{mk}\log_2 \hat p_{mk}$.
- **Misclassification error:** $E = 1 - \max_k \hat p_{mk}$.

For two classes ($p=\hat p_{m1}$): $G=2p(1-p)$, $H=-p\log_2 p-(1-p)\log_2(1-p)$, $E=1-\max(p,1-p)$. All are 0 at $p\in\{0,1\}$ (pure) and maximal at $p=0.5$.

A split of parent node into children $L,R$ with weights $w_L,w_R$ (fraction of samples) has **information gain**

$$
\Delta = I(\text{parent}) - \big(w_L\, I(L) + w_R\, I(R)\big),
$$

and CART scans all (feature, threshold) pairs to maximize $\Delta$.

**Why splitting helps (concavity).** Each impurity measure $I(\cdot)$ is **concave** in the class-proportion vector. Since the parent proportions are the weighted average of the children's, Jensen's inequality gives

$$
I(\text{parent}) = I(w_L p_L + w_R p_R) \ge w_L I(p_L) + w_R I(p_R),
$$

so $\Delta \ge 0$ — a split **never increases** weighted impurity. **Gini and entropy are strictly concave** (so gain is strictly positive unless the split is useless), whereas misclassification error is only piecewise linear and can give zero gain for informative splits — which is why Gini/entropy are used to *grow* trees and error is used only to *prune*.

## 3. Regression Trees

Same recursion; leaves predict the mean, and the greedy split minimizes residual sum of squares

$$
\text{RSS} = \sum_{\text{leaves } m}\ \sum_{i\in R_m}(y_i - \bar y_{R_m})^2 .
$$

For a candidate split the best threshold is found by scanning sorted feature values, updating the two side means incrementally in $O(n\log n)$ per feature.

## 4. Overfitting and Cost-Complexity Pruning

A tree grown until leaves are pure has **zero training error** but high variance — it memorizes noise (large train–test gap, Quiz 9). Simple controls: `max_depth`, `min_samples_leaf`, `min_samples_split`. The principled method is **cost-complexity (weakest-link) pruning**: for a full tree $T_0$ and penalty $\alpha \ge 0$, minimize

$$
C_\alpha(T) = \sum_{m=1}^{|T|} \big(\text{impurity of leaf } m\big) + \alpha\, |T|,
$$

where $|T|$ is the number of leaves. Increasing $\alpha$ collapses the weakest branches first, producing a **nested sequence** of subtrees; choose $\alpha$ by cross-validation (Lecture 5). This is the tree analogue of the loss + complexity-penalty idea from ridge/lasso. A single tree's low-bias/high-variance behavior is exactly what ensembles (Lecture 10) exploit.

## 5. Worked Example (Gini gain)

Node of 10 samples, 6 Red / 4 Green → $G = 1-(0.6^2+0.4^2)=0.48$. Split into {4R,0G} ($G=0$) and {2R,4G} ($G=1-(\tfrac26)^2-(\tfrac46)^2=0.444$). Weighted child impurity $=\tfrac{4}{10}(0)+\tfrac{6}{10}(0.444)=0.267$. Gain $=0.48-0.267=0.213>0$ — a useful split.

## 6. Implementation Example

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(Xtr, ytr)
    print(depth, "train:", round(dt.score(Xtr, ytr), 3), "test:", round(dt.score(Xte, yte), 3))
# cost-complexity path: dt.cost_complexity_pruning_path(Xtr, ytr) -> ccp_alphas
```

> **Graduate depth.** ESL Ch. 9.2 develops CART, the concavity of impurity, and weakest-link pruning;
> PRML Ch. 14.4 places trees in the model-combination context. Be able to derive the optimal leaf
> constant, prove $\Delta \ge 0$ via concavity, and explain why misclassification error is unsuitable
> for growing but fine for pruning.

## Connection to This Week

- **Lab 7** — decision tree from scratch (Gini) on digits; regression tree on airfoil.
- **Quiz 7** — tree depth, impurity, overfitting.
- **Homework 7** — Gini/entropy/error plot; due Mon Oct 26.

## References

- Weekly reading map, Week 10.
- ESL Ch. 9.2 on tree-based methods.
- PRML Ch. 14.4 for tree-based models in a model-combination context.
- scikit-learn Decision Trees user guide.
