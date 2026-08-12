---
layout: page
permalink: /teaching/math-4990v5990v/01-introduction-and-workflow/
title: "Lecture 1 — Introduction, Workflow, and Starting Linear Regression"
---


*Part I · Week 1*

## Learning Goals

By the end of this lecture you should be able to:

- Distinguish artificial intelligence, machine learning, and deep learning, and give an example of each.
- Describe, at a high level, how an **AI agent** (an LLM calling tools/making decisions in a loop) differs from the statistical models this course builds.
- Write the **statistical learning** setup: a joint distribution, a loss, **risk**, and **empirical risk**.
- Explain **generalization**, overfitting, and why every learner needs an **inductive bias**.
- Use NumPy for vectors/matrices and pandas for tabular data; represent a dataset as $(X, y)$.
- Prove that held-out test error is an **unbiased estimate of risk**, and see how **leakage** breaks it.
- Explain **mathematically** why standardization helps: it reshapes the loss surface (conditioning).
- Write the linear regression model and its function class — the starting point for Lecture 2.

## 1. AI vs. ML vs. DL

- **Artificial Intelligence (AI):** making machines perform tasks that appear intelligent.
- **Machine Learning (ML):** a subset of AI that *improves with experience* by fitting models to data using statistical/optimization methods.
- **Deep Learning (DL):** a subset of ML that uses neural networks with many layers; responsible for most recent breakthroughs, enabled by large data and GPUs.

**Traditional vs. data-driven.** In the traditional paradigm you write *rules + data → answers*. In ML you supply *data + answers → rules (a model)*. The recipe analogy: instead of following a cheeseburger recipe, you taste enough cheeseburgers and *infer* the recipe.

## 2. Where Do Agents Fit?

An **AI agent** — an LLM that plans, calls tools, and takes actions in a loop toward a goal — sits on top of the machinery this course builds, not beside it. The LLM inside an agent is itself a (very large) statistically trained model: same ERM skeleton (Section 3), same gradient-based training (Lectures 3–4, 11–12), same generalization concerns (Section 4). What is new in an *agent* is orchestration — search, tool use, memory, multi-step decision making — not a different mathematical foundation. This course builds the foundation (loss functions, optimization, evaluation, and eventually the neural-network and generative machinery in Part II) that agentic systems are built on top of; it does not cover agent orchestration itself.

## 3. The Statistical Learning Setup

We assume examples are drawn i.i.d. from an unknown joint distribution,

$$
(x_i, y_i) \sim P(x, y), \qquad i = 1, \dots, n, \quad x_i \in \mathbb{R}^p,\ y_i \in \mathcal{Y}.
$$

- $x_i$ is a **feature vector**; $x_{ij}$ is feature $j$ of example $i$. $y_i$ is the **label/target/response**.
- A **model** is a function $f_\theta : \mathbb{R}^p \to \mathcal{Y}$ with **parameters** $\theta$.
- A **loss** $L(y, \hat y) \ge 0$ measures the cost of predicting $\hat y$ when the truth is $y$.

The quantity we actually care about is the **risk** (expected loss / generalization error):

$$
R(f) = \mathbb{E}_{(x,y)\sim P}\big[\,L(y, f(x))\,\big].
$$

We cannot compute $R$ because $P$ is unknown, so we minimize the **empirical risk** on the training sample,

$$
\hat R(f) = \frac{1}{n}\sum_{i=1}^n L\big(y_i, f(x_i)\big).
$$

This principle — **empirical risk minimization (ERM)** — is the common skeleton of nearly every method in the course: choose a model class, pick a loss, minimize $\hat R$ (Lectures 2–3), and hope $\hat R(f) \approx R(f)$ (Section 5, Lecture 5).

**The optimal predictor.** Under squared loss, minimizing $\mathbb{E}[(y-f(x))^2]$ pointwise gives the **regression function** $f^\star(x) = \mathbb{E}[y\mid x]$ (differentiate $\mathbb{E}[(y-c)^2\mid x]$ in $c$ and set to zero). Under 0–1 loss, minimizing $P(f(x)\ne y)$ gives the **Bayes classifier** $f^\star(x) = \arg\max_k P(y=k\mid x)$. Both are unreachable in practice (they need $P$), so learning is **approximating $f^\star$ from finite data** with a restricted model class $\mathcal F$.

## 4. Generalization and Inductive Bias

Minimizing training error is easy; a lookup table achieves zero. The goal is small **risk** on unseen data. The gap $R(f) - \hat R(f)$ is the **generalization gap**:

- A model too flexible for the data **overfits** — memorizes noise, large gap (high variance).
- A model too rigid **underfits** — cannot represent the signal (high bias).

No method can generalize without assumptions — the **no-free-lunch** idea. Every learner encodes an **inductive bias** (linearity, smoothness, low dimensionality, locality); choosing a method is choosing which bias matches your data. We make this precise as the bias–variance tradeoff in Lecture 2.

## 5. Types of Learning

- **Supervised:** labeled $(x_i,y_i)$. **Regression** (continuous $y$) or **classification** (discrete $y$).
- **Unsupervised:** only $x_i$. **Clustering** (k-means, Lecture 6) and **dimensionality reduction** (PCA/t-SNE, Lecture 7) — estimating structure of $P(x)$.
- **Reinforcement:** an agent takes actions and learns from rewards (mechanically distinct from the agent orchestration in Section 2, though both are called "agents").

## 6. The Two Views of This Course: Math and Engineering

Every method is studied from **two viewpoints**, assessed differently:

| Viewpoint | Question it answers | Assessed by |
|---|---|---|
| **Mathematics** | *Why* does it work? Objective, gradient, geometry, guarantees. | **Exams** |
| **Engineering** | *How* do I train and diagnose it? Splits, scaling, learning rate, over/underfitting, metrics. | **Quizzes / Labs** |

You should be able both to **derive** a method and to **run and debug** its training.

## 7. The Workflow, and the Scientific Python Stack

Every lab follows the same seven steps: (1) get and inspect data, represent as $(X,y)$; (2) preprocess — clean, encode, scale (Section 9); (3) split — estimate risk on held-out data, never fit on the test set (Section 10); (4) choose a model class and loss; (5) train — minimize empirical risk (Lectures 2–3); (6) evaluate with an appropriate metric on held-out data (Lecture 5); (7) iterate and report reproducibly, under Git.

- **NumPy** — n-dimensional arrays and linear algebra; the numerical backbone.
- **pandas** — labeled tabular data (`DataFrame`, `Series`).
- **matplotlib** — plotting. **scikit-learn** — datasets, preprocessing, models, pipelines.

Vectorized NumPy replaces Python loops: `a + b` is elementwise C-level code, $O(n)$ but with a tiny constant, versus a slow interpreted loop. Conda environments pin versions so every machine reproduces results (Lab 0).

```python
import numpy as np, pandas as pd
a = np.array([[1,2,3],[4,5,6]])         # shape (2,3)
a @ a.T                                  # matrix product -> (2,2)
df = pd.read_csv("../datasets/some.csv")
X = df.drop("target", axis=1); y = df["target"]   # features / label
```

## 8. Preprocessing

**Cleaning** (fill/drop missing, remove duplicates, handle outliers), **transformation** (encode categoricals; normalize/standardize numeric features), and **reduction** (feature selection or extraction — PCA/t-SNE, Lecture 7). Missing data is not random noise: the standard taxonomy is MCAR (missing completely at random), MAR (missing at random given observed features), and MNAR (missingness depends on the unobserved value). Mean/median imputation is unbiased only under MCAR. Impute **inside** the pipeline so the imputer is fit on training folds only.

**Categorical encoding.** Convert categories to numbers with **one-hot** encoding (one indicator column per level) or **ordinal** encoding (only for truly ordered categories). One-hot with an intercept creates the **dummy-variable trap**: the indicator columns sum to the all-ones intercept column, making $X$ rank-deficient so $X^\top X$ is singular (Lecture 2 §3). Fix by dropping one level (`drop='first'`) or omitting the explicit intercept.

## 9. Why the Split Works: Risk Estimation (exam material)

We hold out data because we want the **risk** $R(f)=\mathbb{E}_{(x,y)\sim P}[L(y,f(x))]$ (Section 3), not the training error. For a model $f$ that is **fixed independently of the test set**, the average test loss is an **unbiased estimator** of the risk:

$$
\hat R_{\text{test}}(f) = \frac{1}{m}\sum_{i=1}^m L\big(y_i, f(x_i)\big),
\qquad
\mathbb{E}\big[\hat R_{\text{test}}(f)\big] = R(f),
$$

because each test point is an i.i.d. draw from $P$ and expectation is linear. Its variance is $\operatorname{Var}(\hat R_{\text{test}}) = \tfrac{1}{m}\operatorname{Var}_{(x,y)}(L)$, so larger test sets give tighter estimates.

**Leakage breaks the "fixed $f$" assumption.** If any test information influences $f$ — e.g. fitting a scaler, selecting features, or imputing using the full dataset before splitting — then $f$ depends on the test set, the independence argument fails, and $\mathbb{E}[\hat R_{\text{test}}] < R(f)$: the estimate is **optimistically biased**.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)          # fit on TRAIN ONLY
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)              # same mu, sigma
```

**Rule: split first, then fit every preprocessing step on the training set only.**

## 10. Standardization and the Geometry of the Loss (exam material)

Standardize feature $j$ using the **training** mean $\mu_j$ and std $\sigma_j$:

$$
\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
\quad\Longrightarrow\quad \text{each feature has mean 0, variance 1.}
$$

**Why it matters, precisely.** For least squares the loss Hessian is $\nabla^2 L = \tfrac{2}{n}X^\top X$ (Lecture 2). Gradient descent's convergence rate is governed by the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ of this Hessian: the error contracts per step like $\big(\tfrac{\kappa-1}{\kappa+1}\big)$, so a large $\kappa$ (features on wildly different scales → a stretched, elongated bowl) makes descent zig-zag and crawl (Lecture 3). Standardizing equalizes the diagonal of $X^\top X$, shrinking $\kappa$ and letting a single learning rate work in every direction.

Distance-based methods (KNN, k-means, SVM-RBF) are even more sensitive: Euclidean distance $\lVert x-x'\rVert^2 = \sum_j (x_j-x'_j)^2$ is dominated by the largest-scale feature unless standardized (Lecture 6).

**Which scaler?** Standardization (default); **min–max** to a fixed range $[0,1]$ when bounded inputs are needed; **robust** scaling (median/IQR) when heavy outliers would distort $\mu,\sigma$.

## 11. Starting Linear Regression: The Model and Function Class

Given features $x \in \mathbb{R}^p$, linear regression predicts with an **affine** function

$$
\hat{y} = f_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_p x_p = \theta_0 + w^\top x .
$$

Absorb the intercept by appending a constant $1$ to each example ($\tilde{x} = (1, x)$, $\theta = (\theta_0, w) \in \mathbb{R}^{p+1}$). Stacking the $n$ examples as rows of the **design matrix** $X \in \mathbb{R}^{n\times(p+1)}$ gives the vectorized model

$$
\hat{y} = X\theta .
$$

The hypothesis class $\mathcal{F} = \{x \mapsto \theta^\top \tilde x : \theta \in \mathbb{R}^{p+1}\}$ is linear in the **parameters** — the features themselves may be nonlinear transforms (a **basis expansion** $\phi(x)$), which is how the same machinery fits polynomials and, later, kernels (Lecture 8). Linearity in $\theta$, not in $x$, is what makes the problem convex — the key fact Lecture 2 builds on to derive the normal equations.

## 12. Worked Example (conceptual, end to end)

*Predicting a student's final grade from hours studied and attendance.* Two features ($p=2$), continuous target → **supervised regression**. Under squared loss the ideal predictor is $f^\star(x)=\mathbb{E}[\text{grade}\mid x]$ (Section 3); a linear model $f_\theta(x)=\theta_0+\theta_1 x_1+\theta_2 x_2$ is a restricted approximation. Before fitting anything: split the data, standardize the two features using train-set statistics only (Sections 9–10), and only then fit $\theta$ by minimizing empirical risk — the fitting itself is Lecture 2.

## 13. Implementation Example

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
print(iris.data.shape)      # (150, 4)  -> n=150 examples, p=4 features
print(iris.target_names)    # ['setosa' 'versicolor' 'virginica'] -> classification

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)              # fit on TRAIN ONLY
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
print(X_train.mean(axis=0).round(3), X_train.std(axis=0).round(3))  # ~0, ~1
```

> **Graduate depth.** PRML Ch. 1.5 develops decision theory (expected loss, the regression
> function and Bayes classifier as loss-optimal predictors); ESL Ch. 2.1–2.3 formalizes the
> supervised-learning setup. Be able to derive $f^\star(x)=\mathbb{E}[y\mid x]$ for squared loss,
> state the risk / empirical-risk / generalization-gap decomposition, and prove
> $\mathbb{E}[\hat R_{\text{test}}]=R(f)$ for a fixed $f$.

## Connection to This Week

- **Lab 0** — set up Python/Conda, Git, and the GitLab workflow; NumPy/pandas/plotting warm-up; train/test split and scaling on a first dataset.

## References

- Weekly reading map, Week 1.
- ESL Ch. 1–2.3; PRML Ch. 1 introduction and Ch. 1.1 for data/model language.
- scikit-learn Getting Started guide; pandas and NumPy documentation.
