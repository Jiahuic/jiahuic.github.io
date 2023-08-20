---
layout: page
permalink: /teaching/math-3583/1_vector/
title: Vectors
---

### 1.1 Vectors

A vector is a one-dimensional array of numbers. In physics and engineering, vectors often represent quantities that have both magnitude and direction, such as force or velocity. In linear algebra, vectors are elements of vector spaces, which are sets equipped with the operations of vector addition and scalar multiplication.

#### Representation:

For the sake of these notes, we'll represent vectors as column matrices. 

$$ \mathbf{\bf v} = \begin{bmatrix}
v_1 \\
v_2 \\
\vdots \\
v_n \\
\end{bmatrix} $$

where each $$v_i$$ is a component of the vector.

#### Vector Addition:

To add two vectors, simply add their corresponding components.

Given vectors:

$$ \mathbf{u} = \begin{bmatrix}
u_1 \\
u_2 \\
\end{bmatrix} $$

and 

$$ \mathbf{v} = \begin{bmatrix}
v_1 \\
v_2 \\
\end{bmatrix} $$

Their sum is:

$$ \mathbf{u} + \mathbf{v} = \begin{bmatrix}
u_1 + v_1 \\
u_2 + v_2 \\
\end{bmatrix} $$

#### Scalar-Vector Multiplication:

To multiply a vector by a scalar, multiply each component of the vector by that scalar.

Given scalar $$k$$ and vector:

$$ \mathbf{v} = \begin{bmatrix}
v_1 \\
v_2 \\
\end{bmatrix} $$

The product is:

$$ k \mathbf{v} = \begin{bmatrix}
k \cdot v_1 \\
k \cdot v_2 \\
\end{bmatrix} $$

#### Numpy   
Numpy is a common way to represent vectors, and you are suggested to use ```numpy``` unless otherwise specified. The benefit of ```numpy``` is that it can perform the linear algebra operations listed in the previous section.  

For example, the following code uses ```numpy.array``` to define a vector of four elements.


```python
import numpy as np
x = np.array([-1, 0, 2, 3.1])
y = np.array([4, 3, 2, 1])
print(x+y)
print(3*x)
```

    [3.  3.  4.  4.1]
    [-3.   0.   6.   9.3]

### 1.2 Vector Spaces

A **Vector Space** is a set $$V$$ of elements called **vectors**, having operations of addition and scalar multiplication defined on it that satisfy the following conditions ($$u$$, $$v$$, and $$w$$ are arbitrary elements of $$V$$, and c and d are scalars.)

<strong>Closure Axioms</strong>
<ol>
  <li>The sum \(\mathbf{u} + \mathbf{v}\) exists and is an element of \(V\). (\(V\) is closed under addition.)</li>
  <li>\(c\mathbf{u}\) is an element of \(V\). (\(V\) is closed under scalar multiplication.)</li>
</ol>

<strong>Addition Axioms</strong>
<ol start="3">
  <li>\(\mathbf{u}+\mathbf{v} = \mathbf{v}+\mathbf{u}\) commutative property</li>
  <li>\((\mathbf{u}+\mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v}+\mathbf{w})\) associative property</li>
  <li>There exists an element of \(V\), called a <strong>zero vector</strong>, denoted \(0\), such that \(\mathbf{u}+0 = 0 + \mathbf{u} = \mathbf{u}\)</li>
  <li>For every element \(u\) of \(V\), there exists an element called a <strong>negative</strong> of \(u\), denoted \(-\mathbf{u}\), such that \(\mathbf{u} + (-\mathbf{u}) = 0\).</li>
</ol>

<strong>Scalar Multiplication Axioms</strong>
<ol start="7">
  <li>\(c(\mathbf{u}+\mathbf{v}) = c\mathbf{u} + c\mathbf{v}\) distributed properties</li>
  <li>\((c + d)\mathbf{u} = c\mathbf{u} + d\mathbf{u}\) distributed properties</li>
  <li>\(c(d\mathbf{u}) = (cd)\mathbf{u}\) distributed properties</li>
  <li>\(1\mathbf{u} = \mathbf{u}\) scalar multiplication by 1</li>
</ol>

#### Definition of a basis of a vector space
> A finite set of vectors $${v_1,\dots, v_n}$$ is called a **basis** of a *vector space* $$V$$ if the set *spans* $$V$$ and is *linearly independent*. 
> i.e. each vector in $$V$$ can be expressed uniquely as a *linear combination* of the vectors in a basis.


### 1.2.1 $$\mathbb R^n$$ as a Vector Space 

Note that by $$\mathbb R^n$$ we denote the $$n$$-tuples of real numbers. For example, $$(10,20,3.2)\in \mathbb R^3$$, $$(0,-0.7)\in\mathbb R^2$$, $$(1,2,3,500)\in\mathbb R^4$$.

In order for a set to be a vector space, we need to have defined **addition** (i.e., have defined how we add two vectors together) and **scalar multiplication** (i.e. we have defined how we multiply a vector by a scalar). A vector space is

- Closed under addition
- Closed under scalar multiplication 

There are much more general vectors spaces than $$\mathbb R^n$$ - vectos can be almost any type of object as long as it maintains the two above properties. We will get into this concept later in the semester. In the case of $$\mathbb R^n$$, the above concepts can be described as follows:

- Closed under addition means that if we add any two real vectors vectors (i.e. $$\mathbf{u}, \mathbf{v} \in \mathbb R^n$$) then the result is also in $$\mathbb R^n$$). Addition in $$\mathbb R^n$$ is defined by $$(u_1, u_2, \dots, u_n)+(v_1, v_2, \dots, v_n) = (u_1+v_1, u_2+v_2, \dots, u_n+v_n)$$. Closure under addition is easy to understand -  adding any two real n-vectors there is no way to get a result that is not also a real n-vector. A way to say this mathematically is as follows:
if $$\mathbf{u}, \mathbf{v} \in \mathbb R^n$$, then $$ \mathbf{u}+\mathbf{v} \in \mathbb R^n$$
- Closed under scalar multiplication means that if we have any scalar number ($$s \in \mathbb R$$) and we multiply it by a  vector ($$\mathbf{v} \in \mathbb R^n$$) then the result is also a vector in $$\mathbb R^n$$.  Since scalar multiplication is defined by $$s(v_1, v_2, \dots, v_n) = (sv_1, sv_2, \dots, sv_n)$$ and multiplying a real number by a real number results in a real number this one is also true. Or we can say it as follows:
if $$s \in \mathbb R$$ and $$\mathbf{v} \in \mathbb R^n$$, then $$s\mathbf{v} \in \mathbb R^n$$.

The following are some properties of vector addition and scalar multiplication for vectors $$\mathbf{u}$$ and $$\mathbf{v}$$ and scalars $$c$$ and $$d$$:

**Vector Addition**

Two vectors in $$\mathbb R^n$$ (of the same size) can be added together by adding the corresponding elements, to form another vector in $$\mathbb R^n$$, called the sum of the vectors. For example:

$$ 
\left[
\begin{matrix}
    1  \\ 
    20   
 \end{matrix}
 \right]
 +
\left[
\begin{matrix}
    22 \\ 
    -3 
 \end{matrix}
 \right]
  =
\left[
\begin{matrix}
    23 \\ 
    17 
 \end{matrix}
\right]
$$

**Python Vector Addition**

Here is where things get tricky in Python.  If you try to add a list or tuple, Python does not do the vector addition as we defined above. In the following examples, notice that the two lists concatenate instead of adding by element: 


```python
## THIS IS WRONG
a = [1, 20]
b = [22,-3]
c = a+b
c
```
    [1, 20, 22, -3]

```python
## THIS IS ALSO WRONG
a = (1, 20)
b = (22,-3)
c = a+b
c
```
    (1, 20, 22, -3)

To do proper vector math you need either use a special function (we will learn these) or loop over the list.  Here is a very simplistic example:

```python
a = (1, 20)
b = (22,-3)
c = []
for i in range(len(a)):
    c.append(a[i] + b[i])
c
```
    [23, 17]

For fun, we can define this operation as a function in case we want to use it later:


```python
def vecadd(a,b):
    """Function to add two equal size vectors."""
    if (len(a) != len(b)):
        raise Exception('Error - vector lengths do not match')
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return c
```

```python
#Lets test it

vecadd(a,b)
```
    [23, 17]

**Scalar-Vector multiplication**

You can also multiply a scalar by a vector, which is done by multiplying every element of the vector by the scalar. 


$$ 
3
\left[
\begin{matrix}
    3 \\ 
    -7 \\
    10
 \end{matrix}
 \right]
  =
\left[
\begin{matrix}
    9 \\ 
    -21 \\
    30
 \end{matrix}
\right]
$$


**Scalar-Vector Multiplication in Python**

Again, this can be tricky in Python because Python lists do not do what we want.  Consider the following example that just concatenates three copies of the vector. 


```python
##THIS IS WRONG## 
z = 3
a = [3,-7,10]
c = z*a
c
```
    [3, -7, 10, 3, -7, 10, 3, -7, 10]

Again, in order to do proper vector math in Python you need either use a special function (we will learn these) or loop over the list.  

```python
def sv_multiply(z,a):
    c = []
    for i in range(len(a)):
        c.append(z*a[i])
    return c
```

```python
z = 3
a = [3,-7,10]
sv_multiply(z,a)
```
    [9, -21, 30]

### 1.3 Linear Combination

#### Linearity:

Two important properties of vector addition and scalar multiplication are:

1. **Additive Commutativity**: $$ \mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u} $$
2. **Distributive Property**: $$ k (\mathbf{u} + \mathbf{v}) = k\mathbf{u} + k\mathbf{v} $$

These properties make the set of all vectors (of a given dimension), equipped with these two operations, a vector space.

A linear combination is an expression constructed from a set of terms by multiplying each term by a constant and adding the results. 

#### Definition:

Given vectors $$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k $$ in a vector space $$ V $$ and scalars $$ c_1, c_2, \ldots, c_k $$, a linear combination of the vectors with these scalars as coefficients is:

$$ c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k $$

The scalars $$c_1, c_2, \ldots, c_k$$ are often referred to as the **coefficients** of the linear combination.

1. **Span**: The set of all possible linear combinations of a given set of vectors is called the span of those vectors. It describes the subspace of $ V $ that can be reached using the given vectors.

2. **Basis**: A set of vectors that spans a space and is linearly independent (no vector in the set can be written as a linear combination of the others) is called a basis for that space. The concept of a linear combination is crucial in defining a basis.

3. **Representation**: In many problems, especially in physics and engineering, we often represent a vector as a linear combination of some basic, fundamental vectors.

**Example:**

Consider the vectors:

$$ \mathbf{u} = \begin{bmatrix}
1 \\
2 \\
\end{bmatrix} $$

and 

$$ \mathbf{v} = \begin{bmatrix}
3 \\
1 \\
\end{bmatrix} $$

A linear combination of $$\mathbf{u}$$ and $$ \mathbf{v} $$ might be:

$$ 2\mathbf{u} - 3\mathbf{v} = 2\begin{bmatrix}
1 \\
2 \\
\end{bmatrix} - 3\begin{bmatrix}
3 \\
1 \\
\end{bmatrix} = \begin{bmatrix}
-5 \\
3 \\
\end{bmatrix} $$

Here, the coefficients are 2 and -3, respectively.

**Linear combination with Numpy**

In the following example, we are given two vectors (```x_np``` and ```y_np```), and two scalars (```alpha``` and ```beta```), we obtain the linear combination ```alpha*x_np + beta*y_np```.


```python
x_np = np.array([1,2])
y_np = np.array([3,4])
alpha = 0.5
beta = -0.8
c = alpha*x_np + beta*y_np
print(c)
```

    [-1.9 -2.2]


### 1.4 Linear Independence

A set of vectors $$ \{ \mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \} $$ in a vector space $$ V $$ is said to be **linearly independent** if the only linear combination that produces the zero vector is the one where all coefficients $$ c_i $$ are zero:

$$ c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k = \mathbf{0} $$

implies that 
$$ c_1 = c_2 = \ldots = c_k = 0 $$

If a set of vectors is not linearly independent, then it is **linearly dependent**. This means that one of the vectors can be expressed as a linear combination of the others. In many contexts, especially when trying to form a basis for a space, linear dependence is undesired because it means there's redundancy in our set of vectors.

#### Span:

As touched upon earlier, the **span** of a set of vectors is the set of all possible linear combinations of those vectors. Mathematically, if $$ S $$ is a subset of a vector space $$ V $$, the span of $$ S $$ is given by:

$$ \text{span}(S) = \{ c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_k\mathbf{v}_k \mid \mathbf{v}_i \in S, c_i \in \mathbb{F} \} $$

where $$ \mathbb{F} $$ is the field over which the vector space is defined (often the real numbers $$ \mathbb{R} $$).

#### Basis:

A **basis** of a vector space $$ V $$ is a set of linearly independent vectors that span $$ V $$. This means that every vector in $$ V $$ can be written as a unique linear combination of the basis vectors. The number of vectors in a basis of $$ V $$ is called the **dimension** of $$ V $$.

#### Canonical Basis:

The **canonical basis** (or standard basis) for $$ \mathbb{R}^n $$ is the set of vectors where one component is 1 and all other components are 0. For $$ \mathbb{R}^3 $$, the canonical basis is:

$$ \mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} $$

For any vector space that is isomorphic to $$ \mathbb{R}^n $$, there exists a canonical basis, though the specific representation might differ based on the context.
