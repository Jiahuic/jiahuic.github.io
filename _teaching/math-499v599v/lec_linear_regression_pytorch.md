---
layout: page
permalink: /teaching/math-499v599v/lec_lrp/
title: Introduction to Linear Regression
---

## Code Linear Regression from Scratch
In the homework, you have learned how to implement linear regression using scikit-learn. 
Here, we will check how to implement linear regression using Numpy and PyTorch.
```python
# Load the data and split it into train and test
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv('../datasets/Boston.csv')
X = data.drop('medv', axis=1)
y = data['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Call the sklearn linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
```
When print the mean squared error, you will get 21.538929180643528.
The idea for us is to implement the gradient descent algorithm to find the best parameters for the linear regression model.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

class linear_regression:
    def __init__(self, N):
        # N = number of features
        self.intercept_ = 0
        self.coef_ = np.zeros((N,), dtype=float)
        self.scaler = StandardScaler()

    def forward(self, X):
        y = np.dot(X, self.coef_) + self.intercept_
        return y

    def fit(self, X, y, lr = 0.001, epochs = 20000):
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        N, M = X.shape
        error = 100
        for epoch in range(epochs):
            y_pred = self.forward(X)
            error = y_pred - y
            grad_coff = np.dot(X.T, error) / M # the gradient of the cost function with respect to the coefficients
            grad_intercept = np.sum(error) / M # the gradient of the cost function with respect to the intercept
            self.coef_ -= lr * grad_coff
            self.intercept_ -= lr * grad_intercept
        return

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.forward(X)

my_lin_reg = linear_regression(X_train.shape[1], X_train.shape[0])
my_lin_reg.fit(X_train, y_train)
y_pred = my_lin_reg.predict(X_test)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
```
To increase the performance of the model, we can adjust the learning rate `lr` and the number of epochs `epochs`. 
```python
my_lin_reg.fit(X_train, y_train, lr = 0.01, epochs = 100)
```
Now, let's move to the PyTorch implementation. First we need to install PyTorch by `pip3 install torch torchvision torchaudio`.
```python
import torch

class linear_regression_pytorch:
    def __init__(self, X_shape, lr=0.01, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        # you can use nn.Linear instead of defining the coefficients and intercept
        # Here, we want to break down the process to understand the gradient descent algorithm
        self.coef_ = torch.zeros(X_shape, 1, requires_grad=True)
        self.intercept_ = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        y_hat = X @ self.coef_ + self.intercept_
        return y_hat

    def fit(self, X, y):
        for epoch in range(self.epochs):
            y_hat = self.forward(X)
            loss = ((y_hat - y)**2).mean()
            loss.backward()
            error = (y_hat - y)
            with torch.no_grad():
                self.coef_ -= self.lr * self.coef_.grad
                self.intercept_ -= self.lr * self.intercept_.grad

                self.coef_.grad.zero_() # Zero the gradients after updating
                self.intercept_.grad.zero_()

        return self

my_lin_reg = my_linear_regression(X_train.shape[1])
my_lin_reg.fit(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))

y_pred = my_lin_reg.forward(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
print(mean_squared_error(y_test, y_pred))
```
By running the code, you will get a mean squared error of 71.09... Why the error is high?
