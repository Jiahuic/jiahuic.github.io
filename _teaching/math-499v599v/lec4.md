---
layout: page
permalink: /teaching/math-499v599v/lec4/
title: Introduction to Linear Regression
---

# Linear Regression
Linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression. 
The linear regression method is for finding the relationship between the dependent variable and one or more independent variables with the label of a real number,
while the classificaiton method is for finding the relationship between the dependent variable and one or more independent variables.

## The Linear Regression Model
* **Simple Linear Regression (SLR)** involves a single independent variable to predict the dependent variable. Th e model is represented by \(y = \beta_0 + \beta_1x + \epsilon\), where \(y\) is the dependent variable, \(x\) is the independent variable, \(\beta_0\) is the y-intercept, \(\beta_1\) is the slope, and \(\epsilon\) is the error term.
* **Multiple Linear Regression (MLR)** extends SLR by including multiple independent variables, represented as \(y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon\).

## Estimating the Coefficients
The least squares method is the most common approach to estimate the coefficients (\(\beta\)) of the linear regression model. It involves minimizing the sum of the squared differences between the observed and predicted values.

**Gradient Descent** is an alternative optimization algorithm for estimating the coefficients, particularly in large datasets where the least squares method may be computationally expensive.

## Model Evaluation Metrics
- **Coefficient of Determination (\(R^2\))**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Adjusted \(R^2\)**: Adjusts \(R^2\) for the number of predictors in the model, providing a more accurate measure for MLR.
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**: Measure the average of the squares of the errors or deviations.

## Extensions of Linear Regression
Explores advanced forms of linear regression that incorporate regularization (Ridge and Lasso) to prevent overfitting by penalizing large coefficients, and Elastic Net, which combines the penalties of Ridge and Lasso.

## Practical Implementation
Illustrates the application of linear regression using Python's scikit-learn library. It covers the process from data preparation and model fitting to evaluation and result visualization, providing code examples for clarity.
Here is an example for the implementation of linear regression using scikit-learn:
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model using the training set
model.fit(X_train, y_train)

# Make predictions using the test set
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
```
