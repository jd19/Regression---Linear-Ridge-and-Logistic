# Regression---Linear-Ridge-and-Logistic

This contains implementation of the above algorithms from scratch without using any additional libraries or frameworks

## Linear Regression

The data is a matrix (100; 2). Column 1 is x and Column 2 is y.
using linear regression, we find the line which best fits the data.

## Ridge regression with polynomials and cross-validation

The data is a matrix (100; 2). Column 1 is x and Column 2 is y.
We fit a polynomial of degree 15 to the data using ridge regression. i.e. x is converted to [1; x; x2; x3; : : : ; x15]>. Using 5-fold cross
validation, estimate the best fit from the set, s = [0:01; 0:05; 0:1; 0:5; 1:0; 5; 10]

## Logistic Regression

rain a logistic regression classier to classify two set of digits from the MNIST
dataset.
