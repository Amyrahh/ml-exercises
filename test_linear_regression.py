import numpy as np
from linear_regression import cost_function, gradient_descent

univariate_data = np.genfromtxt('data/ex1data1.txt', delimiter=',')


def test_cost_function_univariate_zero_theta():
    bias = 0
    weights = np.zeros((1,1))
    X = univariate_data[:,:1]
    y = univariate_data[:,1].reshape(-1,1)

    cost, grad_weights, grad_bias = cost_function(X, y, weights, bias)

    assert round(cost, 2) == 32.07, "Wrong cost"


def test_cost_function_univariate_random_theta():
    bias = -1
    weights = np.ones((1,1)) * 2
    X = univariate_data[:,:1]
    y = univariate_data[:,1].reshape(-1,1)

    cost, grad_weights, grad_bias = cost_function(X, y, weights, bias)

    assert round(cost, 2) == 54.24, "Wrong cost"


def test_gradient_descent_univariate():
    bias = 0
    weights = np.zeros((1,1))
    X = univariate_data[:,:1]
    y = univariate_data[:,1].reshape(-1,1)

    weights, bias = gradient_descent(X, y, weights, bias, alpha=0.01, iterations=1500)

    assert round(bias, 4) == -3.6303, "Wrong bias"
    assert round(weights.item(), 4) == 1.1664, "Wrong weights"
