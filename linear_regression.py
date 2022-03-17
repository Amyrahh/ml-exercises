import numpy as np

def predict(X, weights, bias):
    """ Predict y_hat given X, weights, and bias

        Input:
        X (np.array): data point(s) to predict for
        weights (np.array): theta_1 to theta_n
        bias (float): theta_0

        Output:
        y_hat: value(s) predicted from given X
    """
    y_hat = np.matmul(X,weights) + bias
    return y_hat


def cost_function(X, y, weights, bias):
    """ Calculate cost and gradient for given X, y, using weights, and bias

        Input:
        X (np.array): data point(s) to predict for
        y (np.array): true y value for given X
        weights (np.array): theta_1 to theta_n
        bias (float): theta_0

        Output:
        cost: cost of predicting y_hat using given weights, and bias
        grad_weights: gradient of cost_function at weights, and bias
        grad_bias: gradient of cost_function at weights, and bias
    """
    m = X.shape[0]
    y_hat = predict(X, weights, bias)
    error = y_hat - y
    cost = 1/(2*m) * (np.sum(error))**2
    grad_bias = 1/m * np.sum(error)
    grad_weights = 1/m * np.sum(error) * X
    return cost, grad_bias, grad_weights


def gradient_descent(X, y, weights, bias, alpha, iterations):
    """ Find optimal values for weights and bias

        Input:
        X (np.array): data point to predict for
        y (np.array): true y value for given X
        weights (np.array): theta_1 to theta_n
        bias (float): theta_0
        alpha (float): learning rate
        iterations (int): number of iterations to run for

        Output:
        weights (np.array): optimal weights
        bias (float): optimal bias
    """
    cost, grad_bias, grad_weights = cost_function(X, y, weights, bias)
    weights = weights - alpha*grad_weights
    bias = bias - alpha*grad_bias
    return weights, bias
