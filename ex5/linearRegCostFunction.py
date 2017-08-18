import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#


# =========================================================================

    h_theta = np.dot(X, theta)
    regular_cost = sum(np.square(h_theta - y)) / (2 * m )
    regularization = sum(np.square(theta[1:,])) * Lambda / (2 * m)

    J = regular_cost + regularization

    inside = h_theta - y
    
    grad = np.dot(inside, X) / m
    grad[1:,] = grad[1:,] + Lambda * theta[1:] / m
    return J, grad
