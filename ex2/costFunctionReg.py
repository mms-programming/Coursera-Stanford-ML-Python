from costFunction import costFunction
import numpy as np

def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    # X = np.array([[1, 8, 1, 6],[1, 3, 5, 7],[1, 4, 9, 2]]);
    # y = np.array([1, 0, 1]);
    # theta = np.array([-2, -1, 1, 2]);

    # Initialize some useful values
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

# =============================================================

    regular_cf = costFunction(theta, X, y)
    regularization = Lambda * np.sum(np.square(theta[1:])) / (2.0 * m)

    return regular_cf + regularization
