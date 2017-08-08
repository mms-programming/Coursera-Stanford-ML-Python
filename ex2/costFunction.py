from numpy import log
from sigmoid import sigmoid
import numpy as np

def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

    # X = np.array([[1, 8, 1, 6],[1, 3, 5, 7],[1, 4, 9, 2]]);
    # y = np.array([1, 0, 1]);
    # theta = np.array([-2, -1, 1, 2]);

# Initialize some useful values
    m = y.size # number of training examples


# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#

    # DO NOT USE -y!!!
    h_theta = sigmoid(np.dot(X, theta))
    part_A = np.dot(np.dot(-1, y), np.log(h_theta))
    part_B = np.dot((1 - y), np.log(1 - h_theta))

    J = (part_A - part_B) / m

    return J
