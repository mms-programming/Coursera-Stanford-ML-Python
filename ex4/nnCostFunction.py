import numpy as np

from ex2.sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def convertLabelToVector(value, num_labels):
  y = np.zeros(num_labels)
  y[value - 1] = 1
  return y

def calculate_cost(output, value):
  part_A = np.multiply(np.dot(-1, value), np.log(output))
  part_B = np.multiply((1 - value), np.log(1 - output))

  return np.sum(part_A) - np.sum(part_B)

#def calculate_regularization(Theta1_NBias, theta2_NBias):

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    """computes the cost and gradient of the neural network. The
  parameters for the neural network are "unrolled" into the vector
  nn_params and need to be converted back into the weight matrices.

  The returned parameter grad should be a "unrolled" vector of the
  partial derivatives of the neural network.
    """

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
# Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                       (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                       (num_labels, (hidden_layer_size + 1)), order='F').copy()

# Setup some useful variables
    m, _ = X.shape

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

    # Calculate the hidden_layer values for all training examples
    # Don't forget to sigmoid the output!
    hidden_layer = sigmoid(np.dot(X, Theta1.transpose()))

    # Add ones to the hidden layer for the basis
    hidden_layer = np.column_stack((np.ones(m), hidden_layer))

    # Calculate the otuput values for all training examples
    # Don't forget to sigmoid the output!
    output = sigmoid(np.dot(hidden_layer, Theta2.transpose()))

    '''
      Iterative approach. For each training example (row) we will convert
      the corresponding y value to a vector of 0's and 1 and then call
      the calculate_cost function for that particular training example. This
      includes everything inside the K summation.

      We then keep a running tally of the cost and sum the cost values across
      all training examples to get a total cost. We then divide by m to get
      J.
    '''
    total_cost = 0
    for row in range(m):
      total_cost = total_cost + \
                   calculate_cost(output[row, :], convertLabelToVector(y[row], num_labels))

    regular_cost = total_cost / m

    Theta1_NBias = np.delete(Theta1, 0, 1)
    Theta2_NBias = np.delete(Theta2, 0, 1)

    partA = np.square(Theta1_NBias)
    partA_sum = np.sum(np.sum(partA, 0))

    partB = np.square(Theta2_NBias)
    partB_sum = np.sum(np.sum(partB, 0))

    total_sum = (partA_sum + partB_sum)
    regularization = total_sum * Lambda / (2 * m)

    J = regular_cost + regularization
# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the code by working through the
#               following parts.
#
# Part 1: Feedforward the neural network and return the cost in the
#         variable J. After implementing Part 1, you can verify that your
#         cost function computation is correct by verifying the cost
#         computed in ex4.m
#
# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.
#
#         Hint: We recommend implementing backpropagation using a for-loop
#               over the training examples if you are implementing it for the
#               first time.
#
# Part 3: Implement regularization with the cost function and gradients.
#
#         Hint: You can implement this around the code for
#               backpropagation. That is, you can compute the gradients for
#               the regularization separately and then add them to Theta1_grad
#               and Theta2_grad from Part 2.
#



    # -------------------------------------------------------------

    # =========================================================================

    # Unroll gradient
    grad = 0 #np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))


    return J, grad
