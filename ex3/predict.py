import numpy as np

from ex2.sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#

# =========================================================================

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.column_stack.html
    # See also predictOneVsAll.py in ex3
    X = np.column_stack((np.ones(shape=(m,1)), X))
    hidden_layer = sigmoid(np.dot(X, Theta1.transpose()))
    hidden_layer = np.column_stack((np.ones(m), hidden_layer))

    output = sigmoid(np.dot(hidden_layer, Theta2.transpose()))
    p = np.argmax(output, axis=1)

    return p + 1        # add 1 to offset index of maximum in A row
