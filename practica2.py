import numpy as np

# sigmoid(z) computes the sigmoid of z.
def sigmoid(z):
# You need to return the following variable correctly
    g = 0 * z 
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar). You may find useful numpy.exp and numpy.power.

    g = 1 / (1 + np.power(np.e, -z))
# =============================================================
    return g


#   computeCost(X, y, theta) computes the cost of using theta as the
#   parameter for logistic regression 
def computeCost(X, y, theta):
    # Initialize some useful values
    m,n = X.shape
    # You need to return the following variable correctly 
    J = 0.0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost. You may find useful numpy.log
#               and the sigmoid function you wrote.
#
    h = sigmoid(X@theta)
    dota = np.dot(np.transpose(-y), np.log(h))
    dotb = np.dot(np.transpose((1 - y)), np.log(1 - h))
    J = (dota - dotb) / m
 # =============================================================

    return J

#   gradientDescent(X, y, theta, alpha, iterations) updates theta by
#   taking "iterations" gradient steps with learning rate alpha

def gradientDescent(X, y, theta, alpha, iterations):
    # Initialize some useful values
    m,n = X.shape
    # ====================== YOUR CODE HERE ======================

    for i in range(iterations):

        h = sigmoid(X @ theta)
        theta = theta - alpha * 1/m * X.T @ (h-y)
    # ============================================================
    return theta


#   predict(theta, X) computes the predictions for X using a threshold at 0.5
#   (i.e., if sigmoid(theta'*x) >= 0.5, predict 1, otherwise predict 0)
def predict(X,theta):
    # Initialize some useful values
    m,p = X.shape 
# You need to return the following variable correctly
    p=np.zeros((m,1))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned logistic regression parameters. 
#               You should set p to a vector of 0's and 1's

    p = sigmoid(X@theta)
    p =np.round(p)

# =========================================================================
    return p




