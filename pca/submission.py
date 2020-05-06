# PACKAGE: DO NOT EDIT THIS LINE
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

import sklearn
from ipywidgets import interact
from load_data import load_mnist
MNIST = load_mnist()
images = MNIST['data'].astype(np.double)
labels = MNIST['target'].astype(np.int)

# GRADED FUNCTION: DO NOT EDIT THIS LINE

def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product"""
    return np.sqrt(np.dot(x0-x1, x0-x1))

def angle(x0, x1):
    """Compute the angle between two vectors x0, x1 using the dot product"""
    mx0 = np.dot(x0, x0)
    mx1 = np.dot(x1, x1)
    return np.arccos(np.dot(x0,x1)/np.sqrt(mx0*mx1))


# GRADED FUNCTION: DO NOT EDIT
def most_similar_image():
    """Find the index of the digit, among all MNIST digits
       that is the second-closest to the first image in the dataset (the first image is closest to itself trivially). 
       Your answer should be a single integer.
    """
    ref = images[0] # reference image
    result = np.linalg.norm(images[1:].astype(np.float) - ref.astype(np.float), axis=1)
    index = np.argmin(result)+1
    return index # 60

# GRADED FUNCTION: DO NOT EDIT

def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)
    
    Returns
    --------
    distance_matrix: matrix of shape (N, M), each entry distance_matrix[i,j] is the distance between
    ith row of X and the jth row of Y (we use the dot product to compute the distance).
    """
    assert X.ndim == 2
    assert Y.ndim == 2 
    return scipy.spatial.distance_matrix(X, Y)

# GRADED FUNCTION: DO NOT EDIT THIS LINE

def KNN(k, X, y, x):
    """K nearest neighbors
    k: number of nearest neighbors
    X: training input locations
    y: training labels
    x: test input
    """
    N, D = X.shape
    num_classes = len(np.unique(y))
    dist = pairwise_distance_matrix(X, x.reshape(1, -1)).reshape(-1)

    # Next we make the predictions
    ypred = np.zeros(num_classes)
    classes = y[np.argsort(dist)][:k] # find the labels of the k nearest neighbors
    for c in np.unique(classes):
        ypred[c] = len(classes[classes == c])
        
    return np.argmax(ypred)

