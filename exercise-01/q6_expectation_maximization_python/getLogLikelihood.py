import numpy as np
from numpy.linalg import inv

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    ar_means = np.array(means)
    ar_weights = np.array(weights)
    ar_covariances = np.array(covariances)
    ar_X = np.array(X)
    K = ar_means.shape[0]
    D = ar_means.shape[1]
    N = ar_X.shape[0]

    logLikelihood = 0

    for n in range(N):
        x_n = ar_X[n,:]
        result = 0
        for k in range(K):
            u = ar_means[k,:]
            v = ar_covariances[:,:,k]
            v_inv = inv(v)
            diff = np.matrix(x_n - u)
            diffT = diff.T
            expo = diff * np.matrix(v_inv) * diffT
            insE = -0.5 * expo[0][0]
            norm = 1 / (np.sqrt(np.linalg.det(v)) * 2 * np.pi) # d/2 is 1
            result += (ar_weights[k] * norm * np.exp(insE))
        logLikelihood += np.log(result)
    return logLikelihood

