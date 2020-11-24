import numpy as np
from getLogLikelihood import getLogLikelihood
from numpy.linalg import inv

def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    ar_means = np.array(means)
    ar_weights = np.array(weights)
    ar_covariances = np.array(covariances)
    ar_X = np.array(X)
    K = ar_means.shape[0]
    D = ar_means.shape[1]
    N = ar_X.shape[0]
    g = np.zeros((N,K))
    gamma = np.matrix(g)
    for n in range(N):
        x_n = ar_X[n,:]
        res_k = []
        res_accumulated = 0
        for k in range(K):
            u = ar_means[k,:]
            v = ar_covariances[:,:,k]
            v_inv = inv(v)
            diff = np.matrix(x_n - u)
            diffT = diff.T
            expo = diff * np.matrix(v_inv) * diffT
            insE = -0.5 * expo[0][0]
            norm = 1 / (np.sqrt(np.linalg.det(v)) * 2 * np.pi) # d/2 is 1
            result = np.array((ar_weights[k] * norm * np.exp(insE)))
            res_k = np.append(res_k, result)
            res_accumulated += result
        res_k /= res_accumulated[0][0]
        gamma[n,:] = res_k
    return [logLikelihood, gamma]
