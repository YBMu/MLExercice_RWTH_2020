import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    logLikelihood = 0
    N = gamma.shape[0]
    K = gamma.shape[1]
    D = X.shape[1]
    matGamma = np.matrix(gamma)
    matGammaT = matGamma.T # KxN
    matX = np.matrix(X) # NxD
    
    matMeans = matGammaT * matX # KxD
    means = np.array(matMeans)
    
    N_new = np.array(np.sum(matGammaT, axis=1)) # Kx1
    
    means = means / N_new
    weights = N_new / N

    #covariances = np.zeros((D, D, K))
    #for k in range(K):
        #kgamma = matGamma[:,k]
        #diff = matX - matMeans[k,:] # NxD
        #gammaDiff = np.multiply(kgamma, diff)
        #diffT = diff.T
        #prod = diff.T * gammaDiff
        #covariances[:, :, k] = prod/N_new[k]
    covariances = np.zeros((D,D, K))    
    for i in range(K):
        auxSigma = np.zeros((D,D))
        for j in range(N):
            meansDiff = X[j] - means[i]
            auxSigma = auxSigma + gamma[j, i] * np.outer(meansDiff.T, meansDiff)
        covariances[:, :, i] = auxSigma/N_new[i]  
    return weights, means, covariances, logLikelihood
