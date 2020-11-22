import numpy as np

def gkde(ele, v, example):
    insE = -0.5 * ((ele - example) / v) ** 2
    norm = 1 / (v * np.sqrt(2 * np.pi))
    res = norm * np.exp(insE)
    prob = np.average(res)
    return prob
        
def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    answer = np.array([])
    estDensity = np.array([])
    sortSamples = np.sort(samples)
    for ele in sortSamples:
        gkernel = gkde(ele, h, sortSamples)
        answer =  np.append(answer, gkernel)
    estDensity = np.stack((sortSamples, answer), axis=1)
    return estDensity
