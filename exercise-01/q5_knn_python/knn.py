import numpy as np

def knn_distance(element, samples, k):
    distArray = np.abs(samples-element)
    sort_distArray = np.sort(distArray)
    distance = sort_distArray[k-1]
    return distance

def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    answer = np.array([])
    sort_samples = np.sort(samples)
    N = np.size(samples)
    print('Value of k is ', k)
    for element in sort_samples:
        radius = knn_distance(element, sort_samples, k)
        volume = np.pi * (radius**2) # 2d circle as only area, no volume.
        prob = (k/N)*(1/volume)
        answer = np.append(answer, prob)
    estDensity = np.stack((sort_samples, answer), axis=1)
    return estDensity
