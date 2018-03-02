import math

import numpy as np

from distance import total_variation_distance


def c_alg2(e2, delta, realTheta, i, k):
    divergence = np.zeros(k)
    for j in range(k):
        divergence[j] = total_variation_distance([realTheta[i], 1 - realTheta[i]], [realTheta[j], 1 - realTheta[j]])

    return (pow(2 * max(divergence) + 1, 2) * (math.log(2) - math.log(delta))) / (2 * pow(e2, 2))


def c_alg3(e2, delta, realTheta, i, k):
    divergence = np.zeros(k)
    for j in range(k):
        divergence[j] = total_variation_distance([realTheta[i], 1 - realTheta[i]], [realTheta[j], 1 - realTheta[j]])

    return (pow(2 * max(divergence) + 1, 2) * pow(k, 2) * (math.log(2) - math.log(delta))) / (2 * pow(e2, 2))


#kullback_divergence[j] = stats.entropy([realTheta[i], 1 - realTheta[i]], [realTheta[j], 1 - realTheta[j]])

def c_alg(e2_arr, delta_arr, realTheta):
    k = len(realTheta)
    max_divergence=0
    for i in range(k):
        for j in range(k):
           max_divergence = max(total_variation_distance([realTheta[i],
                                1 - realTheta[i]], [realTheta[j], 1 - realTheta[j]]), max_divergence)

    c = np.zeros((len(e2_arr), len(delta_arr)))
    for e2_ind, e2, in enumerate(e2_arr):
        for delta_ind, delta in enumerate(delta_arr):
           c[e2_ind, delta_ind] = (pow(2.*max_divergence + 1, 2.)  * (math.log(2) - math.log(delta))) / (2 * pow(e2, 2.))

    return c