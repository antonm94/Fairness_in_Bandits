import math

import numpy as np

from divergence import total_variation_distance


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
           #max_divergence = 1
    c = np.zeros((len(e2_arr), len(delta_arr)))
    for e2_ind, e2, in enumerate(e2_arr):
        for delta_ind, delta in enumerate(delta_arr):
           c[e2_ind, delta_ind] = (pow(2.*max_divergence + 1, 2.)  * (math.log(2) - math.log(delta))) / (2 * pow(e2, 2.))

    return c

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    e1 = [2, 1, 0.5]
    e2 = [0.05, 0.2, 0.001]#np.finfo(float).eps]
    delta = [0.3, 0.7, 0.001]#np.finfo(float).eps]
    e2 = [0.01, 0.05, 0.1, 0.2]
    delta = [0.01, 0.1, 0.3, 0.5]
    print c_alg(e2, delta, [0.0011, 0., 1., 0.9703, 0.9633])
    print c_alg([0.05], [0.1], [0.91883455, 0.77615216, 0.87667304, 0.89201878, 0.96675656])
    print c_alg(e2, delta, [ 0.4994,  0.5002,  0.5039,  0.5058,  0.5071]
)
