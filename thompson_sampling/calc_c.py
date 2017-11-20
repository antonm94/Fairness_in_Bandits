import numpy as np
import math
from total_variation_distance import total_variation_distance


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