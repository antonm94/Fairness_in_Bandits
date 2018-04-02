from divergence import total_variation_distance
from divergence import kl_divergence
import numpy as np
import math
import logging
import normalBandits
logging.basicConfig(filename='fairness.log',level=logging.DEBUG)

# def smooth_fairness(e1, e2, pi, r, distance):
#     not_fair = 0
#     fair = 0
#     k = len(r)
#     # print pi
#     # print r
#
#     for i in range(k):
#         for j in range(i+1, k):
#
#             d_pi = distance([pi[i], 1 - pi[i]], [pi[j], 1 - pi[j]])
#             d_r = distance([r[i], 1 - r[i]], [r[j], 1 - r[j]])
#             if d_pi > (e1 * d_r + e2):
#                 not_fair = not_fair + 1
#             else:
#                 fair = fair + 1
#
#     return not_fair, fair
#
#
def isclose(a, b, rel_tol=1e-05, abs_tol=1e-08):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def smooth_fairness(e1, e2, d_pi, d_r):
    if (d_pi <= (e1*d_r + e2)):# or isclose(d_pi, (e1 * d_r + e2)):
        return 1
    else:
        # s = 'd_pi: {}'.format(d_pi) + 'd_r: {}'.format(d_r) + 'Arm i: {}'.format(i) \
        #     + ' with r: {}'.format(r[i]) + ' and pi: {}'.format(pi[i]) + 'Arm j: {}'.format(j) \
        #     + ' with r: {}'.format(r[j]) + 'and pi: {}'.format(pi[j]) + 'distance: {}'.format(d_pi - (e1 * d_r + e2))
        # logging.info(s)
        # if d_pi == float('Inf'):
        #     print d_r
        #
        #     print pi
        return 0


def get_e1_smooth_fairness(e2, d_pi, d_r):
    # if d_pi == np.Inf:


    if d_pi == 0:
        e1 = 0.
    elif d_r == 0:
        if d_pi > e2:
            e1 = math.inf
            print "smooth fairness not possible"
    else:
        if isclose(d_pi, d_r):
            e1 = max(1.-(e2/d_r), 0.0)
        else:
            e1 = max((d_pi - e2) / d_r, 0.0)
    # print e1
    # # if e1 == 0.:
    # print 'dr: ' + str(d_r)
    # print 'd_pi: ' + str(d_pi)
    # print 'e1: '+ str(e1)
    return e1