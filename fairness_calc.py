from distance import total_variation_distance
import numpy as np
import logging
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
def isclose(a, b, rel_tol=1e-05, abs_tol=np.finfo(float).eps):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def smooth_fairness(e1, e2, i, j, pi, r, distance=total_variation_distance):
    d_pi = distance([pi[i], 1 - pi[i]], [pi[j], 1 - pi[j]]) #2*distance missing *0.5 in calc
    d_r = distance([r[i], 1 - r[i]], [r[j], 1 - r[j]])

    if (d_pi <= (e1*d_r + e2)):# or isclose(d_pi, (e1 * d_r + e2)):
        return 1
    #else:
    #     s = 'd_pi: {}'.format(d_pi) + 'd_r: {}'.format(d_r) + 'Arm i: {}'.format(i) + ' with r: {}'.format(r[i])
    #     + ' and pi: {}'.format(pi[i]) + 'Arm j: {}'.format(j) + ' with r: {}'.format(r[j]) + 'and pi: {}'.format(pi[j])
    #     + 'distance: {}'.format(d_pi - (e1 * d_r + e2))
    #     logging.info(s)

    return 0


def get_e1_smooth_fairness(e2, i, j, pi, r, distance=total_variation_distance):
    # print pi
    # print r
    d_pi = distance([pi[i], 1 - pi[i]], [pi[j], 1 - pi[j]])  # 2*distance missing *0.5 in calc
    d_r = distance([r[i], 1 - r[i]], [r[j], 1 - r[j]])
    # print "d_r: {}".format(d_r)
    # print 'd_pi: {}'.format(d_pi)
    if d_r == 0.:
        e1 = 0.
    else:
        e1 = max((d_pi - e2) / d_r, 0.0)

    return e1