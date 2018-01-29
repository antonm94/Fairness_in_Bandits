from distance import total_variation_distance
import numpy as np
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
    d_pi = distance([pi[i], 1 - pi[i]], [pi[j], 1 - pi[j]])
    d_r = distance([r[i], 1 - r[i]], [r[j], 1 - r[j]])

    if (d_pi <= (e1 * d_r + e2)):# or isclose(d_pi, (e1 * d_r + e2)):
        return 1
    else:
        print "d_pi: {}".format(d_pi)
        print "d_r: {}".format(d_r)


        print "Arm i: {}".format(i)+'with r: {}'.format(r[i]) + "and pi: {}".format(pi[i])
        print "Arm j: {}".format(j)+'with r: {}'.format(r[j]) + "and pi: {}".format(pi[j])
        print "distance: {}".format(d_pi - (e1 * d_r + e2))



        return 0
