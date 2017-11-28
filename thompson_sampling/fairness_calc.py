
def smooth_fairness(e1, e2, pi, r, distance):
    not_fair = 0
    fair = 0
    k = len(r)
    # print pi
    # print r

    for i in range(k):
        for j in range(i+1, k):

            d_pi = distance([pi[i], 1 - pi[i]], [pi[j], 1 - pi[j]])
            d_r = distance([r[i], 1 - r[i]], [r[j], 1 - r[j]])
            if d_pi > (e1 * d_r + e2):
                not_fair = not_fair + 1
            else:
                fair = fair + 1

    return not_fair, fair

