from load_data import load_data
from distance import total_variation_distance
from test_instances.ts_test import TSTest
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.sd_ts_test import SDTest
import numpy as np
import test_instances.plots as test
import itertools
from fairness_calc import smooth_fairness


T = 1000
e1 = 2
e2 = 0
delta = 0
DATA_SET = ['Bar Exam', 'Default on Credit'][0]

def get_r(perm, p):
    r_prop = []
    for i in range(k):
        if perm[i]:
            r_prop.append(p[i])
        else:
            r_prop.append(1 - p[i])

    return r_prop


def check_if_pi_correct():
    s = np.full(k, .5)
    s = np.array([20.5, 10.5, 12.5, 23.5, 17.5])
    f = np.full(k, .5)
    f = np.array([9.5, 11.5, 10.5, 8.5, 9.0])
    n = np.empty(k)
    r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=k)]
    r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]
    perm_prod = np.zeros(len(r_permutations))
    for t in range(T + 1):
        theta = np.random.beta(s, f, k)
        guessed_r = np.random.binomial(1, theta)
        a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])
        reward = bandits.pull(a)

        # if reward:
        #     s[a] = s[a] + 1
        # else:
        #    f[a] = f[a] + 1

        n[a] = n[a] + 1
    r_h = np.divide(s, s + f)
    for perm_i in range(len(r_permutations)):
        perm_prod[perm_i] = np.prod(get_r(r_permutations[perm_i], r_h))
    pi = np.zeros(k)
    for a in range(k):
        for perm_i in range(len(r_permutations)):
            if r_permutations[perm_i][a]:
                pi[a] += perm_prod[perm_i] / r_sum[perm_i]
            elif not r_sum[perm_i]:
                pi[a] += perm_prod[perm_i] / k
    for i in range(k):
        for j in range(k):
            smooth_fairness(2, 0, i, j, pi, r_h)
    print np.divide(n, T)
    print pi


def check_subjective_smooth():
    s = np.full(k, .5)
    f = np.full(k, .5)
    n = np.empty(k)
    r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=k)]
    r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]
    perm_prod = np.zeros(len(r_permutations))
    for t in range(T + 1):
        b = True
        theta = np.random.beta(s, f, k)
        guessed_r = np.random.binomial(1, theta)
        a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])
        reward = bandits.pull(a)

        if reward:
            s[a] = s[a] + 1
        else:
            f[a] = f[a] + 1

        r_h = np.divide(s, s + f)
        for perm_i in range(len(r_permutations)):
            perm_prod[perm_i] = np.prod(get_r(r_permutations[perm_i], r_h))
        pi = np.zeros(k)
        for a in range(k):
            for perm_i in range(len(r_permutations)):
                if r_permutations[perm_i][a]:
                    pi[a] += perm_prod[perm_i] / r_sum[perm_i]
                elif not r_sum[perm_i]:
                    pi[a] += perm_prod[perm_i] / k
        for i in range(k):
            for j in range(k):
                b = b and smooth_fairness(2, 0, i, j, pi, r_h)

        #print 'in step {}'.format(t) + 'subjective smooth fair for all i,j {}'.format(b)


if __name__ == '__main__':
    bandits = load_data(DATA_SET)
    k = bandits.k

    check_subjective_smooth()
