import numpy as np
import random
import itertools
import thompson_sampling.calc_c
from fairness_calc import isclose
from distance import *
from thompson_sampling.calc_c import c_alg

class Bandits:

    def __init__(self, arms, data_set_name='no_name'):
        self.arms = arms
        self.k = len(arms)
        self.theta = self.get_mean()
        self.p_star = self.calc_p_star()
        self.data_set_name = data_set_name


    def get_mean(self):
        t = np.zeros(self.k)
        for i in range(self.k):
            t[i] = np.mean(self.arms[i])
        return t

    def pull(self, a):
        return random.choice(self.arms[a])

    def calc_p_star(self):
        p_star = np.zeros(self.k)

        r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=self.k)]
        r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]
        perm_prod = np.zeros(len(r_permutations))
        for perm_i in range(len(r_permutations)):
            perm_prod[perm_i] = np.prod(self.get_r(r_permutations[perm_i]))

        for a in range(self.k):
            for perm_i in range(len(r_permutations)):
                if r_permutations[perm_i][a]:
                    p_star[a] += perm_prod[perm_i] / r_sum[perm_i]
                elif not r_sum[perm_i]:
                    p_star[a] += perm_prod[perm_i] / self.k

        if not isclose(sum(p_star), 1.0):
            print "p star doesn't sum to one"
        return p_star

    def get_r(self, perm):
        r_prop = []
        for i in range(self. k):
            if perm[i]:
                r_prop.append(self.theta[i])
            else:
                r_prop.append(1 - self.theta[i])

        return r_prop


    def get_max_D(self, distance=total_variation_distance):
        dmax = 0.
        for i in range(self.k):
            for j in range(self.k):
                dmax = max(distance([self.theta[i], 1-self.theta[i]], [self.theta[j], 1-self.theta[j]]), dmax)
        return dmax