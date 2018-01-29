import random

import numpy as np

from calc_c import c_alg2
from fairness_calc import smooth_fairness

from bern_stochastic_dominance_ts import BernStochasticDominance
from bern_ts import BernThompsonSampling
import itertools

class BernFairStochasticDominance(BernStochasticDominance, BernThompsonSampling):

    def __init__(self, bandits, T, e2, delta, lam=1, mod=0):
        BernStochasticDominance.__init__(self, bandits, T, lam)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0
        self.mod = mod
        self.e2 = e2
        self.delta = delta
        self.pi = np.full((self.T, self.k), 1./self.k)
    def reset(self):
        BernStochasticDominance.reset(self)
        self.pi = np.full((self.T, self.k), 1./self.k)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0

    def run(self):
        o = np.ones(self.k)
        exploiting = False
        for t in range(self.T):

            if np.random.binomial(1, [self.lam])[0]:
                # empty dict evaluate to false
                if exploiting:
                    # exploition

                    self.rounds_exploiting = self.rounds_exploiting + 1
                    self.theta[t] = np.random.beta(self.s[t], self.f[t], self.k)
                    a = BernStochasticDominance.get_a(self,  t)

                else:
                    # exploration
                    self.rounds_exploring = self.rounds_exploring + 1
                    # if self.mod:
                    #     a = np.random.choice(tuple(o))
                    #     p = 1. / len(o)
                    #     for i in o:
                    #         self.pi[t][i] = p
                    #     print self.pi[t]

                    a = np.random.choice(self.k)

            else:
                a = BernThompsonSampling.get_a(self, t)

            # real bernoulli reward for each arm
            reward = self.bandits.pull(a)

            BernStochasticDominance.update(self, t, a, reward)
            if o[a] == 1 and (self.n[t, a] > c_alg2(self.e2, self.delta, self.bandits.get_mean(), a, self.k)):
                o[a] = 0
                if np.sum(o) == 0:
                    exploiting = True
        self.calc_r_h()
        self.calc_pi()


    def calc_r_h(self):

        self.r_h = np.divide(self.s[:self.T], self.s[:self.T] + self.f[:self.T])
        print self.r_h
    def get_r(self, perm, t):
        r_prop = []
        for i in range(self.k):
            if perm[i]:
                r_prop.append(self.r_h[t][i])
            else:
                r_prop.append(1-self.r_h[t][i])

        return r_prop



    def calc_pi(self):

        # r := binary reward vector
        r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=self.k)]
        r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]

        perm_prod = np.zeros((self.T, len(r_permutations)))
        print self.theta[self.rounds_exploring]
        print self.theta[self.rounds_exploring - 1]

        for t in range(self.rounds_exploring, self.T):

            for perm_i in range(len(r_permutations)):
                perm_prod[t][perm_i] = np.prod(self.get_r(r_permutations[perm_i], t))

            for a in range(self.k):
                for perm_i in range(len(r_permutations)):
                    if r_permutations[perm_i][a]:
                        self.pi[t][a] += perm_prod[t][perm_i]/r_sum[perm_i]
                    elif not r_sum[perm_i]:
                        self.pi[t][a] += perm_prod[t][perm_i]/self.k


        print self.pi
