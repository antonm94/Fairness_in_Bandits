import random

import numpy as np

from calc_c import c_alg2
from fairness_calc import smooth_fairness

from bern_stochastic_dominance_ts import BernStochasticDominance
from bern_thompson_sampling import BernThompsonSampling

class FairStochasticDominance(BernStochasticDominance, BernThompsonSampling):

    def __init__(self, bandits, T, e2, delta, lam=1, mod=0):
        BernStochasticDominance.__init__(self, bandits, T, lam)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0
        self.mod = mod
        self.e2 = e2
        self.delta = delta

    def reset(self):
        BernStochasticDominance.reset(self)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0

    def run(self):
        exploiting = False
        for t in range(self.T):
            if np.random.binomial(1, [self.lam])[0]:
                # O(t)={i:n_j,i(t) <=C(e2,delta)}

                o = set()
                if not exploiting:
                    for i in range(self.k):
                        if t > 0:
                            if self.n[t-1, i] <= c_alg2(self.e2, self.delta, self.bandits.get_mean(), i, self.k):
                                 o.add(i)
                        else:
                            o.add(i)
                    if len(o) == 0:
                        exploiting = True

                if exploiting:
                    # exploition

                    self.rounds_exploiting = self.rounds_exploiting + 1
                    self.theta[t] = np.random.beta(self.s, self.f, self.k)
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
                    self.pi[t] = np.full(self.k, 1. / self.k)

            else:
                a = BernThompsonSampling.get_a(self, t)

            # real bernoulli reward for each arm
            reward = self.bandits.pull(a)

            BernStochasticDominance.update(self, t, a, reward)
        print self.rounds_exploring
        print self.rounds_exploiting

