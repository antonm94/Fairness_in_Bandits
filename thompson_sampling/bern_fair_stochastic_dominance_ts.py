import random

import numpy as np

from calc_c import c_alg2
from calc_c import c_alg
from fairness_calc import smooth_fairness
from bern_stochastic_dominance_ts import BernStochasticDominance
from bern_ts import BernThompsonSampling
import itertools
from sets import Set


class BernFairStochasticDominance(BernStochasticDominance, BernThompsonSampling):

    def __init__(self, bandits, T, e2, delta, lam=1, mod=0, c=0, smart_exploration=False):
        BernStochasticDominance.__init__(self, bandits, T, lam)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0
        self.mod = mod
        self.e2 = e2
        self.delta = delta
        self.c = c
        self.smart_explore = smart_exploration

    def reset(self):
        BernStochasticDominance.reset(self)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0


    def smart_exploration(self, arms):
        if len(arms) == 0:
            arms = range(self.k)
        a = random.choice(arms)
        arms.remove(a)
        return a, arms


    def run(self):
        arms = range(self.k)
        o = np.ones(self.k)
        exploiting = False
        for t in range(self.T):

            if self.not_ts[t]:
                # empty dict evaluate to false
                if exploiting:
                    # exploition

                    self.rounds_exploiting = self.rounds_exploiting + 1
                    self.theta[t] = np.random.beta(self.s[t], self.f[t], self.k)
                    a = BernStochasticDominance.get_a(self,  t)

                else:
                    self.rounds_exploring = self.rounds_exploring + 1

                    if self.smart_explore:
                        a, arms = self.smart_exploration(arms)
                    else:
                        a = np.random.choice(self.k)

            else:
                a = BernThompsonSampling.get_a(self, t)

            # real bernoulli reward for each arm
            reward = self.bandits.pull(a)

            BernStochasticDominance.update(self, t, a, reward)
            if o[a] == 1 and (self.n[t, a] > c_alg2(self.e2, self.delta, self.bandits.get_mean(), a, self.k)):
                o[a] = 0
                if np.sum(o) == 0:
                   # print self.n[:t]
                    exploiting = True
        self.calc_r_h()
        self.calc_pi(self.rounds_exploring)



