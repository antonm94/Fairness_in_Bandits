import random
import numpy as np
from distance import total_variation_distance


class StochasticDominance(object):

    def __init__(self, bandits, T, lam=1, distance=total_variation_distance, mod=0):
        self.k = bandits.k
        self.arm = bandits.arms
        self.alpha = 0.5
        self.beta = 0.5
        self.T = T
        self.lam = lam
        self.mod = mod
        self.distance = distance
        self.s = np.full((self.T, self.k), self.alpha)
        self.f = np.full((self.T, self.k), self.beta)
        self.theta = np.zeros((self.T, self.k))
        self.n = np.zeros((self.T, self.k))
        self.pi = np.zeros((self.T,self.k))

        if lam == 0.:
            self.name = 'Thompson Sampling'
        elif lam == 1.:
            self.name = 'Stochastic Dominance Thompson Sampling'
        else:
            self.name = 'Thompson Sampling - Stochastic Dominance Thompson Sampling trade-off' \
                        ' with Lambda = {}'.format(self.lam)

    def reset(self):
        self.s = np.full((self.T, self.k), self.alpha)
        self.f = np.full((self.T, self.k), self.beta)
        self.n = np.zeros((self.T, self.k))

    def run(self):
        for t in range(self.T):
            self.theta[t] = np.random.beta(self.s, self.f, self.k)

            b = np.random.binomial(1, [self.lam])[0]

            if b:
                self.pi[t] = self.theta[t] / sum(self.theta[t])

                if self.mod:
                    a = np.random.choice(self.k, 1, p=self.pi[t])

                else:
                    # guessed bernoulli reward for each arm
                    guessed_r = np.random.binomial(1, self.theta[t])
                    # selected arm with random tie - breaking
                    a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])

            else:
                max_theta = np.where(self.theta[t] == self.theta[t].max())[0]
                a = np.random.choice(max_theta)
                for i in range(self.k):
                    if i in max_theta:
                        self.pi[t][i] = 1. / len(max_theta)
                    else:
                        self.pi[t][i] = 0.

            # real bernoulli reward for each arm
            reward = random.choice(self.arm[a])

            if reward:
                self.s[a] = self.s[a] + 1
            else:
                self.f[a] = self.f[a] + 1

            if t > 0:
                self.n[t] = self.n[t - 1]
            self.n[t][a] = self.n[t][a] + 1
