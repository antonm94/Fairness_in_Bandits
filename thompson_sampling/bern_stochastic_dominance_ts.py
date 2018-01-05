import numpy as np
from bern_thompson_sampling import BernThompsonSampling


class BernStochasticDominance(BernThompsonSampling):

    def __init__(self, bandits, T, lam=1):
        BernThompsonSampling.__init__(self, bandits, T)
        self.lam = lam

    def run(self):
        for t in range(self.T):
            self.theta[t] = np.random.beta(self.s, self.f, self.k)

            if np.random.binomial(1, [self.lam])[0]:
                a = self.get_a(t)
            else:
                a = BernStochasticDominance.get_a(self, t)

            reward = self.bandits.pull(a)
            BernThompsonSampling.update(self, t, a, reward)

    def get_a(self, t):
        self.pi[t] = self.theta[t] / sum(self.theta[t])
        # guessed bernoulli reward for each arm
        guessed_r = np.random.binomial(1, self.theta[t])
        # selected arm with random tie - breaking
        a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])
        return a