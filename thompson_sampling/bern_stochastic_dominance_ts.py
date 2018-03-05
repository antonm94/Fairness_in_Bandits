import numpy as np
from bern_ts import BernThompsonSampling


class BernStochasticDominance(BernThompsonSampling):

    def __init__(self, bandits, T, lam=1):
        BernThompsonSampling.__init__(self, bandits, T)
        self.lam = lam
        self.not_ts = np.random.binomial(1, np.full(T, lam)) # decide if TS or SD TS

    def reset(self):
        BernThompsonSampling.reset(self)
        self.not_ts = np.random.binomial(1, np.full(self.T, self.lam)) # decide if TS or SD TS


    def run(self):
        for t in range(self.T):
            self.theta[t] = np.random.beta(self.s[t], self.f[t], self.k)

            if self.not_ts[t]:
                a = BernStochasticDominance.get_a(self, t)
            else:
                a = BernThompsonSampling.get_a(self, t)

            reward = self.bandits.pull(a)
            BernThompsonSampling.update(self, t, a, reward)
        self.calc_r_h()
        self.calc_pi()


    def get_a(self, t):
        # guessed bernoulli reward for each arm
        guessed_r = np.random.binomial(1, self.theta[t])
        # selected arm with random tie - breaking
        a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])
        return a
