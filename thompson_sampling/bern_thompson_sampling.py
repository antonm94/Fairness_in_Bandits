import numpy as np

class BernThompsonSampling(object):

    def __init__(self, bandits, T):
        self.k = bandits.k
        self.bandits = bandits
        self.T = T
        self.s = np.full(self.k, .5)
        self.f = np.full(self.k, .5)
        self.theta = np.zeros((self.T, self.k))
        self.n = np.zeros((self.T, self.k))
        self.pi = np.zeros((self.T, self.k))

    def reset(self):
        self.s = np.full(self.k, .5)
        self.f = np.full(self.k, .5)
        self.n = np.zeros((self.T, self.k))

    def run(self):
        for t in range(self.T):
            self.theta[t] = np.random.beta(self.s, self.f, self.k)

            a = self.get_a(t)

            reward = self.bandits.pull(a)
            self.update(t, a, reward)

    def get_a(self, t):
        max_theta = np.where(self.theta[t] == self.theta[t].max())[0]
        a = np.random.choice(max_theta)
        for i in range(self.k):
            if i in max_theta:
                self.pi[t][i] = 1. / len(max_theta)
            else:
                self.pi[t][i] = 0.

        return a



    def update(self, t, a, reward):
        if reward:
            self.s[a] = self.s[a] + 1
        else:
            self.f[a] = self.f[a] + 1

        if t > 0:
            self.n[t] = self.n[t - 1]
        self.n[t][a] = self.n[t][a] + 1