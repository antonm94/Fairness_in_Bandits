import numpy as np

class BernThompsonSampling(object):

    def __init__(self, bandits, T):
        self.k = bandits.k
        self.bandits = bandits
        self.T = T
        self.prior_a = 0.5
        self.prior_b = 0.5

        self.s = np.full((self.T, self.k), self.prior_a)
        self.f = np.full((self.T, self.k), self.prior_b)
        self.theta = np.zeros((self.T, self.k))
        self.n = np.zeros((self.T, self.k))
        self.pi = np.zeros((self.T, self.k))
        self.r_1_h = np.full((self.T, self.k), .5)
        self.r_0_h = np.full((self.T, self.k), .5)


    def reset(self):

        self.s = np.full((self.T, self.k), .5)
        self.f = np.full((self.T, self.k), .5)
        self.n = np.zeros((self.T, self.k))

    def run(self):
        for t in range(self.T):
            self.theta[t] = np.random.beta(self.s[t], self.f[t], self.k)

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

        if t > 0:
            self.n[t] = self.n[t - 1]
            self.s[t] = self.s[t - 1]
            self.f[t] = self.f[t - 1]
            self.r_1_h[t] = np.divide(self.s[t], self.f[t] + self.s[t])
            self.r_0_h[t] = np.divide(self.f[t], self.f[t] + self.s[t])
        #
        # sum_r = np.sum(self.r_1_h[t])
        # prod_r = np.prod(self.r_1_h[t])
        # for i in range(self.k):
        #     self.pi[t][i] = self.r_1_h[t][i] / (sum_r - self.r_1_h[t][i]) + prod_r


        self.pi[t] = self.r_1_h[t] / np.sum(self.r_1_h[t])

        self.n[t][a] = self.n[t][a] + 1
        if reward:
            self.s[t][a] = self.s[t][a] + 1
        else:
            self.f[a] = self.f[t][a] + 1





    # def calc_pi(self):
    #     for t in range(self.T):
    #         for k in range(self.k):
    #         self.pi[t][k] =