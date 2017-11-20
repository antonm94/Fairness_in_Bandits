import numpy as np
import random
from fairness_calc import smooth_fairness
from calc_c import c_alg2


class FairStochasticDominance:

    def __init__(self, bandits, T, e1, e2, delta, lam):
        self.k = bandits.k
        self.arm = bandits.arms
        self.r_theta = bandits.theta
        self.T = T
        self.e1 = e1
        self.e2 = e2
        self.delta = delta
        self.lam = lam
        self.s = np.full(self.k, .5)
        self.f = np.full(self.k, .5)
        self.not_smooth_fair = np.zeros(self.T)
        self.smooth_fair = np.zeros(self.T)
        self.fairness_regret = np.zeros(self.T)
        self.theta = np.zeros((self.T, self.k))
        self.n = np.zeros((self.T, self.k))
        self.pi = np.zeros(self.k)
        self.p_star = [float(i) / sum(self.r_theta) for i in self.r_theta]
        self.rounds_exploring = 0
        self.rounds_exploiting = 0
        self.average_smooth_fair = np.zeros(T)
        self.average_not_smooth_fair = np.zeros(T)
        self.average_fairness_regret = np.zeros(T)
        self.regret = np.zeros(T)
        self.average_n = np.zeros((self.T, self.k))
        if lam == 0.:
            self.name = 'Thompson Sampling'
        elif lam == 1.:
            self.name = 'Fair Stochastic Dominance Thompson Sampling'
        else:
            self.name = 'Thompson Sampling - Fair Stochastic Dominance Thompson Sampling trade-off' \
                        ' with Lambda = {}'.format(self.lam)

    def reset(self):
        self.s = np.full(self.k, .5)
        self.f = np.full(self.k, .5)
        self.not_smooth_fair = np.zeros(self.T)
        self.smooth_fair = np.zeros(self.T)
        self.fairness_regret = np.zeros(self.T)
        self.n = np.zeros((self.T, self.k))
        self.rounds_exploring = 0
        self.rounds_exploiting = 0

    def update_fairness(self, t):
        # print self.pi
        # print self.theta[t]
        # print self.r_theta
        [self.not_smooth_fair[t], self.smooth_fair[t]] = smooth_fairness(self.e1, self.e2, self.theta[t], self.r_theta)
        self.fairness_regret[t] = sum([max(self.p_star[i] - self.pi[i], 0.) for i in range(self.k)])

    def get_not_fair_ratio(self):
        return np.divide(self.average_not_smooth_fair, self.average_not_smooth_fair + self.average_smooth_fair)

    def get_fair_ratio(self):
        return np.divide(self.average_smooth_fair, self.average_not_smooth_fair + self.average_smooth_fair)

    def get_rounds(self):
        return self.rounds_exploring, self.rounds_exploiting

    def get_regret(self, n_average):
        distance_to_max = max(self.r_theta) - self.r_theta
        return np.apply_along_axis(lambda x: sum(x * distance_to_max), 1, n_average)

    def run(self):
        for t in range(self.T):
            b = np.random.binomial(1, [self.lam])[0]
            if b == 1:
                # O(t)={i:n_j,i(t) <=C(e2,delta)}
                if t > 0:
                    self.n[t] = self.n[t - 1]
                o = set()
                for i in range(self.k):
                    if self.n[t, i] <= c_alg2(self.e2, self.delta, self.r_theta, i, self.k):
                        o.add(i)

                if len(o) == 0:
                    # exploition
                    self.rounds_exploiting = self.rounds_exploiting + 1
                    self.theta[t] = np.random.beta(self.s, self.f, self.k)
                    # guessed bernoulli reward for each arm
                    guessed_r = np.random.binomial(1, self.theta[t])
                    # selected arm with random tie - breaking
                    a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])
                    self.pi = self.theta[t] / sum(self.theta[t])

                else:
                    # exploration
                    self.rounds_exploring = self.rounds_exploring + 1
                    self.theta[t] = np.full(self.k, .5)
                    a = np.random.choice(self.k)
                    self.pi = np.full(self.k, 1. / self.k)

            else:
                self.theta[t] = np.random.beta(self.s, self.f, self.k)
                max_theta = np.where(self.theta[t] == self.theta[t].max())[0]
                a = np.random.choice(max_theta)
                for i in range(self.k):
                    if i in max_theta:
                        self.pi[i] = 1. / len(max_theta)
                    else:
                        self.pi[i] = 0.

            # real bernoulli reward for each arm
            reward = random.choice(self.arm[a])

            if reward:
                self.s[a] = self.s[a] + 1
            else:
                self.f[a] = self.f[a] + 1

            if t > 0:
                self.n[t] = self.n[t - 1]
            self.n[t][a] = self.n[t][a] + 1

            self.update_fairness(t)

    def analyse(self, n_iterations):
        for i in range(int(n_iterations)):
            self.run()
            self.average_fairness_regret = self.average_fairness_regret + np.add.accumulate(self.fairness_regret)
            self.average_smooth_fair = self.average_smooth_fair + np.add.accumulate(self.smooth_fair)
            self.average_not_smooth_fair = self.average_not_smooth_fair + np.add.accumulate(self.not_smooth_fair)
            self.average_n = self.average_n + self.n
            self.reset()
        self.average_n = np.divide(self.average_n, n_iterations)
        self.regret = self.get_regret(self.average_n)
        self.average_fairness_regret = np.divide(self.average_fairness_regret, n_iterations)
        self.average_smooth_fair = np.divide(self.average_smooth_fair, n_iterations)
        self.average_not_smooth_fair = np.divide(self.average_not_smooth_fair, n_iterations)
