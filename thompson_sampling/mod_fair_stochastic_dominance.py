import random

import numpy as np

from calc_c import c_alg2
from fairness_calc import smooth_fairness


class ModFairStochasticDominance(object):

    def __init__(self, bandits, T, e1, e2, delta, lam, distance):
        self.k = bandits.k
        self.arm = bandits.arms
        self.r_theta = bandits.theta
        self.T = T
        self.e1 = e1
        self.e2 = e2
        self.delta = delta
        self.lam = lam
        self.distance = distance
        self.s = np.full(self.k, .5)
        self.f = np.full(self.k, .5)
        self.not_smooth_fair = np.zeros(self.T)
        self.smooth_fair = np.zeros(self.T)
        self.fairness_regret = np.zeros(self.T)
        self.theta = np.zeros((self.T, self.k))
        self.n = np.zeros((self.T, self.k))
        self.pi = np.zeros((self.T,self.k))
        self.p_star = [float(i) / sum(self.r_theta) for i in self.r_theta]
        self.rounds_exploring = 0
        self.rounds_exploiting = 0
        self.average_smooth_fair = np.zeros((len(e1), len(e2), len(delta), self.T, ))
        self.average_not_smooth_fair = np.zeros((len(e1), len(e2), len(delta), self.T, ))
        self.average_fair_ratio = np.zeros((len(e1), len(e2), len(delta), self.T))
        self.average_fairness_regret = np.zeros((len(e2), len(delta), T))
        self.regret = np.zeros((len(e2),len(delta), T))
        self.average_n = np.zeros((len(e2), len(delta), self.T, self.k))
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


    def update_smooth_fairness(self, e1, e2):
        for t in range(self.T):
            [self.not_smooth_fair[t], self.smooth_fair[t]] = smooth_fairness(e1, e2, self.theta[t], self.r_theta,
                                                                         self.distance)

    def update_fairness_regret(self):
        for t in range(self.T):
            # print self.pi[t]
            # print self.p_star

            self.fairness_regret[t] = sum([max(self.p_star[i] - self.pi[t][i], 0.) for i in range(self.k)])


    def get_not_fair_ratio(self):
        return np.divide(self.average_not_smooth_fair, self.average_not_smooth_fair + self.average_smooth_fair)

    def get_fair_ratio(self):
        return np.divide(self.average_smooth_fair, self.average_not_smooth_fair + self.average_smooth_fair)

    def get_rounds(self):
        return self.rounds_exploring, self.rounds_exploiting

    def get_regret(self, n_average):
        distance_to_max = max(self.r_theta) - self.r_theta
        for j in range(len(self.e2)):
            for d in range(len(self.delta)):
                self.regret[j][d] = np.apply_along_axis(lambda x: sum(x * distance_to_max), 1, n_average[j][d])

    def run(self, e2, delta):
        for t in range(self.T):
            b = np.random.binomial(1, [self.lam])[0]
            if b == 1:
                # O(t)={i:n_j,i(t) <=C(e2,delta)}
                if t > 0:
                    self.n[t] = self.n[t - 1]
                o = set()
                for i in range(self.k):
                    if self.n[t, i] <= c_alg2(e2, delta, self.r_theta, i, self.k):
                        o.add(i)

                if len(o) == 0:
                    # exploition
                    self.rounds_exploiting = self.rounds_exploiting + 1
                    self.theta[t] = np.random.beta(self.s, self.f, self.k)
                    # guessed bernoulli reward for each arm
                    guessed_r = np.random.binomial(1, self.theta[t])
                    # selected arm with random tie - breaking
                    a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])
                    self.pi[t] = self.theta[t] / sum(self.theta[t])

                else:
                    # exploration
                    self.rounds_exploring = self.rounds_exploring + 1
                    self.theta[t] = np.full(self.k, .5)
                    a = np.random.choice(o)

                    for i in o:
                        self.pi[t][i] = 1./len(o)

                    print pi[t]

            else:
                self.theta[t] = np.random.beta(self.s, self.f, self.k)
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

        print 'Rounds Exploring: {}'.format(self.rounds_exploring)
        print 'Rounds Exploiting: {}'.format(self.rounds_exploiting)



    def analyse(self, n_iterations):
        for it in range(int(n_iterations)):
            for j in range(len(self.e2)):
                for d in range(len(self.delta)):
                    self.run(self.e2[j], self.delta[d])
                    self.update_fairness_regret()
                    self.average_fairness_regret[j][d] = self.average_fairness_regret[j][d] + np.add.accumulate(
                        self.fairness_regret)
                    self.average_n[j][d] = self.average_n[j][d] + self.n
                    for i in range(len(self.e1)):

                        self.update_smooth_fairness(self.e1[i], self.e2[j])
                        self.average_smooth_fair[i][j][d] = self.average_smooth_fair[i][j][d] + np.add.accumulate(self.smooth_fair)
                        self.average_not_smooth_fair[i][j][d] = self.average_not_smooth_fair[i][j][d] + np.add.accumulate(self.not_smooth_fair)
                    self.reset()

        self.average_n = np.divide(self.average_n, n_iterations)
        self.get_regret(self.average_n)
        self.average_fairness_regret = np.divide(self.average_fairness_regret, n_iterations)
        self.average_smooth_fair = np.divide(self.average_smooth_fair, n_iterations)
        self.average_not_smooth_fair = np.divide(self.average_not_smooth_fair, n_iterations)
        for i in range(len(self.e1)):
            for j in range(len(self.e2)):
                self.average_fair_ratio[i][j] = np.divide(self.average_smooth_fair[i][j],
                                                          self.average_not_smooth_fair[i][j] + self.average_smooth_fair[i][j])
