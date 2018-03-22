import math
import scipy.stats as stats
import numpy as np
from normal_IG_ts import NormalThompsonSampling

class NormalStochasticDominance(NormalThompsonSampling):

    def __init__(self, bandits, T, lam=1, mean_0=0., alpha_0=1., beta_0=0., v_0=-1):
        NormalThompsonSampling.__init__(self, bandits, T, mean_0=mean_0, alpha_0=alpha_0, beta_0=beta_0, v_0=v_0)
        self.sampled_var = np.zeros(self.k)
        self.lam = lam

    def reset(self):
        NormalThompsonSampling.reset(self)
        self.sampled_var = np.zeros(self.k)

    def get_a(self):
        guessed_r = np.random.normal(self.sampled_mu, self.sampled_var)
        a = np.random.choice(np.where(guessed_r == guessed_r.max())[0])

        return a

    def update_samples(self):
        for a in range(self.k):
            self.sampled_var[a] = self.inv_gamma[a].rvs(1)
            self.sampled_mu[a] = self.normal[a].rvs(1)


    def calc_pi(self, t):
        # n_iter_trial = 10
        # n_iter_mu = 100
        # n_iter_var = 100
        # sampled_mu = np.zeros((self.k, n_iter_mu))
        # sampled_var = np.zeros((self.k, n_iter_var))
        # wins = np.zeros(self.k)
        # for a in range(self.k):
        #     sampled_mu[a] = self.normal[a].rvs(n_iter_mu)
        #     sampled_var[a] = self.inv_gamma[a].rvs(n_iter_var)
        # for mu_ind in range(n_iter_mu):
        #     for var_ind in range(n_iter_var):
        #         for i in range(n_iter_trial):
        #             guessed_r = np.random.normal(sampled_mu[:,mu_ind], np.sqrt(sampled_var[:, var_ind]))
        #             wins[np.random.choice(np.where(guessed_r == guessed_r.max())[0])] += 1
        # self.pi[t] = np.divide(wins, (n_iter_trial*n_iter_var*n_iter_mu))
        # print self.pi[t]

        if self.k == 10000:
            if t > 0:
                mu = self.student_t[t-1, 0].mean() - self.student_t[t-1, 1].mean()
                var = self.student_t[t-1, 0].var() + self.student_t[t-1, 1].var()
                self.pi[t][0] = 1 - stats.norm.cdf(-mu/math.sqrt(var))
                mu = self.student_t[t-1, 1].mean() - self.student_t[t-1, 0].mean()
                var = self.student_t[t-1, 1].var() + self.student_t[t-1, 0].var()
                self.pi[t][1] = 1 - stats.norm.cdf(-mu/math.sqrt(var))
        else:
            n_iter = 1000
            guessed_r = np.zeros((self.k, n_iter))
            for a in range(self.k):
                guessed_r[a] = self.student_t[t - 1, a].rvs(n_iter)
            max_guessed_r = np.append(np.argmax(guessed_r, axis=0), np.arange(self.k))
            counts = np.subtract(np.bincount(max_guessed_r).astype(np.float), 1)
            self.pi[t] = np.divide(counts, float(n_iter))


