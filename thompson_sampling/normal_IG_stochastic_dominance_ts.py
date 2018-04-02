import math
import scipy.stats as stats
import numpy as np
from normal_IG_ts import NormalThompsonSampling

class NormalStochasticDominance(NormalThompsonSampling):

    def __init__(self, bandits, T, lam=1, mean_0=0., alpha_0=1., beta_0=0., v_0=-1, init_phase=True):
        NormalThompsonSampling.__init__(self, bandits, T, mean_0=mean_0, alpha_0=alpha_0,
                                        beta_0=beta_0, v_0=v_0, init_phase=init_phase)
        self.sampled_var = np.zeros(self.k)
        self.lam = lam
        self.not_ts = np.random.binomial(1, np.full(T, lam))


    def run(self):
        ''''''''''Initialization'''''''''
        if self.init_phase:
            t_init = 0
            for a in range(self.k):
                for i in range(int(self.v_0)):
                    r = self.bandits.pull(a)
                    self.rewards[a] += r
                    self.update_round_x_s(a, r, t_init)
                    t_init += 1
            if self.v_0 > 0:
                t_init -= 1
                for a in range(self.k):
                    self.update_param(a, t_init)
                    self.update_distribution(a, t_init)
            self.update_samples()
        t_init += 1
        ''''''''''''''''''''''''''''''
        for t in range(t_init, self.T):
            if self.not_ts[t]:
                self.calc_pi(t)
                a = self.get_a()
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.update_round_x_s(a, r, t)
                self.update_param(a, t)
                self.update_distribution(a, t)
                self.update_samples()
            else:
                NormalThompsonSampling.calc_pi(self, t)
                a = NormalThompsonSampling.get_a(self)
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.update_round_x_s(a, r, t)
                self.update_param(a, t)
                self.update_distribution(a, t)
                NormalThompsonSampling.update_samples(self)

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
        if self.k == 2:
            mu = self.student_t[t-1, 0].mean() - self.student_t[t-1, 1].mean()
            var = self.student_t[t-1, 0].var() + self.student_t[t-1, 1].var()
            self.pi[t][0] = 1 - stats.norm.cdf(-mu/math.sqrt(var))
            mu = self.student_t[t-1, 1].mean() - self.student_t[t-1, 0].mean()
            var = self.student_t[t-1, 1].var() + self.student_t[t-1, 0].var()
            self.pi[t][1] = 1 - stats.norm.cdf(-mu/math.sqrt(var))


        else:

            n_iter = 1000
            guessed_r = np.zeros((self.k, n_iter))
            # print [self.student_t[t - 1, i].mean() for i in range(self.k)]

            for a in range(self.k):
                guessed_r[a] = self.student_t[t - 1, a].rvs(n_iter)
            max_guessed_r = np.append(np.argmax(guessed_r, axis=0), np.arange(self.k))
            counts = np.subtract(np.bincount(max_guessed_r).astype(np.float), 1)
            self.pi[t] = np.divide(counts, float(n_iter))


        # print 'pi'
        # print self.pi[t]
        # print 'r_h'
        # print [self.student_t[t-1, i].mean() for i in range(self.k)]
        # print [self.student_t[t-1, i].var() for i in range(self.k)]

