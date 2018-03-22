import math
import scipy.stats as stats
import numpy as np
from normal_inverse_gamme import NormalInverseGamma
class NormalThompsonSampling(object):

    def __init__(self, bandits, T, mean_0=0., alpha_0=1., beta_0=0., v_0=-1):
        self.k = bandits.k
        self.bandits = bandits
        self.T = T

        self.mean_0 = mean_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        if alpha_0 == -1:
            self.v_0 = max(2., 3.-math.floor(3.*self.alpha_0))
        else:
            self.v_0 = v_0
        self.mean = np.full(self.k, self.mean_0)
        self.alpha = np.full(self.k, self.alpha_0)
        self.beta = np.full(self.k, self.beta_0)
        self.v = np.full(self.k, self.v_0)

        self.rewards = np.zeros(self.k)
        self.n = np.zeros((self.T, self.k))
        self.sum_of_squares = np.zeros(self.k)
        self.sample_mean = np.zeros(self.k)
        self.student_t = np.full((self.T, self.k), stats.t(df=self.alpha_0 * 2., loc=self.mean_0,
                                                           scale=math.sqrt((self.beta_0 * (self.v_0 + 1.))
                                                                           / (self.v_0 * self.alpha_0))))
        self.pi = np.full((self.T, self.k), 1./self.k)
        self.inv_gamma = np.full(self.k, stats.invgamma(a=self.alpha_0, scale=self.beta_0))
                                 #stats.invgamma(a=self.alpha_0, loc=0, scale=self.beta_0))
        self.normal = np.full(self.k, stats.t(df=self.v_0, loc=self.mean_0, scale=1))
                              # stats.norm(loc=self.mean_0, scale=math.sqrt(self.sampled_var[0] * self.v_0)))
        self.sampled_mu = np.zeros(self.k)

    def reset(self):
        self.mean = np.full(self.k, self.mean_0)
        self.alpha = np.full(self.k, self.alpha_0)
        self.beta = np.full(self.k, self.beta_0)
        self.v = np.full(self.k, self.v_0)
        self.rewards = np.zeros(self.k)
        self.n = np.zeros((self.T, self.k))
        self.sum_of_squares = np.zeros(self.k)

        self.student_t = np.full((self.T, self.k), stats.t(df=self.alpha_0 * 2., loc=self.mean_0,
                                                    scale=math.sqrt((self.beta_0 * (self.v_0 + 1))
                                                                 / (self.v_0 * self.alpha_0))))
        self.pi = np.full((self.T, self.k), 1. / self.k)
        self.sampled_mu = np.zeros(self.k)

    def run(self):
        ''''''''''Initialization'''''''''
        t = 0
        for a in range(self.k):
            for i in range(int(self.v_0)):
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.update_round_x_s(a, r, t)
                t += 1
        if self.v_0 > 0:
            t -= 1
            for a in range(self.k):
                self.update_param(a, t)
                self.update_distribution(a, t)
                self.update_samples()
            t += 1
        ''''''''''''''''''''''''''''''
        for t in range(int(self.v_0)*self.k, self.T):
            self.calc_pi(t)
            a = self.get_a()
            r = self.bandits.pull(a)
            self.rewards[a] += r
            self.update_round_x_s(a, r, t)
            self.update_param(a, t)
            self.update_distribution(a, t)
            self.update_samples()

    def update_round_x_s(self, a, r, t):
        if t >= self.v_0*self.k:
            self.n[t] = self.n[t - 1]
            self.n[t, a] += 1
        elif t>0:
            self.n[t] = self.n[t - 1]
            self.n[t, a] = self.n[t-1, a] + 1.
        else:
            self.n[t, a] += 1
        self.sample_mean[a] = self.rewards[a] / self.n[t, a]
        self.sum_of_squares[a] += pow(r - self.sample_mean[a], 2)

    def get_a(self):
        max_mu = np.where(self.sampled_mu == self.sampled_mu.max())[0]
        a = np.random.choice(max_mu)
        return a

    def update_param(self, a, t):
        self.update_alpha(a, t)
        self.update_beta(a, t)
        self.update_mean(a, t)
        self.update_df(a, t)

    def update_distribution(self, a, t):
        self.update_normal_inverse_gamma(a, t)
        self.update_student_t(a, t)


    def update_df(self, a, t):
        self.v[a] = self.v_0 + self.n[t, a]

    def update_alpha(self, a, t):
        self.alpha[a] = self.alpha_0 + self.n[t, a]/2

    def update_beta(self, a, t):
        self.beta[a] = self.beta_0 + 0.5 * self.sum_of_squares[a] +\
                    (self.n[t, a]*self.v_0*pow(self.sample_mean[a]-self.mean_0, 2)) / ((self.v_0+self.n[t, a])*2)

    def update_mean(self, a, t):
        self.mean[a] = (self.v[a]*self.mean_0 + self.n[t, a] * self.sample_mean[a])/(self.v_0 + self.n[t, a])

    def update_student_t(self, a, t):
        if t>0:
            self.student_t[t] = self.student_t[t-1]
        self.student_t[t, a] = stats.t(df=self.alpha[a] * 2, loc=self.mean[a],
                                    scale=math.sqrt((self.beta[a] * (self.v[a] + 1)) / (self.v[a] * self.alpha[a])))

    def update_normal_inverse_gamma(self, a, t):
        self.inv_gamma[a] = stats.invgamma(a=self.alpha[a], scale=self.beta[a])
        scale = math.sqrt(self.sum_of_squares[a] / float((self.n[t, a] * (self.n[t, a] - 1))))
        self.normal[a] = stats.t(df=self.v[a], loc=self.sample_mean[a], scale=scale)

    def update_samples(self):
        for a in range(self.k):
            self.sampled_mu[a] = self.normal[a].rvs(1)


    def calc_pi(self, t):
        n_iter = 1000
        sampled_mu = np.zeros((self.k, n_iter))
        for a in range(self.k):
            sampled_mu[a] = self.normal[a].rvs(n_iter)
        max_mu = np.append(np.argmax(sampled_mu, axis=0), np.arange(self.k))
        counts = np.subtract(np.bincount(max_mu).astype(np.float), 1)
        self.pi[t] = np.divide(counts, float(n_iter))

        # if self.k == 2:
        #     if t > 0:
        #         mu = self.student_t[t-1, 0].mean() - self.student_t[t-1, 1].mean()
        #         var = self.student_t[t-1, 0].var() + self.student_t[t-1, 1].var()
        #         self.pi[t][0] = 1 - stats.norm.cdf(-mu/math.sqrt(var))
        #         mu = self.student_t[t-1, 1].mean() - self.student_t[t-1, 0].mean()
        #         var = self.student_t[t-1, 1].var() + self.student_t[t-1, 0].var()
        #         self.pi[t][1] = 1 - stats.norm.cdf(-mu/math.sqrt(var))
        # print self.pi[t]


