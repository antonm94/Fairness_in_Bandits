import numpy as np
import calculations
import itertools
import math
from fairness_calc import isclose
import scipy.stats as stats
from normalBandits import normalBandits
import numpy as np
from scipy.stats import invgamma
from scipy.stats import norm


class NormalThompsonSampling(object):

    def __init__(self, bandits, T):
        self.k = bandits.k
        self.bandits = bandits
        self.T = T

        self.mean_0 = 0.
        self.alpha_0 = 1.
        self.beta_0 = 0.
        self.v_0 = max(2., 3.-math.floor(3.*self.alpha_0))

        self.mean = np.full(self.k, self.mean_0)
        self.alpha = np.full(self.k, self.alpha_0)
        self.beta = np.full(self.k, self.beta_0)
        self.v = np.full(self.k, self.v_0)

        self.rewards = np.zeros(self.k)
        self.n = np.zeros(self.k)
        self.sum_of_squares = np.zeros(self.k)

        self.student_t = np.full(self.k, stats.t(df=self.alpha_0*2, loc=self.mean_0,
                       scale=math.sqrt((self.beta_0*(self.v_0 + 1))/(self.v_0 * self.alpha_0))))

    def run(self):
        ''''''''''Initialization'''''''''

        for a in range(self.k):
            for t in range(int(self.v_0)):
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.n[a] += 1
                self.sum_of_squares[a] += pow(r-self.sample_mean(a), 2)
            self.update(a)
        ''''''''''''''''''''''''''''''

        for t in range(int(self.v_0)*self.k, self.T):
            a = 0
            r = self.bandits.pull(a)
            self.n[a] += 1
            self.rewards[a] += r
            self.sum_of_squares[a] += pow(r-self.sample_mean(a), 2)
        self.update(a)

    def student_t(self, a):
        return stats.t(df=self.alpha[a]*2, loc=self.mean[a],
                       scale=math.sqrt((self.beta[a]*(self.v[a] + 1))/(self.v[a] * self.alpha[a])))
    def get
    def update(self, a):
        self.update_alpha(a)
        self.update_beta(a)
        self.update_mean(a)
        self.update_df(a)

    def sample_mean(self, a):
        return self.rewards[a]/self.n[a]

    def update_df(self, a):
        self.v[a] = self.v_0 + self.n[a]

    def update_alpha(self, a):
        self.alpha[a] = self.alpha_0 + self.n[a]/2

    def update_beta(self, a):
        self.beta[a] = self.beta_0 + 0.5 * self.sum_of_squares[a] +\
                    (self.n[a]*self.v_0*pow(self.sample_mean(a)-self.mean_0, 2)) / ((self.v_0+self.n[a])*2)

    def update_mean(self, a):
        self.mean[a] = (self.v[a]*self.mean_0 + self.n[a] * self.sample_mean(a))/(self.v_0 + self.n[a])



