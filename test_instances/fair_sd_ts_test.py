import thompson_sampling.bern_fair_stochastic_dominance_ts as fair_sd_ts
from ts_test import TSTest
import numpy as np
from distance import total_variation_distance
import fairness_calc
import os
import math
from thompson_sampling.calc_c import c_alg
import pickle

class FairSDTest(TSTest):
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, lam=1, distance=total_variation_distance,
                 smart_exploration=False):
        TSTest.__init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, distance)
        self.test_cases = np.empty((len(e2_arr), len(delta_arr)), object)
        self.c = c_alg(e2_arr, delta_arr, self.r_theta)
        for i in range(len(e2_arr)):
            for d in range(len(delta_arr)):
                self.test_cases[i][d] = fair_sd_ts.BernFairStochasticDominance(bandits, T, e2_arr[i],
                                                                               delta_arr[d], lam, distance,
                                                                               self.c[i,d], smart_exploration)
        self.average_fairness_regret = np.zeros((len(e2_arr), len(delta_arr), T))
        self.average_n = np.zeros((len(e2_arr), len(delta_arr), self.T, self.k))
        self.average_rounds_exploring = np.zeros((len(e2_arr), len(delta_arr)))
        self.average_rounds_exploiting = np.zeros((len(e2_arr), len(delta_arr)))
        self.lam = lam
        self.name = 'Fair SD TS'
        self.average_fairness_regret = np.zeros((len(e2_arr), len(delta_arr), T))
        self.average_regret = np.zeros((len(e2_arr), len(delta_arr), T))

        self.achievable_delta = np.zeros((len(e1_arr), len(e2_arr), len(delta_arr), self.T))
        #self.subjective_achievable_delta = np.zeros((len(e1_arr), len(e2_arr), len(delta_arr), self.T))

        self.smooth_fair = np.zeros((len(e1_arr), len(e2_arr), len(delta_arr), self.T, self.k, self.k))
       # self.subjective_smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T, self.k, self.k))

        self.is_smooth_fair = np.ones((len(e1_arr), len(e2_arr), len(delta_arr), self.T))
        #self.is_subjective_smooth_fair = np.ones((len(e1_arr), len(e2_arr), len(delta_arr), self.T))

        self.frac_smooth_fair = np.ones((len(e1_arr), len(e2_arr), len(delta_arr), self.T, self.k, self.k))
        self.pi = np.zeros((self.T, self.k))
       # self.frac_subjective_smooth_fair = np.ones((len(e1_arr), len(e2_arr), self.T, self.k, self.k))

    def get_rounds(self):
        return self.average_rounds_exploring, self.average_rounds_exploiting


    def get_regret(self):
        distance_to_max = max(self.r_theta) - self.r_theta
        regret = np.zeros((len(self.e2_arr), len(self.delta_arr), self.T))
        for j in range(len(self.e2_arr)):
            for d in range(len(self.delta_arr)):
                regret[j][d] = np.apply_along_axis(lambda x: sum(x * distance_to_max), 1, self.average_n[j][d])
        return regret

    def calc_smooth_fairness(self, e1_ind, e2_ind, d_ind, e2_times=1):
        self.smooth_fair[e1_ind, e2_ind, d_ind,:self.curr_test.rounds_exploring-1] += 1
        for t in range(self.curr_test.rounds_exploring, self.T):
            for i in range(self.k):
                # for j in range(i + 1, self.k):
                for j in range(self.k):
                    # self.r_theta = np.full(k, 0.5)+k[t] n[t]
                    self.smooth_fair[e1_ind, e2_ind, d_ind, t, i, j] += \
                        fairness_calc.smooth_fairness(self.e1_arr[e1_ind], e2_times * self.e2_arr[e2_ind], i, j,
                                                      self.curr_test.pi[t], self.r_theta, self.distance)


    def calc_frac_smooth_fair(self, e1_ind, e2_ind, delta_ind):
        for t in range(1, self.T):
            self.frac_smooth_fair[e1_ind, e2_ind, delta_ind, t] = \
                np.divide(self.smooth_fair[e1_ind, e2_ind, delta_ind, t], self.n_iter)
            self.achievable_delta[e1_ind, e2_ind, delta_ind, t] = \
                max(1 - np.ndarray.min(self.frac_smooth_fair[e1_ind, e2_ind, delta_ind, t]),
                    self.achievable_delta[e1_ind, e2_ind, delta_ind, t - 1])

    def calc_is_smooth_fair(self, e1_ind, e2_ind):
        for t in range(1, self.T):
            for d_ind in range(len(self.delta_arr)):
                b = (self.frac_smooth_fair[e1_ind, e2_ind, t] >= 1 - self.delta_arr[d_ind])
                self.is_smooth_fair[e1_ind, e2_ind, d_ind, t] = \
                    np.all(b) and self.is_smooth_fair[e1_ind, e2_ind, d_ind, t - 1]


    def analyse(self, regret=True, fair_regret=True, smooth_fair = True, e2_times=1, minimum_e1=True):


        if minimum_e1:
            min_e1 = np.zeros((len(self.e2_arr), len(self.delta_arr), self.T, int(self.n_iter)))
        for e2_ind, e2 in enumerate(self.e2_arr):
            for d_ind, d in enumerate(self.delta_arr):
                self.curr_test = self.test_cases[e2_ind, d_ind]
                for it in range(int(self.n_iter)):
                    self.curr_test.run()

                    # self.average_rounds_exploiting[j][d] += self.curr_test.rounds_exploiting
                    self.average_rounds_exploring[e2_ind][d_ind] += self.curr_test.rounds_exploring
                    if fair_regret:
                        self.average_fairness_regret[e2_ind][d_ind] += self.calc_fairness_regret()
                    self.average_n[e2_ind][d_ind] += self.curr_test.n
                    if smooth_fair:
                        for e1_ind in range(len(self.e1_arr)):
                            self.calc_smooth_fairness(e1_ind, e2_ind, d_ind, e2_times)
                    if minimum_e1:
                        for t in range(self.curr_test.rounds_exploring, self.T):
                            e1 = 0
                            for i in range(self.k):
                                for j in range(self.k):
                                    curr_e1 = fairness_calc.get_e1_smooth_fairness(e2, i, j, self.curr_test.pi[t],
                                                                                   self.r_theta,
                                                                                   self.distance)
                                    # self.r_theta, self.distance)
                                    e1 = max(e1, curr_e1)
                            min_e1[e2_ind, d_ind, t, it] = e1

                    self.pi = self.curr_test.pi
                    self.curr_test.reset()


                if minimum_e1:
                    for t in range(self.T):
                        min_e1.sort(axis=-1)
                        self.min_e1[e2_ind, d_ind, t] \
                            = min_e1[
                            e2_ind, d_ind, t, min(int(math.ceil((1 - d) * self.n_iter)), int(self.n_iter - 1))]
                if smooth_fair:
                    for e1_ind in range(len(self.e1_arr)):

                        self.calc_frac_smooth_fair(e1_ind, e2_ind, d_ind)
                        # self.calc_is_smooth_fair(i, j)

        self.average_n = np.divide(self.average_n, self.n_iter)
        if regret:
                self.average_regret = self.get_regret()
        if fair_regret:
            self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)
        self.average_rounds_exploring = np.divide(self.average_rounds_exploring, self.n_iter)
        self.average_rounds_exploiting = np.divide(self.average_rounds_exploiting, self.n_iter)

        self.save_object()

