import fairness_calc
import thompson_sampling.bern_ts as ts
import numpy as np
from distance import total_variation_distance
import os
import sys
import math

class TSTest:
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, distance=total_variation_distance):
        self.curr_test = ts.BernThompsonSampling(bandits, T)
        self.n_iter = n_iter
        self.bandits = bandits
        self.k = bandits.k
        self.T = T
        self.e1_arr = e1_arr
        self.e2_arr = e2_arr
        self.delta_arr = delta_arr
        self.distance = distance
        self.n_iter = n_iter
        self.r_theta = bandits.theta
        self.p_star = [float(i) / sum(self.r_theta) for i in self.r_theta]
        self.achievable_delta = np.zeros((len(e1_arr), len(e2_arr), self.T))
        self.subjective_achievable_delta = np.zeros((len(e1_arr), len(e2_arr), self.T))

        self.smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T, self.k, self.k))
        self.subjective_smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T, self.k, self.k))

        self.is_smooth_fair = np.ones((len(e1_arr), len(e2_arr), len(delta_arr), self.T))
        self.is_subjective_smooth_fair = np.ones((len(e1_arr), len(e2_arr), len(delta_arr), self.T))

        self.frac_smooth_fair = np.ones((len(e1_arr), len(e2_arr), self.T, self.k, self.k))
        self.frac_subjective_smooth_fair = np.ones((len(e1_arr), len(e2_arr), self.T,  self.k, self.k))

        self.average_fairness_regret = np.zeros(T)
        self.average_regret = np.zeros(T)
        self.average_n = np.zeros((self.T, self.k))
        self.name = 'TS'
        self.lam = 0
        self.min_e1 = np.zeros((len(delta_arr), len(e2_arr), self.T))

    def get_name(self, e1=-1, e2=-1, delta=-1):
        s = self.name
        if not e1 == -1:
            s = s + ' e1={}'.format(e1)
        if not e2 == -1:
            s = s + ' e2={}'.format(e2)
        if not delta == -1:
            s = s + ' delta={}'.format(delta)
        return s

    def calc_smooth_fairness(self, e1_ind, e2_ind, e2_times=1):

        for t in range(self.T):
            for i in range(self.k):
                for j in range(self.k):
                    self.smooth_fair[e1_ind, e2_ind, t, i, j] += fairness_calc.smooth_fairness(
                        self.e1_arr[e1_ind], e2_times*self.e2_arr[e2_ind], i, j, self.curr_test.pi[t],
                        self.r_theta, self.distance)





    def calc_subjective_smooth_fairness(self, e1_ind, e2_ind, e2_times=1):
        for t in range(self.T):
            for i in range(self.k):
                for j in range(self.k):
                        self.subjective_smooth_fair[e1_ind, e2_ind, t, i, j] += fairness_calc.smooth_fairness(
                            self.e1_arr[e1_ind], e2_times*self.e2_arr[e2_ind], i, j, self.curr_test.pi[t],
                            self.curr_test.r_h[t], self.distance)

    def calc_frac_smooth_fair(self, e1_ind, e2_ind):
        for t in range(1, self.T):
            self.frac_smooth_fair[e1_ind, e2_ind, t] = np.divide(self.smooth_fair[e1_ind, e2_ind, t], self.n_iter)
            self.achievable_delta[e1_ind, e2_ind, t] = max(1 - np.ndarray.min(self.frac_smooth_fair[e1_ind, e2_ind, t]),
                                                                      self.achievable_delta[e1_ind, e2_ind, t-1])

    def calc_is_smooth_fair(self, e1_ind, e2_ind):
        for t in range(1, self.T):
            for delta_ind in range(len(self.delta_arr)):
                b = (self.frac_smooth_fair[e1_ind, e2_ind, t] >= 1 - self.delta_arr[delta_ind])
                self.is_smooth_fair[e1_ind, e2_ind, delta_ind, t] = np.all(b) and \
                                                                self.is_smooth_fair[e1_ind, e2_ind, delta_ind, t-1]

    def calc_frac_subjective_smooth_fair(self, e1_ind, e2_ind):
        for t in range(1, self.T):
            self.frac_subjective_smooth_fair[e1_ind, e2_ind, t] = np.divide(self.subjective_smooth_fair[e1_ind, e2_ind, t], self.n_iter)
            self.subjective_achievable_delta[e1_ind, e2_ind, t] = max(1 - np.ndarray.min(self.frac_subjective_smooth_fair[e1_ind, e2_ind, t]),
                                                                      self.subjective_achievable_delta[e1_ind, e2_ind, t-1])



    def calc_is_subjective_smooth_fair(self, e1_ind, e2_ind):
        for t in range(1, self.T):
            for delta_ind in range(len(self.delta_arr)):
                b = (self.frac_subjective_smooth_fair[e1_ind, e2_ind, t] >= 1 - self.delta_arr[delta_ind])
                self.is_subjective_smooth_fair[e1_ind, e2_ind, delta_ind, t] = np.all(b) and \
                                                                               self.is_subjective_smooth_fair[e1_ind, e2_ind, delta_ind, t-1]

    def calc_fairness_regret(self):
        fairness_regret = np.zeros(self.T)
        for t in range(self.T):
            fairness_regret[t] = sum([max(self.p_star[i] - self.curr_test.pi[t][i], 0.) for i in range(self.k)])
        return fairness_regret


    def get_regret(self):
        distance_to_max = max(self.r_theta) - self.r_theta
        return np.apply_along_axis(lambda x: np.sum(x * distance_to_max), 1, self.average_n)

    def analyse(self, regret=True, fair_regret=True, smooth_fair = True, subjective_smooth_fair = False, minimum_e1=True):
        # file_name = self.bandits.data_set_name + '/' + self.name + '/N_ITER_{}'.format(
        #     int(self.n_iter)) + '_T_{}'.format(self.T)
        # cwd = os.getcwd()
        # last_dir = cwd.split('/')[-1]
        # if last_dir == 'notebooks':
        #     os.chdir(cwd.replace('/notebooks', ''))
        #
        #
        # if os.path.exists(file_name):
        #     self.analyse_from_file(regret, fair_regret, smooth_fair, subjective_smooth_fair)
        #     print 'restored data from file'
        #     return

        # pi = np.zeros((int(self.n_iter), self.T, self.k))
        # r_h = np.zeros((int(self.n_iter), self.T, self.k))
        # n = np.zeros((int(self.n_iter), self.T, self.k))
        if minimum_e1:
            min_e1 = np.zeros((len(self.e2_arr), self.T, int(self.n_iter)))
        for it in range(int(self.n_iter)):

            self.curr_test.run()
            # pi[it] = self.curr_test.pi
            # r_h[it] = self.curr_test.r_h
            # n[it] = self.curr_test.n


            if fair_regret:
                self.average_fairness_regret = self.average_fairness_regret + self.calc_fairness_regret()
            self.average_n = self.average_n + self.curr_test.n

            if smooth_fair:
                for i in range(len(self.e1_arr)):
                    for j in range(len(self.e2_arr)):
                        self.calc_smooth_fairness(i, j)

            if subjective_smooth_fair:
                for i in range(len(self.e1_arr)):
                    for j in range(len(self.e2_arr)):
                        self.calc_subjective_smooth_fairness(i, j)
            if minimum_e1:
                for e2_ind, e2 in enumerate(self.e2_arr):
                    for t in range(self.T):
                        e1 = 0
                        for i in range(self.k):
                            for j in range(self.k):
                                curr_e1 = fairness_calc.get_e1_smooth_fairness(e2, i, j, self.curr_test.pi[t],
                                                                                 self.curr_test.r_h[t],  self.distance)
                                                                                # self.r_theta, self.distance)
                                e1 = max(e1, curr_e1)
                        min_e1[e2_ind, t, it] = e1


            self.curr_test.reset()
            if minimum_e1:
               # print min_e1
                min_e1.sort(axis=-1)
             #   print min_e1

                for delta_ind, delta in enumerate(self.delta_arr):
                    for e2_ind, e2 in enumerate(self.e2_arr):
                        for t in range(self.T):
                            self.min_e1[delta_ind, e2_ind, t] \
                                = min_e1[e2_ind, t, min(int(math.ceil((1-delta)*self.n_iter)), int(self.n_iter-1))]




        if smooth_fair:
            for i in range(len(self.e1_arr)):
                for j in range(len(self.e2_arr)):
                    self.calc_frac_smooth_fair(i, j)
                    # self.calc_is_smooth_fair(i, j)

        if subjective_smooth_fair:
            for i in range(len(self.e1_arr)):
                for j in range(len(self.e2_arr)):
                    self.calc_frac_subjective_smooth_fair(i, j)
                    # self.calc_is_subjective_smooth_fair(i, j)

        self.average_n = np.divide(self.average_n, self.n_iter)
        if regret:
            self.average_regret = self.get_regret()
        if fair_regret:
            self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)

        # if not os.path.exists(file_name):
        #     os.makedirs(file_name)
        # np.savez(file_name, pi=pi, r_h=r_h, r_theta=self.bandits.theta, n=n)


    def analyse_from_file(self, regret=True, fair_regret=True, smooth_fair = True, subjective_smooth_fair = False):
        file_name = self.bandits.data_set_name + '/' + self.name + '/N_ITER_{}'.format(int(self.n_iter)) + '_T_{}'.format(self.T)
        if not os.path.exists(file_name):
            print 'no such file'
        npzfile = np.load(file_name+'.npz')

        cwd = os.getcwd()
        last_dir = cwd.split('/')[-1]
        if last_dir == 'notebooks':
            os.chdir(cwd.replace('/notebooks', ''))

        for it in range(int(self.n_iter)):
            self.curr_test.pi = npzfile['pi'][it]
            self.curr_test.r_h = npzfile['r_h'][it]
            self.curr_test.n = npzfile['n'][it]

            if fair_regret:
                self.average_fairness_regret = self.average_fairness_regret + self.calc_fairness_regret()
            self.average_n = self.average_n + self.curr_test.n

            if smooth_fair:
                for i in range(len(self.e1_arr)):
                    for j in range(len(self.e2_arr)):
                        self.calc_smooth_fairness(i, j)

            if subjective_smooth_fair:
                for i in range(len(self.e1_arr)):
                    for j in range(len(self.e2_arr)):
                        self.calc_subjective_smooth_fairness(i, j)

        if smooth_fair:
            for i in range(len(self.e1_arr)):
                for j in range(len(self.e2_arr)):
                    self.calc_frac_smooth_fair(i, j)
                    # self.calc_is_smooth_fair(i, j)

        if subjective_smooth_fair:
            for i in range(len(self.e1_arr)):
                for j in range(len(self.e2_arr)):
                    self.calc_frac_subjective_smooth_fair(i, j)
                    # self.calc_is_subjective_smooth_fair(i, j)

        self.average_n = np.divide(self.average_n, self.n_iter)
        if regret:
            self.average_regret = self.get_regret()
        if fair_regret:
            self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)




