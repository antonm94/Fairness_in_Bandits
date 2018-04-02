import fairness_calc
import thompson_sampling.normal_IG_ts as ts
import numpy as np
import os
import math
import pickle
import datetime
import divergence as div
import time


class NormalTSTest:
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr,  mean_0=0., alpha_0=1., beta_0=0., v_0=-1,
                 pi_div=div.total_variation_distance, r_div=div.cont_total_variation_distance):
        self.curr_test = ts.NormalThompsonSampling(bandits, T,  mean_0=mean_0, alpha_0=alpha_0, beta_0=beta_0, v_0=v_0)
        self.n_iter = n_iter
        self.bandits = bandits
        self.k = bandits.k
        self.T = T
        self.e1_arr = e1_arr
        self.e2_arr = e2_arr
        self.delta_arr = delta_arr
        self.n_iter = n_iter
        self.achievable_delta = np.zeros((len(e1_arr), len(e2_arr), self.T))
        self.subjective_achievable_delta = np.zeros((len(e1_arr), len(e2_arr), self.T))
        self.smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T, self.k, self.k))
        self.subjective_smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T, self.k, self.k))
        self.pi_div = pi_div
        self.r_div = r_div

        self.is_smooth_fair = np.ones((len(e1_arr), len(e2_arr), len(delta_arr), self.T))
        self.is_subjective_smooth_fair = np.ones((len(e1_arr), len(e2_arr), len(delta_arr), self.T))

        self.frac_smooth_fair = np.ones((len(e1_arr), len(e2_arr), self.T, self.k, self.k))
        self.frac_subjective_smooth_fair = np.ones((len(e1_arr), len(e2_arr), self.T,  self.k, self.k))

        self.average_fairness_regret = np.zeros(T)
        self.average_regret = np.zeros(T)
        self.average_n = np.zeros((self.T, self.k))
        self.name = 'TS'
        self.lam = 0
        self.min_e1 = np.zeros((len(e2_arr), len(delta_arr), self.T))
        self.subjective_min_e1 = np.zeros((len(e2_arr), len(delta_arr), self.T))

    def print_info(self):
        print('Iterations:\t{}'.format(self.n_iter))
        print('T:\t\t{}'.format(self.T))
        print('e1:\t\t{}'.format(self.e1_arr))
        print('e2:\t\t{}'.format(self.e2_arr))
        print('delta:\t\t{}'.format(self.delta_arr))
        print('Lambda:\t\t{}'.format(self.lam))





    def get_name(self, e1=-1, e2=-1, delta=-1):
        s = self.name
        if not e1 == -1:
            s = s + ' e1={}'.format(e1)
        if not e2 == -1:
            s = s + ' e2={}'.format(e2)
        if not delta == -1:
            s = s + ' delta={}'.format(delta)
        return s

    def get_label_name(self, e1=-1, e2=-1, delta=-1):
        s = self.name
        if not e1 == -1:
            s = s + ' $\epsilon_1$={}'.format(e1)
        if not e2 == -1:
            s = s + ' $\epsilon_2$={}'.format(e2)
        if not delta == -1:
            s = s + ' $\delta$={}'.format(delta)
        return s

    def calc_smooth_fairness(self, e1_ind, e2_ind, pi_div, e2_times=1):
        self.smooth_fair[e1_ind, e2_ind, :self.curr_test.init_length] += 1
        for t in range(self.curr_test.init_length, self.T):
            for i in range(self.k):
                for j in range(self.k):
                    if i == j:
                        self.smooth_fair[e1_ind, e2_ind, t, i, j] += 1
                    else:
                        self.smooth_fair[e1_ind, e2_ind, t, i, j] += fairness_calc.smooth_fairness(
                            self.e1_arr[e1_ind], e2_times*self.e2_arr[e2_ind],
                            pi_div[t, i, j], self.bandits.divergence[i, j])

    def calc_frac_smooth_fair(self, e1_ind, e2_ind):
        for t in range(1, self.T):
            self.frac_smooth_fair[e1_ind, e2_ind, t] = np.divide(self.smooth_fair[e1_ind, e2_ind, t], self.n_iter)
            self.achievable_delta[e1_ind, e2_ind, t] = max(1 - np.ndarray.min(self.frac_smooth_fair[e1_ind, e2_ind, t]),
                                                                      self.achievable_delta[e1_ind, e2_ind, t-1])


    def calc_is_smooth_fair(self, e1_ind, e2_ind):
        for t in range(1, self.T):
            for delta_ind in range(len(self.delta_arr)):
                b = (self.frac_smooth_fair[e1_ind, e2_ind, t] >= 1 - self.delta_arr[delta_ind])
                self.is_smooth_fair[e1_ind, e2_ind, delta_ind, t] = \
                     np.all(b) and self.is_smooth_fair[e1_ind, e2_ind, delta_ind, t-1]


    def calc_subjective_smooth_fairness(self, e1_ind, e2_ind, pi_div, r_h_div, e2_times=1):
        self.subjective_smooth_fair[e1_ind, e2_ind, :self.curr_test.init_length] += 1
        for t in range(self.curr_test.init_length, self.T):
            for i in range(self.k):
                for j in range(self.k):
                    if i == j:
                        self.subjective_smooth_fair[e1_ind, e2_ind, t, i, j] += 1
                    else:
                        self.subjective_smooth_fair[e1_ind, e2_ind, t, i, j] += fairness_calc.smooth_fairness(
                            self.e1_arr[e1_ind], e2_times*self.e2_arr[e2_ind], pi_div[t, i, j], r_h_div[t, i, j])

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
            fairness_regret[t] = sum([max(self.bandits.p_star[i] - self.curr_test.pi[t][i], 0.) for i in range(self.k)])
        return fairness_regret


    def get_regret(self):
        distance_to_max = max(self.bandits.mean) - self.bandits.mean,
        return np.apply_along_axis(lambda x: np.sum(x * distance_to_max), 1, self.average_n)

    def calc_pi_divergence(self, start_t=0):
        # t = time.time()
        pi_div = np.zeros((self.T, self.k, self.k))
        for t in range(start_t, self.T):
            for i in range(self.k):
                for j in range(i+1, self.k):
                    pi_1 = [self.curr_test.pi[t, i], 1 - self.curr_test.pi[t, i]]
                    pi_2 = [self.curr_test.pi[t, j], 1 - self.curr_test.pi[t, j]]
                    pis_div = self.pi_div(pi_1, pi_2)
                    pi_div[t, i, j] = pis_div
                    pi_div[t, j, i] = pis_div
        # print 'Elapsed pi div: %s' % (time.time() - t)

        return pi_div

    def calc_r_h_divergence(self, start_t=0):
        # t = time.time()
        # print 'tic'
        r_h_div = np.zeros((self.T, self.k, self.k))
        for t in range(start_t+1, self.T):
            for i in range(self.k):
                for j in range(i+1, self.k):
                    p = self.curr_test.student_t[t-1, i]
                    q = self.curr_test.student_t[t-1, j]
                    re_div = self.r_div(p, q)
                    r_h_div[t, i, j] = re_div
                    r_h_div[t, j, i] = re_div

        # print 'Elapsed r h div: %s' % (time.time() - t)
        # print 'toc'
        # print 'pi ' +str(self.curr_test.pi)
        # print 'r_h: ' + str(r_h_div)
        return r_h_div

    def analyse(self, regret=True, fair_regret=True, smooth_fair = True, subjective_smooth_fair = False, minimum_e1=True,
                subjective_minimum_e1=False):

        if minimum_e1:
            min_e1 = np.zeros((len(self.e2_arr), self.T, int(self.n_iter)))
        if subjective_minimum_e1:
            subjective_min_e1 = np.zeros((len(self.e2_arr), self.T, int(self.n_iter)))
        for it in range(int(self.n_iter)):
            # t = time.time()
            self.curr_test.run()
            # print 'Elapsed running: %s' % (time.time() - t)

            if fair_regret:
                self.average_fairness_regret += self.calc_fairness_regret()
            self.average_n = self.average_n + self.curr_test.n

            pi_div = self.calc_pi_divergence(self.curr_test.init_length)

            if subjective_smooth_fair or subjective_minimum_e1:
                r_h_div = self.calc_r_h_divergence(self.curr_test.init_length)

            if smooth_fair:
                # t = time.time()
                for i in range(len(self.e1_arr)):
                    for j in range(len(self.e2_arr)):
                        self.calc_smooth_fairness(i, j, pi_div)
                # print 'Elapsed smooth: %s' % (time.time() - t)

            if subjective_smooth_fair:
                # t = time.time()
                for i in range(len(self.e1_arr)):
                    for j in range(len(self.e2_arr)):
                        self.calc_subjective_smooth_fairness(i, j, pi_div, r_h_div)
                # print 'Elapsed sub smooth: %s' % (time.time() - t)

            if minimum_e1:
                # t = time.time()
                for e2_ind, e2 in enumerate(self.e2_arr):
                    for t in range(self.curr_test.init_length, self.T):
                        e1 = 0
                        for i in range(self.k):
                            for j in range(self.k):
                                if i == j:
                                    curr_e1 = 0.
                                else:
                                    curr_e1 = fairness_calc.get_e1_smooth_fairness(e2, pi_div[t, i, j],
                                                                                 self.bandits.divergence[i, j])
                                e1 = max(e1, curr_e1)
                        min_e1[e2_ind, t, it] = e1
                # print 'Elapsed min %s' % (time.time() - t)


            if subjective_minimum_e1:
                # t = time.time()

                for e2_ind, e2 in enumerate(self.e2_arr):
                    for t in range(self.curr_test.init_length, self.T):
                        e1 = 0
                        for i in range(self.k):
                            for j in range(self.k):
                                if i == j:
                                    curr_e1 = 0.
                                else:
                                    curr_e1 = fairness_calc.get_e1_smooth_fairness(e2, pi_div[t, i, j],
                                                                               r_h_div[t, i, j])
                                e1 = max(e1, curr_e1)
                        subjective_min_e1[e2_ind, t, it] = e1
                # print 'Elapsed sub min %s' % (time.time() - t)


            if it < int(self.n_iter)-1:
                self.curr_test.reset()
            # print 'Elapsed faircalc: %s' % (time.time() - t)

        if minimum_e1:
            min_e1.sort(axis=-1)
            for delta_ind, delta in enumerate(self.delta_arr):
                for e2_ind, e2 in enumerate(self.e2_arr):
                    for t in range(self.T):
                        self.min_e1[e2_ind, delta_ind, t] \
                            = min_e1[e2_ind, t, min(int(math.ceil((1-delta)*self.n_iter)), int(self.n_iter-1))]
        if subjective_minimum_e1:
            subjective_min_e1.sort(axis=-1)
            for delta_ind, delta in enumerate(self.delta_arr):
                for e2_ind, e2 in enumerate(self.e2_arr):
                    for t in range(self.T):
                        self.subjective_min_e1[e2_ind, delta_ind, t] \
                            = subjective_min_e1[e2_ind, t, min(int(math.ceil((1 - delta) * self.n_iter)), int(self.n_iter - 1))]

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
        self.save_object()
        return self

    def save_object(self):
        i = 0
        date_time = 'test-{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())

        directory = 'new_normal_objects/{}'.format(self.T) +'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = directory + self.bandits.data_set_name + '_' + self.name + '_N_ITER_{}'.format(
            int(self.n_iter)) + date_time + str(os.getpid())

        while os.path.exists(file_name):
             i += 1
             file_name = directory + self.bandits.data_set_name + '_' + self.name \
                         + '_N_ITER_{}'.format(int(self.n_iter)) + date_time + '_{}'.format(i)
        with open(file_name + '.file', "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
