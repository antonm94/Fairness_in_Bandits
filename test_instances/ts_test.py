from fairness_calc import smooth_fairness
import thompson_sampling.bern_ts as ts
import numpy as np
from distance import total_variation_distance


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
       # self.average_smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T))
        self.smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T, self.k, self.k))
        self.is_smooth_fair = np.ones((self.T, len(e1_arr), len(e2_arr), len(delta_arr)))
        self.frac_smooth_fair = np.ones((self.T, len(e1_arr), len(e2_arr),  self.k, self.k))

       # self.average_not_smooth_fair = np.zeros((len(e1_arr), len(e2_arr), self.T))
        self.average_fairness_regret = np.zeros(T)
        self.average_regret = np.zeros(T)
        self.average_n = np.zeros((self.T, self.k))
        self.name = 'Thomspon Sampling'
        self.lam = 0

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
                #for j in range(i + 1, self.k):
                for j in range(self.k):
                 # self.r_theta = np.full(k, 0.5)+k[t] n[t]

                    self.smooth_fair[e1_ind][e2_ind][t][i][j] += smooth_fairness(self.e1_arr[e1_ind], e2_times*self.e2_arr[e2_ind]
                                                                                 , i, j, self.curr_test.theta[t], self.r_theta, self.distance)


    def calc_subjective_smooth_fairness(self, e1_ind, e2_ind, e2_times=1):
        for t in range(self.T):
            for i in range(self.k):
                #for j in range(i + 1, self.k):
                for j in range(self.k):

                 # self.r_theta = np.full(k, 0.5)+k[t] n[t]

                        self.smooth_fair[e1_ind][e2_ind][t][i][j] += smooth_fairness(self.e1_arr[e1_ind], e2_times*self.e2_arr[e2_ind], i, j, self.curr_test.pi[t], self.curr_test.r_h[t], self.distance)

    def calc_frac_is_smooth_fair(self):
        for t in range(1, self.T):
            for e1_ind in range(len(self.e1_arr)):
                for e2_ind in range(len(self.e2_arr)):

                        self.frac_smooth_fair[t][e1_ind][e2_ind] = np.divide(self.smooth_fair[e1_ind][e2_ind][t], self.n_iter)
                        for delta_ind in range(len(self.delta_arr)):
                            b = (self.frac_smooth_fair[t][e1_ind][e2_ind] >= 1 - self.delta_arr[delta_ind])
                            self.is_smooth_fair[t][e1_ind][e2_ind][delta_ind] = np.all(b) and \
                                                                            self.is_smooth_fair[t - 1][e1_ind][e2_ind][
                                                                                delta_ind]



    def calc_fairness_regret(self):
        fairness_regret = np.zeros(self.T)
        for t in range(self.T):
            fairness_regret[t] = sum([max(self.p_star[i] - self.curr_test.pi[t][i], 0.) for i in range(self.k)])
        return fairness_regret

    # def get_not_fair_ratio(self):
    #     return np.divide(self.average_not_smooth_fair, self.average_not_smooth_fair + self.average_smooth_fair)
    #
    # def get_fair_ratio(self):
    #     return np.divide(self.average_smooth_fair, self.average_not_smooth_fair + self.average_smooth_fair)

    def get_regret(self):
        distance_to_max = max(self.r_theta) - self.r_theta
        return np.apply_along_axis(lambda x: np.sum(x * distance_to_max), 1, self.average_n)

    def analyse(self, regret=True, fair_regret=True, smooth_fair = True, subjective_smooth_fair = False):
        for it in range(int(self.n_iter)):

            self.curr_test.run()
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

            self.curr_test.reset()

        self.average_n = np.divide(self.average_n, self.n_iter)
        if regret:
            self.average_regret = self.get_regret()
        if fair_regret:
            self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)



