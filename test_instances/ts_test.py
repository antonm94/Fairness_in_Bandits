from fairness_calc import smooth_fairness
import thompson_sampling.stochastic_dominance as sd_ts
import numpy as np


class TSTest:
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, distance):
        self.curr_test = sd_ts.StochasticDominance(bandits, T, lam=0, distance=distance)
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

    def is_smooth_fair(self):
        for e1_ind in range(len(self.e1_arr)):
            for e1_ind in range(len(self.e1_arr)):


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

    def analyse(self):
        for it in range(int(self.n_iter)):

            self.curr_test.run()
            self.average_fairness_regret = self.average_fairness_regret + self.calc_fairness_regret()
            self.average_n = self.average_n + self.curr_test.n
            for i in range(len(self.e1_arr)):
                for j in range(len(self.e2_arr)):
                    self.calc_smooth_fairness(i, j)
            self.curr_test.reset()

        self.average_n = np.divide(self.average_n, self.n_iter)
        self.average_regret = self.get_regret()
        self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)



