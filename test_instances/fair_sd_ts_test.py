import thompson_sampling.bern_fair_stochastic_dominance_ts as fair_sd_ts
from ts_test import TSTest
import numpy as np
from distance import total_variation_distance
from fairness_calc import smooth_fairness


class FairSDTest(TSTest):
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, lam=1, distance=total_variation_distance):
        TSTest.__init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, distance)
        self.test_cases = np.empty((len(e2_arr), len(delta_arr)), object)
        for i in range(len(e2_arr)):
            for d in range(len(delta_arr)):
                self.test_cases[i][d] = fair_sd_ts.FairStochasticDominance(bandits, T, e2_arr[i], delta_arr[d], lam, distance)
        self.curr_test = self.test_cases[0][0]
        self.average_fairness_regret = np.zeros((len(e2_arr), len(delta_arr), T))
        self.average_n = np.zeros((len(e2_arr), len(delta_arr), self.T, self.k))
        self.average_rounds_exploring = np.zeros((len(e2_arr), len(delta_arr)))
        self.average_rounds_exploiting = np.zeros((len(e2_arr), len(delta_arr)))
        self.lam = lam
        self.name = 'Fair SD TS'
        self.smooth_fair = np.zeros((len(e1_arr), len(e2_arr), len(delta_arr), self.T, self.k, self.k))

    def get_rounds(self):
        return self.average_rounds_exploring, self.average_rounds_exploiting


    def get_regret(self):
        distance_to_max = max(self.r_theta) - self.r_theta
        regret = np.zeros((len(self.e2_arr), len(self.delta_arr), self.T))
        for j in range(len(self.e2_arr)):
            for d in range(len(self.delta_arr)):
                regret[j][d] = np.apply_along_axis(lambda x: sum(x * distance_to_max), 1, self.average_n[j][d])
        return regret

    def calc_smooth_fairness(self, e1_ind, e2_ind, d, e2_times=1):

        for t in range(self.T):
            for i in range(self.k):
                # for j in range(i + 1, self.k):
                for j in range(self.k):
                    # self.r_theta = np.full(k, 0.5)+k[t] n[t]

                    self.smooth_fair[e1_ind][e2_ind][d][t][i][j] += smooth_fairness(self.e1_arr[e1_ind],
                                                                                 e2_times * self.e2_arr[e2_ind]
                                                                                 , i, j, self.curr_test.theta[t],
                                                                                 self.r_theta, self.distance)

    def frac_smooth_fair(self):
        for e1_ind in range(len(self.e1_arr)):
            for e2_ind in range(len(self.e2_arr)):
                for delta_ind in range(len(self.delta_arr)):
                   # print (np.divide(self.smooth_fair[e1_ind][e2_ind], self.n_iter) >= 1 - self.delta_arr[delta_ind])
                    self.is_smooth_fair[e1_ind][e2_ind][delta_ind] = np.all(np.divide(self.smooth_fair[e1_ind][e2_ind]
                                                                                      [delta_ind], self.n_iter) >= 1 -
                                                                            self.delta_arr[delta_ind])

    def analyse(self, regret=True, fair_regret=True, smooth_fair = True, e2_times=1):

        for it in range(int(self.n_iter)):
            for j in range(len(self.e2_arr)):
                for d in range(len(self.delta_arr)):
                    self.curr_test = self.test_cases[j][d]
                    self.curr_test.run()
                    self.average_rounds_exploiting[j][d] += self.curr_test.rounds_exploiting
                    self.average_rounds_exploring[j][d] += self.curr_test.rounds_exploring
                    if regret:
                        self.calc_fairness_regret()
                    if fair_regret:
                        self.average_fairness_regret[j][d] += self.calc_fairness_regret()
                    self.average_n[j][d] += self.curr_test.n
                    if smooth_fair:
                        for i in range(len(self.e1_arr)):
                           self.calc_smooth_fairness(i, j, d, e2_times)

                    self.curr_test.reset()

        self.average_n = np.divide(self.average_n, self.n_iter)
        if fair_regret:
                self.average_regret = self.get_regret()
        if fair_regret:
            self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)
        # if smooth_fair:
        #     self.average_smooth_fair = np.divide(self.average_smooth_fair, self.n_iter)
        #     self.average_not_smooth_fair = np.divide(self.average_not_smooth_fair, self.n_iter)
        self.average_rounds_exploring = np.divide(self.average_rounds_exploring, self.n_iter)
        self.average_rounds_exploiting = np.divide(self.average_rounds_exploiting, self.n_iter)


