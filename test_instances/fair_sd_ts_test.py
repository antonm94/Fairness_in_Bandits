import thompson_sampling.bern_fair_stochastic_dominance_ts as fair_sd_ts
from ts_test import TSTest
import numpy as np


class FairSDTest(TSTest):
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, lam, distance):
        TSTest.__init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, distance)
        self.test_cases = np.empty((len(e2_arr), len(delta_arr)), object)
        for i in range(len(e2_arr)):
            for d in range(len(delta_arr)):
                self.test_cases[i][d] = fair_sd_ts.FairStochasticDominance(bandits, T, e2_arr[i], delta_arr[d], lam, distance)
        self.curr_test = self.test_cases[0][0]
        self.average_smooth_fair = np.zeros((len(e1_arr), len(e2_arr), len(delta_arr), self.T,))
        self.average_not_smooth_fair = np.zeros((len(e1_arr), len(e2_arr), len(delta_arr), self.T,))
        self.average_fairness_regret = np.zeros((len(e2_arr), len(delta_arr), T))
        self.average_n = np.zeros((len(e2_arr), len(delta_arr), self.T, self.k))
        self.average_rounds_exploring = 0
        self.average_rounds_exploiting = 0


    def get_rounds(self):
        return self.rounds_exploring, self.rounds_exploiting


    def get_regret(self):
        distance_to_max = max(self.r_theta) - self.r_theta
        regret = np.zeros((len(self.e2_arr), len(self.delta_arr), self.T))
        for j in range(len(self.e2_arr)):
            for d in range(len(self.delta_arr)):
                regret[j][d] = np.apply_along_axis(lambda x: sum(x * distance_to_max), 1, self.average_n[j][d])
        return regret
    def analyse(self):
        for it in range(int(self.n_iter)):
            for j in range(len(self.e2_arr)):
                for d in range(len(self.delta_arr)):
                    self.curr_test = self.test_cases[j][d]
                    self.curr_test.run()
                    self.average_rounds_exploiting = self.average_rounds_exploiting + self.curr_test.rounds_exploiting
                    self.average_rounds_exploring = self.average_rounds_exploring + self.curr_test.rounds_exploring
                    self.calc_fairness_regret()
                    self.average_fairness_regret[j][d] = self.average_fairness_regret[j][d] + self.calc_fairness_regret()
                    self.average_n[j][d] = self.average_n[j][d] + self.curr_test.n
                    for i in range(len(self.e1_arr)):
                        smooth = self.calc_smooth_fairness(self.e1_arr[i], self.e2_arr[j])
                        self.average_smooth_fair[i][j][d] = self.average_smooth_fair[i][j] + smooth[1]
                        self.average_not_smooth_fair[i][j][d] = self.average_not_smooth_fair[i][j] + smooth[0]
                    self.curr_test.reset()

        self.average_n = np.divide(self.average_n, self.n_iter)
        self.average_regret = self.get_regret()
        self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)
        self.average_smooth_fair = np.divide(self.average_smooth_fair, self.n_iter)
        self.average_not_smooth_fair = np.divide(self.average_not_smooth_fair, self.n_iter)
        self.average_rounds_exploring = np.divide(self.average_rounds_exploring, self.n_iter)
        self.average_rounds_exploiting = np.divide(self.average_rounds_exploiting, self.n_iter)


