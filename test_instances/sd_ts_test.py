import thompson_sampling.bern_stochastic_dominance_ts as sd_ts
from ts_test import TSTest
from distance import total_variation_distance


class SDTest(TSTest):
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, lam=1, distance=total_variation_distance):
        TSTest.__init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, distance)
        self.curr_test = sd_ts.BernStochasticDominance(self.bandits, self.T, lam)
        self.name = 'SD TS'
        self.lam = lam
