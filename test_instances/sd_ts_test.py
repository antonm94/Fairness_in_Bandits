import thompson_sampling.stochastic_dominance as sd_ts
from ts_test import TSTest


class SDTest(TSTest):
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, lam, distance):
        TSTest.__init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, distance)
        self.curr_test = sd_ts.StochasticDominance(self.bandits, self.T, lam, distance=distance)


