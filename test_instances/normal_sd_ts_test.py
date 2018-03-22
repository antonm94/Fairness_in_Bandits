import thompson_sampling.normal_IG_stochastic_dominance_ts as sd_ts
from normal_ts_test import NormalTSTest


class NormalSDTest(NormalTSTest):
    def __init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr, lam=1, mean_0=0., alpha_0=1., beta_0=0., v_0=-1):
        NormalTSTest.__init__(self, n_iter, bandits, T, e1_arr, e2_arr, delta_arr,
                              mean_0=mean_0, alpha_0=alpha_0, beta_0=beta_0, v_0=v_0)
        self.curr_test = sd_ts.NormalStochasticDominance(self.bandits, self.T, lam)
        self.name = 'SD TS'
        self.lam = lam