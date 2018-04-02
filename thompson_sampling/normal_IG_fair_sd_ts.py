import numpy as np
from calc_c import c_alg_normal
from normal_IG_stochastic_dominance_ts import NormalStochasticDominance
from normal_IG_ts import NormalThompsonSampling


class NormalFairStochasticDominance(NormalStochasticDominance, NormalThompsonSampling):

    def __init__(self, bandits, T, e2, delta, lam=1, mod=0, smart_exploration=False, mean_0=0., alpha_0=1., beta_0=0.):
        NormalStochasticDominance.__init__(self, bandits, T, lam, mean_0=mean_0, alpha_0=alpha_0,  beta_0=beta_0)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0
        self.mod = mod
        self.e2 = e2
        self.delta = delta
        self.c = c_alg_normal(e2, delta, self.bandits)
        self.smart_explore = smart_exploration

    def reset(self):
        NormalStochasticDominance.reset(self)
        self.rounds_exploring = 0
        self.rounds_exploiting = 0
        if self.init_phase:
            self.init_length = int(self.v_0) * self.k - 1
        else:
            self.init_length = 0

    def run(self):
        ''''''''''Exploration'''''''''
        t_exp=0
        o = np.ones(self.k)
        while np.sum(o) and t_exp<self.T:
            if self.not_ts[t_exp] or t_exp <= self.init_length:
                a = np.random.choice(self.k)
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.update_round_x_s(a, r, t_exp)

            else:
                NormalThompsonSampling.calc_pi(self, t_exp)
                a = NormalThompsonSampling.get_a(self)
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.update_round_x_s(a, r, t_exp)
                self.update_param(a, t_exp)
                self.update_distribution(a, t_exp)
                NormalThompsonSampling.update_samples(self)
            if self.n[t_exp, a] > self.c:
                o[a] = 0
            self.rounds_exploring += 1
            t_exp += 1

        t_exp -= 1
        self.init_length = t_exp
        for a in range(self.k):
            self.update_param(a, t_exp)
            self.update_distribution(a, t_exp)
        t_exp += 1
        '''''''''Exploitation'''''''''

        for t in range(t_exp, self.T):
            if self.not_ts[t]:
                NormalStochasticDominance.calc_pi(self, t)
                a = NormalStochasticDominance.get_a(self)
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.update_round_x_s(a, r, t)
                self.update_param(a, t)
                self.update_distribution(a, t)
                NormalStochasticDominance.update_samples(self)
                self.rounds_exploiting += 1

            else:
                NormalThompsonSampling.calc_pi(self, t)
                a = NormalThompsonSampling.get_a(self)
                r = self.bandits.pull(a)
                self.rewards[a] += r
                self.update_round_x_s(a, r, t)
                self.update_param(a, t)
                self.update_distribution(a, t)
                NormalThompsonSampling.update_samples(self)
                self.rounds_exploiting += 1





        # print self.pi
        # for i in range(self.k):
        #     print self.inv_gamma[i].mean()
        #     print self.normal[i].mean()
        #     print self.student_t[-1, i].stats('mv')
