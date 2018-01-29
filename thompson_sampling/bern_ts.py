import numpy as np
import calculations
import itertools
import math
class BernThompsonSampling(object):


    def __init__(self, bandits, T):
        self.k = bandits.k
        self.bandits = bandits
        self.T = T
        self.prior_a = 0.5
        self.prior_b = 0.5

        self.s = np.full((self.T+1, self.k), self.prior_a)
        self.f = np.full((self.T+1, self.k), self.prior_b)
        self.theta = np.zeros((self.T, self.k))
        self.n = np.zeros((self.T, self.k))
        self.pi = np.zeros((self.T, self.k))
        self.r_h = np.full((self.T, self.k), .5)



    def reset(self):
        self.pi = np.zeros((self.T, self.k))
        self.r_h = np.full((self.T, self.k), .5)
        self.s = np.full((self.T+1, self.k), .5)
        self.f = np.full((self.T+1, self.k), .5)
        self.n = np.zeros((self.T, self.k))

    def run(self):
        for t in range(self.T):
            self.theta[t] = np.random.beta(self.s[t], self.f[t], self.k)

            a = self.get_a(t)

            reward = self.bandits.pull(a)
            self.update(t, a, reward)

        self.calc_r_h()
        self.calc_pi()

    def get_a(self, t):
        max_theta = np.where(self.theta[t] == self.theta[t].max())[0]
        a = np.random.choice(max_theta)

        return a



    def update(self, t, a, reward):


        if t>0:
            self.n[t] = self.n[t-1]



        self.s[t+1] = self.s[t]
        self.f[t+1] = self.f[t]
        if reward:
            self.s[t + 1][a] = self.s[t+1][a] + 1
        else:
            self.f[t + 1][a] = self.f[t+1][a] + 1

        # sum_r = np.sum(self.r_1_h[t])
        # prod_r = np.prod(self.r_1_h[t])
        # for i in range(self.k):
        #     self.pi[t][i] = self.r_1_h[t][i] / (sum_r - self.r_1_h[t][i]) + prod_r


       # self.pi[t] = self.r_1_h[t] / np.sum(self.r_1_h[t])

        self.n[t][a] = self.n[t][a] + 1


    def calc_r_h(self):

        self.r_h = np.divide(self.s[:self.T], self.s[:self.T] + self.f[:self.T])

    def get_r_log(self, perm, t):
        r_prop = []
        for i in range(self.k):
            if perm[i]:
                r_prop.append(np.log(self.r_h[t][i]))
            else:
                r_prop.append(np.log(1.-self.r_h[t][i]))

        return r_prop

    def get_r(self, perm, t):
        r_prop = []
        for i in range(self.k):
            if perm[i]:
                r_prop.append(self.r_h[t][i])
            else:
                r_prop.append(1.-self.r_h[t][i])

        return r_prop


    def calc_pi(self):
        # r := binary reward vector
        r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=self.k)]
        r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]

        perm_prod = np.zeros((self.T, len(r_permutations)))
        for t in range(self.T):
            for perm_i in range(len(r_permutations)):
                #np.set_printoptions(100)
                # print np.prod(self.get_r(r_permutations[perm_i], t))
                # print np.exp(math.fsum(self.get_r_log(r_permutations[perm_i], t)))
                perm_prod[t][perm_i] = np.prod(self.get_r(r_permutations[perm_i], t))
                # perm_prod[t][perm_i] = np.exp(math.fsum(self.get_r_log(r_permutations[perm_i], t)))

            for a in range(self.k):
                for perm_i in range(len(r_permutations)):
                    if r_permutations[perm_i][a]:
                        # self.pi[t][a] = self.pi[t][a] + math.fsum(perm_prod[t][perm_i]/r_sum[perm_i])
                        self.pi[t][a] += perm_prod[t][perm_i]/r_sum[perm_i]
                    elif not r_sum[perm_i]:
                        # self.pi[t][a] = self.pi[t][a] + perm_prod[t][perm_i]/self.k
                        self.pi[t][a] += perm_prod[t][perm_i]/self.k

