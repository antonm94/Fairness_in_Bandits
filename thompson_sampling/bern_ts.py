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
        self.not_ts = np.zeros(self.T)

    def reset(self):
        self.pi = np.zeros((self.T, self.k))
        self.r_h = np.full((self.T, self.k), .5)
        self.s = np.full((self.T+1, self.k), .5)
        self.f = np.full((self.T+1, self.k), .5)
        self.n = np.zeros((self.T, self.k))
        self.not_ts = np.zeros(self.T)

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

        self.n[t, a] = self.n[t, a] + 1


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


    def calc_pi(self, starting_round=0):
        """""workaround for floating point issues: same probs have to result in same pi"""

        # r := binary reward vector
        r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=self.k)]
        r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]

        perm_prod = np.zeros((self.T, len(r_permutations)))
        self.pi[:starting_round] = 1./self.k

        for t in range(starting_round, self.T):
            if self.not_ts[t]:
                for perm_i in range(len(r_permutations)):
                    #np.set_printoptions(100)
                    # print np.prod(self.get_r(r_permutations[perm_i], t))
                    # print np.exp(math.fsum(self.get_r_log(r_permutations[perm_i], t)))
                    perm_prod[t][perm_i] = np.prod(self.get_r(r_permutations[perm_i], t))
                    # perm_prod[t][perm_i] = np.exp(math.fsum(self.get_r_log(r_permutations[perm_i], t)))


                arms_with_same_prob = self.get_arms_with_same_prob(t)
                arms = range(self.k)

                for duplicate in arms_with_same_prob:
                    a = duplicate[0]
                    arms.remove(a)
                    self.pi_from_perm_prob(a, perm_prod, r_permutations, r_sum, t)
                    for i in np.delete(duplicate, 0):
                        arms.remove(i)
                        self.pi[t][i] = self.pi[t][a]

                for a in arms:
                    self.pi_from_perm_prob(a, perm_prod, r_permutations, r_sum, t)
            else:
                """every arm need entry in counts otherwise dim pi  not k -> add and subtract """
                n_iter = 1000
                theta = np.random.beta(self.s[t], self.f[t], (n_iter, self.k))
                max_theta = np.append(np.argmax(theta, axis=1), np.arange(self.k))
                counts = np.subtract(np.bincount(max_theta).astype(np.float), 1)
                self.pi[t] = np.divide(counts, n_iter)
        # print self.pi

    def pi_from_perm_prob(self, a, perm_prod, r_permutations, r_sum, t):
        for perm_i in range(len(r_permutations)):

            if r_permutations[perm_i][a]:
                # self.pi[t][a] = self.pi[t][a] + math.fsum(perm_prod[t][perm_i]/r_sum[perm_i])
                self.pi[t][a] += perm_prod[t][perm_i] / r_sum[perm_i]
            elif not r_sum[perm_i]:
                self.pi[t][a] += perm_prod[t][perm_i] / self.k

    def get_arms_with_same_prob(self, t):
        records_array = self.r_h[t]
        idx_sort = np.argsort(records_array)
        sorted_records_array = records_array[idx_sort]
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                        return_index=True)

        # sets of indices
        res = np.split(idx_sort, idx_start[1:])
        # filter them with respect to their size, keeping only items occurring more than once

        vals = vals[count > 1]
        res = filter(lambda x: x.size > 1, res)
        return res
