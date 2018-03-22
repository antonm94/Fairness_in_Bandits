import numpy as np
import scipy.stats as stats
import math
import divergence

class NormalBandits:
    def __init__(self, means, variances, data_set_name='no_name', divergence_fun=divergence.normal_kl):
        self.k = len(means)
        self.mean = np.asarray(means, dtype=np.float)
        self.variance = np.asarray(variances, dtype=np.float)
        # self.p_star = self.calc_p_star()
        self.data_set_name = data_set_name
        self.divergence_fun = divergence_fun
        self.divergence = np.zeros((self.k, self.k))

        self.p_star = self.calc_p_star()
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    self.divergence[i, j] = divergence_fun(self.mean[i], self.variance[i], self.mean[j], self.variance[j])
        self.normal = np.full(self.k, stats.norm(self.mean[0], math.sqrt(self.variance[0])))
        for a in range(1, self.k):
            self.normal[a] = stats.norm(self.mean[a], math.sqrt(self.variance[a]))

    def pull(self, a):
        return self.normal[a].rvs(1)

    def calc_p_star(self):
        n_iter = 10000.
        wins = np.zeros(self.k)
        for i in range(int(n_iter)):
            guessed_r = np.random.normal(self.mean, np.sqrt(self.variance))
            wins[np.random.choice(np.where(guessed_r == guessed_r.max())[0])] += 1
        p_star = np.divide(wins, n_iter)
        return p_star
# def calc_p_star(self):
    #     p_star = np.zeros(self.k)
    #
    #     r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=self.k)]
    #     r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]
    #     perm_prod = np.zeros(len(r_permutations))
    #     for perm_i in range(len(r_permutations)):
    #         perm_prod[perm_i] = np.prod(self.get_r(r_permutations[perm_i]))
    #
    #     for a in range(self.k):
    #         for perm_i in range(len(r_permutations)):
    #             if r_permutations[perm_i][a]:
    #                 p_star[a] += perm_prod[perm_i] / r_sum[perm_i]
    #             elif not r_sum[perm_i]:
    #                 p_star[a] += perm_prod[perm_i] / self.k
    #
    #     if not isclose(sum(p_star), 1.0):
    #         print "p star doesn't sum to one"
    #     return p_star
    #
    # def get_r(self, perm):
    #     r_prop = []
    #     for i in range(self. k):
    #         if perm[i]:
    #             r_prop.append(self.theta[i])
    #         else:
    #             r_prop.append(1 - self.theta[i])
    #
    #     return r_prop
    #
    #
    # def get_max_D(self, distance=total_variation_distance):
    #     dmax = 0.
    #     for i in range(self.k):
    #         for j in range(self.k):
    #             dmax = max(distance([self.theta[i], 1-self.theta[i]], [self.theta[j], 1-self.theta[j]]), dmax)
    #     return dmax





