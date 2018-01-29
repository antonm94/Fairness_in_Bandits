import numpy as np
import itertools

k= 5
def get_r(perm, p):
    r_prop = []
    for i in range(k):
        if perm[i]:
            r_prop.append(p[i])
        else:
            r_prop.append(1 - p[i])

    return r_prop


if __name__ == '__main__':

    p = np.random.uniform(0, 1, k)
    p_norm = np.divide(p, np.sum(p))


    # r := binary reward vector
    r_permutations = [np.asarray(seq, dtype=np.int8) for seq in itertools.product("01", repeat=k)]
    r_sum = [np.count_nonzero(r_permutations[perm_i]) for perm_i in range(len(r_permutations))]
    pi = np.zeros(k)
    perm_prod = np.zeros(len(r_permutations))
    for perm_i in range(len(r_permutations)):
        perm_prod[perm_i] = np.prod(get_r(r_permutations[perm_i], p))

    for a in range(k):
        for perm_i in range(len(r_permutations)):
            if r_permutations[perm_i][a]:
                pi[a] += perm_prod[perm_i] / r_sum[perm_i]
            elif not r_sum[perm_i]:
                pi[a] += perm_prod[perm_i] / k


    print sum(pi)
    print sum(p_norm)