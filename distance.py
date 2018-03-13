import operator
import scipy.stats


def total_variation_distance(p, q):
    diff_abs = map(abs, map(operator.sub, p, q))
    return sum(diff_abs) * 0.5


def kl_divergence(p, q):
    return scipy.stats.entropy(p, q)



if __name__ == '__main__':
    p = [0.4, 0.6]
    q = [0.6, 0.4]
    print total_variation_distance(p,q)
    print kl_divergence(p,q)