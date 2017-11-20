import operator


def total_variation_distance(p, q):
    diff_abs = map(abs, map(operator.sub, p, q))
    return sum(diff_abs) * 0.5



