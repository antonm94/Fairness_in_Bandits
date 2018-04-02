import operator
import scipy.stats
import math
import scipy
import numpy as np
import time
def total_variation_distance(p, q):
    diff_abs = map(abs, map(operator.sub, p, q))
    return sum(diff_abs) * 0.5


def kl_divergence(p, q):

    kl = scipy.stats.entropy(p, q)
    # print kl
    return kl


def cont_kl(p, q, norm=0):
    if norm:
        pm = p.mean()
        qm = q.mean()
        pv = p.var()
        qv = q.var()
        a = pow(pm - qm, 2) / (2 * qv)
        var_div = (pv / qv)
        b = var_div - 1 - math.log(var_div)
        return a + 0.5 * b
    else:
        f = lambda x: -p.pdf(x) * (q.logpdf(x) - p.logpdf(x))
        kl = scipy.integrate.quad(f, -np.inf, np.inf)[0]
        return kl





def cont_total_variation_distance(p, q, norm=1):
    fun = lambda x: abs(p.pdf(x) - q.pdf(x))
    tv = 0.5 * scipy.integrate.quad(fun, -np.inf, np.inf)[0]
    return tv



# Copyright (c) 2008 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III

"""
Divergence and distance measures for multivariate Gaussians and
multinomial distributions.

This module provides some functions for calculating divergence or
distance measures between distributions, or between one distribution
and a codebook of distributions.
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision$"

import numpy

def gau_bh(pm, pv, qm, qv):
    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = numpy.log(pv).sum()
    ldqv = numpy.log(qv).sum(axis)
    # Log-determinant of pqv
    ldpqv = numpy.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)
    return dist + norm


def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (numpy.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)

             - len(pm)))                     # - N

def gau_js(pm, pv, qm, qv):
    """
    Jensen-Shannon divergence between two Gaussians.  Also computes JS
    divergence between a single Gaussian pm,pv and a set of Gaussians
    qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverses of diagonal covariances pv, qv
    iqv = 1./qv
    ipv = 1./pv
    # Difference between means pm, qm
    diff = qm - pm
    # KL(p||q)
    kl1 = (0.5 *
           (numpy.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)))                     # - N
    # KL(q||p)
    kl2 = (0.5 *
           (numpy.log(dpv / dqv)            # log |\Sigma_p| / |\Sigma_q|
            + (ipv * qv).sum(axis)          # + tr(\Sigma_p^{-1} * \Sigma_q)
            + (diff * ipv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p)
            - len(pm)))                     # - N
    # JS(p,q)
    return 0.5 * (kl1 + kl2)

def multi_kl(p, q):
    """Kullback-Liebler divergence from multinomial p to multinomial q,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Clip before taking logarithm to avoid NaNs (but still exclude
    # zero-probability mixtures from the calculation)
    return (p * (numpy.log(p.clip(1e-10,1))
                 - numpy.log(q.clip(1e-10,1)))).sum(axis)

def multi_js(p, q):
    """Jensen-Shannon divergence (symmetric) between two multinomials,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # D_{JS}(P\|Q) = (D_{KL}(P\|Q) + D_{KL}(Q\|P)) / 2
    return 0.5 * ((q * (numpy.log(q.clip(1e-10,1))
                        - numpy.log(p.clip(1e-10,1)))).sum(axis)
                      + (p * (numpy.log(p.clip(1e-10,1))
                              - numpy.log(q.clip(1e-10,1)))).sum(axis))
if __name__ == '__main__':
    import scipy.special
    import numpy as np

    qfunc = lambda x: 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))
    fun1 = lambda x: p.pdf(x)
    fun2 = lambda x: q.pdf(x)
    df1 = 123
    loc1 = 1
    scale1 = 0.1
    df2 = 1230
    loc2 = 2
    scale2 = 0.5
    p = scipy.stats.t(df=df1, loc=loc1, scale=scale1)
    q = scipy.stats.t(df=df2, loc=loc2, scale=scale2)
    int1 = scipy.integrate.quad(fun1, -np.inf, np.inf)[0]
    int2 = scipy.integrate.quad(fun2,  -np.inf, np.inf)[0]
    tv1 = 0.5*abs(int1 - int2)
    # print tv1
    t = time.time()
    for i in range(10):
        a = cont_total_variation_distance(p, q)
    print time.time() - t
    print a

    t = time.time()
    for i in range(10):
        s = [abs(q.pdf(x) - p.pdf(x)) for x in np.lin]
        a = 0.5*np.sum(s)
    print time.time() - t
    print a


    # print abs((qfunc((-np.inf)) - loc1)/scale1 - (qfunc((-np.inf)) - loc2)/scale2)



   # print a
    #
    # # fun = lambda x: abs(p.pdf(x) - q.pdf(x))
    #
    # fun = lambda x: abs(qfunc((x-loc1)/scale1) - qfunc((x-loc2)/scale2))
    #
    # # fun = lambda x: abs(((qfunc(x)-loc1)/scale1) - ((qfunc(x)-loc2)/scale2))
    # tv = 0.5 * scipy.integrate.quad(fun, -np.inf, np.inf)[0]
    # print tv