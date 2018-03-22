import scipy.stats as st
import numpy as np
from math import *
from scipy import special

class NormalInverseGamma(st.rv_continuous):
    """A Normal Inverse Gaussian continuous random variable.
    %(before_notes)s
    Notes
    -----
    The probability density for a Normal Inverse Gaussian distribution is
        nig.pdf(x, m, a, b, d)=
            (a*d/pi) * kv(1,a*np.sqrt(d**2+(x-m)**2))*
            exp(d*sqrt(a**2-b**2)-b*(m-x))/
            sqrt(d**2+(x-m)**2)
    where kv is a Bessel function of second order.
    m is mean, a is tail heaviness, b is asymmetry parameter
    and d is scale parameter (see Wikipedia)
    %(example)s
    """
    def _argcheck(self, mu, a, b, d):
        return (a > 0) & (d > 0) & (np.absolute(b) < a)

    def _pdf(self, x, mu, a, b, d):
        return (a * d / pi) * \
            special.kv(1, a * sqrt(d ** 2 + (x - mu) ** 2)) * \
            exp(d * sqrt(a ** 2 - b ** 2) - b * (mu - x)) / \
            sqrt(d ** 2 + (x - mu) ** 2)

    def _logpdf(self, x, mu, a, b, d):
        return log((a * d / pi)) + \
            log(special.kv(1, a * sqrt(d ** 2 + (x - mu) ** 2))) + \
            d * sqrt(a ** 2 - b ** 2) - b * (mu - x) - \
            log(sqrt(d ** 2 + (x - mu) ** 2))

    def _stats(self, mu, a, b, d):
        gamma = sqrt(a ** 2 - b ** 2)
        mean = mu + d * b / gamma
        variance = d * a ** 2 / gamma ** 3
        skewness = 3 * b / (a * sqrt(d * gamma))
        kurtosis = 3 * (1 + 4 * b ** 2 / a ** 2) / (d * gamma)
        return mean, variance, skewness, kurtosis


nig = NormalInverseGamma(shapes="m, a, b, d")
if __name__ == '__main__':
    pdf = NormalInverseGamma(0, 1, 1, 1).rvs(1, 1, 1, 1, 1)
    print pdf