import scipy.stats as stats



"""""""""
 def NormalInverseGaussian(mean, df, alpha, beta):
        inv_gamma = invgamma(a=alpha, loc=0, scale=math.sqrt(beta))
        var_sq = inv_gamma.rvs(1)
        normal = norm(loc=mean, scale=math.sqrt(var_sq * df))
        mean = normal.rvs(1)

        return mean, var_sq
"""""""""

