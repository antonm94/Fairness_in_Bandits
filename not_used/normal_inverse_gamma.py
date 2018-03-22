import scipy.stats as stats



"""""""""
 def NormalInverseGaussian(mean, df, alpha, beta):
        inv_gamma = invgamma(a=alpha, loc=0, scale=math.sqrt(beta))
        var_sq = inv_gamma.rvs(1)
        normal = norm(loc=mean, scale=math.sqrt(var_sq * df))
        mean = normal.rvs(1)

        return mean, var_sq
"""""""""


def update_normal_inverse_gamma(self):
    for a in range(self.k):
        self.inv_gamma[a] = stats.invgamma(a=self.alpha[a], loc=0, scale=math.sqrt(self.beta[a]))
        self.sample_var[a] = self.inv_gamma[a].rvs(1)
        self.normal[a] = stats.norm(loc=self.mean[a], scale=math.sqrt(self.sample_var[a] * self.v[a]))
        self.sampled_mu[a] = self.normal[a].rvs(1)