import scipy.stats as st
import scipy


n = 200
k = 100
a = 0.5
b = 0.5

# class my_pdf(st.rv_continuous):
#     def _pdf(self,x):
#         return pow(x, a+k-1)*pow(1-x, b+n-k-1)


if __name__ == '__main__':

    # my_rv = my_pdf(a=0, b=1, name='my_pdf')
    # rvs = my_rv.rvs(size=10)
    # print rvs
    # print rvs.mean()
    rvs = scipy.stats.beta.rvs(a, b, size=´´)
    print rvs.mean()
    print [theta*pow(theta, k)*pow(1-theta, n-k) for theta in rvs]
    c = scipy.special.beta(a+k, b+n-k)
    print c
    print sum([theta*pow(theta, k)*pow(1-theta, n-k) for theta in rvs])/ c


