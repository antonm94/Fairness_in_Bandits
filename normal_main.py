from normalBandits import normalBandits
from thompson_sampling.normal_IG_prior import NormalThompsonSampling
import numpy as np
# np.set_printoptions(suppress=True)
from scipy.stats import invgamma
np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':
    bandits = normalBandits([1], [10])
    ts = NormalThompsonSampling(bandits, 10000)
    ts.run()
    print ts.student_t(0).mean()
    print ts.student_t(0).var()

    # print ts.variance_sq

    # t(df=self.v, loc=self.mean, scale=1).rvs(1)

