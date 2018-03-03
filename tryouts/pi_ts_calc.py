import numpy as np



if __name__ == '__main__':


    n_iter = 4
    k = 5
    n = 3
    s = np.random.randint(1, 8, (k, n))
    f = np.random.randint(1, 8, (k, n))
    # s = [1000.5, 2.5, 0.5, 3.5, 1000.5]
    # f = [200.5, 1.5, 2.5, 2.5, 200.5]

    theta = np.random.beta(s, f, (n_iter, k, n))

    max_theta = np.argmax(theta, axis=2)
    print theta
    print max_theta
    # counts = np.bincount(max_theta).astype(np.float)


    #     max_theta =
    #     max_theta = np.where(self.theta == self.theta.max())[0]
    #     ++a[np.random.choice(max_theta)]
    # np.divide(a, n_iter)
