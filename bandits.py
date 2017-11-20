import numpy as np


class Bandits:

    def __init__(self, arms):
        self.arms = arms
        self.k = len(arms)
        self.theta = self.get_mean()

    def get_mean(self):
        t = np.zeros(self.k)
        for i in range(self.k):
            t[i] = np.mean(self.arms[i])
        return t
