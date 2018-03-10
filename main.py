
import random
from load_data import load_data
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.ts_test import  TSTest
from test_instances.sd_ts_test import SDTest
import numpy as np
np.set_printoptions(threshold=np.nan)


SEED = False
TEST_THOMPSON = True
TEST_SD_TS = True
TEST_FAIR_SD_TS = True
PLOT = False

N_ITERATIONS = 10.
T = 20
SETS = ['0', '1', '2', '3']
# SETS = ['2', '3']
# 'Bar Exam'
# 'Default on Credit'


if __name__ == '__main__':

    e1 = [2., 1., 0.5, 0.4]
    e2 = [0., 0.3, 0.1, np.finfo(float).eps]
    delta = [0., 0.1, 0.3, 0.4, 0.7, np.finfo(float).eps]

    if SEED:
        random.seed(0)
        np.random.seed(0)

    ts_test = []
    if TEST_THOMPSON:
        for set in SETS:
            ts_test.append(TSTest(N_ITERATIONS, load_data(set), T, e1, e2[:-1], delta[:-1]))

    sd_ts_test = []
    if TEST_SD_TS:
        for set in SETS:
            sd_ts_test.append(SDTest(N_ITERATIONS, load_data(set), T, e1, e2[:-1], delta[:-1], lam=1))

    fair_sd_ts_test = []
    if TEST_FAIR_SD_TS:
        for set in SETS:
            fair_sd_ts_test.append(FairSDTest(N_ITERATIONS, load_data(set), T, e1, e2[1:], delta[1:], lam=1))

    for test in ts_test:
        test.analyse(regret=True, fair_regret=True, smooth_fair=True,
                     subjective_smooth_fair=True, minimum_e1=True,
                     subjective_minimum_e1=True)

    for test in sd_ts_test:
        test.analyse(regret=True, fair_regret=True, smooth_fair=True,
                     subjective_smooth_fair=True, minimum_e1=True,
                     subjective_minimum_e1=True)

    for test in fair_sd_ts_test:
        test.analyse(regret=True, fair_regret=True, smooth_fair=True, minimum_e1=True)

    if PLOT:
        import plot.plots as plt
        if TEST_THOMPSON or TEST_SD_TS:
            test_cases = ts_test + sd_ts_test
            plt.plot_delta_subjective_fair(test_cases)
            plt.plot_subjective_min_e1(test_cases)

        if TEST_FAIR_SD_TS:
            test_cases = ts_test + sd_ts_test + fair_sd_ts_test
            plt.plot_delta_smooth_fair(test_cases)
            plt.plot_min_e1(test_cases)
            plt.plot_average_total_regret(test_cases)
            plt.plot_fairness_regret(test_cases)








