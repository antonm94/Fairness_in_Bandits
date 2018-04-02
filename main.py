import random
from load_data import load_data
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.ts_test import  TSTest
from test_instances.sd_ts_test import SDTest
import numpy as np


np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)


SEED = True
TEST_THOMPSON = False
TEST_SD_TS = False
TEST_FAIR_SD_TS = True
PLOT = False

N_ITERATIONS = 10.
T = 50000
SETS = ['0', 'Bar Exam', 'Default on Credit', '3']
# SETS = ['2']
#SETS = ['Bar Exam']
SETS = ['0']
# SETS = ['Default on Credit']


if __name__ == '__main__':

    e1 = [2, 1, 0.5]
    e2 = [0., 0.05, 0.2, np.finfo(float).eps]
    delta = [0.1, 0.3, 0.7]
    e2 = [0.01, 0.05, 0.1]
    if SEED:
        random.seed(0)
        np.random.seed(0)


    load_data('Bar Exam')
    bandits = []
    for set in SETS:
        bandits.append(load_data(set))

    ts_test = []
    if TEST_THOMPSON:
        for bandit in bandits:
            ts_test.append(TSTest(N_ITERATIONS, bandit, T, e1, e2, delta))

    sd_ts_test = []
    if TEST_SD_TS:
        for bandit in bandits:
            print bandit.get_mean()
            print bandits
            sd_ts_test.append(SDTest(N_ITERATIONS, bandit, T, e1, e2[:-1], delta, lam=1))

    fair_sd_ts_test = []
    if TEST_FAIR_SD_TS:
        for bandit in bandits:
            fair_sd_ts_test.append(FairSDTest(N_ITERATIONS, bandit, T, e1, e2, delta, lam=1))

    for test in ts_test:
        test.analyse(regret=False, fair_regret=False, smooth_fair=True,
                     subjective_smooth_fair=False, minimum_e1=True,
                     subjective_minimum_e1=False)

    for test in sd_ts_test:
        test.analyse(regret=False, fair_regret=False, smooth_fair=True,
                     subjective_smooth_fair=False, minimum_e1=True,
                     subjective_minimum_e1=False)

    for test in fair_sd_ts_test:
        test.analyse(regret=False, fair_regret=False, smooth_fair=True, minimum_e1=True, e2_times=2.)

    if PLOT:
        import plot.plot_data_seperated as plt
        if TEST_THOMPSON or TEST_SD_TS:
            test_cases = ts_test + sd_ts_test
            # plt.plot_delta_subjective_fair(test_cases)
            # plt.plot_subjective_min_e1(test_cases)

        if TEST_FAIR_SD_TS:
            test_cases = ts_test + sd_ts_test + fair_sd_ts_test
            plt.plot_delta_smooth_fair(test_cases)
            plt.plot_min_e1(test_cases)
            plt.plot_average_total_regret(test_cases)
            plt.plot_fairness_regret(test_cases)


def infos_print(tests):
    if TEST_FAIR_SD_TS:
        print [test.c for test in fair_sd_ts_test]
        print [test.average_round_exploring for test in fair_sd_ts_test]