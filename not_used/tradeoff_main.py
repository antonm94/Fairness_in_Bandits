
import random
from load_data import load_data
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.ts_test import  TSTest
from test_instances.sd_ts_test import SDTest
import numpy as np


np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

SEED = True

N_ITERATIONS = 1000.
T = 100
SETS = ['0', 'Bar Exam', 'Default on Credit', '3']
# SETS = ['2']
# 'Bar Exam'
# SETS = [ 'Bar Exam', '0']


if __name__ == '__main__':

    e2_arr= [0.2]
    delta_arr = [0.2]
    n = 100
    lam_arr = np.linspace(0, 1., n)

    if SEED:
        random.seed(0)
        np.random.seed(0)

    bandits = []
    for set in SETS:
        bandits.append(load_data(set))

    sd_regret = np.zeros(n)
    sd_fregret = np.zeros(n)


    sd_tests = []
    for lam_ind, lam in enumerate(lam_arr):
        test = SDTest(N_ITERATIONS, bandits[0], T, [0], [0], [0], lam=lam)
        test.analyse(fair_regret=True, regret=True, subjective_smooth_fair=False, smooth_fair=False,
                             subjective_minimum_e1=False,
                             minimum_e1=False)

    fair_collection = np.zeros((len(e2_arr), len(delta_arr)), dtype=np.ndarray)


    for lam in lam_arr:

    for e2_ind, e2 in enumerate(e2_arr):
        for d_ind, d in enumerate(delta_arr):
            fair_sd_tests = []
            for lam in lam_arr:
                fair_sd_tests.append(FairSDTest(N_ITERATIONS, bandits[0], T, [0], [e2], [d], lam=lam))
                fair_sd_tests[-1].analyse(regret=True, fair_regret=True, smooth_fair=False, minimum_e1=False)
            fair_collection[e2_ind, d_ind] = fair_sd_tests




    # fair_sd_tests.append(TSTest(N_ITERATIONS, bandits[0], T, e1, e2, delta))
    # fair_sd_tests[-1].analyse(regret=True, fair_regret=True, smooth_fair=False,
    #                           subjective_smooth_fair=False, minimum_e1=False,
    #                           subjective_minimum_e1=False)


# def infos_print(tests):
#     if TEST_FAIR_SD_TS:
#         print [test.c for test in fair_sd_ts_test]
#         print [test.average_round_exploring for test in fair_sd_ts_test]