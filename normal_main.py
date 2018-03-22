import random
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.normal_ts_test import  NormalTSTest
from test_instances.normal_sd_ts_test import NormalSDTest
from thompson_sampling.normal_IG_ts import NormalThompsonSampling
from thompson_sampling.normal_IG_stochastic_dominance_ts import NormalStochasticDominance
import numpy as np
from normalBandits import NormalBandits

np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)


SEED = True
TEST_THOMPSON = True
TEST_SD_TS = True
TEST_FAIR_SD_TS = False
PLOT = 1


SF = True
SSF = 0
MIN_E1 = True
SMIN_E1 = 0
REGRET = True
FAIR_REGRET = True

N_ITERATIONS = 1.
T = 100




if __name__ == '__main__':

    e1 = [1.]
    e2 = [0.]
    delta = [0.]

    if SEED:
        random.seed(0)
        np.random.seed(0)

    bandits = [NormalBandits([1, 1],[4, 3])]

    if False:
    ts_test = []
    if TEST_THOMPSON:
        for bandit in bandits:
            ts_test.append(NormalTSTest(N_ITERATIONS, bandit, T, e1, e2, delta))

    sd_ts_test = []
    if TEST_SD_TS:
        for bandit in bandits:
            sd_ts_test.append(NormalSDTest(N_ITERATIONS, bandit, T, e1, e2, delta, lam=1))

    fair_sd_ts_test = []
    if TEST_FAIR_SD_TS:
        for bandit in bandits:
            fair_sd_ts_test.append(FairSDTest(N_ITERATIONS, bandit, T, e1, e2, delta, lam=0.5))

    for test in ts_test:
        test.analyse(regret=REGRET, fair_regret=FAIR_REGRET , smooth_fair=SF,
                     subjective_smooth_fair=SSF, minimum_e1=MIN_E1,
                     subjective_minimum_e1=SMIN_E1)

    for test in sd_ts_test:
        test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF,
                     subjective_smooth_fair=SSF, minimum_e1=MIN_E1,
                     subjective_minimum_e1=SMIN_E1)

    for test in fair_sd_ts_test:
        test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF, minimum_e1=MIN_E1)

    if PLOT:
        import plot.plot_data_seperated as plt
        if SSF and SMIN_E1:
            test_cases = ts_test + sd_ts_test
            plt.plot_delta_subjective_fair(test_cases)
            plt.plot_subjective_min_e1(test_cases)

        if SF and MIN_E1:
            test_cases = ts_test + sd_ts_test + fair_sd_ts_test
            plt.plot_min_e1(test_cases)
            plt.plot_delta_smooth_fair(test_cases)

        if REGRET and FAIR_REGRET:
            test_cases = ts_test + sd_ts_test + fair_sd_ts_test
            plt.plot_average_total_regret(test_cases)
            plt.plot_fairness_regret(test_cases)



def infos_print(tests):
    if TEST_FAIR_SD_TS:
        print [test.c for test in fair_sd_ts_test]
        print [test.average_round_exploring for test in fair_sd_ts_test]


def n_info(test_cases):
    for test in test_cases:
        print test.name + 'n' + str(test.average_n[-1])
        print test.name + 'pi prediction' + str(np.average(test.curr_test.pi, axis=0) * T)


def simple_tests(bandits, T):
    ts = NormalThompsonSampling(bandits, T)
    sd_ts = NormalStochasticDominance(bandits, T)
    ts.run()
    sd_ts.run()
    for i in range(bandits.k):
        print 'ts mean_norm: ' + str(ts.normal[i].mean())
        print 'ts var: ' + str(ts.inv_gamma[i].mean())
        print 'sd_ts mean_norm: ' + str(sd_ts.normal[i].mean())
        print 'sd_ts var: ' + str(sd_ts.inv_gamma[i].mean())
    print 'ts n: ' + str(ts.n[-1])
    print 'ts reward: ' + str(sum(ts.rewards))
    print 'sd_ts n: ' + str(sd_ts.n[-1])
    print 'sd_ts reward: ' + str(sum(sd_ts.rewards))
    print np.average(ts.pi, axis=0) * T
    print np.average(sd_ts.pi, axis=0) * T



