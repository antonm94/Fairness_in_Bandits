import random
from test_instances.normal_fair_sd_ts_test import NormalFairSDTest
from test_instances.normal_ts_test import NormalTSTest
from test_instances.normal_sd_ts_test import NormalSDTest
from thompson_sampling.normal_IG_ts import NormalThompsonSampling
from thompson_sampling.normal_IG_stochastic_dominance_ts import NormalStochasticDominance
import numpy as np
from normalBandits import NormalBandits
import divergence as div
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)



SEED = 1
TEST_THOMPSON = 1
TEST_SD_TS = 1
TEST_FAIR_SD_TS = 0
PLOT = 1

SF = 1
SSF = 1
MIN_E1 = 1
SMIN_E1 = 1
REGRET = 0
FAIR_REGRET = 0

N_ITERATIONS = 10.
T = 100

mean_0 = 0
alpha_0 = 0.5
beta_0 = 0.
init_phase = True

PI_DIV = div.total_variation_distance
# PI_DIV = div.kl_divergence

R_DIV = div.cont_total_variation_distance
# R_DIV = div.cont_kl


def infos_print(tests):
    if TEST_FAIR_SD_TS:
        print [test.c for test in fair_sd_ts_test]
        print [test.average_round_exploring for test in fair_sd_ts_test]


def n_info(test_cases):
    for test in test_cases:
        print test.name + 'n' + str(test.average_n[-1])
        print test.name + 'pi prediction' + str(np.average(test.curr_test.pi, axis=0) * T)


def simple_tests(bandits, T):
    # ts = NormalThompsonSampling(bandits, T, mean_0=mean_0, alpha_0=alpha_0, beta_0=beta_0, init_phase=init_phase)
    sd_ts = NormalStochasticDominance(bandits, T, mean_0=mean_0, alpha_0=alpha_0, beta_0=beta_0, init_phase=init_phase)
    # ts.run()
    sd_ts.run()

    print 'ts mean: ' + str([ts.normal[i].mean() for i in range(bandits.k)])
    print 'ts var: ' + str([ts.inv_gamma[i].mean() for i in range(bandits.k)])
    print 'sd_ts mean ' + str([sd_ts.normal[i].mean() for i in range(bandits.k)])
    print 'sd_ts var: ' + str([sd_ts.inv_gamma[i].mean() for i in range(bandits.k)])

    print 't-sd_ts mean_norm: ' + str([sd_ts.student_t[-1,i].stats('m')for i in range(bandits.k)])
    print 't-sd_ts var: ' + str([sd_ts.student_t[-1,i].stats('v') for i in range(bandits.k)])

    print 'ts n: ' + str(ts.n[-1])
    print 'ts reward: ' + str(sum(ts.rewards))
    print 'sd_ts n: ' + str(sd_ts.n[-1])
    print 'sd_ts reward: ' + str(sum(sd_ts.rewards))
    print np.average(ts.pi, axis=0) * T
    print np.average(sd_ts.pi, axis=0) * T





if __name__ == '__main__':

    e1 = [1.]
    e2 = [0.]
    delta = [0.0]

    if SEED:
        random.seed(0)
        np.random.seed(0)

    bandits = [NormalBandits([0, 0, 0, 0], [1, 0.5, 0.3, 0.2], data_set_name='D0', divergence_fun=R_DIV),
               NormalBandits([1., 0.95, 0.3, -0.1], [1., 1., 1., 1.], data_set_name='D1', divergence_fun=R_DIV),
               NormalBandits([0.5, -1, 1, 0.5], [6, 2, 0.1, 1], data_set_name='D2', divergence_fun=R_DIV),
               NormalBandits([0, 0], [5, 1], data_set_name='D3', divergence_fun=R_DIV)]
    # bandits = [NormalBandits([0, 0], [4, 1], data_set_name='D3', divergence_fun=R_DIV)]
    if True:
        ts_test = []
        if TEST_THOMPSON:
            for bandit in bandits:
                ts_test.append(NormalTSTest(N_ITERATIONS, bandit, T, e1, e2, delta, mean_0=mean_0, alpha_0=alpha_0,
                                            beta_0=beta_0, pi_div=PI_DIV, r_div=R_DIV))

        sd_ts_test = []
        if TEST_SD_TS:
            for bandit in bandits:
                sd_ts_test.append(NormalSDTest(N_ITERATIONS, bandit, T, e1, e2, delta, mean_0=mean_0, alpha_0=alpha_0,
                                            beta_0=beta_0, pi_div=PI_DIV, r_div=R_DIV))

        fair_sd_ts_test = []
        if TEST_FAIR_SD_TS:
            for bandit in bandits:
                fair_sd_ts_test.append(NormalFairSDTest(N_ITERATIONS, bandit, T, e1, e2[1:], delta[1:], mean_0=mean_0, alpha_0=alpha_0,
                                        beta_0=beta_0, pi_div=PI_DIV))

        for test in ts_test:
            test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF,
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




