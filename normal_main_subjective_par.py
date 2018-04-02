import random
from test_instances.normal_fair_sd_ts_test import NormalFairSDTest
from test_instances.normal_ts_test import  NormalTSTest
from test_instances.normal_sd_ts_test import NormalSDTest
from thompson_sampling.normal_IG_ts import NormalThompsonSampling
from thompson_sampling.normal_IG_stochastic_dominance_ts import NormalStochasticDominance
import numpy as np
from normalBandits import NormalBandits
from multiprocessing import Process
import divergence as div
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)


SEED = 1
TEST_THOMPSON = 1
TEST_SD_TS = 1
TEST_FAIR_SD_TS = 0


SF = 0
SSF = 1
MIN_E1 = 0
SMIN_E1 = 1
REGRET = 0
FAIR_REGRET = 0

N_ITERATIONS = 50.
T = 500

R_DIV = div.cont_total_variation_distance
PI_DIV = div.total_variation_distance


mean_0 = 0
alpha_0 = 0.5
beta_0 = 0.
init_phase = True

def infos_print(tests):
    if TEST_FAIR_SD_TS:
        print [test.c for test in fair_sd_ts_test]
        print [test.average_round_exploring for test in fair_sd_ts_test]


def n_info(test_cases):
    for test in test_cases:
        print test.name + 'n' + str(test.average_n[-1])
        print test.name + 'pi prediction' + str(np.average(test.curr_test.pi, axis=0) * T)


def simple_tests(bandits, T):
    ts = NormalThompsonSampling(bandits, T, mean_0=mean_0, alpha_0=alpha_0, beta_0=beta_0, init_phase=init_phase)
    sd_ts = NormalStochasticDominance(bandits, T, mean_0=mean_0, alpha_0=alpha_0, beta_0=beta_0, init_phase=init_phase)
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


def ts_analyse():
    ts_test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF,
                 subjective_smooth_fair=SSF, minimum_e1=MIN_E1,
                 subjective_minimum_e1=SMIN_E1)
    # print procnum
    # send_end.send(result)


def sd_ts_analyse():
    sd_test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF,
                 subjective_smooth_fair=SSF, minimum_e1=MIN_E1,
                 subjective_minimum_e1=SMIN_E1)
    # print procnum
    # send_end.send(result)


def fair_sd_ts_analyse():
    fair_test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF, minimum_e1=MIN_E1)


if __name__ == '__main__':

    e1 = [1]
    e2 = [0, 0.05, 0.15, 0.2, 0.3, 0.4]
    delta = [0, 0.01, 0.05, 0.1, 0.2, 0.4]

    if SEED:
        random.seed(0)
        np.random.seed(0)

    bandits = [NormalBandits([0, 0, 0, 0], [1, 0.5, 0.3, 0.2], data_set_name='D0', divergence_fun=R_DIV),
               NormalBandits([1., 0.95, 0.3, -0.1], [1., 1., 1., 1.], data_set_name='D1', divergence_fun=R_DIV),
               NormalBandits([0.5, -1, 1, 0.5], [6, 2, 0.1, 1], data_set_name='D2', divergence_fun=R_DIV),
               NormalBandits([0, 0], [5, 1], data_set_name='D3', divergence_fun=R_DIV)]
    #   --> implied by previous  NormalBandits([1, 0], [1, 10], data_set_name='D3', di

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
            fair_sd_ts_test.append(
                NormalFairSDTest(N_ITERATIONS, bandit, T, e1, e2[1:], delta[1:], mean_0=mean_0, alpha_0=alpha_0,
                                 beta_0=beta_0, pi_div=PI_DIV))


    jobs = []
    for ts_test in ts_test:
        p = Process(target=ts_analyse)
        jobs.append(p)
        p.start()

    for sd_test in sd_ts_test:
        p = Process(target=sd_ts_analyse)
        jobs.append(p)
        p.start()

    for fair_test in fair_sd_ts_test:
        p = Process(target=fair_sd_ts_analyse)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()




