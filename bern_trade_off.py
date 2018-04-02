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

import random
from load_data import load_data
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.ts_test import  TSTest
from test_instances.sd_ts_test import SDTest
import numpy as np
from divergence import total_variation_distance
SEED = 1
TEST_THOMPSON = 0
TEST_SD_TS = 1
TEST_FAIR_SD_TS = 1


SF = 0
SSF = 0
MIN_E1 = 0
SMIN_E1 = 0
REGRET = 1
FAIR_REGRET = 1

N_ITERATIONS = 50.
T = 5000

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




def sd_ts_analyse():
    sd_test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF,
                 subjective_smooth_fair=SSF, minimum_e1=MIN_E1,
                 subjective_minimum_e1=SMIN_E1)
    # print procnum
    # send_end.send(result)


def fair_sd_ts_analyse():
    fair_test.analyse(regret=REGRET, fair_regret=FAIR_REGRET, smooth_fair=SF, minimum_e1=MIN_E1)


if __name__ == '__main__':


    e1 = []
    e2 = [0.1, 0.2]
    delta = [0.01, 0.2]
    n = 11
    lam_arr = np.linspace(0, 1, n)




    if SEED:
        random.seed(1)
        np.random.seed(1)

    bandits = [load_data('Bar Exam')]


    sd_ts_test = []
    if TEST_SD_TS:
        for lam in lam_arr:
            for bandit in bandits:
                sd_ts_test.append(SDTest(N_ITERATIONS, bandit, T, e1, e2, delta, lam=lam, distance=total_variation_distance))

    fair_sd_ts_test = []
    if TEST_FAIR_SD_TS:
        for lam in lam_arr:
            for bandit in bandits:
                for e2_v in e2:
                    for delta_v in delta:
                        # print e2_v
                        # print delta_v
                        fair_sd_ts_test.append(
                            FairSDTest(N_ITERATIONS, bandit, T, e1, [e2_v], [delta_v], lam=lam))


    jobs = []

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




