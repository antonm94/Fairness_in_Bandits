import random
from load_data import load_data
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.ts_test import  TSTest
from test_instances.sd_ts_test import SDTest
import numpy as np
from multiprocessing import Process, Value, Array



SEED = True

TEST_FAIR_SD_TS = True

N_ITERATIONS = 100.
T = 10000
SETS = ['0', 'Bar Exam', 'Default on Credit', '3']

e1 = [2, 1, 0.5, 0.3]
e2 = [0.01, 0.05, 0.1]
delta = [0.1, 0.3, 0.7]
if SEED:
    random.seed(0)
    np.random.seed(0)

def run_fair(bandit):
    FairSDTest(N_ITERATIONS, bandit, T, e1, e2, delta, lam=1).analyse(regret=False, fair_regret=False, smooth_fair=True,
                                                                      minimum_e1=True, e2_times=2.)


if __name__ == '__main__':

    bandits = []
    for set in SETS:
        bandits.append(load_data(set))



    ps = []
    for bandit in bandits:
        ps.append(Process(target=run_fair(), args=(bandit)))

    for p in ps:
        p.start()


    for p in ps:
        p.join()




