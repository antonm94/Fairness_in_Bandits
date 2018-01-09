from load_data import load_data
from distance import total_variation_distance
from test_instances.ts_test import TSTest
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.sd_ts_test import SDTest
import numpy as np
import test_instances.test as test

TEST_THOMPSON = 1
TEST_SD_TS = 0
TEST_FAIR_SD_TS = 0


TEST = 0
N_ITERATIONS = 10.
DATA_SET = ['Bar Exam', 'Default on Credit'][0]
METHODS = TEST_THOMPSON*['Thompson Sampling'] + TEST_SD_TS*['Stochastic Dominance Thompson Sampling'] + \
          TEST_FAIR_SD_TS*['Fair Stochastic Dominance Thompson Sampling']

if __name__ == '__main__':
    bandits = load_data(DATA_SET)

    T = 1000
    e1 = [2, 1]
    e2 = [0.01, 0.2]
    delta = [0.1]
    lam = [1]

    test0 = TSTest(N_ITERATIONS, bandits, T, e1, e2, delta, distance=total_variation_distance)
    test0.analyse()
    test1 = SDTest(N_ITERATIONS, bandits, T, e1, e2, delta, lam[0], distance=total_variation_distance)
    test1.analyse()
    test2 = FairSDTest(N_ITERATIONS, bandits, T, e1, e2, delta, lam[0], distance=total_variation_distance)
    test2.analyse()
    test_cases = [test0, test1, test2]
    test.print_result(test_cases)
    test.plot_fairness_regret(test_cases)
    test.plot_average_total_regret(test_cases)


    # test1 = TSTest(bandits, METHODS, N_ITERATIONS, T, e1, e2, delta, lam)
    # test1.add_test_case(bandits, 'Fair Stochastic Dominance Thompson Sampling', e1, e2, delta, mod=1)
    # test1.print_result()
    # test1.plot_fairness_regret()
    # test1.plot_average_total_regret()


    # print test.average_smooth_fair[:][:][-1]
    # print test.average_not_smooth_fair[:][:][-1]
    # print test.average_n[T-1]


    # print test.average_fairness_regret[0][0][-1]
    # print test.average_not_smooth_fair[0][0][0][-1]
    # print test.average_n[0][0][ -1]
    # plot1 = plt.plot([1,2,3],[3,5,7])
    # plot2 = plt.plot([1,2,3],[3,5,10])
    #
    #
    # plot1.ylabel('Average Regret')
    # plot2.xlabel('T')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()


    # TEST_THOMPSON = 0
    # TEST_SD_TS = 0
    # TEST_FAIR_SD_TS = 0
    #
    # TEST = 0
    # N_ITERATIONS = 10.
    # DATA_SET = ['Bar Exam', 'Default on Credit'][0]
    # METHODS = TEST_THOMPSON * ['Thompson Sampling'] + TEST_SD_TS * [
    #     'Stochastic Dominance Thompson Sampling'] + TEST_FAIR_SD_TS * ['Fair Stochastic Dominance Thompson Sampling']
    # print METHODS