
TEST_THOMPSON = 1
TEST_SD_TS = 0
TEST_FAIR_SD_TS = 0
import random
from load_data import load_data
from distance import total_variation_distance
from test_instances.ts_test import TSTest
from test_instances.fair_sd_ts_test import FairSDTest
from test_instances.sd_ts_test import SDTest
import numpy as np
import test_instances.plots as plot
import test_instances.final_plot as f_plot
np.set_printoptions(threshold=np.nan)

DATA_SET = ['Bar Exam', 'Default on Credit'][0]
METHODS = TEST_THOMPSON*['Thompson Sampling'] + TEST_SD_TS*['Stochastic Dominance Thompson Sampling'] + \
          TEST_FAIR_SD_TS*['Fair Stochastic Dominance Thompson Sampling']

if __name__ == '__main__':
    bandits = load_data(DATA_SET)


   #  T = 1000
   #  e1 = [0.0000001]
   #  e2 = [0.5]
   #  delta = [0.6]
   #  lam = [1]
   #
   #  # test0 = TSTest(N_ITERATIONS, bandits, T, e1, e2, delta, distance=total_variation_distance)
   # # test0.analyse()
   #  test1 = SDTest(N_ITERATIONS, bandits, T, e1, e2, delta, lam[0], distance=total_variation_distance)
   #  test1.analyse()
  #   test2 = FairSDTest(N_ITERATIONS, bandits, T, e1, e2, delta, lam[0], distance=total_variation_distance)
  #   test2.analyse(e2_times=1)
  #   test_cases = [test0, test1, test2]
  #   # print test0.smooth_fair
  #   # print test1.smooth_fair
  #   test0.frac_smooth_fair()
  #   test1.frac_smooth_fair()
  #   test2.frac_smooth_fair()
  #   print test0.is_smooth_fair
  #   print test1.is_smooth_fair
  #   print test2.is_smooth_fair
  #   print test2.average_rounds_exploring
    #print test2.smooth_fair

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
    #     'Stochastic Dominance Thompson S.ampling'] + TEST_FAIR_SD_TS * ['Fair Stochastic Dominance Thompson Sampling']
    # print METHODS
    N_ITERATIONS = 1.
    DATA_SET = ['Bar Exam', 'Default on Credit'][0]
    bandits = load_data(DATA_SET)
    T = 20

    e1 = [2.]
    e2 = [0.]
    delta = [0.]
    # random.seed(0)
    # np.random.seed(0)
    sd_test0 = SDTest(N_ITERATIONS, bandits, T, e1, e2, delta, lam=1, distance=total_variation_distance)
    sd_test0.analyse(fair_regret=True, regret=True, subjective_smooth_fair=True, smooth_fair=True, subjective_minimum_e1=True,
                     minimum_e1=True)
    # fair_sd_test = FairSDTest(N_ITERATIONS, bandits, T, e1, e2, delta, lam=1, distance=total_variation_distance)
    # fair_sd_test.analyse()
    # fair_sd_test.average_rounds_exploring
    # DATA_SET = ['Bar Exam', 'Default on Credit'][1]
    # bandits = load_data(DATA_SET)
    # sd_test1 = SDTest(N_ITERATIONS, bandits, T, e1, e2, delta, lam=1, distance=total_variation_distance)
    # sd_test1.analyse(fair_regret=True, regret=True, subjective_smooth_fair=True, smooth_fair=True)
    # test_cases = [sd_test0, sd_test1]
    # test_cases = [sd_test0]
    # test_cases = [fair_sd_test, sd_test0]
    # plot.plot_delta_subjective_fair(test_cases)
    # plot.plot_delta_smooth_fair(test_cases)
    # f_plot.plot_min_e1(test_cases)
    # f_plot.plot_subjective_min_e1(test_cases)