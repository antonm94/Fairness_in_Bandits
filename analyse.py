from load_data import load_data
import thompson_sampling.stochastic_dominance as sd_ts
import thompson_sampling.fair_stochastic_dominance as fair_sd_ts
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time

TEST_THOMPSON = False
TEST_SD_TS = False
TEST_FAIR_SD_TS = True
TESTS = [TEST_THOMPSON, TEST_SD_TS, TEST_FAIR_SD_TS]
N_TESTS = sum(TESTS)
TEST = 0
N_ITERATIONS = 1.
DATA_SET = ['Bar Exam', 'Default on Credit'][0]


# def print_fairness(method):
#     if method == 'Thompson Sampling':
#
#     print('\n' + '#' * 20 + '\n')
#     print(method)
#     if smooth_fairness[0] > delta:
#         print(
#         'not ({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: {}'.format(smooth_fairness[0])
#         + '> {}'.format(delta))
#     else:
#         print('({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: '.format(smooth_fairness[0])
#               + '<= {}'.format(delta))
#     print('Cumulative Fairness Regret of Thompson Sampling:\t{}'.format(cumulative_fairness_regret[0]))
#     print('\n' + '#' * 20 + '\n')
#     print('Stochastic Dominance Thompson Sampling with Lambda: {}'.format(lam))
#     if smooth_fairness[1] > delta:
#         print(
#         'not ({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: {}'.format(smooth_fairness[1])
#         + '> {}'.format(delta))
#     else:
#         print('({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: {}'.format(smooth_fairness[1])
#               + '<= {}'.format(delta))
#     print('Cumulative Fairness Regret of Stochastic Dominance Thompson Sampling with Lambda \t{}'.format(lam) +
#           ':\t{}'.format(cumulative_fairness_regret[1]))
#
#
# def test_thompson():
#     global thompson_sampling
#     thompson_sampling = sd_ts.StochasticDominance(bandits, T, e1, e2, delta, 0)
#     thompson_sampling.run()
#     print np.add.accumulate(thompson_sampling.fairness_regret)[T - 1]
#     print np.add.accumulate(thompson_sampling.smooth_fair)[T - 1]
#     print np.add.accumulate(thompson_sampling.not_smooth_fair)[T - 1]
#     thompson_sampling.reset()


def analyse(method):
    global TEST
    for i in range(N_ITERATIONS):
        method.run()
        fairness_regret[TEST] = fairness_regret[TEST] + np.add.accumulate(method.fairness_regret)
        smooth_fair[TEST] = smooth_fair[TEST] + np.add.accumulate(method.smooth_fair)
        not_smooth_fair[TEST] = not_smooth_fair[TEST] + np.add.accumulate(method.not_smooth_fair)
        n = n + method.n
        method.reset()
    regret[TEST] = method.get_regret(np.divide(n,  N_ITERATIONS))
    fairness_regret[TEST] = np.divide(fairness_regret[TEST],  N_ITERATIONS)
    smooth_fair[TEST] = np.divide(smooth_fair[TEST],  N_ITERATIONS)
    not_smooth_fair[TEST] = np.divide(not_smooth_fair[TEST],  N_ITERATIONS)
    TEST = TEST + 1

    
if __name__ == '__main__':

    start_time = time.time()
    bandits = load_data(DATA_SET)

    T = 1000
    e1 = 2
    e2 = 0.1
    delta = 0.001
    lam = 1
    print('Analysing '+DATA_SET+' data set')
    print('Iterations:\t{}'.format(N_ITERATIONS))
    print('T:\t\t{}'.format(T))
    print('e1:\t\t{}'.format(e1))
    print('e2:\t\t{}'.format(e2))
    print('delta:\t{}'.format(delta))
    print('e2:\t\t{}'.format(e2))
    print('Lambda: {}'.format(lam))

    fairness_regret = np.zeros((N_TESTS, T))
    smooth_fair = np.zeros((N_TESTS, T))
    not_smooth_fair = np.zeros((N_TESTS, T))
    regret = np.zeros((N_TESTS, T))

    thompson_sampling = sd_ts.StochasticDominance(bandits, T, e1, e2, delta, 0)
    analyse(thompson_sampling)

    stochastic_dominance = sd_ts.StochasticDominance(bandits, T, e1, e2, delta, lam)
    analyse(stochastic_dominance)

    fair_stochastic_dominance = fair_sd_ts.FairStochasticDominance(bandits, T, e1, e2, delta, lam)
    analyse(fair_sd_ts)

    # if TEST_THOMPSON:
    #     test_thompson('Thompson Sampling')
    #     print_fairness('Thompson Sampling', )
    #
    # if TEST_SD_TS:
    #     test_thompson()
    #     print_fairness('Thompson Sampling', )
    #
    # if TEST_FAIR_SD_TS:






    # not_fair_ratio = np.zeros((3, N_ITERATIONS))
    # cumulative_fairness_regret = np.zeros((3, N_ITERATIONS))
    # for i in range(N_ITERATIONS):
    #     if TEST_THOMPSON:
    #         result = sd_ts.run(T, arm, e1, e2, delta, 0)
    #         not_fair_ratio[0][i] = result[0]
    #         cumulative_fairness_regret[0][i] = np.add.accumulate(result[1])[T - 1]
    #     if TEST_SD_TS:
    #         result = sd_ts.run(T, arm, e1, e2, delta, lam)
    #         not_fair_ratio[1][i] = result[0]
    #         cumulative_fairness_regret[1][i] = np.add.accumulate(result[1])[T - 1]
    #     if TEST_FAIR_SD_TS:
    #         result = fair_sd_ts.run(T, arm, e1, e2, delta, lam)
    #         not_fair_ratio[2][i] = result[0]
    #         cumulative_fairness_regret[2][i] = np.add.accumulate(result[1])[T - 1]
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # smooth_fairness = np.mean(not_fair_ratio, axis=1)
    # cumulative_fairness_regret = np.mean(cumulative_fairness_regret, axis=1)
    #
    # if TEST_THOMPSON:
    #     print_fairness('Thompson Sampling', )
    #
    # if TEST_SD_TS:
    #
    #
    # if TEST_FAIR_SD_TS:
    #     print('\n' + '#' * 20+'\n')
    #     print('Fair Stochastic Dominance Thompson Sampling with Lambda: {}'.format(lam))
    #     if smooth_fairness[2] > delta:
    #         print('not ({}'.format(e1)+',{}'.format(e2)+',{}'.format(delta)+')smooth fair: {}'.format(smooth_fairness[2])
    #               + '> {}'.format(delta))
    #     else:
    #         print('({}'.format(e1)+',{}'.format(e2)+',{}'.format(delta)+')smooth fair: {}'.format(smooth_fairness[2])
    #               + '<= {}'.format(delta))
    #     print('Cumulative Fairness Regret of Fair Stochastic Dominance Thompson Sampling with Lambda \t{}'.format(lam) +
    #           ':\t{}'.format(cumulative_fairness_regret[2]))
    #
    #


