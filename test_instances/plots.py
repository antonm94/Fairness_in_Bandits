import matplotlib.pyplot as plt
import thompson_sampling.bern_fair_stochastic_dominance_ts as fair_sd_ts
import logging
import warnings
import numpy as np
from matplotlib import rc



warnings.simplefilter(action='ignore', category=FutureWarning)
LOGGING = 1
if LOGGING > 0:
    logger = logging.getLogger(__name__)

# def analyse(method):
#     global TEST
#     n = 0
#     for i in range(int(N_ITERATIONS)):
#         method.run()
#         fairness_regret[TEST] = fairness_regret[TEST] + np.add.accumulate(method.fairness_regret)
#         smooth_fair[TEST] = smooth_fair[TEST] + np.add.accumulate(method.smooth_fair)
#         not_smooth_fair[TEST] = not_smooth_fair[TEST] + np.add.accumulate(method.not_smooth_fair)
#         n = n + method.n
#         method.reset()
#     regret[TEST] = method.get_regret(np.divide(n,  N_ITERATIONS))
#     fairness_regret[TEST] = np.divide(fairness_regret[TEST],  N_ITERATIONS)
#     smooth_fair[TEST] = np.divide(smooth_fair[TEST],  N_ITERATIONS)
#     not_smooth_fair[TEST] = np.divide(not_smooth_fair[TEST],  N_ITERATIONS)
#     TEST = TEST + 1

# def print_results(method, ):
#     for i in range(N_TESTS):
#         method = TESTS[TESTS_INDEX[i]]
#         if (method == 'Thompson Sampling') | (lam == 1):
#             print('\n' + '#' * 20 + '\n')
#             print(method)
#         s = smooth_fair[i][T-1]
#         n = not_smooth_fair[i][T-1]
#         print ('Smooth Fair:\t{}'.format(s))
#         print ('Not Smooth Fair:\t{}'.format(n))
#         print ('=> Smooth Fair with Prob:\t{}'.format(np.divide(s, s+n)))
#         print ('Needed Probability: 1-delta\t= {}'.format(1-delta))
#         print ('Cumulative Fairness Regret\t{}'.format(fairness_regret[i][T-1]))
#         print ('Regret\t{}'.format(regret[i][T-1]))


# def get_label(test, i, j):
#     if test.name == 'Thompson Sampling':
#         return test.name
#     else:
#         return test.name + ' e1= {}'.format(test.e1_arr) + ' e2= {}'.format(test.e2_arr) + ' delta= {}'.format(test.delta_arr)


# class Test:
#
#     def __init__(self, bandits, methods, n_iterations, T):
#         self.T = T
#         self.test_cases = []
#         self.n_iterations = n_iterations
#         self.bandtis = bandits
#         self.k = bandits.k
#         self.tested_thompson = False

    # def add_test_cases(self, delta_arr, distance, e1_arr, e2_arr, lam_arr, methods, mod, n_iterations):
    #     for method in methods:
    #         for lam in lam_arr:
    #             if method == 'Thompson Sampling':
    #                 if not self.tested_thompson:
    #                     self.test_cases.append(sd_ts.StochasticDominance(self.bandits, self.T, e1_arr, e2_arr,
    #                                                                      delta_arr, 0, distance))
    #                     self.tested_thompson = True
    #             elif method == 'Stochastic Dominance Thompson Sampling':
    #                 self.test_cases.append(sd_ts.StochasticDominance(self.bandits, self.T, e1_arr, e2_arr,
    #                                                                  delta_arr, lam, distance, mod))
    #             elif method == 'Fair Stochastic Dominance Thompson Sampling':
    #                 self.test_cases.append(fair_sd_ts.FairStochasticDominance(self.bandits, self.T, e1_arr, e2_arr,
    #                                                                           delta_arr, lam, distance, mod))
    #             else:
    #                 print 'unknown method'
    #     for i in range(len(self.test_cases)):
    #         self.test_cases[i].analyse(n_iterations)
    #         if LOGGING > 0:
    #             logger.info('test {}'.format(i), ' out of {}'.format({len(self.test_cases)}, ' is done.'))


def print_result(test_cases):
    for test in test_cases:

        print('\n' + '#' * 20)

        print('Iterations:\t{}'.format(test.n_iter))
        print('T:\t\t{}'.format(test.T))
        print('#' * 20)
        for e1 in range(len(test.e1_arr)):
            for e2 in range(len(test.e1_arr)):
                for d in range(len(test.delta_arr)):
                    delta = test.delta_arr[d]
                    print('e1:\t\t{}'.format(test.e1_arr[e1]))
                    print('e2:\t\t{}'.format(test.e2_arr[e2]))
                    print('delta:\t{}'.format(delta))
                    if (test.lam > 0) and (test.lam < 1):
                        print('Lambda: {}'.format(test.lam))
                    if test.name == 'Fair SD TS':
                        print('#' * 20 + '\n')
                        print test.get_name(e2=test.e2_arr[e2], delta=delta)
                        print('#' * 20 + '\n')
                        print ('Smooth Fair:\t{}'.format(test.average_smooth_fair[e1][e2][d][-1]))
                        print ('Not Smooth Fair:\t{}'.format(test.average_not_smooth_fair[e1][e2][d][-1]))
                        # print ('=> Smooth Fair with Prob:\t{}'.format(test.average_fair_ratio[e1][e2][d][-1]))
                        print ('Needed Probability: 1-delta\t= {}'.format(1 - delta))
                        print ('Cumulative Fairness Regret\t{}'.format(np.add.accumulate(test.average_fairness_regret[e2][d])[-1]))
                        print ('Regret\t{}'.format(test.average_regret[e2][d][-1]))
                        print ('Average Rounds exploring: {}'.format(test.average_rounds_exploring[e2][d]))
                        print ('Average Rounds exploiting: {}'.format(test.average_rounds_exploiting[e2][d]))

                    else:
                        print('#' * 20 + '\n')
                        print test.get_name()
                        print('#' * 20 + '\n')
                        print ('Smooth Fair:\t{}'.format(test.average_smooth_fair[e1][e2][-1]))
                        print ('Not Smooth Fair:\t{}'.format(test.average_not_smooth_fair[e1][e2][-1]))
                        # print ('=> Smooth Fair with Prob:\t{}'.format(test.average_fair_ratio[e1][e2][-1]))
                        print ('Needed Probability: 1-delta\t= {}'.format(1 - delta))
                        print ('Cumulative Fairness Regret\t{}'.format(np.add.accumulate(test.average_fairness_regret)[-1]))
                        print ('Regret\t{}'.format(test.average_regret[-1]))

                    print('#' * 20)


def plot_smooth_fairness(test_cases):
    x = range(test_cases[0].T)
    for test in test_cases:
        if test.name == 'Fair SD TS':
            for i in range(len(test.average_not_smooth_fair)):
                for j in range(len(test.average_not_smooth_fair[i])):
                    for d in range(len(test.average_not_smooth_fair[i][j])):
                        plt.plot(x, np.add.accumulate(test.average_not_smooth_fair[i][j][d]), label=test.get_name(test.e1_arr[i],
                                                                                               test.e2_arr[j], test.delta_arr[d]))
        else:
            for i in range(len(test.average_not_smooth_fair)):
                for j in range(len(test.average_not_smooth_fair[i])):
                 plt.plot(x, np.add.accumulate(test.average_not_smooth_fair[i][j]), label=test.get_name( test.e1_arr[i], test.e2_arr[j]))


    for delta in test.delta_arr:
        plt.plot(x, [delta * t for t in x], label='allowed number of unfair with delta= {}'.format(delta))
    plt.xlabel('T')
    plt.ylabel('number of unfair w.r.t total variation distance')
    plt.legend()
    #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def plot_delta_subjective_fair(test_cases, start_index=0):
    x = range(test_cases[0].T)
    for test in test_cases:
        # if test.name == 'Thompson Sampling':
        for e1_ind in range(len(test.e1_arr)):
            for e2_ind in range(len(test.e2_arr)):
                if test.name == 'Fair SD TS':
                    for d in range(len(test.delta_arr)):
                        plt.plot(x[start_index:],
                                 np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind, d], axis=1), axis=1)
                                 [start_index:],
                                 label=test.get_name(test.e1_arr[e1_ind], test.e2_arr[e2_ind], test.delta_arr[d]))
                else:
                        plt.plot(x[start_index:],
                                 np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1),  axis=1)
                                 [start_index:],
                                     label=test.get_name(test.e1_arr[e1_ind], test.e2_arr[e2_ind]))


                    # xmin = np.amin(np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1))
                    # ymin = np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1)
                    # plt.plot(xmin, ymin)
                    # plt.annotate((1 - d), xy=(2, 1), xytext=(3, 1.5),
                    #              arrowprops=dict(facecolor='black', shrink=0.05),
                    #              )

    plt.xlabel(r'T \sigma')
    plt.ylabel('Subjective Smooth fair with probability')

    plt.legend()
    # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def plot_delta_smooth_fair(test_cases, start_index=0):
    T = test_cases[0].T
    x = range(T)
    for test in test_cases:
        # if test.name == 'Thompson Sampling':
        for e1_ind in range(len(test.e1_arr)):
            for e2_ind in range(len(test.e2_arr)):
                if test.name == 'Fair SD TS':
                    for d in range(len(test.delta_arr)):
                        y = np.min(np.min(test.frac_smooth_fair[e1_ind, e2_ind, d], axis=2), axis=1)
                        plt.plot(x[start_index:],y[start_index:],
                                 label=test.get_name(test.e1_arr[e1_ind], test.e2_arr[e2_ind], test.delta_arr[d]))
                        if test.average_rounds_exploring[e2_ind, d] < T:
                            plt.plot(test.average_rounds_exploring[e2_ind, d]-1, y[int(test.average_rounds_exploring[e2_ind, d]) -1], 'g*')

                else:
                    plt.plot(x[start_index:],
                                 np.min(np.min(test.frac_smooth_fair[e1_ind, e2_ind], axis=1),  axis=1)
                                 [start_index:],
                                     label=test.get_name(test.e1_arr[e1_ind], test.e2_arr[e2_ind]))


                # xmin = np.amin(np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1))
                # ymin = np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1)
                # plt.plot(xmin, ymin)
                # plt.annotate((1 - d), xy=(2, 1), xytext=(3, 1.5),
                #              arrowprops=dict(facecolor='black', shrink=0.05),
                #              )

    plt.xlabel(r'T \sigma')
    plt.ylabel('Smooth fair with probability')
    plt_name = 'delta_smooth_fair_{}'.format(T)+'.png'
    plt.savefig('delta_smooth_fair.png', bbox_inches='tight')

    plt.legend()
    # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def plot_fairness_regret(test_cases):
    x = range(test_cases[0].T)
    for test in test_cases:
        if test.name == 'Fair SD TS':
            for e2_ind in range(len(test.e2_arr)):
                for delta_ind in range(len(test.delta_arr)):
                    e2 = test.e2_arr[e2_ind]
                    delta = test.delta_arr[delta_ind]
                    plt.plot(x, np.add.accumulate(test.average_fairness_regret[e2_ind][delta_ind]), label=test.get_name(e2=e2, delta=delta))
        else:
            plt.plot(x, np.add.accumulate(test.average_fairness_regret), label=test.get_name())

    plt.plot(x, [pow(test_cases[0].k * t, 2. / 3) for t in x], label='regret bound O((k*T)^2/3)')
    plt.xlabel('T')
    plt.ylabel('Cumulative Fairness Regret')
    plt.legend()
    #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def plot_average_total_regret(test_cases):
    x = range(test_cases[0].T)
    for test in test_cases:
        if test.name == 'Fair SD TS':
            for e2_ind in range(len(test.e2_arr)):
                for delta_ind in range(len(test.delta_arr)):
                    e2 = test.e2_arr[e2_ind]
                    delta = test.delta_arr[delta_ind]
                    plt.plot(x, test.average_regret[e2_ind][delta_ind], label=test.get_name(e2=e2, delta=delta))
        else:
            plt.plot(x, test.average_regret, label=test.get_name())
    plt.xlabel('T')
    plt.ylabel('Average Total Regret')
    plt.legend()
    #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_lambda_regret_tradeoff(self, lam):
    x = range(test_cases[0].T)
    n_lam = len(lam)
    total_regret = np.zeros(n_lam)
    fairness_regret = np.zeros(n_lam)
    for i in range(n_lam):
        if type(self.test_cases[i]) is not fair_sd_ts.FairStochasticDominance:
            total_regret[i] = self.test_cases[i].regret[-1]
            fairness_regret[i] = self.test_cases[i].average_fairness_regret[-1]
        else:
            total_regret[i] = self.test_cases[i].regret[0][0][-1]
            fairness_regret[i] = self.test_cases[i].average_fairness_regret[0][0][-1]
    plt.plot(lam, total_regret, label='modification regret')
    plt.plot(lam, fairness_regret, label='fairness regret')
    plt.xlabel('Lambda')
    plt.ylabel('regret')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_regret_tradeoff(self, lam):
    x = range(self.T)
    n_lam = len(lam)
    total_regret = np.zeros(n_lam)
    fairness_regret = np.zeros(n_lam)

    if type(self.test_cases[0]) is not fair_sd_ts.FairStochasticDominance:
        for i in range(n_lam):
            test = self.test_cases[i]
            total_regret[i] = test.regret[-1]
            fairness_regret[i] = test.average_fairness_regret[-1]
        plt.plot(fairness_regret, total_regret, label=test.name)

    else:
        for i in range(n_lam):
            test = self.test_cases[i]
            total_regret[i] = test.regret[0][0][-1]
            fairness_regret[i] = test.average_fairness_regret[0][0][-1]
        plt.plot(fairness_regret, total_regret, label=test.name
                                                      + ' e2= {}'.format(test.e2_arr[0])
                                                      + ' delta= {}'.format(test.delta_arr[0]))

    plt.xlabel('Lambda')
    plt.ylabel('regret')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

        # if method == 'Thompson Sampling':
#     thompson_sampling = sd_ts.StochasticDominance(bandits, T, e1, e2, delta, 0)
#     thompson_sampling.analyse(N_ITERATIONS)
#     print_result(thompson_sampling)
# elif method == 'Stochastic Dominance':
#
#
# elif method == 'Fair Stochastic Dominance':
#
# else
#     print 'Unknown Method'
# if TEST_SD_TS:
#     stochastic_dominance = sd_ts.StochasticDominance(bandits, T, e1, e2, delta, 1)
#     stochastic_dominance.analyse(N_ITERATIONS)
#     print_result(stochastic_dominance)
#
# if TEST_SD_TS:
#     stochastic_dominance = sd_ts.StochasticDominance(bandits, T, e1, e2, delta, 0.5)
#     stochastic_dominance.analyse(N_ITERATIONS)
#     print_result(stochastic_dominance)
#
# if TEST_FAIR_SD_TS:
#     fair_stochastic_dominance = fair_sd_ts.FairStochasticDominance(bandits, T, e1, e2, delta, 1)
#     fair_stochastic_dominance.analyse(N_ITERATIONS)
#     print_result(fair_stochastic_dominance)


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

# if smooth_fairness[j] > delta:
#                print(
#                'not ({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: {}'.format(smooth_fairness[0])
#                + '> {}'.format(delta))
#            else:
#                print('({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: '.format(smooth_fairness[0])
#                      + '<= {}'.format(delta))
#            print('Cumulative Fairness Regret of Thompson Sampling:\t{}'.format(cumulative_fairness_regret[0]))
#            print('\n' + '#' * 20 + '\n')
#            print('Stochastic Dominance Thompson Sampling with Lambda: {}'.format(lam))
#            if smooth_fairness[1] > delta:
#                print(
#                'not ({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: {}'.format(smooth_fairness[1])
#                + '> {}'.format(delta))
#            else:
#                print('({}'.format(e1) + ',{}'.format(e2) + ',{}'.format(delta) + ')smooth fair: {}'.format(smooth_fairness[1])
#                      + '<= {}'.format(delta))
#            print('Cumulative Fairness Regret of Stochastic Dominance Thompson Sampling with Lambda \t{}'.format(lam) +
#                  ':\t{}'.format(cumulative_fairness_regret[1]))

