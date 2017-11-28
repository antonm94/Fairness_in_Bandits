import matplotlib.pyplot as plt
import thompson_sampling.stochastic_dominance as sd_ts
import thompson_sampling.fair_stochastic_dominance as fair_sd_ts
import warnings
import numpy as np
from distance import total_variation_distance
warnings.simplefilter(action='ignore', category=FutureWarning)




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


def get_label(test):
    fair = ['Fair Stochastic Dominance Thompson Sampling',
            'Thompson Sampling - Fair Stochastic Dominance Thompson Sampling trade-off' \
            ' with Lambda = {}'.format(test.lam)]
    if test.name not in fair:
        return test.name
    else:
        return test.name + ' e1= {}'.format(test.e1) + ' e2= {}'.format(test.e2) + ' delta= {}'.format(test.delta)


class Test:

    def __init__(self, bandits, methods, n_iterations, T, e1_arr, e2_arr, delta_arr, lam_arr,
                 distance=total_variation_distance):
        self.T = T
        self.test_cases = []
        self.n_iterations = n_iterations
        self.k = bandits.k
        self.tested_thompson = False
        for method in methods:
            for e1 in e1_arr:
                for e2 in e2_arr:
                    for delta in delta_arr:
                        for lam in lam_arr:
                            if method == 'Thompson Sampling':
                                if not self.tested_thompson:
                                    self.test_cases.append(sd_ts.StochasticDominance(bandits, T, e1, e2,
                                                                                     delta, 0, distance))
                                    self.tested_thompson = True
                            elif method == 'Stochastic Dominance Thompson Sampling':
                                self.test_cases.append(sd_ts.StochasticDominance(bandits, T, e1, e2,
                                                                                 delta, lam, distance))
                            elif method == 'Fair Stochastic Dominance Thompson Sampling':
                                self.test_cases.append(fair_sd_ts.FairStochasticDominance(bandits, T, e1, e2,
                                                                                          delta, lam, distance))
                            else:
                                print 'Unknown Method'
        for test in self.test_cases:
            test.analyse(n_iterations)

    def add_test_case(self, bandits, method, e1, e2, delta, lam=1, distance=total_variation_distance):
        if method == 'Thompson Sampling':
            if not self.tested_thompson:
                self.test_cases.append(sd_ts.StochasticDominance(bandits, self.T, e1, e2,
                                                                 delta, 0, distance))
                self.tested_thompson = True
        elif method == 'Stochastic Dominance Thompson Sampling':
            self.test_cases.append(sd_ts.StochasticDominance(bandits, self.T, e1, e2,
                                                             delta, lam, distance))
        elif method == 'Fair Stochastic Dominance Thompson Sampling':
            self.test_cases.append(fair_sd_ts.FairStochasticDominance(bandits, self.T, e1, e2,
                                                                      delta, lam, distance))
        else:
            print 'Unknown Method'

    def print_result(self):
        for test in self.test_cases:
            print('\n' + '#' * 20)
            print test.name
            print('#' * 20 + '\n')
            print('Iterations:\t{}'.format(self.n_iterations))
            print('T:\t\t{}'.format(test.T))
            print('e1:\t\t{}'.format(test.e1))
            print('e2:\t\t{}'.format(test.e2))
            print('delta:\t{}'.format(test.delta))
            print('Lambda: {}'.format(test.lam))
            print ('Smooth Fair:\t{}'.format(test.average_smooth_fair[-1]))
            print ('Not Smooth Fair:\t{}'.format(test.average_not_smooth_fair[-1]))
            print ('=> Smooth Fair with Prob:\t{}'.format(test.get_fair_ratio()[-1]))
            print ('Needed Probability: 1-delta\t= {}'.format(1 - test.delta))
            print ('Cumulative Fairness Regret\t{}'.format(test.average_fairness_regret[-1]))
            print ('Regret\t{}'.format(test.regret[-1]))

    def plot_smooth_fairness(self):
        x = range(self.T)
        delta = set([])
        for test in self.test_cases:
            plt.plot(x, test.average_not_smooth_fair, label=get_label(test))
            if test.delta not in delta:
                plt.plot(x, [test.delta * t for t in x], label='allowed number of unfair with delta= {}'.format(test.delta))
                delta.add(test.delta)
        plt.xlabel('T')
        plt.ylabel('number of unfair w.r.t total variation distance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot_fairness_regret(self):
        x = range(self.T)
        for test in self.test_cases:
            plt.plot(x, test.average_fairness_regret, label=get_label(test))

        plt.plot(x, [pow(self.k * t, 2. / 3) for t in x], label='regret bound O((k*T)^2/3)')
        plt.xlabel('T')
        plt.ylabel('cumulative fairness regret')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot_average_total_regret(self):
        x = range(self.T)
        for test in self.test_cases:
            plt.plot(x, test.regret, label=get_label(test))
        plt.xlabel('T')
        plt.ylabel('average total regret')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    def plot_regret_tradeoff(self, lam):
        n_lam = len(lam)
        modification_regret = np.zeros(n_lam)
        fairness_regret = np.zeros(n_lam)
        for i in range(n_lam):
            modification_regret[i] = abs(self.test_cases[i].regret[-1] - self.test_cases[0].regret[-1])
            fairness_regret[i] = (self.test_cases[i].average_fairness_regret[-1])

        plt.plot(lam, modification_regret, label='modification regret')
        plt.plot(lam, fairness_regret, label='fairness regret')
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

