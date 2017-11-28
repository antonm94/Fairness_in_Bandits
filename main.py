from test import Test
from load_data import load_data

TEST_THOMPSON = 1
TEST_SD_TS = 1
TEST_FAIR_SD_TS = 1


TEST = 0
N_ITERATIONS = 1.
DATA_SET = ['Bar Exam', 'Default on Credit'][0]
METHODS = TEST_THOMPSON*['Thompson Sampling'] + TEST_SD_TS*['Stochastic Dominance Thompson Sampling'] + \
          TEST_FAIR_SD_TS*['Fair Stochastic Dominance Thompson Sampling']

if __name__ == '__main__':
    bandits = load_data(DATA_SET)

    T = 1000
    e1 = [2]
    e2 = [0.01]
    delta = [0.1]
    lam = [1]
    test1 = Test(bandits, METHODS, N_ITERATIONS, T, e1, e2, delta, lam)
    test1.print_result()
    test1.plot_smooth_fairness()
    test1.plot_fairness_regret()
    test1.plot_average_total_regret()

    import matplotlib.pyplot as plt

    # plot1 = plt.plot([1,2,3],[3,5,7])
    # plot2 = plt.plot([1,2,3],[3,5,10])
    #
    #
    # plot1.ylabel('Average Regret')
    # plot2.xlabel('T')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()


    TEST_THOMPSON = 0
    TEST_SD_TS = 0
    TEST_FAIR_SD_TS = 0

    TEST = 0
    N_ITERATIONS = 10.
    DATA_SET = ['Bar Exam', 'Default on Credit'][0]
    METHODS = TEST_THOMPSON * ['Thompson Sampling'] + TEST_SD_TS * [
        'Stochastic Dominance Thompson Sampling'] + TEST_FAIR_SD_TS * ['Fair Stochastic Dominance Thompson Sampling']
    print METHODS