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
METHODS = TEST_THOMPSON*['Thompson Sampling']+TEST_SD_TS*['Stochastic Dominance Thompson Sampling']+TEST_FAIR_SD_TS*['Fair Stochastic Dominance Thompson Sampling']
print METHODS