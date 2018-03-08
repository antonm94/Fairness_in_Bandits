# file_name = self.bandits.data_set_name + '/' + self.name + '/N_ITER_{}'.format(
        #     int(self.n_iter)) + '_T_{}'.format(self.T)
        # cwd = os.getcwd()
        # last_dir = cwd.split('/')[-1]
        # if last_dir == 'notebooks':
        #     os.chdir(cwd.replace('/notebooks', ''))
        #
        #
        # if os.path.exists(file_name):
        #     self.analyse_from_file(regret, fair_regret, smooth_fair, subjective_smooth_fair)
        #     print 'restored data from file'
        #     return

        # pi = np.zeros((int(self.n_iter), self.T, self.k))
        # r_h = np.zeros((int(self.n_iter), self.T, self.k))
        # n = np.zeros((int(self.n_iter), self.T, self.k))
# pi[it] = self.curr_test.pi
# r_h[it] = self.curr_test.r_h
# n[it] = self.curr_test.n


# if not os.path.exists(file_name):
#     os.makedirs(file_name)
# np.savez(file_name, pi=pi, r_h=r_h, r_theta=self.bandits.theta, n=n)
# print self.frac_smooth_fair[-1, -1]
# print np.min(self.frac_smooth_fair[-1, -1], axis=1)

# def analyse_from_file(self, regret=True, fair_regret=True, smooth_fair = True, subjective_smooth_fair = False):
#     file_name = self.bandits.data_set_name + '/' + self.name + '/N_ITER_{}'.format(int(self.n_iter)) + '_T_{}'.format(self.T)
#     if not os.path.exists(file_name):
#         print 'no such file'
#     npzfile = np.load(file_name+'.npz')
#
#     cwd = os.getcwd()
#     last_dir = cwd.split('/')[-1]
#     if last_dir == 'notebooks':
#         os.chdir(cwd.replace('/notebooks', ''))
#
#     for it in range(int(self.n_iter)):
#         self.curr_test.pi = npzfile['pi'][it]
#         self.curr_test.r_h = npzfile['r_h'][it]
#         self.curr_test.n = npzfile['n'][it]
#
#         if fair_regret:
#             self.average_fairness_regret = self.average_fairness_regret + self.calc_fairness_regret()
#         self.average_n = self.average_n + self.curr_test.n
#
#         if smooth_fair:
#             for i in range(len(self.e1_arr)):
#                 for j in range(len(self.e2_arr)):
#                     self.calc_smooth_fairness(i, j)
#
#         if subjective_smooth_fair:
#             for i in range(len(self.e1_arr)):
#                 for j in range(len(self.e2_arr)):
#                     self.calc_subjective_smooth_fairness(i, j)
#
#     if smooth_fair:
#         for i in range(len(self.e1_arr)):
#             for j in range(len(self.e2_arr)):
#                 self.calc_frac_smooth_fair(i, j)
#                 # self.calc_is_smooth_fair(i, j)
#
#     if subjective_smooth_fair:
#         for i in range(len(self.e1_arr)):
#             for j in range(len(self.e2_arr)):
#                 self.calc_frac_subjective_smooth_fair(i, j)
#                 # self.calc_is_subjective_smooth_fair(i, j)
#
#     self.average_n = np.divide(self.average_n, self.n_iter)
#     if regret:
#         self.average_regret = self.get_regret()
#     if fair_regret:
#         self.average_fairness_regret = np.divide(self.average_fairness_regret, self.n_iter)
#



# cwd = os.getcwd()
        # last_dir = cwd.split('/')[-1]
        # if last_dir == 'notebooks':
        #     os.chdir(cwd.replace('/notebooks', ''))
        #
        # if os.path.exists(file_name):
        #     self.analyse_from_file(regret, fair_regret, smooth_fair, subjective_smooth_fair)
        #     print 'restored data from file'
        #     return

        # pi = np.zeros((int(self.n_iter), len(self.e2_arr), len(self.delta_arr), self.T, self.k))
        # r_h = np.zeros((int(self.n_iter), len(self.e2_arr), len(self.delta_arr), self.T, self.k))
        # n = np.zeros((int(self.n_iter), len(self.e2_arr), len(self.delta_arr), self.T, self.k))


# pi[it][j][d] = self.curr_test.pi
# r_h[it][j][d] = self.curr_test.r_h
# n[it][j][d] = self.curr_test.n
#
# file_name = self.bandits.data_set_name + '/' + self.name + '/N_ITER_{}'.format(
#     int(self.n_iter)) + '_T_{}'.format(self.T)+'_e2_{}'.format(self.e2_arr[j])\
#             +'_delta_{}'.format(self.delta_arr[d])


# if not os.path.exists(file_name):
#     os.makedirs(file_name)
# np.savez(file_name, pi=pi, r_h=r_h, r_theta=self.bandits.theta, n=n)
