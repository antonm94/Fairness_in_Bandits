import thompson_sampling.bern_fair_stochastic_dominance_ts as fair_sd_ts
import logging
import warnings
import numpy as np
import os
import matplotlib.lines as mlines
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.simplefilter(action='ignore', category=FutureWarning)
LOGGING = 1
if LOGGING > 0:
    logger = logging.getLogger(__name__)

# sns.set_palette("husl")
# colors = sns.color_palette()
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1.0, 60))

colors = sns.husl_palette(10) + sns.color_palette("cubehelix", 8)

linestyles = ['-', '--', '-.', ':']

def plot_delta_subjective_fair(test_cases, start_index=0):
    T = test_cases[0].T
    x = range(T)


    for test in test_cases:
        if test.name == 'Fair SD TS':
            continue

        for e1_ind, e1 in enumerate(test.e1_arr):
            for e2_ind, e2 in enumerate(test.e2_arr):
                algo_name = test.get_name(e1=e1, e2=e2)
                plt.plot(x[start_index:],
                         np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1),  axis=1)
                         [start_index:], label=algo_name)


                    # xmin = np.amin(np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1))
                    # ymin = np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1)
                    # plt.plot(xmin, ymin)
                    # plt.annotate((1 - d), xy=(2, 1), xytext=(3, 1.5),
                    #              arrowprops=dict(facecolor='black', shrink=0.05),
                    #              )

    plt.xlabel('T')
    plt.ylabel('Subjective Smooth fair with probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    name = 'seperate_data_delta_smooth_fair_{}'.format(T)
    save_plot(name)

    plt.show()


def plot_delta_smooth_fair(test_cases, start_index=0):
    T = test_cases[0].T
    x = range(T)

    for test in test_cases:
        ds_name = test.bandits.data_set_name
        for e1_ind, e1 in enumerate(test.e1_arr):
            for e2_ind, e2 in enumerate(test.e2_arr):
                if test.name == 'Fair SD TS':
                    for delta_ind, delta in enumerate(test.delta_arr):
                        algo_name = test.get_name(e1=e1, e2=e2, delta=delta)
                        y = np.min(np.min(test.frac_smooth_fair[e1_ind, e2_ind, delta_ind], axis=1),
                                        axis=1)[start_index:]

                        explore_end_x = test.average_rounds_exploring[e2_ind, delta_ind]
                        if explore_end_x < T:
                            plt.plot(x[start_index:], y, label=algo_name, marker='o', markevery=[int(explore_end_x)])
                        else:
                            plt.plot(x[start_index:], y, label=algo_name)
                else:
                    algo_name = test.get_name(e1=e1, e2=e2)
                    plt.plot(x[start_index:],
                             np.min(np.min(test.frac_smooth_fair[e1_ind, e2_ind], axis=1), axis=1)
                             [start_index:], label=algo_name)

                # xmin = np.amin(np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1))
                # ymin = np.min(np.min(test.frac_subjective_smooth_fair[e1_ind, e2_ind], axis=1), axis=1)
                # plt.plot(xmin, ymin)
                # plt.annotate((1 - d), xy=(2, 1), xytext=(3, 1.5),
                #              arrowprops=dict(facecolor='black', shrink=0.05),
                #              )

    plt.xlabel('T')
    plt.ylabel('Smooth fair with probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    name = 'delta_smooth_fair_{}'.format(T)
    save_plot(name)


    # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()




def plot_fairness_regret(test_cases, start_index=0):
    T = test_cases[0].T
    x = range(T)

    algo_colors, ds_styles = get_colors_and_styles_regret(test_cases)
    labels = get_labels(algo_colors, ds_styles)

    for test in test_cases:
        ds_name = test.bandits.data_set_name

        if test.name == 'Fair SD TS':
            for e2_ind, e2 in enumerate(test.e2_arr):
                for delta_ind, delta in enumerate(test.delta_arr):
                    algo_name = test.get_name(e2=e2, delta=delta)
                    y = np.add.accumulate(test.average_fairness_regret[e2_ind][delta_ind])
                    explore_end_x = test.average_rounds_exploring[e2_ind, delta_ind]
                    if explore_end_x < T:
                        plt.plot(x[start_index:], y, label=algo_name, marker='o', markevery=[int(explore_end_x)])
                    else:
                        plt.plot(x[start_index:], y, label=algo_name)
        else:
            algo_name = test.get_name()
            plt.plot(x, np.add.accumulate(test.average_fairness_regret), label=algo_name, linestyle=ds_styles[ds_name],
                         color=algo_colors[algo_name]),


    plt.plot(x, [pow(test_cases[0].k * t, 2. / 3) for t in x], linestyle=':', color='k')
    # labels.append(mlines.Line2D([], [], color='k',
    #                             linestyle=':', label='regret bound O((k*T)^2/3)'))
    plt.xlabel('T')
    plt.ylabel('Cumulative Fairness Regret')
    # Create a legend for the first line.
    # bound = plt.legend(
    #     handles=[mlines.Line2D([], [], color='k', linestyle=':', label='regret bound O((k*T)^2/3)')], loc=4)
    #
    # # Add the legend manually to the current Axes.
    # plt.gca().add_artist(bound)
    labels.append(mlines.Line2D([], [], color='k', linestyle=':', label='regret bound O((k*T)^2/3)'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)




    name = 'cumulative_fairness_regret{}'.format(T)
    save_plot(name)
    plt.show()



def plot_average_total_regret(test_cases, start_index=0):
    T = test_cases[0].T
    x = range(T)

    algo_colors, ds_styles = get_colors_and_styles_regret(test_cases)
    labels = get_labels(algo_colors, ds_styles)

    for test in test_cases:
        ds_name = test.bandits.data_set_name

        if test.name == 'Fair SD TS':
            for e2_ind, e2 in enumerate(test.e2_arr):
                for delta_ind, delta in enumerate(test.delta_arr):
                    algo_name = test.get_name(e2=e2, delta=delta)
                    y = test.average_regret[e2_ind, delta_ind]
                    explore_end_x = test.average_rounds_exploring[e2_ind, delta_ind]
                    if explore_end_x < T:
                        plt.plot(x[start_index:], y, label=algo_name, marker='o', markevery=[int(explore_end_x)])
                    else:
                        plt.plot(x[start_index:], y, label=algo_name)
        else:
            algo_name = test.get_name()
            plt.plot(x, test.average_regret, label=algo_name, linestyle=ds_styles[ds_name],
                     color=algo_colors[algo_name]),

    plt.xlabel('T')
    plt.ylabel('Total Regret')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    name = 'total_regret{}'.format(T)
    save_plot(name)
    plt.show()

def plot_new_lambda_regret_tradeoff(test_cases):

    Fair_SD_TS_FR = np.zeros()
    lam_arr = []
    fr = []
    r = []
    for test in test_cases:
        fr.append(test.average_fairness_regret[-1])
        r.append(test.average_regret[-1])

    plt.plot(fr, r)






    # lam_arr.append(test.lam)
    # if test.name == 'Fair SD TS':
    #     for e2_ind, e2 in enumerate(test.e2_arr):
    #         for delta_ind, delta in enumerate(test.delta_arr):









def plot_lambda_regret_tradeoff(test_cases):
    sns.set_palette("husl")
    colors = sns.color_palette()
    linestyles = ['-', '--', '-.', ':']

    ds_style = {}
    algo_color = {}
    ds_ind = 0
    color_ind = 0


    lam_arr = []
    total_regret = []
    fairness_regret = []

    for test in test_cases:
        lam_arr.append(test.lam)
        if test.name == 'Fair SD TS':
            for e2_ind, e2 in enumerate(test.e2_arr):
                for delta_ind, delta in enumerate(test.delta_arr):
                    total_regret.append(test.average_regret[-1])
                    fairness_regret.append(test.average_fairness_regret[-1])

        else:
            total_regret.append(test.average_regret[0][0][-1])
            fairness_regret.append(test.average_fairness_regret[0][0][-1])

    fig, ax1 = plt.subplots()
    ax1.plot(lam_arr, total_regret, 'b-')
    ax1.set_xlabel('$/lambda$')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Regret', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(lam_arr, fairness_regret, 'r-')
    ax2.set_ylabel('Cumulative Fairness Regret', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()

    name = 'regret_tradeoff_{}'.format(test_cases[0].T)
    save_plot(name)
    plt.show()

def plot_regret_tradeoff(self, lam):
    n_lam = len(lam)
    total_regret = np.zeros(n_lam)
    fairness_regret = np.zeros(n_lam)
    sns.set_palette("husl")

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
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., )
    plt.show()



def plot_min_e1(test_cases):
    T = test_cases[0].T
    x = range(T)

    for test_ind, test in enumerate(test_cases):
        ds_name = test.bandits.data_set_name
        for e2_ind, e2 in enumerate(test.e2_arr):
            for delta_ind, delta in enumerate(test.delta_arr):
                algo_name = test.get_name(delta=delta, e2=e2)
                y = test.min_e1[e2_ind, delta_ind]
                plt.plot(x, y, label=algo_name)
                # if test.name == 'Fair SD TS':
                #     explore_end_x = int(test.average_rounds_exploring[e2_ind, delta_ind])
                #     if explore_end_x < T:
                #         explore_end_y = y[explore_end_x]
                #         plt.plot(explore_end_x, explore_end_y, marker='o', color=algo_colors[algo_name])


    plt.xlabel('T')
    plt.ylabel('minimum $\epsilon_1$ to be Smooth Fair')
    name = 'subjective_min_e1_{}_'.format(T)


    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    save_plot(name)
    plt.show()



def plot_subjective_min_e1(test_cases):
    T = test_cases[0].T
    x = range(T)

    for test_ind, test in enumerate(test_cases):
        if test.name == 'Fair SD TS':
            continue
        ds_name = test.bandits.data_set_name
        for e2_ind, e2 in enumerate(test.e2_arr):
            for delta_ind, delta in enumerate(test.delta_arr):
                algo_name = test.get_name(delta=delta, e2=e2)
                plt.plot(x, test.subjective_min_e1[e2_ind, delta_ind],label=algo_name)

    plt.xlabel('T')
    plt.ylabel('minimum $\epsilon_1$ to be Subjective Smooth Fair')
    name = 'subjective_min_e1_{}_'.format(T)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    save_plot(name)
    plt.show()


def get_labels(algo_colors, ds_style):
    labels = []
    for algo_name, color in algo_colors.iteritems():
        labels.append(mpatches.Patch(color=color, label=algo_name))
    for ds_name, style in ds_style.iteritems():
        labels.append(mlines.Line2D([], [], color='black',
                                    linestyle=ds_style[ds_name], label=ds_name))
    return labels


def get_colors_and_styles(test_cases):
    ds_styles = {}
    algo_colors = {}
    ds_ind = 0
    color_ind = 0

    for test_ind, test in enumerate(test_cases):

        ds_name = test.bandits.data_set_name
        if ds_name not in ds_styles:
            ds_styles[ds_name] = linestyles[ds_ind]
            ds_ind = ds_ind + 1

        for e2_ind, e2 in enumerate(test.e2_arr):
            for delta_ind, delta in enumerate(test.delta_arr):

                algo_name = test.get_name(delta=delta, e2=e2)
                if algo_name not in algo_colors:
                    algo_colors[algo_name] = colors[color_ind]
                    color_ind = color_ind + 1

    return algo_colors, ds_styles


def get_colors_and_styles_delta(test_cases):
    ds_styles = {}
    algo_colors = {}
    ds_ind = 0
    color_ind = 0

    for test_ind, test in enumerate(test_cases):
        ds_name = test.bandits.data_set_name
        if ds_name not in ds_styles:
            ds_styles[ds_name] = linestyles[ds_ind]
            ds_ind = ds_ind + 1
        for e1_ind, e1 in enumerate(test.e1_arr):
            for e2_ind, e2 in enumerate(test.e2_arr):
                if test.name == 'Fair SD TS':
                    for delta_ind, delta in enumerate(test.delta_arr):
                        algo_name = test.get_name(e1=e1, delta=delta, e2=e2)
                        if algo_name not in algo_colors:
                            algo_colors[algo_name] = colors[color_ind]
                            color_ind = color_ind + 1
                else:
                    algo_name = test.get_name(e1=e1, e2=e2)
                    if algo_name not in algo_colors:
                        algo_colors[algo_name] = colors[color_ind]
                        color_ind = color_ind + 1

    return algo_colors, ds_styles


def get_colors_and_styles_regret(test_cases):
    ds_styles = {}
    algo_colors = {}
    ds_ind = 0
    color_ind = 0

    for test_ind, test in enumerate(test_cases):
        ds_name = test.bandits.data_set_name
        if ds_name not in ds_styles:
            ds_styles[ds_name] = linestyles[ds_ind]
            ds_ind = ds_ind + 1
        if test.name == 'Fair SD TS':
            for e2_ind, e2 in enumerate(test.e2_arr):
                for delta_ind, delta in enumerate(test.delta_arr):
                    algo_name = test.get_name(e2=e2, delta=delta)
                    if algo_name not in algo_colors:
                        algo_colors[algo_name] = colors[color_ind]
                        color_ind = color_ind + 1
        else:
            algo_name = test.get_name()
            if algo_name not in algo_colors:
                algo_colors[algo_name] = colors[color_ind]
                color_ind = color_ind + 1

    return algo_colors, ds_styles



def save_plot(name):
    i = 0
    name = 'sep_data_' + name
    name_i = name + '_' + format(i) + '.png'
    while os.path.exists(name):
        i += 1
        name_i = name + '_' + format(i) + '.png'
    plt.savefig('/Users/antonm/Desktop/BachelorThesis/Plots/'+ name_i, bbox_inches='tight')
