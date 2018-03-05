import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def plot_min_e1(tests):
    fig, ax = plt.subplots()
    x = range(tests[0].T)
    sns.set_palette("cubehelix", 10)
    colors = sns.color_palette()
    linestyles = ['-', '--', '-.', ':']
    algo_labels = []
    data_labels = []
    got_labels = False
    for test_ind, test in enumerate(tests):
        if test.name == 'Fair SD TS':
            print test.average_rounds_exploring


        color_ind = 0
        for e2_ind, e2 in enumerate(test.e2_arr):
            for delta_ind, delta in enumerate(test.delta_arr):
                ax.plot(x, test.min_e1[e2_ind, delta_ind], linestyle=linestyles[test_ind],
                        color=colors[color_ind], label=test.get_name(delta=delta, e2=e2))
                # algo_labels.append(mpatches.Patch(color=colors[color_ind], label=test.get_name(delta=delta, e2=e2)))
                color_ind = color_ind + 1
        got_labels = True
    plt.xlabel('T')
    plt.ylabel('minimum e1')
    # plt.legend(handles=algo_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend()

    i = 0
    name = 'min_e1_{}'.format(i) + '.png'
    while os.path.exists(name):
        i += 1
        name = 'min_e1_{}'.format(i) + '.png'
    plt.savefig(name, bbox_inches='tight')

    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_subjective_min_e1(tests):
    fig, ax = plt.subplots()
    x = range(tests[0].T)
    sns.set_palette("cubehelix", 10)
    colors = sns.color_palette()
    linestyles = ['-', '--', '-.', ':']
    algo_labels = []
    data_labels = []
    got_labels = False
    for test_ind, test in enumerate(tests):
        if test.name == 'Fair SD TS':
            print test.average_rounds_exploring


        color_ind = 0
        for e2_ind, e2 in enumerate(test.e2_arr):
            for delta_ind, delta in enumerate(test.delta_arr):
                ax.plot(x, test.subjective_min_e1[e2_ind, delta_ind], linestyle=linestyles[test_ind],
                        color=colors[color_ind], label=test.get_name(delta=delta, e2=e2))
                # algo_labels.append(mpatches.Patch(color=colors[color_ind], label=test.get_name(delta=delta, e2=e2)))
                color_ind = color_ind + 1
        got_labels = True
    plt.xlabel('T')
    plt.ylabel('minimum e1')
    # plt.legend(handles=algo_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend()
    i = 0
    name = 'subjective_min_e1_{}'.format(i)+'.png'

    while os.path.exists(name):
        i += 1
        name = 'subjective_min_e1_{}'.format(i) + '.png'
    plt.savefig(name, bbox_inches='tight')

    plt.show()