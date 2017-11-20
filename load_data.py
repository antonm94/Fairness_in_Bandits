import pandas as pd
import numpy as np
import bandits


def bar_exam_data():
    df = pd.read_sas('/Users/antonm/bachelorThesis/DataSets/LawSchool/BarPassage/LSAC_SAS/lsac.sas7bdat')
    bar = df[['ID', 'sex', 'race1', 'pass_bar' , 'bar']]
    bar = bar.dropna()
    race_grouped = bar[['race1','pass_bar']].groupby('race1')
    arm = np.empty((5,), dtype=object)
    arm[0] = list(race_grouped.get_group('asian')['pass_bar'])
    arm[1] = list(race_grouped.get_group('black')['pass_bar'])
    arm[2] = list(race_grouped.get_group('hisp')['pass_bar'])
    arm[3] = list(race_grouped.get_group('other')['pass_bar'])
    arm[4] = list(race_grouped.get_group('white')['pass_bar'])

    return bandits.Bandits(arm)


def default_credit_data():
    df = pd.read_excel('/Users/antonm/bachelorThesis/DataSets/Default/default.xls')
    sex_default = df[['X2','Y']][1:]
    sex_default_grouped = sex_default.groupby('X2')
    arm = np.empty((2,), dtype=object)
    arm[0] = list(sex_default_grouped.get_group(1)['Y'])
    arm[1] = list(sex_default_grouped.get_group(2)['Y'])

    return bandits.Bandits(arm)


def load_data(s):
    if s == 'Bar Exam':
        return bar_exam_data()
    else:
        if s == 'Default on Credit':
            return default_credit_data()
        else:
            print 'unknown data set'
