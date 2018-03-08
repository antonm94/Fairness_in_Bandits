import pandas as pd
import numpy as np
import bandits
import os
path = os.path.dirname(os.path.abspath(__file__))




def bar_exam_data():
    df = pd.read_sas(path +'/DataSets/LawSchool/BarPassage/LSAC_SAS/lsac.sas7bdat')
    bar = df[['ID', 'sex', 'race1', 'pass_bar' , 'bar']]
    bar = bar.dropna()
    race_grouped = bar[['race1','pass_bar']].groupby('race1')
    arm = np.empty((5,), dtype=object)
    arm[0] = list(race_grouped.get_group('asian')['pass_bar'])
    arm[1] = list(race_grouped.get_group('black')['pass_bar'])
    arm[2] = list(race_grouped.get_group('hisp')['pass_bar'])
    arm[3] = list(race_grouped.get_group('other')['pass_bar'])
    arm[4] = list(race_grouped.get_group('white')['pass_bar'])

    return bandits.Bandits(arm, 'bar_exam')


def default_credit_data():
    df = pd.read_excel(path +'/DataSets/Default/default.xls')
    sex_default = df[['X2','Y']][1:]
    sex_default_grouped = sex_default.groupby('X2')
    arm = np.empty((2,), dtype=object)
    arm[0] = list(sex_default_grouped.get_group(1)['Y'])
    arm[1] = list(sex_default_grouped.get_group(2)['Y'])

    return bandits.Bandits(arm, 'default_credit')


def adult_data():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                    'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'income_class']


    df = pd.read_csv(path +'./DataSets/Adult/adult.data.csv', names = column_names)

    df['income_class'] = df['income_class'].astype('str')
    df = df.replace({'income_class': {df['income_class'][7]: 1, df['income_class'][0]: 0}})
    age_grouped = df.groupby('marital_status')
    for name, group in age_grouped:
               print(name)
               print(len(group))

    # arm = np.empty((5,), dtype=object)
    # arm[0] = list(race_grouped.get_group('asian')['pass_bar'])
    # arm[1] = list(race_grouped.get_group('black')['pass_bar'])
    # arm[2] = list(race_grouped.get_group('hisp')['pass_bar'])
    # arm[3] = list(race_grouped.get_group('other')['pass_bar'])
    # arm[4] = list(race_grouped.get_group('white')['pass_bar'])
    #
    # return bandits.Bandits(arm)

    #return bandits.Bandits(arm)

def load_data(s):
    data = {
        'Bar Exam': 'Bar Exam',
        'Default on Credit': 'Default on Credit',
        '0': [0.001, 0.00001, 0.98, 0.97, 0.96],
        '1': [0.12, 0.2, 0.13, 0.04, 0.10]
    }

    if data[s] == 'Bar Exam':
        return bar_exam_data()
    elif data[s] == 'Default on Credit':
            return default_credit_data()
    else:
        p = data[s]
        arm = np.empty((len(p), 1000), dtype=object)
        for i in range(len(p)):
            arm[i] = np.random.binomial(1, p[i], 1000)

        return bandits.Bandits(arm, 'Data'+s)

if __name__ == '__main__':
    #adult_data()
    bandits = load_data('0')
    print bandits.get_mean()