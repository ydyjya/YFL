# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        
# Author:           
# Version:          
# Created:          
# Description:      
# Function List:               
# History:
#       <author>        <version>       <time>      <desc>
# ------------------------------------------------------------------
from typing import Optional
from sklearn.preprocessing import OneHotEncoder
from data_loader import Adult_Loader as AL
import scipy.optimize as spo
import numpy as np
import pandas as pd
import time

BUDGET_S = 0.1
BUDGET_N = 1
DELTA_S = 0.001
DELTA_N = 0.0001

"""
没给delta
budget_delta_dict = {
    'race': (0.008, 0.0096),
    'native-country': (0.008, 0.0096),
    'capital-loss': (0.011, 0.0132),
    'workclass': (0.022, 0.0265),
    'capital-gain': (0.0429, 0.0516),
    'hours-per-week': (0.0539, 0.0648),
    'age': (0.0837, 0.1007),
    'occupation': (0.0929, 0.1117),
    'education': (0.0936, 0.1126),
    'education-num': (0.0936, 0.1126),
    'marital-status': (0.1565, 0.1882),
    'relationship': (0.1653, 0.1998)
}"""
budget_delta_dict = {
    'race': (0.0096, 0.0001),
    'native-country': (0.0096, 0.001),
    'capital-loss': (0.0132, 0.001),
    'workclass': (0.0265, 0.001),
    'capital-gain': (0.0516, 0.001),
    'hours-per-week': (0.0648, 0.001),
    'age': (0.1007, 0.001),
    'occupation': (0.1117, 0.001),
    'education': (0.1126, 0.001),
    'education-num': (0.1126, 0.001),
    'marital-status': (0.1882, 0.001),
    'relationship': (0.1998, 0.001)
}


def ent(my_data):
    prob1 = pd.value_counts(my_data) / len(my_data)
    return sum(np.log2(prob1) * prob1 * (-1))


def gain(my_data, str1, str2):
    e1 = my_data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(my_data[str1]) / len(my_data[str1])
    e2 = sum(e1 * p1)
    return ent(my_data[str2]) - e2


def num_data2cat_data(my_data: pd.DataFrame, num_col):
    def age_process(x):
        if x < 28:
            return "17-27"
        elif x < 37:
            return "28-36"
        elif x < 48:
            return "37-47"
        else:
            return "48-90"

    def capital_process(x):
        if x > 0:
            return "have_capital_gain/loss"
        else:
            return "don't_have"

    def educational_process(x):
        if x < 9:
            return "lt9"
        elif x == 9:
            return "9"
        elif x == 10:
            return "10"
        elif x < 14:
            return "11-13"
        else:
            return "mt13"

    def hours_per_week_process(x):
        if x < 36:
            return "lt36"
        elif x < 41:
            return "36-40"
        else:
            return "mt40"

    original_data = my_data
    other_data = original_data.drop(columns=num_col)
    process_data = my_data[num_col]
    tmp_df = pd.DataFrame()
    tmp_df['age'] = process_data['age'].apply(age_process).astype('object')
    tmp_df['capital-loss'] = process_data['capital-loss'].apply(capital_process).astype('object')
    tmp_df['capital-gain'] = process_data['capital-gain'].apply(capital_process).astype('object')
    tmp_df['hours-per-week'] = process_data['hours-per-week'].apply(hours_per_week_process).astype('object')
    # tmp_df['educational-num'] = process_data['educational-num'].apply(educational_process).astype('object')
    tmp_df['educational-num'] = process_data['educational-num'].astype('object')

    return pd.concat([other_data, tmp_df], axis=1)


def fun_gender(gender):
    if gender == "Female":
        return 1
    else:
        return 0


def fun_income(income):
    if income == "<=50K":
        return 0
    else:
        return 1


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class Differential_Privacy(object):
    def __init__(self, privacy_budget: float, relaxed_budget: Optional[float]):
        self.privacy_budget = privacy_budget
        if relaxed_budget:
            self.relaxed_budget = relaxed_budget
        else:
            self.relaxed_budget = 0

        self.weight, bias = self.compute_weight()

    def compute_weight(self):
        weight, bias = None, None
        return weight, bias

    def set_budget(self, budget):
        self.privacy_budget = budget

    def set_relaxed(self, relaxed):
        self.relaxed_budget = relaxed

    def get_weight(self):
        return self.weight


class ADFC(Differential_Privacy):

    def __init__(self, privacy_budget, relaxed_budget, data: pd.DataFrame, labels: pd.Series,
                 protected: pd.Series, special_col: list, **kwargs):
        super().__init__(privacy_budget, relaxed_budget)

        if "feature_name" in kwargs:
            self.feature_name = kwargs["feature_name"]

        # 除去被保护列的数据, df和array
        self.data = data
        self.original_data = data
        self.data = self.__processing_data()
        self.dataset = self.data.to_numpy()

        # 目标列
        vfunc = np.vectorize(fun_income)
        self.labels = vfunc(labels.to_numpy())

        # 把被保护列直接分出来
        vfunc = np.vectorize(fun_gender)
        self.protected = vfunc(protected.to_numpy())

        # 特殊列的列表
        self.special_col = special_col

        self.dimension = len(self.dataset[0])
        self.data_size = len(self.dataset)

        self.budget = None
        self.delta = None

        # 计算敏感性
        if 'sensitivity' not in kwargs:
            self.sensitivity = np.sqrt((self.dimension * self.dimension) / 16 + 9 * self.dimension)
        else:
            self.sensitivity = np.sqrt((kwargs['sensitivity'] * kwargs['sensitivity']) / 16 + 9 * kwargs['sensitivity'])

        self.w_0, self.w_1, self.w_2 = self.__get_obj_fun()

        # self.__add_entropy_noise()
        # self.__add_noise()
        # 对目标函数求解最优
        self.__optimize()

    def __get_obj_fun(self):
        """
        :return: 展开函数
        """
        # weight是一个d维向量

        # w的0次系数
        w_0 = 0
        # w的1次系数
        w_1 = np.array([[0] * self.dimension])
        # w的2次系数
        w_2 = np.array([[0] * self.dimension] * self.dimension)

        # 被保护列的平均值，用于进行公平性约束
        pro_mean = np.mean(self.protected.astype("int64"))

        # 根据泰勒公式进行二次展开近似为一个多项式函数
        for i, r in enumerate(self.dataset):
            # item1 计算
            row = np.array(r)
            f0 = np.log(2)
            f1 = 1 / 2 * row
            f2 = 1 / 2 * 1 / 4 * np.dot(row.reshape(1, self.dimension).transpose(), row.reshape(1, self.dimension))
            w_0 = w_0 + f0
            w_1 = w_1 + f1
            w_2 = w_2 + f2

            # item2 计算
            w_1 = w_1 - self.labels[i] * row

        # item3-公平性约束计算
        temp_w_1 = np.array([[0] * self.dimension])
        for i, r in enumerate(self.protected):
            delta_z = r - pro_mean
            temp_w_1 = temp_w_1 + delta_z * np.array(self.dataset[i])
        w_1 = w_1 + np.abs(temp_w_1)

        # 返回加噪后的数据
        return w_0, w_1, w_2

    def __obj_fun(self, x):
        """
        通过w_0,w_1,w_2的系数进行计算，得到一个多项式函数，根据函数机制的原始要求，这个函数经过加噪处理后，可能是无界的，可能需要多跑几次
        :param x: 输入的变量--->权重
        :return: 计算的loss
        """
        # 计算二次系数与乘积
        temp = np.dot(x.reshape(1, self.dimension).transpose(), x.reshape(1, self.dimension))
        res2 = temp * self.w_2
        # 目前计算0次项
        return self.w_0 + np.dot(x, self.w_1.transpose()) + np.sum(res2)

    def __add_noise(self):

        self.budget = (1 / self.dimension) * BUDGET_S + ((self.dimension - 1) / self.dimension) * BUDGET_N
        self.delta = 1 - (1 - DELTA_S) * (1 - DELTA_N)
        tmp_w_1 = self.w_1
        tmp_w_2 = self.w_2
        gau_delta_s = np.sqrt(1 / 2) * self.sensitivity / BUDGET_S * \
                      (np.sqrt(
                          np.log(
                              np.sqrt(2 / np.pi) * 1 / DELTA_S)) +
                       np.sqrt(
                           np.log(
                               np.sqrt(2 / np.pi) * 1 / DELTA_S) + BUDGET_S))
        gau_delta_n = np.sqrt(1 / 2) * self.sensitivity / BUDGET_N * \
                      (np.sqrt(
                          np.log(
                              np.sqrt(2 / np.pi) * 1 / DELTA_N)) +
                       np.sqrt(
                           np.log(
                               np.sqrt(2 / np.pi) * 1 / DELTA_N) + BUDGET_N))

        for special_idx in self.special_col:
            for i, r in enumerate(self.w_1[0]):
                if i == special_idx:
                    tmp_w_1[0][i] = tmp_w_1[0][i] + np.random.normal(0, gau_delta_s)
                else:
                    tmp_w_1[0][i] = tmp_w_1[0][i] + np.random.normal(0, gau_delta_n)

            for i, ri in enumerate(self.w_2):
                for j, rj in enumerate(ri):
                    if i == special_idx or j == special_idx:
                        tmp_w_2[i][j] += np.random.normal(0, gau_delta_s)
                    else:
                        tmp_w_2[i][j] += np.random.normal(0, gau_delta_n)

        self.w_1 = tmp_w_1
        self.w_2 = tmp_w_2

    def __add_entropy_noise(self):
        def gau_delta_noise(budget, delta):
            gau_delta_para = np.sqrt(1 / 2) * self.sensitivity / budget * \
                             (np.sqrt(
                                 np.log(
                                     np.sqrt(2 / np.pi) * 1 / delta)) +
                              np.sqrt(
                                  np.log(
                                      np.sqrt(2 / np.pi) * 1 / delta) + budget))
            return np.random.normal(0, gau_delta_para)

        num_col = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        tmp = list(self.feature_name)
        num_col.extend(tmp)
        data_col_name = np.array(num_col)
        add_dict = {}
        for i, r in enumerate(data_col_name):
            for key, value in budget_delta_dict.items():
                if key in r:
                    if i not in add_dict:
                        add_dict[i] = key
        add_dict[1] = 'education-num'
        add_dict[41] = 'occupation'

        tmp_w_1 = self.w_1
        tmp_w_2 = self.w_2

        for i, r in enumerate(self.w_1[0]):
            tmp_w_1[0][i] = \
                tmp_w_1[0][i] + gau_delta_noise(budget_delta_dict[add_dict[i]][0], budget_delta_dict[add_dict[i]][1])

        for i, ri in enumerate(self.w_2):
            for j, rj in enumerate(ri):
                if i == j:
                    tmp_w_2[i][j] += gau_delta_noise(budget_delta_dict[add_dict[i]][0],
                                                     budget_delta_dict[add_dict[i]][1])
                else:
                    tmp_w_2[i][j] = tmp_w_2[i][j] + \
                                    (1 / 2 * gau_delta_noise(budget_delta_dict[add_dict[i]][0],
                                                             budget_delta_dict[add_dict[i]][1])) + \
                                    (1 / 2 * gau_delta_noise(budget_delta_dict[add_dict[j]][0],
                                                             budget_delta_dict[add_dict[j]][1]))

        self.w_1 = tmp_w_1
        self.w_2 = tmp_w_2

    def __optimize(self):
        minimum = spo.minimize(self.__obj_fun, np.array([0] * self.dimension), method="L-BFGS-B")
        # minimum = spo.basinhopping(self.__obj_fun, np.array([0] * self.dimension),
        #                            minimizer_kwargs={"method": "L-BFGS-B"})
        res = minimum.x
        loss = self.__obj_fun(np.array([0] * self.dimension))
        return res, loss

    def get_res(self):
        """
        获取ADFC系数的主函数
        :return: 参数w，隐私预算b，松弛约束δ
        """
        res_list = []
        loss_list = []
        rd_list = []
        for i in range(20):
            time.sleep(0.001)
            self.w_0, self.w_1, self.w_2 = self.__get_obj_fun()
            self.__add_entropy_noise()
            res, loss = self.__optimize()
            rd = self.get_risk_difference(res)
            if rd > 0.025:
                res_list.append(res)
                loss_list.append(loss)
                rd_list.append(rd)
        result = res_list[loss_list.index(min(loss_list))]
        rd = rd_list[loss_list.index(min(loss_list))]
        self.weight = result
        self.weight_list = res_list
        return result, rd, self.budget, self.delta

    def get_risk_difference(self, weight):
        positive_1 = 0
        negative_1 = 0
        positive = 0
        negative = 0
        ans = 0
        for j in self.labels:
            if j == 1:
                ans += 1
        w = weight
        for i, r in enumerate(self.dataset):
            row = np.array(r)
            res = sigmoid(np.dot(row.transpose(), w))
            if self.protected[i] == 0:
                negative += 1
            else:
                positive += 1

            if res > 0.5 and self.protected[i] == 1:
                positive_1 += 1
            elif res > 0.5 and self.protected[i] == 0:
                negative_1 += 1
        num = positive + negative
        pr_pos_and_1 = positive_1 / num
        pr_pos_and_0 = negative_1 / num
        pr_1 = positive / num + 1
        pr_0 = negative / num + 1
        return np.abs(pr_pos_and_1 / pr_1 - pr_pos_and_0 / pr_0)

    def __processing_data(self):

        tmp_df = self.data
        ohe = OneHotEncoder(handle_unknown='ignore')

        tmp_df.drop(columns=["fnlwgt"], axis=1, inplace=True)
        # 离散数据
        cat_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                       'native-country']
        # 连续数据
        num_columns = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

        ad = pd.read_csv("./src/adult.data")
        ad.columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
                      'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                      'hours-per-week', 'native-country', "income"]

        ohe.fit(ad[cat_columns])

        encode_df = pd.DataFrame(ohe.transform(tmp_df[cat_columns]).toarray(), columns=self.feature_name)
        tmp_df[num_columns] = tmp_df[num_columns].apply(pd.to_numeric)

        num_max = tmp_df[num_columns].max()
        num_min = tmp_df[num_columns].min()

        tmp_df = (tmp_df[num_columns] - num_min) / (num_max - num_min)

        tmp_df = pd.concat([tmp_df, encode_df], axis=1)
        return tmp_df

    def get_entropy_gain(self):
        Y = pd.Series(self.labels).rename('income')
        num_columns = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        not_process_data = pd.concat([self.original_data, Y], axis=1)
        cat_data = num_data2cat_data(self.original_data, num_columns)
        X_Y_data = pd.concat([cat_data, Y], axis=1)
        print("连续数据离散化的信息增益：")
        p_list = []
        for col in X_Y_data.columns[:-1]:
            p_list.append((col, gain(X_Y_data, col, 'income')))
        for i in sorted(p_list, key=lambda x: x[1]):
            print(i)
        o_list = []
        print("\n\n原始数据的信息增益：")
        for col in not_process_data.columns[:-1]:
            o_list.append((col, gain(not_process_data, col, 'income')))
        for i in sorted(o_list, key=lambda x: x[1]):
            print(i)


if __name__ == '__main__':
    t_data = AL(["./src/adult.data"]).get_data()
    feature_name = ['workclass_?', 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked',
                    'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov',
                    'workclass_Without-pay', 'education_10th', 'education_11th', 'education_12th', 'education_1st-4th',
                    'education_5th-6th', 'education_7th-8th', 'education_9th', 'education_Assoc-acdm',
                    'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate', 'education_HS-grad',
                    'education_Masters', 'education_Preschool', 'education_Prof-school', 'education_Some-college',
                    'marital-status_Divorced', 'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
                    'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated',
                    'marital-status_Widowed', 'occupation_?', 'occupation_Adm-clerical', 'occupation_Armed-Forces',
                    'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing',
                    'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service',
                    'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv',
                    'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
                    'relationship_Husband', 'relationship_Not-in-family', 'relationship_Other-relative',
                    'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife', 'race_Amer-Indian-Eskimo',
                    'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'native-country_?',
                    'native-country_Cambodia', 'native-country_Canada', 'native-country_China',
                    'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic',
                    'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England',
                    'native-country_France', 'native-country_Germany', 'native-country_Greece',
                    'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands',
                    'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India',
                    'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica',
                    'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua',
                    'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines',
                    'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico',
                    'native-country_Scotland', 'native-country_South', 'native-country_Taiwan',
                    'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States',
                    'native-country_Vietnam', 'native-country_Yugoslavia']
    test = ADFC(privacy_budget=1, relaxed_budget=1, data=t_data.drop(columns=["income", "gender"]),
                labels=t_data["income"],
                protected=t_data["gender"], special_col=[0], feature_name=np.array(feature_name))
    # test.get_entropy_gain()
    print(test.get_res())

"""
['?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private',
                     'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
                    ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
                     'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
                     'Masters', 'Preschool', 'Prof-school', 'Some-college'],
                    ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                     'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
                    ['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair',
                     'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners',
                     'Machine-op-inspct', 'Other-service', 'Priv-house-serv',
                     'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support',
                     'Transport-moving'],
                    ['Husband', 'Not-in-family', 'Other-relative', 'Own-child',
                     'Unmarried', 'Wife'],
                    ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other',
                     'White'],
                    ['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba',
                     'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England',
                     'France', 'Germany', 'Greece', 'Guatemala', 'Haiti',
                     'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',
                     'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
                     'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',
                     'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South',
                     'Taiwan',
                     'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',
                     'Yugoslavia']"""
