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
import differencial_privacy as dp
from data_loader import Adult_Loader as AL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

LEARN_RATE = 0.01


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class Client(object):
    def __init__(self, data, epsilon, delta):
        self.data = data

        self.epsilon = epsilon
        self.delta = delta

        self.weight = None
        self.server_weight = None

    def get_weight(self):
        return self.weight

    def download_weight(self, weight):
        self.server_weight = weight

    def update_weight(self, weight):
        self.weight = weight
        return None


class ADFC_Client(Client):

    def __init__(self, data: pd.DataFrame, epsilon, delta, **kwargs):
        super().__init__(data, epsilon, delta)
        self.kwargs = kwargs
        self.all_data = data
        self.all_label = data["income"]
        self.protected = data["gender"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.all_data, self.all_label,
                                                                                test_size=0.2)

    def get_weight(self):
        return self.weight

    def init_weight(self):

        self.x_train, self.x_test = self.x_train.reset_index(), self.x_test.reset_index()
        dp_method = dp.ADFC(privacy_budget=self.epsilon, relaxed_budget=self.delta,
                            data=self.x_train.drop(columns=["income", "gender"]), labels=self.y_train,
                            protected=self.x_train['gender'], special_col=[0], feature_name=self.kwargs["feature_name"])
        dp_method.get_res()
        self.weight = dp_method.get_weight()
        wl = dp_method.weight_list
        res_list = []
        max_acc_rd = [0, 0]
        for w in wl:
            self.weight = w
            acc = self.test_acc(self.x_test.drop(columns=["income", "gender"]), self.y_test, self.weight)
            rd = self.test_rd(self.x_test.drop(columns=["income", "gender"]), self.y_test, self.x_test['gender'])
            res_list.append(acc)
            if acc > max_acc_rd[0]:
                max_acc_rd = [acc, rd]
        result = wl[res_list.index(max(res_list))]
        print("准确率%f, rd%f" % (max_acc_rd[0], max_acc_rd[1]))
        self.update_weight(result)

    def update_server_weight(self, weight):
        """update = np.array([1e5] * len(self.weight)).astype('float')
        for idx, num in enumerate(update):
            if num > weight[idx] * LEARN_RATE:
                num = weight[idx] * LEARN_RATE"""
        update = LEARN_RATE * weight
        new_weight = update + self.weight
        new_acc = self.test_acc(self.x_test.drop(columns=["income", "gender"]), self.y_test, new_weight)
        old_acc = self.test_acc(self.x_test.drop(columns=["income", "gender"]), self.y_test, self.weight)
        if old_acc < new_acc:
            print("客户端进行了一次更新， 准确率由%.3f提升至%.3f" % (old_acc, new_acc))
            self.weight = new_weight

    def __processing_data(self, tmp_data):

        tmp_df = tmp_data

        ohe = OneHotEncoder(handle_unknown='ignore')

        temp = tmp_df.drop(columns=["fnlwgt"], axis=1)
        tmp_df = temp
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

        encode_df = pd.DataFrame(ohe.transform(tmp_df[cat_columns]).toarray(), columns=self.kwargs["feature_name"])

        tmp_df[num_columns] = tmp_df[num_columns].apply(pd.to_numeric)

        num_max = tmp_df[num_columns].max()
        num_min = tmp_df[num_columns].min()

        tmp_df = (tmp_df[num_columns] - num_min) / (num_max - num_min)
        tmp_df = pd.concat([tmp_df, encode_df], axis=1)

        return tmp_df

    def test_rd(self, dataset, protected, labels):
        dataset = self.__processing_data(dataset).to_numpy().astype('float')
        positive_1 = 0
        negative_1 = 0
        positive = 0
        negative = 0
        ans = 0
        for j in labels:
            if j == 1:
                ans += 1
        protected = protected.to_numpy()
        w = np.array(self.weight).astype('float')
        for i, r in enumerate(dataset):
            row = np.array(r)
            res = sigmoid(np.dot(row.transpose(), w))
            if protected[i] == 0:
                negative += 1
            else:
                positive += 1
            if res > 0.5 and protected[i] == 1:
                positive_1 += 1
            elif res > 0.5 and protected[i] == 0:
                negative_1 += 1
        num = positive + negative
        pr_pos_and_1 = positive_1 / num
        pr_pos_and_0 = negative_1 / num
        pr_1 = positive / num
        pr_0 = negative / num
        return np.abs(pr_pos_and_1 / pr_1 - pr_pos_and_0 / pr_0)

    def test_acc(self, x_test, y_test, weight):
        x_test = self.__processing_data(x_test).to_numpy().astype('float')
        w = np.array(weight).astype('float')
        right_0 = 0
        right_1 = 0
        all_y = len(y_test)
        y_test = y_test.to_numpy()
        for i, r in enumerate(x_test):
            row = np.array(r)
            res = sigmoid(np.dot(row.transpose(), w))
            if res > 0.5 and y_test[i] == 1:
                right_1 += 1
            elif res < 0.5 and y_test[i] == 0:
                right_0 += 1
            else:
                pass
        right = right_0 + right_1
        return right / all_y

    def cmp_lr(self, x_train, y_train, x_test, y_test):
        clf = LogisticRegression(
            penalty="l2", C=1.0, random_state=None, solver="lbfgs", max_iter=3000,
            multi_class='ovr', verbose=0,
        )
        clf.fit(x_train, y_train)
        # 使用测试数据来预测，返回值预测分类数据
        y_pred = clf.predict(x_test)

        # 打印主要分类指标的文本报告
        print('--- report ---')
        print(classification_report(y_test, y_pred))

        # 打印模型的参数
        print('--- params ---')
        print(clf.coef_, clf.intercept_)


if __name__ == '__main__':
    t_data = AL(["./src/adult.data"]).get_data()
    t_data.loc[t_data['income'] == '<=50K', 'income'] = 0
    t_data.loc[t_data['income'] == '>50K', 'income'] = 1
    t_data.loc[t_data['gender'] == 'Male', 'gender'] = 0
    t_data.loc[t_data['gender'] == 'Female', 'gender'] = 1
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
    test_client = ADFC_Client(data=t_data, epsilon=1, delta=0.001, feature_name=feature_name)
    # test.get_entropy_gain()
    test_client.init_weight()
