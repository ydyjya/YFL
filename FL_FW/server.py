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
from client import ADFC_Client
from data_loader import Adult_Loader as AL
import numpy as np

feature_name = ['workclass_?', 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked',
                        'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
                        'workclass_State-gov',
                        'workclass_Without-pay', 'education_10th', 'education_11th', 'education_12th',
                        'education_1st-4th',
                        'education_5th-6th', 'education_7th-8th', 'education_9th', 'education_Assoc-acdm',
                        'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate', 'education_HS-grad',
                        'education_Masters', 'education_Preschool', 'education_Prof-school', 'education_Some-college',
                        'marital-status_Divorced', 'marital-status_Married-AF-spouse',
                        'marital-status_Married-civ-spouse',
                        'marital-status_Married-spouse-absent', 'marital-status_Never-married',
                        'marital-status_Separated',
                        'marital-status_Widowed', 'occupation_?', 'occupation_Adm-clerical', 'occupation_Armed-Forces',
                        'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing',
                        'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service',
                        'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv',
                        'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
                        'relationship_Husband', 'relationship_Not-in-family', 'relationship_Other-relative',
                        'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
                        'race_Amer-Indian-Eskimo',
                        'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'native-country_?',
                        'native-country_Cambodia', 'native-country_Canada', 'native-country_China',
                        'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic',
                        'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England',
                        'native-country_France', 'native-country_Germany', 'native-country_Greece',
                        'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands',
                        'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary',
                        'native-country_India',
                        'native-country_Iran', 'native-country_Ireland', 'native-country_Italy',
                        'native-country_Jamaica',
                        'native-country_Japan', 'native-country_Laos', 'native-country_Mexico',
                        'native-country_Nicaragua',
                        'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru',
                        'native-country_Philippines',
                        'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico',
                        'native-country_Scotland', 'native-country_South', 'native-country_Taiwan',
                        'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States',
                        'native-country_Vietnam', 'native-country_Yugoslavia']


class Server(object):

    def __init__(self, client_num):
        self.global_weight = None
        self.client_num = client_num
        self.server_weight = None

    def update_weight(self):
        pass

    def connect_client(self):
        print("please rewrite")


class Adult_Server(Server):
    def __init__(self, client_num):
        super().__init__(client_num)
        self.client = []

    def connect_client(self):
        for i in range(self.client_num):
            t_data = AL(["./src/data_piece_%d.csv" % i]).get_data()
            t_data.loc[t_data['income'] == '<=50K', 'income'] = 0
            t_data.loc[t_data['income'] == '>50K', 'income'] = 1
            t_data.loc[t_data['gender'] == 'Male', 'gender'] = 0
            t_data.loc[t_data['gender'] == 'Female', 'gender'] = 1
            test_client = ADFC_Client(data=t_data, epsilon=1, delta=0.001, feature_name=feature_name)
            test_client.init_weight()
            self.client.append(test_client)

    def update_weight(self):
        new_server_weight = np.array([0] * len(self.client[0].get_weight())).astype("float")
        for client in self.client:
            new_server_weight += (client.get_weight() / self.client_num)
        self.server_weight = new_server_weight
        for client_idx, client in enumerate(self.client):
            print("当前是%d个客户端正在尝试更新" % client_idx)
            client.update_server_weight(self.server_weight)


if __name__ == '__main__':
    test = Adult_Server(10)
    test.connect_client()
    communication_times = 10
    for i in range(communication_times):
        print("当前是第%d轮通信" % i)
        test.update_weight()
    print()
