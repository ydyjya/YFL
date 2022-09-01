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
from sklearn.preprocessing import OneHotEncoder
from data_spilt import Dirichlet_Spilt
import pandas as pd
import numpy as np


class Data_Loader(object):
    def __init__(self, data_road_list: list = None):
        self.data_road_list = data_road_list
        self.data = self.read_data()

    def set_data_road(self, data_road_list):
        self.data_road_list = data_road_list

    def read_data(self):
        print("please write your own function")
        data = []
        return data


class Adult_Loader(Data_Loader):

    def __init__(self, data_road: list = None):
        super().__init__(data_road)
        self.all_data = self.read_data()
        self.feature_name = self.read_feature_name()

    def read_data(self):
        file = self.data_road_list[0]
        data = pd.read_csv(file, sep=',')
        data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
                        'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country', 'income']
        return data

    @staticmethod
    def read_feature_name():
        file_name = "./src/feature_name.txt"
        with open(file_name, "r") as fn:
            cat_feature_name = fn.read()
        cat_feature_name = np.array(cat_feature_name.split(","))
        return cat_feature_name

    def __get_onehot_feature_name(self):
        ohe = OneHotEncoder(handle_unknown='ignore')
        cat_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                       'native-country']
        cat_data = self.all_data[cat_columns]
        ohe.fit(cat_data)
        feature_name = ohe.get_feature_names(cat_data.columns)
        return feature_name

    def get_data(self):
        return self.all_data


if __name__ == '__main__':
    test = Adult_Loader(["./src/adult.data"])
    spilt = Dirichlet_Spilt(
        test.get_data().to_numpy(), alpha=10, client_num=10, labels=test.get_data()['race'].to_numpy())
    spilt.get_file_data()
