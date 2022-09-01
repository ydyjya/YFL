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
import numpy as np
from numpy.random import default_rng


class Data_Spilt(object):
    def __init__(self, data, client_num: int):
        self.data = data
        self.client_num = client_num
        self.data_spilt = self.spilt_data()
        self.data_spilt_list = self.get_data_piece()

    def spilt_data(self):
        print("please rewrite this function")
        spilt = {"data_1": []}
        return spilt

    def get_data_piece(self):
        data_list = []
        for i in range(self.client_num):
            data_list.append([i])
        return data_list

    def get_file_data(self):
        for i in range(self.client_num):
            with open(r"./src/data_piece_%d.csv" % i, "w+") as data_file:
                data = ""
                for _ in self.data_spilt_list[i]:
                    print(_)
                    data += str(_)
                data_file.write(data)


class Dirichlet_Spilt(Data_Spilt):
    def __init__(self, data, client_num, alpha, labels):
        self.alpha = alpha
        self.labels = labels
        self.rng = default_rng()
        super().__init__(data, client_num)

    def spilt_data(self):
        n_classes = []
        set_labels = set(self.labels)
        for i, j in enumerate(set_labels):
            n_classes.append(j)
        train_labels_np = np.array(self.labels)
        label_distribution = self.rng.dirichlet([self.alpha] * self.client_num, len(n_classes))

        class_idcs = [np.argwhere(train_labels_np == y) for y in n_classes]

        client_idcs = [[] for _ in range(self.client_num)]

        for c, fracs in zip(class_idcs, label_distribution):
            # np.split按照比例将类别为k的样本划分为了N个子集
            # for i, idcs 为遍历第i个client对应样本集合的索引
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs

    def get_data_piece(self):
        client_data_list = [None] * self.client_num
        for i, r in enumerate(self.data_spilt):
            data = []
            for idx in r:
                data.append(self.data[idx])
            client_data_list[i] = data
        return client_data_list

    def get_file_data(self):
        for i in range(self.client_num):
            with open(r"./src/data_piece_%d.csv" % i, "w+") as data_file:
                data = ""
                for _ in self.data_spilt_list[i]:
                    tmp = ""
                    for item in _[0]:
                        if type(item) == type("test"):
                            tmp += (item.replace("\'", "") + ",")
                        else:
                            tmp += (str(item) + ",")
                    data += (str(tmp[:-1]) + "\n")
                data_file.write(data)
