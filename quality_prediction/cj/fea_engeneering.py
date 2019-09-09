# -*- coding: utf-8 -*-
# @Time         : 2019/9/6 2:25 PM
# @Author       : Chenjie
# @File         : fea_engeneering.py
# @Description  : 
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

train = pd.read_csv('./first_round_training_data.csv')
test = pd.read_csv('./first_round_testing_data.csv')
submit = pd.read_csv('./submit_example.csv')
save_path = './w2v/'
# data = train.append(test).reset_index(drop=False)
dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
train['label'] = train['Quality_label'].map(dit)
train.drop(['Quality_label'], axis=1, inplace=True)
# print(data.info())

fea_list = [i for i in test.columns if i not in ['Group']]
tmp_list = [i for i in train.columns if i not in fea_list]
print(fea_list)
tmp_test_data = train
tmp_train_data = test
others = train[['label']]
# print(others.info())

for i in tmp_list:
    # print(i)
    others[i] = train[[i]]
tmp_test_data.drop(tmp_list, axis=1,inplace=True)
tmp_train_data = tmp_train_data.rename(columns={'Group':'Group1'})
others['Group1']=tmp_train_data[['Group1']]
tmp_train_data.drop(['Group1'],axis=1,inplace=True)
print(others.head())

print(tmp_train_data.info(), tmp_test_data.info())

cluster = KMeans(n_clusters=119)
cluster = cluster.fit(tmp_train_data)
train.insert(0,'Group', cluster.labels_)
# train['Group'] = cluster.labels_
print(train.info())


max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
fea_list.append('Group')
# for i in ['Quality_label', 'Group', 'label']:
#     fea_list.append(i)
for i in fea_list:
    train[[i]] =train[[i]].apply(max_min_scaler)


for i in fea_list:
    test[[i]] = test[[i]].apply(max_min_scaler)

train['label'] = others['label']
train['Group1'] = others['Group1']

# for i in intmp_list:
# #     try :
# #         test[[i]] = others[[i]]
# #     except KeyError:
# #         pass

print(train.info())
print(test.info())

train.to_csv("./fea_training_data.csv")
test.to_csv("./fea_testing_data.csv")