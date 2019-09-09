# !pip install catboost==0.15.2
# !pip install tqdm
# !pip install xlrd
import numpy as np
import pandas as pd
import catboost as cbt#据说性能超过lgb和xgboost
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss
import gc
import math
import time
from tqdm import tqdm
import datetime
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import warnings
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

train = pd.read_csv('./fea_training_data.csv')
test = pd.read_csv('./fea_testing_data.csv')
submit = pd.read_csv('./submit_example.csv')
save_path = './w2v/'


fea_list = [i for i in test.columns]
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
fea_list.append('Group')
for i in fea_list:
    train[[i]] =train[[i]].apply(max_min_scaler)

data = train.append(test).reset_index(drop=False)
# dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
# data['label'] = data['Quality_label'].map(dit)
print(data.info())

fea_list = [i for i in test.columns if i not in ['Group','Group1']]
cat_list = []
lbl = LabelEncoder()
for i in tqdm(fea_list):
#     print(i,'{}'.format(i))
    if train['{}'.format(i)].nunique() < 6000:
        cat_list.append('{}'.format(i))

print(cat_list)

feature_name = [i for i in data.columns if i not in ['Attribute1','Attribute10', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5',
 'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'label']]
tr_index = ~data['label'].isnull()
X_train = data[tr_index][feature_name].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(str)
X_test = data[~tr_index][feature_name].reset_index(drop=True)
X_train_size = len(X_train)
data = pd.concat([X_train, X_test])


w2v_features = []
for col in ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
                       'Parameter8', 'Parameter9', 'Parameter10']:
    df = pd.read_csv('./w2v/' + col + '.csv')
    df = df.drop_duplicates([col])
    fs = list(df)
    fs.remove(col)
    w2v_features += fs
    data = pd.merge(data, df, on=col, how='left')
    print(len(data))
print(w2v_features)
# data.drop(['A'])

X_train = data[:X_train_size]
X_test = data[X_train_size:]
print(X_train.shape,X_test.shape)
print(X_train.info(), X_test.info())

# print(X_train.shape,X_test.shape)
X_train = X_train.astype(str)
X_test =X_test.astype(str)
# print(X_train.head())
# print(X_test.head())
# print(y.head())
oof = np.zeros((X_train.shape[0],4))
prediction = np.zeros((X_test.shape[0],4))
seeds = [20190903, 2048 * 2 + 1024, 4096, 2048, 1024]
num_model_seed = 5
for model_seed in range(num_model_seed):
    print(model_seed + 1)
    oof_cat = np.zeros((X_train.shape[0],4))
    prediction_cat=np.zeros((X_test.shape[0],4))
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train.iloc[train_index], X_train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        gc.collect()
        cbt_model = cbt.CatBoostClassifier(iterations=100000,learning_rate=0.001,verbose=300,max_depth=8,#eval_metric='Accuracy',
                                        early_stopping_rounds=1000,task_type='GPU',cat_features=cat_list,
                                        loss_function='MultiClass')
       
        train_x.drop(['index'], axis=1)
        print(train_x.info())
        test_x.drop(['index'], axis=1)
        print(test_x.info())
        cbt_model.fit(train_x, train_y ,eval_set=(test_x,test_y))
        oof_cat[test_index] += cbt_model.predict_proba(test_x)
        prediction_cat += cbt_model.predict_proba(X_test)/5
        gc.collect()
    oof += oof_cat / num_model_seed
    prediction += prediction_cat / num_model_seed
    print('logloss',log_loss(pd.get_dummies(y).values, oof_cat))
    print('ac',accuracy_score(y, np.argmax(oof_cat,axis=1)))
print('logloss',log_loss(pd.get_dummies(y).values, oof))
print('ac',accuracy_score(y, np.argmax(oof,axis=1)))

sub = test[['Group']]
prob_cols = [i for i in submit.columns if i not in ['Group']]
for i, f in enumerate(prob_cols):
    sub[f] = prediction[:, i]
for i in prob_cols:
    sub[i] = sub.groupby('Group')[i].transform('mean')
sub = sub.drop_duplicates()
sub['Group'] = train['Group1']

sub.to_csv("./submission.csv",index=False)