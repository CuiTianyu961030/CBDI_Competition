import pandas as pd
import numpy as np
import os
import warnings
import time
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm, tqdm_notebook
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import math
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
# from gensim.models import Word2Vec
import json
import gc
import re
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

train = pd.read_csv('../data/first_round_training_data.csv')
test = pd.read_csv('../data/first_round_testing_data.csv')

features = [c for c in test.columns if c!='Group']
cat_feats = ['Attribute4',
'Attribute5',
'Attribute6',
'Attribute7',
'Attribute8',
'Attribute9',
'Attribute10',
'Parameter5',
'Parameter6',
'Parameter7',
'Parameter8',
'Parameter9',
'Parameter10']

use_cate = [c for c in cat_feats if 'Para' in c]  #使用到的6个变量
col_only_train = [c for c in test.columns if c != 'Group']  #训练集中的所有变量
all_feat = [c for c in train.columns if c != 'Quality_label']  #数据的所有特征
data = pd.concat([train,test])

def nnq_encode(data,en_col,use_col):
    data[en_col + '_nnq_of_' + use_col] = data[en_col].map( data.groupby([en_col])[use_col].nunique() )
    features.append( en_col + '_nnq_of_' + use_col )
    return data


classMap = {'Excellent':0,
'Good':1,
'Pass':2,
'Fail':3,}
tr_index = ~data.Quality_label.isnull()
train_df = data[tr_index][use_cate +['Quality_label']].reset_index(drop=True)
train_df['Quality_label'] = train_df['Quality_label'].map(classMap)

test_df = data[~tr_index].reset_index(drop=True)
id_test = test_df.Group.values


X_train = train_df[use_cate]    
y = train_df.Quality_label
X_test = test_df[use_cate]
del train,test
del train_df,test_df
gc.collect()


lgb_paras = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'learning_rate': 0.01,
        'num_leaves': 32,
        # 'lambda_l1': 0.01,
        # 'lambda_l2': 10,
        'num_class': 4,
        'max_depth': -1,
        'seed': 2019,
        'feature_fraction': 0.8,
        # 'device': 'gpu',
                        #         'bagging_fraction': 0.8,
                                #         'bagging_freq': 4,
        'verbose': 1
        }
from sklearn.metrics import f1_score, confusion_matrix
all_preads = []
skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
def f1_weighted(preds, train_data):
    y_true = train_data.label
    preds = np.argmax(preds.reshape(4, -1), axis=0)
    score = f1_score(y_true, preds, average='weighted')
    return 'f1_weighted', score, True
evals_result = {}
for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
   train_x, test_x, train_y, test_y = X_train[use_cate].iloc[train_index], X_train[use_cate].iloc[test_index], y.iloc[train_index], y.iloc[test_index]
   dtrain = lgb.Dataset(train_x, train_y)

   dvalid = lgb.Dataset(test_x, test_y)

   lgb_modelall = lgb.train(lgb_paras, dtrain,
                            valid_sets=[dtrain, dvalid],
                            num_boost_round=1000,
                            early_stopping_rounds=100,
                            valid_names=["train", "valid"],
                            evals_result=evals_result,
                            verbose_eval=50,
                            feval=f1_weighted)
   pred = lgb_modelall.predict(X_test)
   all_preads.append( pred )

mean_pread = np.mean(all_preads,axis=0)
cols = ['Excellent ratio','Good ratio','Pass ratio','Fail ratio']

sub_prob  = pd.DataFrame(mean_pread,columns=cols)
len(sub_prob),len(id_test)
sub_prob['Group'] = id_test

sub_prob['Group'] = sub_prob['Group'].map(int)
sub_prob = sub_prob.groupby([ 'Group'])[cols].median().reset_index()

sub_prob.to_csv('../logs/lgb_second.csv',index=False)
