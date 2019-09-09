# 预处理复赛数据
import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import warnings
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

train = pd.read_csv('./first_round_training_data.csv')
test = pd.read_csv('./first_round_testing_data.csv')
submit = pd.read_csv('./submit_example.csv')
save_path = './w2v'
data = train.append(test).reset_index(drop=False)
dit = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
data['label'] = data['Quality_label'].map(dit)
print(data.info())

fea_list = [i for i in test.columns if i not in ['Group']]
print(fea_list)
cat_list = []
lbl = LabelEncoder()
for i in tqdm(fea_list):
    #     print(i,'{}'.format(i))
    if train['{}'.format(i)].nunique() < 6000:
        cat_list.append('{}'.format(i))

print(cat_list)

feature_name = [i for i in data.columns if
                i not in ['Attribute1', 'Attribute10', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5',
                          'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Quality_label', 'Group', 'label']]
tr_index = ~data['label'].isnull()
X_train = data[tr_index][feature_name].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(str)
X_test = data[~tr_index][feature_name].reset_index(drop=True)
X_train_size = len(X_train)
data = pd.concat([X_train, X_test])

print(X_train.shape, X_test.shape)
sentence = []
for line in list(data[['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
                       'Parameter8', 'Parameter9', 'Parameter10']].values):
    sentence.append([str(float(l)) for idx, l in enumerate(line)])

print('training...')
model = Word2Vec(sentence, size=10, window=2, min_count=1, workers=multiprocessing.cpu_count(),
                 iter=10)
print('outputing...')
for fea in ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5', 'Parameter6', 'Parameter7',
            'Parameter8', 'Parameter9', 'Parameter10']:
    values = []
    for line in list(data[fea].values):
        values.append(line)
    values = set(values)
    print(len(values))
    w2v = []
    for i in values:
        a = [i]
        a.extend(model[str(float(i))])
        w2v.append(a)
    out_df = pd.DataFrame(w2v)

    name = [fea]
    for i in range(10):
        name.append(name[0] + 'W' + str(i))
    out_df.columns = name
    out_df.to_csv(save_path + '/' + fea + '.csv', index=False)