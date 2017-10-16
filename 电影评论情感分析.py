# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:14:06 2017

@author: lx
'(?::|;|=)(?:-)?(?:\)|\(|D|P)'
"""

# Python对象的持久化
"""
import pickle
# 将对象保存到文件
pickle.dump(obj, f)
# f必须以二进制写进行打开
obj = pickle.load(f)
# f必须以二进制读进行打开

# 使用sklearn内部支持模块保存模型
from sklearn.externals import joblib
joblib.dump(obj, 'file_name')
obj = joblib.load('file_name')

"""

import os
import re
import numpy as np
import pandas as pd

def load_imdb(path, typ='train'):
    """
    处理数据得到程序可读格式
    """
    imdb_df = pd.DataFrame()
    labels = {'pos':1, 'neg':0}
    for label in labels.keys():
        label_path = '%s/%s/%s' % (path, typ, label)
        for file in os.listdir(label_path):
            with open(os.path.join(label_path, file), 'r', encoding='utf-8') as f:
                review = f.read()
            imdb_df = imdb_df.append([[review, labels[label]]], ignore_index=True)
    return imdb_df
"""
imdb_train = load_imdb('./imdb/aclImdb', 'train')
imdb_test = load_imdb('./imdb/aclImdb', 'test')

imdb = imdb_train.append(imdb_test, ignore_index=True)
imdb.columns = ['评论', '情感']
imdb.to_csv('./imdb/imdb.csv', index=None)
"""
# 读取数据集
imdb = pd.read_csv('imdb.csv', encoding='gbk')

# 词袋模型
from sklearn.feature_extraction.text import CountVectorizer
docs = ['xiaoy is xiaoy', 'is fine', 'hello xiaoy']
count = CountVectorizer()
bag = count.fit_transform(docs)
print(bag.toarray())

"""
词频：tf(t,d)为词汇 t 在文档 d 中出现的次数
逆文档频率：idf(t,d) = log(nd/(1+df(d,t))，nd为文档总数，df(d,t)为包含词汇 t 的文档 d 的数量
tf-idf(t,d)=tf(t,d)*idf(t,d) 称为词频-逆文档频率 （注意：不是词频减逆文档频率的意思）
***（词频-逆文档频率）越高，具备更好的类别区分能力

# is的词频-逆文档频率
tf = 1
idf = log(3/(1+2)) = 0
tf_idf = tf*idf = 0
"""
# 计算词频-逆文档频率
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
print(tfidf.fit_transform(bag).toarray())

def clear_review(text):
    """
    评论内容的清洗
    """
    text = re.sub('\<[^\>]*\>', '', text)
     # 匹配表情，(?:不是代表分组匹配，其中的()不是代表元组匹配
    bq = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(bq)
    return text

imdb['评论'] = imdb['评论'].apply(clear_review)

def doc_split(text):
    """
    拆分文档
    """
    return text.split()   # 默认空格分隔

from nltk.stem.porter import PorterStemmer
def doc_split_atom(text):
    words = []
    porter = PorterStemmer()
    for word in text.split():
        words.append(porter.stem(word))
    return words

#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')   # 英文常用停止词集


X_train = imdb.iloc[:25000, 0]
y_train = imdb.iloc[:25000, 1]
X_test = imdb.iloc[25000:, 0]
y_test = imdb.iloc[25000:, 1]

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer # 
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

# 流水线
pipe_param = [('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)), \
               ('lr', LogisticRegression(random_state=0))]
# 流水
pipe_lr = Pipeline(pipe_param)

param_grid = [{'vect__ngram_range':[(1,1)], 'vect__stop_words':[stop, None], 'vect__tokenizer':[doc_split, doc_split_atom], 'lr__penalty':['l1', 'l2'], 'lr__C':[0.1, 1.0, 10, 100]}, \
                                    {'vect__use_idf':[False], 'vect__norm':[None], 'lr__penalty':['l1', 'l2'], 'lr__C':[0.1, 1.0, 10, 100]}]

gscv = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, cv=3)

gscv.fit(X_train, y_train)

estimator = gscv.best_estimator_
print(gscv.best_params_)












