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
                imdb_df = imdb_df.append([[review, labels[label]]])
    return imdb_df

imdb_train = load_imdb('./aclImdb', 'train')
imdb_test = load_imdb('./aclImdb', 'test')

imdb = imdb_train.append(imdb_test, ignore_index=None)
imdb.columns = ['评论', '情感']
imdb.to_csv('./imdb11.csv', index=None)














