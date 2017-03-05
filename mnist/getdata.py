# -*- coding:utf-8 -*-

import numpy as np
from mnist.mnist import load_mnist

# mnistデータの取得の４パターン    

# 画像ファイルへの変換をする場合に利用する
def get_testdata():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False,one_hot_label=False)
    return x_test,t_test

# 画像ファイルへの変換をする場合に利用する
def get_traindata():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False,one_hot_label=False)
    return x_train,t_train


# 画像ファイルへの変換時に利用。ミニバッチで指定した件数のみをランダムに抽出したものを返す
def get_testdata_choiced(size):
    x,t = get_testdata()
    test_size = x.shape[0]
    batch_mask = np.random.choice(test_size,size)
    x_batch = x[batch_mask]
    t_batch = t[batch_mask]
    return x_batch,t_batch

# 画像ファイルへの変換時に利用。ミニバッチで指定した件数のみをランダムに抽出したものを返す
def get_traindata_choiced(size):
    x,t = get_traindata()
    train_size = x.shape[0]
    batch_mask = np.random.choice(train_size,size)
    x_batch = x[batch_mask]
    t_batch = t[batch_mask]
    return x_batch,t_batch

# 学習・推論処理時に利用する。
# 正規化（0　～　1.0）に変換　と　教師データの正解ラベルのみ1にする変換を行う。
def get_normed_testdata():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_test,t_test


# 学習・推論処理時に利用する。
def get_normed_traindata():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_train,t_train


# 学習・推論処理時に利用する。
# ミニバッチで指定した件数のみをランダムに抽出したものを返す
def get_normed_testdata_choiced(size):
    x,t = get_normed_testdata()
    test_size = x.shape[0]
    batch_mask = np.random.choice(test_size,size)
    x_batch = x[batch_mask]
    t_batch = t[batch_mask]
    return x_batch,t_batch

# 学習・推論処理時に利用する。
def get_normed_traindata_choiced(size):
    x,t = get_normed_traindata()
    train_size = x.shape[0]
    batch_mask = np.random.choice(train_size,size)
    x_batch = x[batch_mask]
    t_batch = t[batch_mask]
    return x_batch,t_batch

    
    


    
          
    
