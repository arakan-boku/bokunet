# -*- coding:utf-8 -*-

import numpy as np

#  アフィン変換を行う

#  行列の内積を求めるのを幾何学の分野ではそう呼ぶらしいので。
#
#　self.original_x_shape：
#　  ３次元以上の配列のとき、learnで戻すために保存する。
#
#  x = x.reshape(x.shape[0], -1):
#　  -1を指定すると、３次元以上の配列を２次元に展開してくれる        
#　  これを行うことで、後ろの処理でループを回す必要がなくなる
#
#  dx = np.dot(dout, self.W.T):
#       doutに対して演算する時、Wは転置しないと行列数不一致エラーになる
#       doutに対して演算する時、self.xは転置しないと行列数不一致エラーになる

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def classify(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def learn(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx
    

