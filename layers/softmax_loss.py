# -*- codeing:utf-8 -*-

import numpy as np

class SoftmaxLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def classify(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        # オーバーフロー対策をしない方を試したいときだけ
        #self.y = self.softmax_no_of(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        
        return self.loss
    

    def learn(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            
        return dx
    

    #　損失関数
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        
        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
             
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


    def softmax(self ,x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x) # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x))


    # 入力の正規化を期待して、オーバーフロー対策をしない
    def softmax_no_of(self ,x):
        if x.ndim == 2:
            x = x.T
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        return np.exp(x) / np.sum(np.exp(x))
        

