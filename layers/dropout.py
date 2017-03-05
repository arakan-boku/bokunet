# -*- codeing:utf-8 -*-

import numpy as np

#　ニューラルネットを間引く
#　学習時：ランダムにネットワークを選んで指定したレシオより大きいものだけ
#　を通す。
#　学習時以外は、学習時消去した割合を乗算して出力する。
#
class Dropout:

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def classify(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def learn(self, dout):
        return dout * self.mask
