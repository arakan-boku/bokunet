# -*- coding:utf-8 -*-

import numpy as np

# 活性化関数
# 入力値をなめらかな実数に変換する
#

class Sigmoid:
    def __init__(self):
        self.out = None

    def classify(self, x):
        out = self.sigmoid(x)
        self.out = out
        return out

    def learn(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

