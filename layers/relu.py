# -*- coding:utf-8 -*-

import numpy as np

#　活性化関数
#　0以上なら値をそのまま保持
#　0以下なら0に置き換える。
#　スイッチのように働く
#-------------------------------------------
#classify
#x：Numpy配列
#out[self.mask] = 0は、配列の添字ではなく、Trueの時だけ更新するIF文的に働く
#--------------------------------------------
#learn
#dout：Numpy配列。classifyのアウトプットを想定（逆伝播）
#dout[self.mask] = 0は、配列の添字ではなく、Trueの時だけ更新するIF文的に働く
#--------------------------------------------

class ReLu:
    def __init__(self):
        self.mask = None

    def classify(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def learn(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

