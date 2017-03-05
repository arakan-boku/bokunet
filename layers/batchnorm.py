# -*- codeing:utf-8 -*-

import numpy as np

#　学習時にバッチサイズごとに正規化を行う。
#　入力データに対して、平均と分散を求めて、平均0、分散1になるように調整する。
#　（正規分布に近似させるということかな？）
#　そのあとで固有のスケールでシフト変換を行う。
#　利用する時は、Affineレイアの後続におく。
#　この出力をReluレイアのインプットにすることで学習効率を向上させる

class BatchNormalization:
  
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # learn時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def classify(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.transpose(1, 0, 2, 3).reshape(C, -1) 

        out = self.__classify(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __classify(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def learn(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.transpose(1, 0, 2, 3).reshape(C, -1) 

        dx = self.__learn(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __learn(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
