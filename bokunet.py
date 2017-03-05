# -*- coding:utf-8 -*-

import numpy as np
from collections import OrderedDict
import layers.affine as af
import layers.batchnorm as bn
import layers.relu as re
import layers.sigmoid as si
import layers.softmax_loss as sl
import layers.dropout as dr
import pickle

# 全結合による多層ニューラルネットワーク

class BokuNet:
    def __init__(self, input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],output_size=10,dropout=False,batchnorm=True,pklname='bokuparams.pkl'):    
        # 学習結果保存用ピックルファイル名
        self.pkl_file_name = pklname

        # 前回の学習結果を読み込むか否か
        self.useload = False
        
        # 入力サイズ（MNISTの場合は784）
        self.input_size = input_size
        
        # 出力サイズ（MNISTの場合は10）
        self.output_size = output_size
        
        # 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
        self.hidden_size_list = hidden_size_list
        
        # 隠れ層の数
        self.hidden_layer_num = len(hidden_size_list)
        
        # Dropout を使うかどうか（通常は　True）
        self.use_dropout = dropout
        
        # Dropoutの割り合い(どの程度間引くか）
        self.dropout_ration = 0.5
        
        # Weight Decay（L2ノルム）の強さ
        # 過学習防止のためすべての損失関数に　0.5 * lambda * W**2 を加算する
        self.weight_decay_lambda = 0.05
        
        # Batch Normalizationを使用するかどうか（通常はTrue)
        self.use_batchnorm = batchnorm
        
        # 重みの初期値を計算する方法の指定
        # 'relu'または'he'を指定した場合は「Heの初期値」を設定
        # 'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        self.weight_init_std ='relu'
        
        # 活性化関数に何を使うか relu　または　rで　Relu　以外は　sigmoid
        self.activation = 'relu'
        
       
        if self.useload:
            with open(self.pkl_file_name,'rb') as f:
                self.params = pickle.load(f)
        else:
            self.params = {}
            self.__init_weight(self.weight_init_std)
        
        # レイヤの生成
        #activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            
            # Affineレイヤ
            self.layers['Affine' + str(idx)] = af.Affine(self.params['W' + str(idx)],self.params['b' + str(idx)])
            
            # BatchNormレイヤ　（AffineとReluの間にはさむ）
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = bn.BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
                
            # 活性化関数レイヤ
            
            if str(self.activation).lower() in ['relu','r']:
                self.layers['Activation_function' + str(idx)] = re.ReLu()
            else:
                self.layers['Activation_function' + str(idx)] = si.Sigmoid()
            
            # Dropoutレイヤ
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = dr.Dropout(self.dropout_ration)

        idx = self.hidden_layer_num + 1
        
        # 最終レイヤの設定　Affine　→　SoftMax
        self.layers['Affine' + str(idx)] = af.Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = sl.SoftmaxLoss()
        

    # 重みの初期値設定
    def __init_weight(self, weight_init_std):        
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            #scale = weight_init_std
            
            if str(weight_init_std).lower() in ('relu', 'he','r'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            else:
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値
                
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    # 推論部　Dropout　と　BatchNormの場合は、トレーニングフラグが必要
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.classify(x, train_flg)
            else:
                x = layer.classify(x)
                
        return x
    

    # 損失関数の結果を求める。最終レイヤのclassifyで同時に損失関数　交差エントロピー関数を処理している
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)        
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
            
        return self.last_layer.classify(y, t) + weight_decay

    # 推論部の一致率を計算して返す
    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1 : T = np.argmax(T, axis=1)
        
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    # 外部ウエイトで動作確認    
    def acc(self,X,T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1 : T = np.argmax(T, axis=1)
        
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy        

    # 勾配    
    def gradient(self, x, t):
        # classify
        self.loss(x, t, train_flg=True)
        
        # learn
        dout = 1
        dout = self.last_layer.learn(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.learn(dout)
            
        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
                
        return grads
    

    # ---------------------  ここから下はおそらく使わない　---------------------------
    # optimizer を使わない場合に簡易的に勾配法で学習するためのもの
    # --------------------------------------------------------------------------------
    def numerical_gradient(self, X, T):
        """勾配を求める（数値微分）

        Parameters
        ----------
        X : 入力データ
        T : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
