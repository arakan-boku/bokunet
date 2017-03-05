# -*- coding:utf-8 -*-

import numpy as np
import bokunet as boku
import optims.ada as ada
import optims.adam as adam
import optims.moment as mom
import optims.nest as nes
import optims.rmsprop as rms
import optims.sgd as sgd
import mnist.deformation as de
import pickle
import copy

# ニューラルネットの訓練を行うクラス

class Trainer:

    def __init__(self, network, x_train, t_train, x_test, t_test, epoch=20, batch_size=100, pklname='bokuparams.pkl'):
        
        # 学習結果を保存するピックルファイルの名前
        self.pkl_file_name = pklname
        
        # 利用するネットワーク 呼び出し側で初期化して渡される
        self.network = network
        
        # 保存するテストの評価結果を保持する
        self.test_acc_max = 0
        
        # 学習済パラメータ保存
        self.pklsave = True
        
        # optimzer
        self.optimizer_class = 'adagrad'
        
    	# 学習状況を出力するか否か。Trueなら出力する。
        self.verbose = True
        
        # エポック数（学習で訓練データを使い切ったときが1エポック）
        # バッチサイズ100で、10000枚のデータなら、100回で1エポック
        self.epoch = epoch
        
        # バッチサイズ
        self.batch_size = batch_size
        self.batch_size_w = batch_size
        
        # 学習データ
        self.x_train = x_train
        
        # 学習データ　正解ラベル
        self.t_train = t_train
        
        # 正解を確認するためのテストデータ
        self.x_test = x_test
        
        # 正解を確認するときの正解ラベル
        self.t_test = t_test
        
        # self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        
        # optimzer
        if str(self.optimizer_class).lower() in ['sgd','s']:
            self.optimizer = sgd.SGD(lr=0.01)
        elif str(self.optimizer_class).lower() in ['momentum','m']:
            self.optimizer = mom.Momentum(lr=0.01, momentum=0.9)
        elif str(self.optimizer_class).lower() in ['nesterov','n']:
            self.optimizer = nest.Nesterov(lr=0.01, momentum=0.9)
        elif str(self.optimizer_class).lower() in ['adam','a']:
            self.optimizer = adam.Adam(lr=0.001, beta1=0.9, beta2=0.999)
        elif str(self.optimizer_class).lower() in ['rmsprpo','r']:
            self.optimizer = rms.RMSprop(lr=0.01, decay_rate = 0.99)
        elif str(self.optimizer_class).lower() in ['adagrad','g']:
            self.optimizer = ada.AdaGrad(lr=0.01)
        else:
            self.optimizer = sgd.SGD(lr=0.01)
            
        # optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
        #                         'addgrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        # self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / self.batch_size, 1)
        self.max_iter = int(self.epoch * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # バッチマスクの設定
        batch_mask = np.random.choice(self.train_size, self.batch_size_w )
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
   
        
        # 勾配
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        #if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
                
            #train_acc = self.network.accuracy(self.x_train, self.t_train)
            self.test_acc_max = self.network.accuracy(self.x_test, self.t_test)           
            
            # 後でグラフ等を書く場合に利用する。テスト確認用のリスト
            #self.train_acc_list.append(train_acc)
            self.test_acc_list.append(self.test_acc_max)
            
            #if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", test acc:" + str(self.test_acc_max) + " ===")
        self.current_iter += 1

    def train(self):
        self.test_acc_max = self.network.accuracy(self.x_test, self.t_test)
        if self.verbose: print("=== epoch:0" + ", test acc:" + str(self.test_acc_max) + " ===")
        
        for i in range(self.max_iter):
            self.train_step()
            
        save_last_params = copy.deepcopy(self.network.params)
        self.test_acc_max = self.network.accuracy(self.x_test, self.t_test)
        

        if self.pklsave:
            with open(self.pkl_file_name,'wb') as f:
                pickle.dump(self.network.params,f)
                
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(self.test_acc_max))
