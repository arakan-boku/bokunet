# -*- codeing:utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
import unittest
import numpy as np
import mnist.getdata as gd
import bokunet as ai
import trainer as tr
#import matplotlib.pyplot as plt

class TestLayer(unittest.TestCase):

    def test_train(self):
        (x_test,t_test) = gd.get_normed_testdata()

        # 件数を絞りたいときに使う。以下ならテストデータをランダムに100件だけとりだす。
        #(x_test,t_test) = gd.get_normed_testdata_choiced(100)
        
        (x_train,t_train) = gd.get_normed_traindata()
        
        # 件数を絞りたいときに使う。以下ならテストデータをランダムに1000件だけとりだす。
        #(x_train,t_train) = gd.get_normed_traindata_choiced(1000)
        
        # 動かすネットワークを構築する
        network = ai.BokuNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],output_size=10,dropout=False,batchnorm=True,pklname='bokuparams.pkl')

        # ネットワークを使って学習する。epochは学習の実行単位のこと。テストデータが10000件で、batch_sizeが100なら、100回で1epochになる。
        trainer= tr.Trainer(network, x_train, t_train, x_test, t_test,epoch=10, batch_size=100,pklname='bokuparams.pkl')
        trainer.train()
        
        # 学習済データで推論だけ試したいときは、上の2行をコメントアウトして、こちらを活かす
        #acc = network.acc(x_train,t_train)
        #print("----- acc=" + str(acc) + " -----")


suite = unittest.TestLoader().loadTestsFromTestCase(TestLayer)
unittest.TextTestRunner(verbosity=2).run(suite)
