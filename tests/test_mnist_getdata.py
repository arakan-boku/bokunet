# -*- codeing:utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
import unittest
import numpy as np
import mnist.getdata as gd
#import matplotlib.pyplot as plt

class TestLayer(unittest.TestCase):
    def test_get_test(self):
        (xt,xl) = gd.get_testdata()
        print(xt.shape)
        self.assertEquals(xt.shape[0],10000)
        self.assertEquals(xt.shape[1],784)
        

    def test_get_train(self):
        (xt,xl) = gd.get_traindata()
        print(xt.shape)
        self.assertEquals(xt.shape[0],60000)
        self.assertEquals(xt.shape[1],784)
        

    def test_test_choiced(self):
        (xt,xl) = gd.get_testdata_choiced(10)
        print(xt.shape)
        self.assertEquals(xt.shape[0],10)
        self.assertEquals(xt.shape[1],784)
        

    def test_train_choiced(self):
        (xt,xl) = gd.get_traindata_choiced(10)
        print(xt.shape)
        self.assertEquals(xt.shape[0],10)
        self.assertEquals(xt.shape[1],784)
       
    def test_get_normed_test(self):
        (xt,xl) = gd.get_normed_testdata()
        print(xt[0])
        print(xt.shape)
        self.assertEquals(xt.shape[0],10000)
        self.assertEquals(xt.shape[1],784)
        

    def test_get_normed_train(self):
        (xt,xl) = gd.get_normed_traindata()
        print(xt.shape)
        self.assertEquals(xt.shape[0],60000)
        self.assertEquals(xt.shape[1],784)
        

    def test_normed_test_choiced(self):
        (xt,xl) = gd.get_normed_testdata_choiced(10)
        print(xt.shape)
        self.assertEquals(xt.shape[0],10)
        self.assertEquals(xt.shape[1],784)
        

    def test_normed_train_choiced(self):
        (xt,xl) = gd.get_normed_traindata_choiced(10)
        print(xt.shape)
        self.assertEquals(xt.shape[0],10)
        self.assertEquals(xt.shape[1],784)

suite = unittest.TestLoader().loadTestsFromTestCase(TestLayer)
unittest.TextTestRunner(verbosity=2).run(suite)
