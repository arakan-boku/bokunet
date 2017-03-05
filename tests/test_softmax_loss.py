# -*- codeing:utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
import unittest
import numpy as np
import layers.softmax_loss as sl
#import matplotlib.pyplot as plt

class TestLayer(unittest.TestCase):

    def test_softmax_loss(self):
        x = np.array([0.1 ,0.2 ,0.7])
        t = np.array([0 ,0 ,1])
        swl = sl.SoftmaxLoss()
        y = swl.forward(x,t)
        print(y)
            

suite = unittest.TestLoader().loadTestsFromTestCase(TestLayer)
unittest.TextTestRunner(verbosity=2).run(suite)
