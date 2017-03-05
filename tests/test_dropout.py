# -*- codeing:utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
import unittest
import numpy as np
import matplotlib.pyplot as plt
import layers.dropout as af

class TestLayer(unittest.TestCase):
    x = np.array([[1 ,2 ,3],])
    #w = np.array([[0.1 ,0.1],[0.2 ,0.2],[0.3 ,0.3]])
       
    so = af.Dropout(0.6)
    y1 = so.forward(x)
    print(y1)
    y2 = so.backward(y1)
    print(y2)
    #plt.plot(x,y1,label='forward')
    #plt.plot(x,y2,linestyle="--",label = 'backward')
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.title('ReLu')
    #plt.legend()
    #plt.show()


suite = unittest.TestLoader().loadTestsFromTestCase(TestLayer)
unittest.TextTestRunner(verbosity=2).run(suite)
