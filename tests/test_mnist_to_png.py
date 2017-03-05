# -*- codeing:utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
import unittest
import numpy as np
import mnist.to_png as png
#import matplotlib.pyplot as plt

class TestLayer(unittest.TestCase):
    def test_from_train(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        SIZE = 10
        png.save_train(SIZE,OUTPUT)
    
    def test_from_test(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        SIZE = 10
        png.save_test(SIZE,OUTPUT)

suite = unittest.TestLoader().loadTestsFromTestCase(TestLayer)
unittest.TextTestRunner(verbosity=2).run(suite)
