# -*- codeing:utf-8 -*-

import sys
import os
sys.path.append(os.pardir)
import unittest
import numpy as np
import mnist.deformation as de
import mnist.getdata as gd
import mnist.to_png as png
#import matplotlib.pyplot as plt

class TestLayer(unittest.TestCase):

    def test_rotate90(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        y = de.rotate90(x[0],28)
        png.to_png(y,OUTPUT + "test90",28,28)
        print(x)
        print(y)
        
    def test_rotate180(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        y = de.rotate180(x[0],28)
        png.to_png(y,OUTPUT + "test180",28,28)
        print(x)
        print(y)
        
    def test_rotate270(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        y = de.rotate270(x[0],28)
        png.to_png(y,OUTPUT + "test270",28,28)
        print(x)
        print(y)

    def test_left(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        y = de.left(x[0],4,28)
        png.to_png(y,OUTPUT + "left",28,28)
        print(x)
        print(y)
    

    def test_right(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        y = de.right(x[0],4,28)
        png.to_png(y,OUTPUT + "right",28,28)
        print(x)
        print(y)
            

    def test_upto(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        y = de.upto(x[0],2,28)
        png.to_png(y,OUTPUT + "upto",28,28)
        print(x)
        print(y)
        

    def test_downto(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        y = de.downto(x[0],2,28)
        png.to_png(y,OUTPUT + "downto",28,28)
        print(x)
        print(y)

    def test_mix_fig(self):
        (x,t) = gd.get_normed_traindata_choiced(10)
        for i in range(len(x)):
            y = de.downto(x[0],2,28)
            print(y)

    def test_all(self):
        OUTPUT = "C:\\myWork\\00_python\\output\\tmp\\unittest\\"
        (x,t) = gd.get_testdata_choiced(1)
        png.to_png(x,OUTPUT + "orign02",28,28)
        y1 = de.downto(x[0],4,28)
        png.to_png(y1,OUTPUT + "downto02",28,28)
        y2 = de.upto(x[0],4,28)
        png.to_png(y2,OUTPUT + "upto02",28,28)
        y3 = de.left(x[0],4,28)
        png.to_png(y3,OUTPUT + "left02",28,28)
        y4 = de.right(x[0],4,28)
        png.to_png(y4,OUTPUT + "right02",28,28)
        y5 = de.rotate90(x[0],28)
        png.to_png(y5,OUTPUT + "rotate90_02",28,28)
        y6 = de.rotate180(x[0],28)
        png.to_png(y6,OUTPUT + "rotate180_02",28,28)
        y7 = de.rotate270(x[0],28)
        png.to_png(y7,OUTPUT + "rotate270_02",28,28)
        y8 = de.noise(x[0],4,28)
        png.to_png(y8,OUTPUT + "noise02",28,28)

suite = unittest.TestLoader().loadTestsFromTestCase(TestLayer)
unittest.TextTestRunner(verbosity=2).run(suite)
