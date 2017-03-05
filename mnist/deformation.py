# -*- coding:utf-8 -*-

import numpy as np

#画像変換前の配列データを変形する。


def rotate90(x,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for j in range(0,w):
        for i in reversed(range(0,w)):
            outb = w * i + j
            outwk[outb] = x[b]
            b = b + 1
 
    out = np.array(outwk)
    return out


def rotate180(x,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for i in reversed(range(0,w)):
        for j in reversed(range(0,w)):
            outb = w * i + j
            outwk[outb] = x[b]
            b = b + 1
 
    out = np.array(outwk)
    return out


def rotate270(x,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for j in reversed(range(0,w)):
        for i in range(0,w):
            outb = w * i + j
            outwk[outb] = x[b]
            b = b + 1
 
    out = np.array(outwk)
    return out


def right(x,n,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for b in range(0,size):
        a = b % w
        if( a < (w - n)):
            outb = b + n
            outwk[outb] = x[b]
 
    out = np.array(outwk)
    return out


def left(x,n,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for b in range(0,size):
        a = b % w
        if( a >= n):
            outb = b - n
            outwk[outb] = x[b]
 
    out = np.array(outwk)
    return out


def upto(x,n,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for b in range(0,size):
        if( b >= (w * n)):
            outb = b - (w * n)
            outwk[outb] = x[b]
 
    out = np.array(outwk)
    return out


def downto(x,n,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for b in range(0,size):
        if( b <= (size - 1 - (w * n))):
            outb = b + (w * n)
            outwk[outb] = x[b]
 
    out = np.array(outwk)
    return out

def noise(x,n,w=28):
    size = len(x)
    outwk = [0]*size
    b = 0
    for b in range(0,size):
        a = (b % w) % n
        if( a == 0):
            outwk[b] = 0
        else:
            outwk[b] = x[b]
 
    out = np.array(outwk)
    return out


