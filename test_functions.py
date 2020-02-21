#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:37:36 2020

@author: macbookthibaultlahire
"""

import numpy as np
import math

def griewank(xs):
    summ = 0
    for x in xs:
        summ += x * x
    product = 1
    for i in range(len(xs)):
        product *= math.cos(xs[i] / math.sqrt(i + 1))
    return 1 + summ / 4000 - product

def g07(x):
    assert len(x)==10
    aux = x[0]**2 + x[1]**2 +x[0]*x[1] - 14*x[0] - 16*x[1] + (x[2]-10)**2 + 4*(x[3]-5)**2
    aux2 = (x[4]-3)**2 + 2*(x[5]-1)**2 + 5*x[6]**2 + 7*(x[7]-11)**2 + 2*(x[8]-10)**2 + (x[9]-7)**2
    return aux + aux2 + 45

def sinus(x):
    return np.sin(2*np.pi*x)

