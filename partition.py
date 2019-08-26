# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:52:08 2019

@author: Andrés
"""

import random
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]