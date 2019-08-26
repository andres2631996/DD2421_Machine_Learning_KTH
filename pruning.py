# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:14:08 2019

@author: Andrés
"""

import dtree


def pruning(input_tree,validation):
    error=1
    aux=error
    err=list()
    while aux<=error:
        aux=error
        alt=dtree.allPruned(input_tree)
        for i in range(len(alt)):
            err[i]=1-dtree.check(alt[i], validation)
        error=min(err)
        ind=err.index(min(err))
        input_tree=alt[ind]
    return error