# -*- coding: utf-8 -*-
"""
@author: Maher Deeb
"""
import numpy as np

sample_x = np.multiply(np.random.randint(2, size=100),np.array(range(100)))
sample_pre_x = np.multiply(np.random.randint(2, size=100),np.array(range(100)))


def _calculate_scoring(sample_x,sample_pre_x):
    
    TP = np.sum(1*np.subtract(sample_x,sample_pre_x)==0)
    FP = np.sum(1*np.subtract(sample_x,sample_pre_x)<0)
    FN = np.sum(1*np.subtract(sample_x,sample_pre_x)>0)
    if (TP+FP+FN)!=0:
        IoU_t = TP/(TP+FP+FN)
    else:
        IoU_t = 0
        
    return IoU_t

print( _calculate_scoring(sample_x,sample_x))