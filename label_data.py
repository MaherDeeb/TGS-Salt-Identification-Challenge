# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:18:28 2018

@author: Maher Deeb
"""

#label data as 0 or 1
#0 it has no salt
#1 it has salt

from ETL import ETL_data_loading
from submission import _reshape_image
import numpy as np


def contain_salts(train_ids, train_x,train_y):
    train_ids_L = []
    train_y_f = np.zeros((len(train_y)))
    for count_image in  range(len(train_y)):
        train_y_f[count_image] = (np.sum(_reshape_image(train_y[count_image])) > 0).astype('bool')
        if train_y_f[count_image]  == 1:
            train_ids_L.append(train_ids[count_image])
    train_x = train_x[train_y_f==1]
    train_y = train_y[train_y_f==1]
    return train_ids_L, train_x, train_y