# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:15:58 2018

@author: Maher Deeb
"""

from ETL import ETL_data_loading
import numpy as np


def image_diff (train_x,test_x):
    img_size_target = 128
    padding = True
    padding_type = 'constant'
    
    train_ids, dataframe_depth, train_x, train_y, test_x = \
        ETL_data_loading(img_size_target,False,False,padding, padding_type)

    diff_train_x = np.zeros((len(train_ids), img_size_target-1,
                                    img_size_target-1, 1), dtype=np.uint8)
    diff_train_x_p = np.zeros((len(train_ids), img_size_target,
                                    img_size_target, 1), dtype=np.uint8)
    
    diff_test_x = np.zeros((len(test_x), img_size_target-1,
                                    img_size_target-1, 1), dtype=np.uint8)
    diff_test_x_p = np.zeros((len(test_x), img_size_target,
                                    img_size_target, 1), dtype=np.uint8)
    for count_image in range(len(train_x)):
        
        diff_train_x[count_image] = train_x[count_image,1:128,1:128,:] - train_x[count_image,0:127,0:127,:]
        diff_train_x_p[count_image] =np.pad(diff_train_x[count_image], ((0,1),(0,1),(0,0)), 'constant',
                                 constant_values=(0, 0))
    for count_image in range(len(test_x)):
        
        diff_test_x[count_image] = test_x[count_image,1:128,1:128,:] - test_x[count_image,0:127,0:127,:]
        diff_test_x_p[count_image] =np.pad(diff_test_x[count_image], ((0,1),(0,1),(0,0)), 'constant',
                                 constant_values=(0, 0))
    return diff_train_x_p, diff_test_x_p,

