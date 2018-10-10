# -*- coding: utf-8 -*-
"""

@author: Maher Deeb
"""

import numpy as np

def _create_new_image(X_train,y_train,number_image):
    new_train_x=np.zeros((number_image,128,128,1))
    new_train_y=np.zeros((number_image,128,128,1))
    
    for x in range(number_image):
        count_i=0
        rand1=np.random.randint(400, size=64)
        rand2 =np.random.randint(16, size=(8,8))
        for y in range(8):
            for z in range(8):
                start=rand2[y,z]
                L = X_train[rand1[count_i],8*start:8*(start+1),
                            8*start:8*(start+1),:]
                n = y_train[rand1[count_i],8*start:8*(start+1),
                            8*start:8*(start+1),:]
                new_train_x[x,y*8:8*(y+1),z*8:8*(z+1),:] = L
                new_train_y[x,y*8:8*(y+1),z*8:8*(z+1),:] = n
                count_i+=1
    X_train = np.append(X_train, [x for x in new_train_x], axis=0)
    y_train = np.append(y_train, [x for x in new_train_y], axis=0)
    return X_train,y_train