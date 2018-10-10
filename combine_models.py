# -*- coding: utf-8 -*-
"""
@author: Maher Deeb
"""

from keras.models import load_model
from padding import _padding_image
import numpy as np
def _combine_models(model_list,X_train, Predict,new_size):
    
    X_train_resized = np.zeros((len(X_train), new_size,
                                new_size, 1), dtype=np.uint8)
    for model_i in model_list:
        
        model = load_model('keras_random_state_0_{}.model'.format(model_i))
        for image_i in range(len(X_train)):
            
            X_train_resized[image_i]= _padding_image(X_train[image_i,14:128-13,
                      14:128-13,0:1],model_i)
            
        Predict += model.predict(X_train_resized)
        
    Predict /= len(model_list)

    return Predict
    