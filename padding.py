# -*- coding: utf-8 -*-
"""

@author: Maher Deeb
"""
import numpy as np

def _padding_image_constant_0(array_to_pad):
    
    padded_array =np.pad(array_to_pad, ((14,13),(14,13),(0,0)), 'constant',
                         constant_values=(0, 0))
    return padded_array


#array_to_pad = np.random.randint(10, size=(101,101,1))

#padded_array = _padding_image_constant_0(array_to_pad)
