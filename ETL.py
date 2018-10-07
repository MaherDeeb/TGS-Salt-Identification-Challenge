# -*- coding: utf-8 -*-
"""
@author: Maher Deeb
"""
# Tensorflow-gpu should be installed as well
import os
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

# the data should be in the same folder of the script
path_train = './data/images/'
path_train_mask = './data/masks/'
path_test = './data/images_test/'
path_depth = './data/depths.csv'


# read all the image file name from the folder
def _images_id(path):
    
    _ids = next(os.walk(path))[2]
    
    return _ids

#load the depth csv file
def _load_depth(path):
    
    df = pd.read_csv(path)
    
    return df

# load an image using a function loaded from keras
def _load_an_image(path,_id):
    
    loaded_image = load_img(path + _id)
    
    return loaded_image

# change the type of the data so it can fit in the CNN model later
def _img_to_array(loaded_image):
    
    _array = img_to_array(loaded_image)[:,:,1]
    
    return _array

# changing the size of the image if needed 
def _resize_image_size(image_as_array,new_size):
    
    new_image = resize(image_as_array, (new_size, new_size, 1),
                       mode='constant', preserve_range=True)
    
    return new_image
# in order to test if all functions work, it is possible to plot samples 
# using this function
def plot_sample_image(train_x,train_x_resized,train_y,train_y_resized,dataset):
    
    plt.figure(1)
    plt.imshow(train_x)
    plt.title("original image")
    plt.show()
    plt.figure(2)
    plt.imshow(train_x_resized[:,:,0])
    plt.title("resized image")
    plt.show()
    if dataset is "train":
        plt.figure(3)
        plt.imshow(train_y)
        plt.title("original mask")
        plt.show()
        plt.figure(4)
        plt.imshow(train_y_resized[:,:,0])
        plt.title("resized image")
        plt.show()

# This the main function that should be loaded and called from other scripts    
def ETL_data_loading(new_size,plot_train_sample = False,
                     plot_test_sample = False):
    
    # get the file names of the images
    train_ids = _images_id(path_train)
    train_mask_ids = _images_id(path_train_mask)
    test_ids = _images_id(path_test)
    # load the depth csv file
    dataframe_depth = _load_depth(path_depth)

    print("There are {} images in the train dataset".format(len(train_ids)))
    print("There are {} images in the test dataset".format(len(test_ids)))
    print("The depth dataset contains {} data points".format(
            len(dataframe_depth)))
    print("The first 5 rows of the depth dataframe:")
    print(dataframe_depth.head())
    
    # this can be changed between 0 and 3999 to print a sample figure from the
    # traning dataset
    sample_id = 110

    # initiate the variables for train data
    train_x_resized = np.zeros((len(train_ids), new_size,
                                new_size, 1), dtype=np.uint8)
    train_y_resized = np.zeros((len(train_ids),
                                new_size, new_size, 1), dtype=np.bool)
    count_image = 0
    
    # go through each image in the train dataset and resize it
    for train_id_i in train_ids:
        
        loaded_train_image = _load_an_image(path_train,train_id_i)
        train_x = _img_to_array(loaded_train_image)
        train_x_resized[count_image] = _resize_image_size(train_x,new_size)
        
        loaded_train_mask = _load_an_image(path_train_mask,train_id_i)
        train_y = _img_to_array(loaded_train_mask)
        train_y_resized[count_image] =  _resize_image_size(train_y,new_size)
        
        # plot sample images from train dataset
        if count_image == sample_id and plot_train_sample:
            
            plot_sample_image(train_x,
                            train_x_resized[count_image],
                            train_y,
                            train_y_resized[count_image],
                            "train")
        count_image +=1

    # this can be changed between 0 and 17999 to print a sample figure from the
    # traning dataset
    sample_id_test = 100
    # initiate the variable for test data
    test_x_resized = np.zeros((len(test_ids), new_size,
                                new_size, 1), dtype=np.uint8)
    count_image = 0
    # go through each image in the test dataset and resize it
    for test_id_i in test_ids:
        
        loaded_test_image = _load_an_image(path_test,test_id_i)
        test_x = _img_to_array(loaded_test_image)
        test_x_resized[count_image] = _resize_image_size(test_x,new_size)
        
        # plot sample images from test dataset
        if count_image == sample_id_test and plot_test_sample:
            plot_sample_image(test_x,
                            test_x_resized[count_image],
                            0,
                            0,
                            "test")
        count_image +=1
    
    return train_ids, dataframe_depth, train_x_resized,\
     train_y_resized.astype(int), test_x_resized

# to test the script uncomment the next two lines
# dataframe_depth, train_x, train_y, test_x = \
#    ETL_data_loading(128,False,False)

    

