# -*- coding: utf-8 -*-
"""
@author: Maher Deeb
"""
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from ETL import ETL_data_loading,_resize_image_size,plot_sample_image
from U_Net_layers import build_model
from submission import submit_results,_reshape_image
from scoring import _calculate_scoring
from combine_models import _combine_models
from create_new_image import _create_new_image
def plot_kbis(history):
    
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"],
                 label="Validation loss")
    ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
    ax_acc.plot(history.epoch, history.history["val_acc"],
                label="Validation accuracy")

def run_model(combine_models,random_state,epochs,batch_size,loss="binary_crossentropy", 
              optimizer="adam", metrics=["accuracy"],
              plot_KBI=False):
    if combine_models is False:
        input_layer = Input((img_size_target, img_size_target, 1))
        output_layer = build_model(input_layer, 16)
        model = Model(input_layer, output_layer)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.summary()
        early_stopping = EarlyStopping(patience=10, verbose=1)
        model_checkpoint = ModelCheckpoint(
                "./keras_random_state_{}.model".format(random_state),
                                           save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5,
                                      min_lr=0.00001, verbose=1)
        history = model.fit(X_train, y_train,
                            validation_data=[X_cv, y_cv], 
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping,
                                       model_checkpoint, reduce_lr])
        
        if plot_KBI:
            plot_kbis(history)
    
        return history,model
    else:
        history ='' 
        model = ''
        return history ,model


def calculate_score(combine_models,model_list,id_train, X_train,
                    y_train,padding):
    
    if combine_models:
        preds_cv = np.zeros((X_train.shape))
        preds_cv = _combine_models(model_list, X_train, preds_cv, 
                                   img_size_target)
    else:
        preds_cv = model.predict(X_train)
    
    for threshold in np.arange(0.5, 1.0, 0.05):
        preds_cv_original_size = np.zeros((len(X_train), img_size_original,
                                        img_size_original, 1), dtype=np.uint8)
        y_train_orginal = np.zeros((len(X_train), img_size_original,
                                        img_size_original, 1), dtype=np.uint8)
        for count_image in range(len(preds_cv)):
            if padding:
                 preds_cv_original_size[count_image] = (preds_cv[
                         count_image,14:128-13,14:128-13,
                         0:1]>threshold).astype(int)
                 y_train_orginal [count_image] = y_train[count_image,
                                 14:128-13,14:128-13,0:1]
            else:
                preds_cv_original_size[count_image] = _resize_image_size(
                        (preds_cv[count_image]>threshold).astype(int),
                              img_size_original)
                
                y_train_orginal [count_image] = _resize_image_size(
                        y_train[count_image],
                              img_size_original)
        iou_mean =0
        for count_image in  range(len(X_train)):
            
            sample_pred_cv_flat =_reshape_image(
                    preds_cv_original_size[count_image])
            sample_real_cv_flat = _reshape_image(
                    y_train_orginal[count_image])
            iou_mean += _calculate_scoring(
                    sample_real_cv_flat,sample_pred_cv_flat)
        
        iou_mean /= len(X_train)
        print(iou_mean)

def _extend_train_dataset(train_ids,train_x,train_y):
    train_x = np.append(train_x, [np.fliplr(x) for x in train_x], axis=0)
    train_y = np.append(train_y, [np.fliplr(x) for x in train_y], axis=0)
    for id_i in range(len(train_ids)):
        train_ids.append(train_ids[id_i])
    return train_ids, train_x,train_y
# Run everything from here
img_size_target = 128
img_size_original = 101
padding = True
combine_models = False
padding_type = 'constant'
model_list = ['wrap',
              'symmetric',
              'reflect']
# 1. Load data
train_ids, dataframe_depth, train_x, train_y, test_x = \
    ETL_data_loading(img_size_target,True,False,padding, padding_type)

# 2. Split the data
random_state=0
id_train, id_cv, X_train, X_cv, y_train, y_cv = train_test_split(
    train_ids, train_x, train_y, test_size=0.1, random_state=random_state)

#id_train,X_train,y_train = _extend_train_dataset(id_train,X_train,y_train)
#X_train,y_train = _create_new_image(X_train,y_train,15000)
print(len(id_train),len(X_train),len(y_train))

# 3. load the model and train it    
history,model = run_model(combine_models, random_state,epochs = 200,
                          batch_size = 32, loss="binary_crossentropy",
                          optimizer="adam", metrics=["accuracy"],
                          plot_KBI=False)
# 4. predict and calculate the score
calculate_score(combine_models,model_list,id_cv, X_cv, y_cv,padding =padding)
# 5. submitt
threshold = 0.6

if combine_models:
    preds_test = np.zeros((test_x.shape))
    preds_test = _combine_models(model_list, test_x, preds_test,
                                 img_size_target)
else:
    preds_test = model.predict(test_x)
    
preds_test_org = np.zeros((len(preds_test), img_size_original,
                                img_size_original, 1), dtype=np.uint8)
for count_image in range(len(preds_test)):
    if padding:
        preds_test_org[count_image] = (preds_test[count_image,14:128-13,
                      14:128-13,0:1]> threshold).astype(int)
    else:
        preds_test_org[count_image] =  _resize_image_size(
                (preds_test[count_image]> threshold).astype(int),
                      img_size_original) 
    
submit_results(preds_test_org)  