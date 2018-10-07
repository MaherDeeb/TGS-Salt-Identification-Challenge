# -*- coding: utf-8 -*-
"""
@author: Maher Deeb
"""
from ETL import ETL_data_loading, _images_id
import datetime, time
import pandas as pd

path = './'
path_test = './data/images_test/'
test_ids = _images_id(path_test)

def _reshape_image(predicted_mask):
    
    
    reshaped_mask = predicted_mask.reshape(predicted_mask.shape[0] *
                                           predicted_mask.shape[1], order='F')
    
    return reshaped_mask

def _arrage_data_for_submission(reshaped_mask):
    
    # this a list of start positions and length where the mask value >0
    salt_positions = []
    #initiate the start postion of 1s values of the mask
    start_pos = 1
    # this is a tracker to count how many 1s after the start positon
    tracker = 0
    
    #check it value of the mask picture after reshaping it
    for pixel_i in reshaped_mask:
        if (pixel_i == 0):
            if tracker != 0:
                # the current bunch of value>0 finish here and we add it to the
                # the finial results
                salt_positions.append((start_pos, tracker))
                # the next position may start from here if the next value >0
                # otherwise it will be corrected in other places
                start_pos += tracker
                # reset the tracker
                tracker = 0
            # if the current value is 0, the start positon it not in this point
            start_pos += 1
        else:
            # if the tracker is >0 we go further
            tracker += 1
    # in case of the last value of pixel_i, if tracker still >0, it means we
    # miss registering the last record
    if tracker != 0:
        salt_positions.append((start_pos, tracker))
        start_pos += tracker
        tracker = 0
    
    return salt_positions

def _reshape_finial_results_for_submission(salt_positions):
    
    reshaped_results = ''

    for salt_position_i in salt_positions:
        reshaped_results += '{} {} '.format(salt_position_i[0],
                             salt_position_i[1])
    return reshaped_results[:-1]

def _make_it_ready_to_submission(predicted_mask):
    
    reshaped_mask = _reshape_image(predicted_mask)
    salt_positions = _arrage_data_for_submission(reshaped_mask)
    reshaped_prediction_i = _reshape_finial_results_for_submission(
            salt_positions)
    
    return reshaped_prediction_i

def submit_results(predicted_y):
    
    pred_dict = {}
    for id_n in range(len(test_ids)):
        
        # this has to be changed after we get predicted_y
        predicted_mask_i = predicted_y[id_n]
    
        pred_dict.update({test_ids[id_n].split('.')[0]: _make_it_ready_to_submission(
                predicted_mask_i)})
        # this has to be changed after we get predicted_y
        #if id_n > 20:
            
            #break
    
    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(path + '{}_submit.csv'.format(str(round(time.mktime((datetime.datetime.now().timetuple()))))))
    
#submit_results(preds_test_org)    
#dataframe_depth, train_x, train_y, test_x = ETL_data_loading(128,False,False)