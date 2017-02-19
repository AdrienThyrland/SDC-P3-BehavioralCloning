# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:04:41 2017

@author: thyrland
"""
# Import all the needed libraries
import pickle
import tensorflow as tf
import numpy as np
import cv2
import csv
import time
import math
import h5py
import json
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Activation, Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


    
# Preprocess the images 
# Convert to YUV, resize to 1/2 the original size
def preprocess(image):
    #yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yuv_resized = cv2.resize(image, (160, 80))
    yuv_resized = yuv_resized[20:70, 0:160]
    return yuv_resized


# Model/Training parameters
NUM_EPOCH   = 5
BATCH_SIZE  = 128
H, W, CH    = 50, 160, 3
CROP_TOP    = 30
CROP_BOTTOM = 10
MODEL_TEST  = False


# Create a generator 
def data_generator(driving_log):
    X_batch = np.ndarray(shape = (BATCH_SIZE, H, W, CH), dtype=float)
    y_batch = np.ndarray(shape = BATCH_SIZE, dtype=float)
    index = 0
    while True:
        for i in range(BATCH_SIZE):
            if index == 0:
                driving_log = shuffle(driving_log)
            elif index >= len(driving_log):
                index = 0
            if (driving_log[index][2] == 'False'):
                # Attention: remove empty space from the image name using str.strip
                X_batch[i] = (preprocess(mpimg.imread(str.strip(driving_log[index][0]))))
                #X_batch[i] = mpimg.imread(str.strip(driving_log[i][0]))
                y_batch[i] = (float(driving_log[index][1])) 
            else:
                # Attention: remove empty space from the image name using str.strip
                X_batch[i] = (cv2.flip(preprocess(mpimg.imread(str.strip(driving_log[index][0]))), 1))
                #X_batch[i] = mpimg.imread(str.strip(driving_log[i][0]))
                y_batch[i] = -(float(driving_log[index][1])) 
            # Increment the driving log index
            index += 1
        yield (X_batch, y_batch)


# Calculate the epoch size
def epoch_size(data_size, batch_size):
    num_batches = data_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


# Implement the E2E Nvidia pipeline using Keras
def get_model(H, W, CH, time_len=1):

    model = Sequential()
    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(H, W, CH), output_shape=(H, W, CH)))
    # Crop
    #model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0)), input_shape=(H, W, CH)))
    # First convolution layer 24@23x78    
    model.add(Convolution2D(24, 5, 5, subsample= (2, 2), name='Conv1_5x5_24_23x78'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    # Second convolution layer 36@10x37     
    model.add(Convolution2D(36, 5, 5, subsample= (2, 2), name='Conv2_5x5_36_10x37'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    # Third convolution layer 48@6x33
    model.add(Convolution2D(48, 5, 5, subsample= (1, 1), name='Conv3_5x5_48_6x33'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    # Fourth convolution layer 64@4x31
    model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='Conv4_3x3_64_4x31'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    # Fifth convolution layer 64@1x18    
    #model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='Conv5_3x3_64_1x18'))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    # Fully-connected layer 7936 neurons
    model.add(Flatten())
    #model.add(Dense(1164, init = normal_init, name = "Dense1_1164"))
    #model.add(Activation('relu'))
    #model.add(Dropout(p))
    # Fully-connected layer 100 neurons
    model.add(Dense(100,  name = "Dense1_100"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    # Fully-connected layer 50 neurons
    model.add(Dense(50, name = "Dense2_50"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    # Fully-connected layer 10 neurons
    model.add(Dense(10, name = "Dense3_10"))
    model.add(Activation('relu'))
    # Output
    model.add(Dense(1, name = "dense_4"))
    #model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "atan_0"))

    model.compile(optimizer="adam", loss="mse")

    return model


# BEHAVIORAL CLONING PROJECT


# CSV DATA
# Open the driving log CSV file and insert the content in a list
with open("driving_log.csv", 'r') as csvfile:
     reader = csv.DictReader(csvfile, delimiter=',')
     driving_log = list(reader)


# TRAINING SET
## Use the list to create the training set
data = []
for i in range(len(driving_log)):
    data.append((str.strip(driving_log[i]['center']), float(driving_log[i]['steering']), False))
    data.append((str.strip(driving_log[i]['center']), float(driving_log[i]['steering']), True))
    data.append((str.strip(driving_log[i]['left']), (float(driving_log[i]['steering']) + 0.1), False))
    data.append((str.strip(driving_log[i]['left']), (float(driving_log[i]['steering']) + 0.1), True))
    data.append((str.strip(driving_log[i]['right']), (float(driving_log[i]['steering']) - 0.1), False))
    data.append((str.strip(driving_log[i]['right']), (float(driving_log[i]['steering']) - 0.1), True))

data = shuffle(data)
data_train, data_val, data_test = np.split(data, [int(.8*len(data)), int(.9*len(data))])


# MODEL
# Create the model
modelE2E = get_model(H, W, CH)
modelE2E.summary()


# TRAIN
# Keras fit generator
if MODEL_TEST:
    data_trainMod = data_train[:5]
    data_valMod   = data_val[:2]
    Start_time = time.time()
    history = modelE2E.fit_generator(data_generator(data_trainMod), 
                                        samples_per_epoch  = epoch_size(len(data_trainMod), 2), 
                                        nb_epoch = NUM_EPOCH, 
                                        verbose=1, 
                                        validation_data = data_generator(data_valMod), 
                                        nb_val_samples = epoch_size(len(data_trainMod), 2),
                                        class_weight=None, 
                                        max_q_size=10)
    #history.history()
    Total_time = time.time() - Start_time
    print('Total training time: %.2f sec (%.2f min)' % (Total_time, Total_time/60))
else:
    Start_time = time.time()
    history = modelE2E.fit_generator(data_generator(data_train), 
                                        samples_per_epoch  = epoch_size(len(data_train), BATCH_SIZE), 
                                        nb_epoch = NUM_EPOCH, 
                                        verbose=1, 
                                        validation_data = data_generator(data_val), 
                                        nb_val_samples = epoch_size(len(data_val), BATCH_SIZE),
                                        class_weight=None, 
                                        max_q_size=10)
    #history.History()
    Total_time = time.time() - Start_time
    print('Total training time: %.2f sec (%.2f min)' % (Total_time, Total_time/60))


# TEST
# Keras predict
if MODEL_TEST:
    X_testMod = X_test[:2]
    y_testMod = y_test[:2]
    score = modelE2E.evaluate_generator(
                        generator = data_generator(data_test),
                        val_samples = epoch_size(len(data_test), BATCH_SIZE), # How many batches to run in one epoch
                        )
    print("Test score {}".format(score))
else:
    score = modelE2E.evaluate_generator(
                    generator = data_generator(data_test),
                    val_samples = epoch_size(len(data_test), BATCH_SIZE), # How many batches to run in one epoch
                    )
    print("Test score {}".format(score))


# SAVE THE MODEL
print("Saving model weights and configuration file.")
modelE2E.save('model.h5')  # creates a HDF5 file 'my_model.h5'
print ("Model saved")


del modelE2E  # deletes the existing model