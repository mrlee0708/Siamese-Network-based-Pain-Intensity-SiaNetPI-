# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 02:36:34 2021

@author: Miran
"""

import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from configuration import setup_conf #setup
from train_model import train_siamese_model #train_siamese_model
import cv2
from datetime import datetime
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

MODEL_W_SAVE = "SiaNetPI_Weight.h5"
MODEL_SAVE = "SiaNetPI_model"

def imgload_func():
    facialimg_dir = "D://Coding/Pain_Intensity/data/UNBC/PainGroup"
    categories = ["PG1","PG2","PG3","PG4"]
    nb_classes = len(categories)
    
    # Load Image data 
    X = []
    Y = []
    for idx, expre in enumerate(categories):
        # Label
        label = [0 for i in range(nb_classes)]
        label[idx] = 1
        # Imgae
        image_dir = facialimg_dir + "/" + expre
        files = glob.glob(image_dir+"/*.png")
        for i, f in enumerate(files):
            img = Image.open(f) 
            img = img.convert("RGB")
            img = img.resize((setup_conf.IMAGE_W, setup_conf.IMAGE_H))
            data = np.asarray(img)      
            X.append(data)
            Y.append(label)
            
    X = np.array(X)
    Y = np.array(Y)
    
    # Image show
    cv2.imshow("Test", X[1])
      
    #Labeling
    data_Y_label = []
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if(Y[i][j]==1):
                tmp = j
                break
        data_Y_label.append(tmp)
        
    return X, data_Y_label

def pair_samples(trainX, trainY):
    pairImages = []
    pairLabels = []

    labels = trainY
    images = trainX
    
    numClasses = len(np.unique(labels))
    labels = np.array(labels, dtype=np.uint8)
      
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]

        idxB = np.random.choice(idx[label])
        posImage = images[idxB]

        pairImages.append([currentImage, posImage])
        pairLabels.append([1])

        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]

        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    
    return (np.array(pairImages), np.array(pairLabels))

if __name__=="__main__":
    X, Y_label = imgload_func()
    trainX, testX, trainY, testY = train_test_split(X, Y_label, test_size = 0.3, random_state=0)
    trainX = trainX / 255.0
    testX = testX / 255.0
       
    (pairTrain, labelTrain) = pair_samples(trainX, trainY)
    (pairTest, labelTest) = pair_samples(testX, testY)
        
    pairA = Input(shape=setup_conf.IMG_SHAPE)
    pairB = Input(shape=setup_conf.IMG_SHAPE)
    print(setup_conf.IMG_SHAPE)
    
    ConvNetModel = train_siamese_model.siamese_network()
    
    InputModelA = ConvNetModel(pairA)
    InputModelB = ConvNetModel(pairB)
    
    # Define the Keras TensorBoard callback.
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    checkpointer = ModelCheckpoint(filepath='./SiaNetPI.model.bet.hdf5', verbose=1, save_best_only=True)
    
    sum_distance = K.sum(K.square(InputModelA - InputModelB), axis=1, keepdims=True)
    sqrt_distance= K.sqrt(K.maximum(sum_distance, K.epsilon()))
    distance = Lambda(sqrt_distance)
	
    model = Model(inputs=[pairA, pairB], outputs=distance)
    
    model.compile(loss=setup_conf.contrastive_loss, optimizer="adam")

    #validation_split = 30%
    history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
    	batch_size=setup_conf.BATCH_SIZE, epochs=setup_conf.EPOCHS, callbacks=[checkpointer])

    model.save_weights(MODEL_W_SAVE)
    model.save(MODEL_SAVE)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    