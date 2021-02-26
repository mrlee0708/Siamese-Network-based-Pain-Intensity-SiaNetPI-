# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:40:07 2021

@author: Miran
"""

from tensorflow.keras.models import Model, Input, Conv2D, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
from configuration import setup_conf #setup 

def train_siamese_model():
	dropout_val=0.3
	inputs = Input(setup_conf.IMG_SHAPE)

    # ConvNet-1
	x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
	x = Conv2D(64, (3, 3), activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropout_val)(x)
    
    # ConvNet-2
	x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
	x = Conv2D(128, (3, 3), activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropout_val)(x)
    
    # ConvNet-3
	x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
	x = Conv2D(256, (3, 3), activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(dropout_val)(x)
    
    # Global Average Pooing and dense
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(setup_conf.EMBEDDING_DIM)(pooledOutput)
    
	model = Model(inputs, outputs)
    
    #Summary 
	model.summary()

	return model
