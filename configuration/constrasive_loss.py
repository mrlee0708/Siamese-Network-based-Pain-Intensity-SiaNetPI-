# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 02:51:12 2021

@author: Miran
"""
import tensorflow as tf
import tensorflow.keras.backend as K

def contrastive_loss(Yacut, Ypred, margin=1):
	Yacut = tf.cast(Yacut, Ypred.dtype)

	margin_temp = K.square(K.maximum(margin - Ypred, 0))
	cont_loss_value = K.mean(Yacut * K.square(Ypred) + (1 - Yacut) * margin_temp)

	return cont_loss_value