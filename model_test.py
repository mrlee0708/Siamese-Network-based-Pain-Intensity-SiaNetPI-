# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:52:27 2021

@author: Miran
"""

import matplotlib.pyplot as plt
import numpy
import glob
from PIL import Image
import openpyxl
from configuration import setup_conf #setup 
from tensorflow.keras.models import load_model
import numpy as np

MODEL_SAVE = "SiaNetPI_model"
model = load_model(MODEL_SAVE, compile=False)

def imgfolderload_func(subject):
    facialimg_dir = "D://Database/Datasets/UNBC/"+personalID+"/"+subject

    # Load Image data 
    gray_X = []
    files = glob.glob(facialimg_dir+"/"+"*.png")
    for i, f in enumerate(files):
        img = Image.open(f)
        gray_img = img.convert("RGB")
        gray_img = gray_img.resize((setup_conf.IMAGE_W, setup_conf.IMAGE_H))
        gray_img = np.asarray(gray_img)
        gray_X.append(gray_img)
            
    gray_X = np.array(gray_X)
    
    plt.imshow(gray_X[1], cmap=plt.cm.gray)

    return gray_X
    
if __name__=="__main__":
    personalID = "107-hs107"
    subject = "hs107t2aaaff"
    dataX = imgfolderload_func(subject)
    
    pi = [0,]
    for i in range(len(dataX)-1):
        TestimageA = dataX[0]
        TestimageB = dataX[i+1]
        
        # add a batch dimension to both images
        TestimageA = np.expand_dims(TestimageA, axis=-1)
        TestimageB = np.expand_dims(TestimageB, axis=-1)
        
        TestimageA = np.expand_dims(TestimageA, axis=0)
        TestimageB = np.expand_dims(TestimageB, axis=0)
           
        TestimageA = TestimageA / 255.0
        TestimageB = TestimageB / 255.0
         
        Ypred = model.predict([TestimageA, TestimageB])
        Ypred = float(Ypred)
        pi.append(Ypred)
        print(pi)
  
    exl_filedir = "./data/pearson/"+personalID+"/intensity/"+subject+"_intensity.xlsx"
    
    result_pi = np.asarray(pi)
    wb = openpyxl.Workbook()
    sheet1 = wb.active
    sheet1.append(pi)
    wb.save(exl_filedir)
    