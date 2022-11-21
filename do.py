import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# 用來後續將 label 標籤轉為 one-hot-encoding
from keras.utils import np_utils
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np    
import os 
import csv
from PIL import Image
import numpy as np
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
# 用來後續將 label 標籤轉為 one-hot-encoding
from keras.utils import np_utils

fo = open("1090646.txt", "w")

yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Evaluation Dataset/'
allFileList = os.listdir(yourPath)
reload_model = tf.keras.models.load_model('./model.h5')
fo.write("ImageName,label\n")
for file in allFileList:
    if(file[-3]=="c"):
        
        melbourne_data = pd.read_csv(yourPath+file)
        DD=melbourne_data.values.reshape(1,4510).astype('float32')
        features=reload_model.predict(DD)
        num=0
        for i in range(8):
            if(features[0][i]>=0.5):
                fo.write(str(file[:-4])+","+str(i)+"\n")
                print(file,i)
fo.close()