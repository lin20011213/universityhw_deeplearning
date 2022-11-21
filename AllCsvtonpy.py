import pandas as pd
import numpy as np    
import os 
import csv
from PIL import Image
import numpy as np


list=[]
listlabel=[]
for i in range(7182):
    listlabel=listlabel+[0]
for i in range(1754):
    listlabel=listlabel+[1]
for i in range(40):
    listlabel=listlabel+[2]
for i in range(836):
    listlabel=listlabel+[3]
for i in range(1824):
    listlabel=listlabel+[4]
for i in range(59):
    listlabel=listlabel+[5]
for i in range(113):
    listlabel=listlabel+[6]
for i in range(192):
    listlabel=listlabel+[7]
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/0csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    
    list1=[melbourne_data]
    list=list+list1
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/1csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    list1=[melbourne_data]
    list=list+list1
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/2csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    list1=[melbourne_data]
    list=list+list1
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/3csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    list1=[melbourne_data]
    list=list+list1
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/4csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    list1=[melbourne_data]
    list=list+list1
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/5csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    list1=[melbourne_data]
    list=list+list1
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/6csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    list1=[melbourne_data]
    list=list+list1
yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Datasetcsv/7csv/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    melbourne_data = pd.read_csv(yourPath+file)
    list1=[melbourne_data]
    list=list+list1
print(list[0].astype)




#print(list)





X = np.stack((list), axis=0)
label = np.stack((listlabel), axis=0)
np.save('testX.npy', X)
np.save('testY.csv', label)
print(X.shape)
print(label.shape)
# 讀取資料          