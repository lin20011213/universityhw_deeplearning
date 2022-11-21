import os 
import csv
from PIL import Image
import numpy as np

yourPath = 'C:/Users/LIN20/Desktop/deeplearning/Training Dataset/7/'
allFileList = os.listdir(yourPath)
for file in allFileList:
    #print(file)
    img = Image.open(yourPath+file).convert('L')
    arr = np.asarray(img)
    #print(arr.shape)
    lst = []
    for row in arr:
        tmp = []
        for col in row:
            tmp.append(str(col))
        lst.append(tmp)
    # 4. Save list of lists to CSV
    with open(yourPath+file[:-4]+".csv", 'w') as f:
        for row in lst:
            f.write(','.join(row) + '\n')

    
#把png to gray to CSV