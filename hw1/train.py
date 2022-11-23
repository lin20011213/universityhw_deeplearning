import numpy as np  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils   
from sklearn.model_selection import train_test_split


train_data=np.load('testX.npy',allow_pickle=True)
train_target=np.load('testY.csv.npy',allow_pickle=True)
X_train,X_test, y_train, y_test =train_test_split(train_data,train_target,test_size=0.4, random_state=0)

model = Sequential()
model.add(Dense(units=256, input_dim=4510, kernel_initializer='normal', activation='relu')) 
model.add(Dense(units=8, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

y_TrainOneHot = np_utils.to_categorical(y_train) 
y_TestOneHot = np_utils.to_categorical(y_test) 
X_train_2D = X_train.reshape(7200, 41*110).astype('float32')  
X_test_2D = X_test.reshape(4800, 41*110).astype('float32')  
x_Train_norm = X_train_2D/255
x_Test_norm = X_test_2D/255
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.1, epochs=400, batch_size=700, verbose=2)  
scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

X = x_Test_norm[0:30,:]
predictions = np.argmax(model.predict(X), axis=-1)
print(predictions)

model.save('model.h5')