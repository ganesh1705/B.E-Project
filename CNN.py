import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from docutils.nodes import header
from envs.test.Lib.unittest.mock import inplace
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from scipy import dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import float32
from numpy import reshape
from numba.targets.arrayobj import np_reshape

#Read training and test data files
train = pd.read_csv("E:/Work/B.E Project/TensorFlow Speech Recognition Challenge/trainData.csv")
# print(train.head(5))
# print(train.describe())
# print(train.columns)
y_data=train['Label']
print(y_data.shape)
x_data=train.iloc[:,1:41]
print(x_data.shape)

from  sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
  
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
print(y_train.shape)
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
print(y_test.shape)
x_traincnn =np.expand_dims(x_train, axis=2)
print(x_traincnn.shape)
x_testcnn= np.expand_dims(x_test, axis=2)
print(x_testcnn.shape)

#  from keras.models import Sequential
#  from keras.layers import Dense
#     
#  model = Sequential()
#  model.add(Dense(units=64, activation='relu', input_shape=(40,)))
#  model.add(Dense(30, activation='softmax'))
#  model.compile(loss='categorical_crossentropy',
#                optimizer='sgd',
#                metrics=['accuracy'])
#  # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
#  model.fit(x_train, y_train, epochs=5, batch_size=32)
#  loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#  classes = model.predict(x_test, batch_size=128)

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD, Adam
#    
# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 40-dimensional vectors.
# model.add(Dense(40, activation='relu', input_shape=(40,)))
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation='softmax'))
# 
# adam = Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)   
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#    
# model.fit(x_train, y_train,
#           epochs=100,
#           batch_size=512)
# score = model.evaluate(x_test, y_test, batch_size=512)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD, Adam

seq_length = 40

model = Sequential()
model.add(Conv1D(40, 3, activation='relu', input_shape=(40, 3)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(40, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.8, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)   
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)