import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from docutils.nodes import header
from envs.test.Lib.unittest.mock import inplace

# Read training and test data files
train = pd.read_csv("E:/Work/B.E Project/TensorFlow Speech Recognition Challenge/trainData.csv", header=None)
test = pd.read_csv("E:/Work/B.E Project/TensorFlow Speech Recognition Challenge/testData.csv", header = None)
print(train.head(5))
print(test.head(5))
print(train.shape)
print(test.shape)

y_train=train[0]
print(y_train.shape)
x_train=train.iloc[:,1:41]
print(x_train.shape)

y_test=test[0]
print(y_test.shape)
x_test=test.iloc[:,1:41]
print(x_test.shape)

from keras.models import Sequential
model = Sequential()
from keras.layers import Dense

model.add(Dense(units=64, activation='relu'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)
