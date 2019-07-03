import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.utils import to_categorical

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import History
from keras.utils import plot_model
from keras.optimizers import SGD

# Import and split data
data = pd.read_csv("voice.csv")
print(data.shape)
data['label'].replace({'male': 0, 'female': 1}, inplace=True)

data = data.astype('float64')
scaler = MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
y = data.label.values
x = data.drop(["label"], axis=1)

# Test-train split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# Build Neural Network

n_cols = x_train.shape[1]
y_train = to_categorical(y_train, 2)

hist = History()

model = Sequential()

model.add(Dense(1000, activation='relu', input_dim=n_cols))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

model.fit(x_train, y_train, epochs=40, validation_split=.2, callbacks=[hist])

y_pred = model.predict(x_test)
y_pred = np.round(y_pred[:, 1])
print(metrics.accuracy_score(y_pred, y_test))

plt.plot(hist.history['acc'], color='red')
plt.plot(hist.history['val_acc'], color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()