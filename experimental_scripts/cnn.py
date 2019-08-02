import os.path

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten
from keras.utils import to_categorical

# this is the size of our encoded representations


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img = X_train[0].reshape((28, 28))
    # reshape data to fit model
    plt.imshow(X_train[0], cmap="Greys", vmin=0, vmax=255)
    plt.savefig(os.path.join("imgs","example-digit.png"))

    """for CNN"""
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    # one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(y_train[0])
    return X_train/255.0, X_test/255.0, y_train, y_test


def get_cnn():
    model = keras.models.Sequential()
    # add model layers
    model.add(Conv2D(64, kernel_size=3,
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


X_train, X_test, y_train, y_test = get_data()
filepath = "model/cnn.h5"
if os.path.isfile(filepath):
    model = keras.models.load(filepath)
else:
    model = get_cnn()

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
    model.save(filepath)


score = model.evaluate(X_test, y_test)
