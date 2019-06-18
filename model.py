import os.path

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.utils import to_categorical

# plot the first image in the dataset

ENCODING_DIM = 10


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img = X_train[0].reshape((28, 28))
    # reshape data to fit model
    plt.imshow(X_train[0], cmap="Greys", vmin=0, vmax=255)
    plt.savefig("example_img.png")
    # X_train = X_train.reshape(60000, 28, 28, 1)
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    # X_test = X_test.reshape(10000, 28, 28, 1)

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


def build_aue():
    input_img = Input(shape=(784,))

    encoded = Dense(ENCODING_DIM, activation="relu")(input_img)
    decoded = Dense(784, activation="sigmoid")(encoded)
    autoencoder = keras.models.Model(input_img, decoded)

    encoder = keras.models.Model(input_img, encoded)

    encoded_input = Input(shape=(ENCODING_DIM,))
    decoder_layer = autoencoder.layers[-1]

    decoder = keras.models.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
    return autoencoder, encoder, decoder


X_train, X_test, y_train, y_test = get_data()
# filepath = "model/model.h5"
# if os.path.isfile(filepath):
#     encoder = keras.models.load_model(filepath)
# else:
#     encoder = get_cnn()

#     encoder.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
#     encoder.save(filepath)


# score = encoder.evaluate(X_test, y_test)
autoencoder, encoder, decoder = build_aue()
autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=8192,
                shuffle=True,
                validation_data=(X_test, X_test))
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("autoencoder.png")