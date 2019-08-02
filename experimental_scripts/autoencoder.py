import os.path

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.utils import to_categorical

# this is the size of our encoded representations
ENCODING_DIM = 10


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img = X_train[0].reshape((28, 28))
    # reshape data to fit model
    plt.imshow(X_train[0], cmap="Greys", vmin=0, vmax=255)
    plt.savefig("example_img.png")

    """for Autoencoder"""
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    np.random.shuffle(X_test)
    X_validate = X_test[len(X_test)//2:]
    X_test = X_test[:len(X_test)//2]

    return X_train/255.0, X_test/255.0, X_validate/255.0


def build_aue():
    INPUT_DIM = 28 * 28
    MID_DIM = 512
    # this is our input placeholder
    input_img = Input(shape=(INPUT_DIM,))
    # layer between input and middle layer
    encode_1 = Dense(MID_DIM, activation="relu")(input_img)

    # "encoded" is the encoded representation of the input, middle layer of the aue
    encoded = Dense(ENCODING_DIM, activation="relu")(encode_1)

    # layer between middle and output layer
    decode_1 = Dense(MID_DIM, activation="relu")(encoded)
    # "decoded" is the lossy reconstruction of the input; output
    decoded = Dense(INPUT_DIM, activation="sigmoid")(decode_1)

    # this model maps an input to its reconstruction
    autoencoder = keras.models.Model(inputs=input_img, outputs=decoded)

    # this model maps an input to its encoded representation; Big image to small rep
    encoder = keras.models.Model(inputs=input_img, outputs=encoded)

    # create a placeholder for an encoded (ENCODING_DIM-dimensional) input
    encoded_input = Input(shape=(ENCODING_DIM,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model; small rep to big image
    decoder = keras.models.Model(
        inputs=encoded_input, outputs=autoencoder.layers[-1](autoencoder.layers[-2](encoded_input)))

    # build (aka "compile") the model
    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
    return autoencoder, encoder, decoder


X_train, X_test, X_validate = get_data()

# autoencoder, encoder, decoder = build_aue()
# autoencoder.fit(X_train, X_train,
#                 epochs=50,
#                 batch_size=100,
#                 shuffle=True,
#                 validation_data=(X_validate, X_validate))
# autoencoder.save("model/autoencoder.h5")
# encoder.save("model/encoder.h5")
# decoder.save("model/decoder.h5")
autoencoder = keras.models.load("model/autoencoder.h5")
decoder = keras.models.load("model/decoder.h5")
encoder = keras.models.load("model/encoder.h5")

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
plt.savefig(os.path.join("imgs","autoencoder.png"))
