import os.path

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, UpSampling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# this is the size of our encoded representations
ENCODING_DIM = 10


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img = X_train[0].reshape((28, 28))
    # reshape data to fit model
    plt.imshow(X_train[0], cmap="Greys", vmin=0, vmax=255)
    plt.savefig("example_img.png")

    X_train = X_train.reshape((len(X_train), 28, 28, 1))
    X_test = X_test.reshape((len(X_test), 28, 28, 1))
    np.random.shuffle(X_test)
    X_validate = X_test[len(X_test)//2:]
    X_test = X_test[:len(X_test)//2]

    return X_train/255.0, X_test/255.0, X_validate/255.0


def build_conv_aue():
    INPUT_SHAPE = (28, 28, 1)
    MID_DIM = 32
    DEFAULT_KERNEL = (3, 3)
    DEFAULT_STRIDE = (2, 2)
    # this is our input placeholder
    input_img = Input(shape=INPUT_SHAPE)
    # layer between input and middle layer
    encode = Conv2D(16, DEFAULT_KERNEL, activation="relu",
                    padding="same")(input_img)
    encode = MaxPooling2D(DEFAULT_STRIDE, padding="same")(encode)
    encode = Conv2D(8, DEFAULT_KERNEL, activation="relu",
                    padding="same")(encode)
    encode = MaxPooling2D(DEFAULT_STRIDE, padding="same")(encode)
    encode = Conv2D(8, DEFAULT_KERNEL, activation="relu",
                    padding="same")(encode)
    # "encoded" is the encoded representation of the input, middle layer of the aue
    encoded = MaxPooling2D(
        DEFAULT_STRIDE, name="encoder", padding="same")(encode)

    # layer between middle and output layer
    decode = Conv2D(8, DEFAULT_KERNEL, activation="relu", padding="same"
                    )(encoded)
    decode = UpSampling2D(DEFAULT_STRIDE)(decode)
    decode = Conv2D(8, DEFAULT_KERNEL, activation="relu", padding="same"
                    )(decode)
    decode = UpSampling2D(DEFAULT_STRIDE)(decode)
    decode = Conv2D(16, DEFAULT_KERNEL, activation="relu")(decode)
    decode = UpSampling2D(DEFAULT_STRIDE)(decode)
    decoded = Conv2D(1, DEFAULT_KERNEL, activation="sigmoid",
                     padding="same")(decode)

    # this model maps an input to its reconstruction
    autoencoder = keras.models.Model(inputs=input_img, outputs=decoded)

    # this model maps an input to its encoded representation; Big image to small rep
    encoder = keras.models.Model(
        inputs=autoencoder.input, outputs=autoencoder.get_layer("encoder").output)

    # create a placeholder for an encoded (ENCODING_DIM-dimensional) input
    encoded_input = Input(shape=(4, 4, 8))
    decoder = autoencoder.layers[-7](encoded_input)
    decoder = autoencoder.layers[-6](decoder)
    decoder = autoencoder.layers[-5](decoder)
    decoder = autoencoder.layers[-4](decoder)
    decoder = autoencoder.layers[-3](decoder)
    decoder = autoencoder.layers[-2](decoder)
    decoder = autoencoder.layers[-1](decoder)
    # create the decoder model; small rep to big image
    decoder = keras.models.Model(encoded_input, decoder)

    # build (aka "compile") the model
    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
    return autoencoder, encoder, decoder


def get_codec_from_aue(autoencoder):
    # this model maps an input to its encoded representation; Big image to small rep
    encoder = keras.models.Model(
        inputs=autoencoder.input, outputs=autoencoder.get_layer("encoder").output)

    # create a placeholder for an encoded (ENCODING_DIM-dimensional) input^
    #TODO: remove hard coded shape
    encoded_input = Input(shape=(4, 4, 8))
    decoder = autoencoder.layers[-7](encoded_input)
    decoder = autoencoder.layers[-6](decoder)
    decoder = autoencoder.layers[-5](decoder)
    decoder = autoencoder.layers[-4](decoder)
    decoder = autoencoder.layers[-3](decoder)
    decoder = autoencoder.layers[-2](decoder)
    decoder = autoencoder.layers[-1](decoder)
    # create the decoder model; small rep to big image
    decoder = keras.models.Model(encoded_input, decoder)
    return encoder, decoder


X_train, X_test, X_validate = get_data()

ckpt_loc = "ckpts/.conv-aue.hdf5"

if(os.path.isfile(ckpt_loc)):
    autoencoder = keras.models.load_model(ckpt_loc)
    encoder, decoder = get_codec_from_aue(autoencoder)
else:
    autoencoder, encoder, decoder = build_conv_aue()
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(ckpt_loc,
                               save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(X_validate, X_validate), callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

    autoencoder.save("model/conv/autoencoder.h5")
    encoder.save("model/conv/encoder.h5")
    decoder.save("model/conv/decoder.h5")
# autoencoder = keras.models.load_model("model/conv/autoencoder.h5")
# decoder = keras.models.load_model("model/conv/decoder.h5")
# encoder = keras.models.load_model("model/conv/encoder.h5")

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
plt.savefig("conv-autoencoder.png")
