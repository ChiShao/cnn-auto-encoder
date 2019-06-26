import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import (
    Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D)
from keras.models import Model, Sequential, load_model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


# this is the size of our encoded representations
ENCODING_DIM = 10
THRESHOLD = 0.5
# np.set_printoptions(threshold=sys.maxsize)

CUR_DIR = os.path.curdir


def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


    X_train = X_train.reshape((len(X_train), 28, 28, 1))
    test_len = len(X_test)

    X_test = X_test.reshape((test_len, 28, 28, 1))
    # print(X_train.shape, X_test.shape)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]


    x_len = len(X)
    boundaries = [int(x_len * 0.7), int(x_len*0.85)]
    # print("boundaries %s" % str(boundaries))

    [X_train, X_test, X_validate] = np.split(X, boundaries)
    # print(X_train.shape, X_test.shape, X_validate.shape)
    
    [y_train, y_test, y_validate] = np.split(y, boundaries)
    # print(y_train.shape,y_test.shape, y_validate.shape)



    # one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_validate = to_categorical(y_validate)

    return (X_train/255.0, X_test/255.0, X_validate/255.0), (y_train, y_test, y_validate)


def build_classifier():
    classifier = Sequential()
    # classifier.add(Dense(16, activation="relu"))
    classifier.add(Dense(10, activation="softmax", input_dim=128,
                         kernel_initializer="random_normal"))

    classifier.compile(optimizer='adam', loss='mean_squared_error',
                       metrics=['accuracy'])
    return classifier

# TODO: test regularizing the model


def build_conv_aue():
    INPUT_SHAPE = (28, 28, 1)
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
        DEFAULT_STRIDE, padding="same", name="encoder")(encode)

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
    autoencoder = Model(inputs=input_img, outputs=decoded)

    encoder, decoder = get_codec_from_aue(autoencoder)
    # build (aka "compile") the model
    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
    return autoencoder, encoder, decoder


def get_codec_from_aue(autoencoder, mid_shape=(4, 4, 8)):
    # this model maps an input to its encoded representation; Big image to small rep
    encoder = Model(
        inputs=autoencoder.input, outputs=autoencoder.get_layer("encoder").output)

    # create a placeholder for an encoded (ENCODING_DIM-dimensional) input^
    # TODO: remove hard coded shape
    encoded_input = Input(shape=mid_shape)
    decoder = autoencoder.layers[-7](encoded_input)
    decoder = autoencoder.layers[-6](decoder)
    decoder = autoencoder.layers[-5](decoder)
    decoder = autoencoder.layers[-4](decoder)
    decoder = autoencoder.layers[-3](decoder)
    decoder = autoencoder.layers[-2](decoder)
    decoder = autoencoder.layers[-1](decoder)

    # create the decoder model; encoded(small) rep to big image
    decoder = Model(encoded_input, decoder)
    return encoder, decoder


print("Getting data...")
(X_train, X_test, X_validate), (y_train, y_test, y_validate) = get_data()
print("Done")
#####################################################################
print("AUTOENCODER")
#####################################################################
ckpt_loc = os.path.join(CUR_DIR, "ckpts", "conv-aue.hdf5")

if(os.path.isfile(ckpt_loc)):
    print("Loading Autoencoder from directory %s..." % ckpt_loc)
    autoencoder = load_model(ckpt_loc)
    encoder, decoder = get_codec_from_aue(autoencoder)

else:
    print("Training Autoencoder...")
    autoencoder, encoder, decoder = build_conv_aue()
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='min', min_delta=0.01)
    mcp_save = ModelCheckpoint(ckpt_loc,
                               save_best_only=True, verbose=1, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='min')
    autoencoder.fit(X_train, X_train,
                    epochs=100,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(X_validate, X_validate), callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
    model_path = os.path.join(CUR_DIR, "model", "conv")
    autoencoder.save(os.path.join(model_path, "autoencoder.h5"))
    encoder.save(os.path.join(model_path, "encoder.h5"))
    decoder.save(os.path.join(model_path, "decoder.h5"))
# autoencoder = load_model("model/conv/autoencoder.h5")
# decoder = load_model("model/conv/decoder.h5")
# encoder = load_model("model/conv/encoder.h5")
print("Done")

eval_train = autoencoder.evaluate(X_train, X_train)
eval_validate = autoencoder.evaluate(X_validate, X_validate)
eval_test = autoencoder.evaluate(X_test, X_test)
print("Evaluation: Train", eval_train, "Validate",
      eval_validate, "Test",  eval_test)

encoded_imgs_train = encoder.predict(X_train)
encoded_imgs_validate = encoder.predict(X_validate)
encoded_imgs_test = encoder.predict(X_test)

decoded_imgs = decoder.predict(encoded_imgs_test)
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
plt.savefig(os.path.join("imgs","conv-autoencoder.png"))

#####################################################################
print("CLASSIFIER")
#####################################################################
encoded_imgs_train = encoded_imgs_train.reshape(len(encoded_imgs_train), 128)
encoded_imgs_validate = encoded_imgs_validate.reshape(
    len(encoded_imgs_validate), 128)
encoded_imgs_test = encoded_imgs_test.reshape(len(encoded_imgs_test), 128)

ckpt_loc = os.path.join(CUR_DIR, "ckpts", "classifier.hdf5")
if(os.path.isfile(ckpt_loc)):
    print("Loading classifier from directory %s..." % ckpt_loc)

    classifier = load_model(ckpt_loc)
else:
    print("Training classifier...")
    classifier = build_classifier()
    earlyStopping = EarlyStopping(
        monitor='val_acc', patience=3, verbose=1, mode='max', min_delta=0.01)
    mcp_save = ModelCheckpoint(ckpt_loc,
                               save_best_only=True, verbose=1, monitor='val_acc', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_acc', factor=0.3, patience=3, verbose=1, mode='max')
    classifier.fit(encoded_imgs_train, y_train, validation_data=(
        encoded_imgs_validate, y_validate), batch_size=10, epochs=10, shuffle=True, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
print("Done")

eval_train = classifier.evaluate(encoded_imgs_train, y_train)
eval_validate = classifier.evaluate(encoded_imgs_validate, y_validate)
eval_test = classifier.evaluate(encoded_imgs_test, y_test)
print("Evaluation: Train", eval_train, "Validate",
      eval_validate, "Test",  eval_test)


def get_cm(input, y, TRESHOLD):
    y_pred = classifier.predict(input)
    y_pred = (y_pred > THRESHOLD)

    y_pred = tf.argmax(y_pred.astype(np.float32), axis=1)
    y = tf.argmax(y, axis=1)
    c = tf.keras.backend.eval(y_pred)
    d = tf.keras.backend.eval(y)
    # print(c, d)
    cm = confusion_matrix(c, d)
    return cm


print(get_cm(encoded_imgs_train, y_train, THRESHOLD))
print(get_cm(encoded_imgs_validate, y_validate, THRESHOLD))
print(get_cm(encoded_imgs_test, y_test, THRESHOLD))
