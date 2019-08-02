import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import (
    Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D)
from keras.models import Model, Sequential, load
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

# this is the size of our encoded representations
ENCODING_DIM = 10

# decision boundary for classifier
THRESHOLD = 0.7

# working directory
CUR_DIR = os.path.curdir

np.random.seed(42)


def get_data(train_split=.7, test_split=.85):
    """retrieves data MNIST data set and rebalances dataset, such that train=.7, test=.15 and validation=.15"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((len(X_train), 28, 28, 1))
    test_len = len(X_test)

    X_test = X_test.reshape((test_len, 28, 28, 1))

    # divide X values bei 255.0 since MNIST data set changed such that pixel values are in [0,255]
    X = np.concatenate((X_train, X_test)) / 255.0
    y = np.concatenate((y_train, y_test))

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # define boundaries for train,validation and test set at .7 and .85 % of the MNIST data set
    x_len = len(X)
    boundaries = [int(x_len * 0.7), int(x_len*0.85)]

    [X_train, X_test, X_validate] = np.split(X, boundaries)

    [y_train, y_test, y_validate] = np.split(y, boundaries)

    # non-anomalies
    zero_indices = np.where(y_train == 0)
    zeros_train = X_train[zero_indices]
    # anomalies
    zero_indices = np.where(y_train == 8)
    eights_train = X_train[zero_indices]

    # one-hot encode target columns
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_validate = to_categorical(y_validate)

    return (X_train, X_test, X_validate), (y_train, y_test, y_validate)


def build_classifier(input_dim):
    """Builds classifier for classification of MNIST encoded representation."""
    classifier = Sequential()
    classifier.add(Dense(32, activation="relu", input_dim=input_dim,
                         kernel_initializer="random_normal"))
    classifier.add(Dense(ENCODING_DIM, activation="softmax",
                         kernel_initializer="random_normal"))

    classifier.compile(optimizer='adam', loss='mean_squared_error',
                       metrics=['accuracy'])
    return classifier


def build_conv_aue():
    INPUT_SHAPE = (28, 28, 1)
    DEFAULT_KERNEL = (3, 3)
    DEFAULT_POOL_SIZE = (2, 2)
    # this is our input placeholder
    input_img = Input(shape=INPUT_SHAPE)
    # layer between input and middle layer
    encode = Conv2D(16, DEFAULT_KERNEL, activation="relu",
                    padding="same")(input_img)
    encode = MaxPooling2D(DEFAULT_POOL_SIZE, padding="same")(encode)
    encode = Conv2D(8, DEFAULT_KERNEL, activation="relu",
                    padding="same")(encode)
    encode = MaxPooling2D(DEFAULT_POOL_SIZE, padding="same")(encode)
    encode = Conv2D(8, DEFAULT_KERNEL, activation="relu",
                    padding="same")(encode)
    encode = MaxPooling2D(DEFAULT_POOL_SIZE, padding="same")(encode)
    encode = Conv2D(4, DEFAULT_KERNEL, activation="relu",
                    padding="same")(encode)

    # "encoded" is the encoded representation of the input, middle layer of the aue
    encoded = MaxPooling2D(
        DEFAULT_POOL_SIZE, padding="same", name="encoder")(encode)

    # layer between middle and output layer
    decode = Conv2D(4, DEFAULT_KERNEL, activation="relu", padding="same"
                    )(encoded)
    decode = UpSampling2D(DEFAULT_POOL_SIZE)(decode)
    decode = Conv2D(8, DEFAULT_KERNEL, activation="relu", padding="same"
                    )(decode)
    decode = UpSampling2D(DEFAULT_POOL_SIZE)(decode)
    decode = Conv2D(8, DEFAULT_KERNEL, activation="relu", padding="same"
                    )(decode)
    decode = UpSampling2D(DEFAULT_POOL_SIZE)(decode)
    decode = Conv2D(16, DEFAULT_KERNEL, activation="relu")(decode)
    decode = UpSampling2D(DEFAULT_POOL_SIZE)(decode)
    decoded = Conv2D(1, DEFAULT_KERNEL, activation="sigmoid",
                     padding="same")(decode)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_img, outputs=decoded)

    encoder, decoder = get_codec_from_aue(autoencoder)

    # build (aka "compile") the model
    autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
    return autoencoder, encoder, decoder


def get_codec_from_aue(autoencoder):
    encoder_layer = autoencoder.get_layer("encoder")
    # this model maps an input to its encoded representation; Big image to small rep
    encoder = Model(
        inputs=autoencoder.input, outputs=encoder_layer.output)

    # create a placeholder for an encoded (ENCODING_DIM-dimensional) input
    encoded_input = Input(shape=encoder_layer.output_shape[1:])

    # getting the middle of the autoencoder
    start = (len(autoencoder.layers))//2
    decoder = autoencoder.layers[-start](encoded_input)
    # stacking the decoder layers
    for i in range(start-1, 0, -1):
        decoder = autoencoder.layers[-i](decoder)

    # create the decoder model; "<": encoded(small) representation to big image
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
    autoencoder = load(ckpt_loc)
    encoder, decoder = get_codec_from_aue(autoencoder)

else:
    print("Training Autoencoder...")
    autoencoder, encoder, decoder = build_conv_aue()
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='min', min_delta=0.0005)
    mcp_save = ModelCheckpoint(ckpt_loc,
                               save_best_only=True, verbose=1, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='min')
    autoencoder.fit(X_train, X_train,
                    epochs=128,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_validate, X_validate), callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
    model_path = os.path.join(CUR_DIR, "model", "conv")
    autoencoder.save(os.path.join(model_path, "autoencoder.h5"))
    encoder.save(os.path.join(model_path, "encoder.h5"))
    decoder.save(os.path.join(model_path, "decoder.h5"))
print(autoencoder.summary())
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
plt.savefig(os.path.join("imgs", "conv-autoencoder.png"))

#####################################################################
print("CLASSIFIER")
#####################################################################
flat = np.prod(encoded_imgs_train.shape[1:], dtype=np.int64)
encoded_imgs_train = encoded_imgs_train.reshape(len(encoded_imgs_train), flat)
encoded_imgs_validate = encoded_imgs_validate.reshape(
    len(encoded_imgs_validate), flat)
encoded_imgs_test = encoded_imgs_test.reshape(len(encoded_imgs_test), flat)

ckpt_loc = os.path.join(CUR_DIR, "ckpts", "classifier.hdf5")
if(os.path.isfile(ckpt_loc)):
    print("Loading classifier from directory %s..." % ckpt_loc)
    classifier = load(ckpt_loc)
else:
    print("Training classifier...")
    classifier = build_classifier(input_dim=flat)
    earlyStopping = EarlyStopping(
        monitor='val_acc', patience=5, verbose=1, mode='max',  min_delta=0.0005)
    mcp_save = ModelCheckpoint(ckpt_loc,
                               save_best_only=True, verbose=1, monitor='val_acc', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_acc', factor=0.3, patience=3, verbose=1, mode='max')
    classifier.fit(encoded_imgs_train, y_train, validation_data=(
        encoded_imgs_validate, y_validate), batch_size=16, epochs=32, shuffle=True, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
print("Done")

eval_train = classifier.evaluate(encoded_imgs_train, y_train)
eval_validate = classifier.evaluate(encoded_imgs_validate, y_validate)
eval_test = classifier.evaluate(encoded_imgs_test, y_test)
print("Evaluation: Train", eval_train, "Validate",
      eval_validate, "Test",  eval_test)


def get_cm(input, y_true):
    """Computes confusion matrix."""
    y_pred = tf.argmax(classifier.predict(input), axis=1)
    y_true = tf.argmax(y_true, axis=1)

    c = tf.keras.backend.eval(y_pred)
    d = tf.keras.backend.eval(y_true)

    return confusion_matrix(c, d)


def precision(cm):
    results = []
    for i in range(len(cm)):  # rows
        TP = cm[i][i]
        fp_tp = np.sum(cm[i])
        results.append(TP/fp_tp)
    return results + [np.mean(results)]


def recall(cm):
    results = []
    for i in range(len(cm)):  # rows
        TP = cm[i][i]
        tp_fn = 0
        for j in range(len(cm[i])):
            tp_fn += cm[j][i]
        results.append(TP/tp_fn)
    return results + [np.mean(results)]


cm_train = get_cm(encoded_imgs_train, y_train)
print(cm_train, precision(cm_train), recall(cm_train), sep="\n")

cm_validate = get_cm(encoded_imgs_validate, y_validate)
print(cm_validate, precision(cm_validate), recall(cm_validate), sep="\n")

cm_test = get_cm(encoded_imgs_test, y_test)
print(cm_test, precision(cm_test), recall(cm_test), sep="\n")
