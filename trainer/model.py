import io
import os
import time
from contextlib import redirect_stdout
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.multi_gpu_utils import multi_gpu_model

import cv2
import trainer.plot_tools

from .gpu_utils import ModelCkptMultiGPU, get_available_gpus

# set random seed for redproducible results
np.random.seed(42)
tf.set_random_seed(42)

# set CONSTANTS
AUTOTUNE = tf.data.experimental.AUTOTUNE

ANOMALY_STR = "anomaly"
NORMAL_STR = "normal"
USE_CASE = "cable"

# define paths for reading data, saving imgs, ckpts and logs
img_dir = os.path.join("imgs", USE_CASE)
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)

cpkt_dir = os.path.join("ckpts", USE_CASE)
if not os.path.isdir(cpkt_dir):
    os.mkdir(cpkt_dir)
data_path = os.path.join(os.getcwd(), "data", "mvtec_anomaly_detection")

base_log_dir = os.path.join("logs", USE_CASE)
if not os.path.isdir(base_log_dir):
    os.mkdir(base_log_dir)


def read_image_file_names(dir_path):
    """Returns a list of absolute file paths for relative dir input with all relevant file names."""
    path_to_dir = os.path.join(os.getcwd(), dir_path)
    return [os.path.join(path_to_dir, p) for p in os.listdir(dir_path)]


def get_path_to_data(mode):
    return os.path.join(data_path, USE_CASE, mode)


def load_data():
    normal_path = os.path.join(data_path, USE_CASE, "train", "good")
    normal_train_data = read_image_file_names(normal_path)

    test_path = os.path.join(data_path, USE_CASE, "test")
    normal_test_data = read_image_file_names(os.path.join(test_path, "good"))
    anomaly_test_data = []

    # iterate of test list dir because anomalies are named dynamically
    for p in filter(lambda x: x != "good", os.listdir(test_path)):
        anomaly_test_data += read_image_file_names(os.path.join(test_path, p))
    return normal_train_data, normal_test_data,  anomaly_test_data


def train_img_generator(file_paths, batch_size, input_only=False):
    # yields batch_size-d arrays of images indefinetly
    # order of images is random.
    # ONLY FOR TRAIN AND EVALUATE PURPOSES SUITABLE
    """DEPRECATED: keras img preproc is used for train purposes.
    Still used for evaluation purposes."""

    while True:
        inds = (np.random.randint(0, len(file_paths), batch_size))
        imgs = np.array([])
        for i in inds:
            fp = file_paths[i]
            if os.path.isfile(fp):
                img = filepath_to_image(fp)
                imgs = np.concatenate(
                    (imgs, np.array([img]))) if imgs.size > 0 else np.array([img])
        if not input_only:
            yield imgs / 255, imgs / 255
        else:
            yield imgs / 255


def filepath_to_image(fp):
    img = cv2.imread(fp)  # reads an image in the BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img = cv2.resize(img, dsize=(256, 256),
                     interpolation=cv2.INTER_NEAREST)
    return img


def img_generator(file_paths, n):
    # returns generator of n images
    if len(file_paths) < n:
        # prevent an IndexError
        n = len(file_paths)

    for i in range(n):
        fp = file_paths[i]
        if os.path.isfile(fp):
            yield filepath_to_image(fp) / 255


def get_codec_from_ae(autoencoder):
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


def build_conv_ae(filters, input_shape=(256, 256, 3)):
    n = 9
    if len(filters) != n:
        raise ValueError("%d Filters must be given. Sorry." % n)
    # this is our input placeholder
    input_img = Input(shape=input_shape)
    # layer between input and middle layer
    i = 0
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(input_img)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(encode)
    i += 1
    encode = Conv2D(
        filters[i], (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(encode)
    i += 1
    encode = Conv2D(
        filters[i], (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(encode)
    i += 1
    encode = Conv2D(
        filters[i], (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(encode)
    i += 1
    # "encoded" is the encoded representation of the input, middle layer of the aue
    encoded = Conv2D(
        filters[i], (8, 8), strides=(1, 1), activation="softmax", name="encoder"
    )(encode)

    i -= 1
    # layer between middle and output layer
    decode = Conv2DTranspose(filters[i], (8, 8), strides=(1, 1), activation="relu")(
        encoded
    )
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (3, 3), strides=(1, 1), activation="relu", padding="same"
    )(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2), activation="relu", padding="same"
    )(decode)
    i -= 1

    decoded = Conv2D(
        input_shape[-1], (3, 3), activation="sigmoid", padding="same"
    )(decode)

    # this model maps an input to its reconstruction
    autoencoder_single = Model(inputs=input_img, outputs=decoded)
    n_gpus = len(get_available_gpus())
    if n_gpus > 1:
        autoencoder = multi_gpu_model(autoencoder_single, n_gpus)
    else:
        autoencoder = Model(inputs=input_img, outputs=decoded)

    encoder, decoder = get_codec_from_ae(autoencoder_single)

    # build (aka "compile") the model
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder_single.compile(optimizer="adam", loss="mse")
    return autoencoder, autoencoder_single, encoder, decoder


def preproc_data(input_shape, batch_size):
    print("Preprocessing data to shape:", input_shape)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2,
        rotation_range=180,
        brightness_range=(0.2, 1),
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        get_path_to_data("train"),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='input',
        subset="training"
    )

    validation_generator = train_datagen.flow_from_directory(
        get_path_to_data("train"),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='input',
        subset="validation"
    )
    test_generator = ImageDataGenerator(rescale=1./255.0).flow_from_directory(
        get_path_to_data("test"),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='input',
        classes=["good"])

    return train_generator, validation_generator, test_generator


def preproc_data(input_shape, batch_size):
    print("Preprocessing data to shape:", input_shape)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2,
        rotation_range=45,
        brightness_range=(0.5, 1),
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        get_path_to_data("train"),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='input',
        subset="training"
    )

    validation_generator = train_datagen.flow_from_directory(
        get_path_to_data("train"),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='input',
        subset="validation"
    )
    test_generator = ImageDataGenerator(rescale=1./255.0).flow_from_directory(
        get_path_to_data("test"),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='input',
        classes=["good"])

    return train_generator, validation_generator, test_generator


# latent_space_d=128, batch_size=8, ):
def train(train_generator, validation_generator,  train_args, log_dir="logs", ckpt_dir="ckpts", input_shape=(256, 256, 3)):
    print("Writing logs to %s" % log_dir)
    print("Training Autoencoder for %s feature extraction..." % USE_CASE)
    d = train_args.latent_space
    batch_size = train_args.batch_size
    #filters = [16, 16, 16, 32, 64, 64, 32, 32, d]

    ae, ae_single, encoder, decoder = build_conv_ae(
        input_shape=input_shape, filters=train_args.filters)
    # define callbacks for logging and optimized training and ckpt saving

    earlyStopping = EarlyStopping(
        monitor="val_loss", patience=15, verbose=1, mode="min", min_delta=(1/10**5)
    )
    # saves model which was trained on a single cpu since saving the other model threw some kind of error
    mcp_save = ModelCkptMultiGPU(
        ckpt_dir, ae_single, save_best_only=True, verbose=1, monitor="val_loss", mode="min"
    )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1, mode="min"
    )

    tb = TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )

    f = io.StringIO()
    with redirect_stdout(f):
        ae.summary()
    summary = f.getvalue()
    print(summary)

    with open(os.path.join(log_dir, "train-specs%.0f.txt" % time.time()), "w") as f:
        f.write("\n".join(("Filters: {}".format(train_args.filters),
                           "Batch Size: %d" % batch_size,
                           "Latent Space Dim: %d" % d,
                           "Auto Encoder Network %s" % summary)))
    ae.fit_generator(
        generator=train_generator,  # needs to produce data infinitely
        epochs=train_args.epochs,
        steps_per_epoch=len(train_generator) * batch_size,  # every element once on average
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)*batch_size,
        callbacks=[mcp_save, earlyStopping, reduce_lr_loss, tb]
    )
    return ae, encoder, decoder


def load(ckpt_dir):
    print("Loading Autoencoder for %s feature extraction from directory %s..." % (
        USE_CASE, ckpt_dir))
    ae = load_model(ckpt_dir)

    # this needs to be compiled since the untrained, single-GPU model is saved instead of the multi-GPU model
    ae.compile(optimizer="adadelta", loss="mse")
    encoder, decoder = get_codec_from_ae(ae)
    # ae.summary()


def main(args):
    X_normal_train, X_normal_test, X_anomaly_test = load_data()
    input_shape = (256, 256, 3)
    batch_size = args.batch_size  # 10

    train_generator, validation_generator, test_generator = preproc_data(
        input_shape, batch_size)

    ckpt_path = os.path.join(ckpt_dir, "cae%.0f.hdf5" % time.time())
    log_dir = os.path.join(base_log_dir, str(len(os.listdir(base_log_dir))))
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # if False and os.path.isfile(ckpt_dir):
    if os.path.isfile(ckpt_path):
        ae, encoder, decoder = train(
            train_generator, validation_generator, args)
    else:
        ae, encoder, decoder = load(ckpt_path)


# main()
