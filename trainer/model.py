import io
import json
import os
import random
import subprocess
from contextlib import redirect_stdout
from urllib.parse import urlparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Conv2D, Conv2DTranspose, Input, LeakyReLU
from keras.models import Model, load_model
from keras.utils.multi_gpu_utils import multi_gpu_model
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn import svm
from tensorflow.python.lib.io import file_io
import tensorflow.io.gfile as gfile


import trainer.data_spm
# import trainer.data_mvtec

from trainer.gpu_utils import get_available_gpus
from trainer.model_ckpt_gc import ModelCheckpointGC
from trainer.plot_utils import plot_hist, plot_mvtec, plot_samples, savefig

# set random seed for redproducible results
RAND = 42
np.random.seed(RAND)
tf.set_random_seed(RAND)
random.seed(RAND)

# initialise var USE_CASE for using it globally
USE_CASE = None


def train_img_generator(file_paths, batch_size, target_size, preproc=True):
    while True:
        inds = (np.random.randint(0, len(file_paths), batch_size))
        imgs = np.array([])
        for i in inds:
            fp = file_paths[i]
            # print(fp)
            if gfile.Exists(fp):
                img = filepath_to_image(fp)
                img = cv2.resize(img, target_size[:-1])
                if preproc:
                    np_img = preproc_img(img, flip_top_bottom=False, hsv=None)
                else:
                    np_img = np.array(img) / 255

                imgs = np.concatenate(
                    (imgs, np.array([np_img]))) if imgs.size > 0 else np.array([np_img])
            yield imgs, imgs


def filepath_to_image(fp):
    with gfile.Gfile(fp, mode="rb") as f:
        # img = Image.open(f)
        img = cv2.imdecode(np.asarray(bytearray(f.read())), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert2rgb
    return img


def rand(a=0, b=1):
    """:returns random value between a and b"""
    return np.random.rand()*(b-a) + a


def preproc_img(image, flip_top_bottom=True, flip_left_right=True, hsv=(.1, 1.5, 1.5)):
    """https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/utils.py"""
    # flip image or not
    if flip_left_right and rand() < .5:
        # image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = cv2.flip(image, flipCode=0)
    if flip_top_bottom and rand() < .5:
        # image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = cv2.flip(image, flipCode=1)

    # image brighDtness enhancer
    # if brightness:
        # brightness = ImageEnhance.Brightness(image)

        # image = brightness.enhance(rand(.5, 1.5))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # brightness and contrast

    image = np.array(image)
    image = np.clip((rand(.5, 1.5) * image + rand(-50, 100)), 0, 255)
    image = image.astype(np.uint8)

    # # increase or decrease contrast
    # if contrast:
    #     contrast = ImageEnhance.Contrast(image)
    #     image = contrast.enhance(rand(.5, 1.5))

    image = image / 255.

    if hsv != None:
        hue, sat, val = hsv
        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)

        x = rgb_to_hsv(image)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)
    return image  # numpy array, 0 to 1


def img_generator(file_paths, n, target_size):
    # returns generator of n images
    if len(file_paths) < n:
        # prevent an IndexError
        n = len(file_paths)
    for i in range(n):
        fp = file_paths[i]
        if gfile.Exists(fp):
            img = filepath_to_image(fp)
            # img = img.resize(target_size[:-1], Image.BICUBIC)
            img = cv2.resize(img, target_size[:-1])
            yield np.array(img) / 255


def split_ae(autoencoder):
    encoder_layer = autoencoder.get_layer("encoder")
    # this model maps an input to its encoded representation; Big image to small rep
    encoder = Model(
        inputs=autoencoder.get_input_at(0), outputs=encoder_layer.output)

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
    n = 10
    print(filters)
    if len(filters) != n:
        raise ValueError("%d Filters must be given. Sorry." % n)
    # this is our input placeholder
    input_img = Input(shape=input_shape)
    # layer between input and middle layer
    i = 0
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2), padding="same"
    )(input_img)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)

    i += 1
    encode = Conv2D(
        filters[i], (3, 3), strides=(1, 1),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    encode = Conv2D(
        filters[i], (3, 3), strides=(1, 1),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    encode = Conv2D(
        filters[i], (3, 3), strides=(1, 1),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    encode = Conv2D(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(encode)
    encode = LeakyReLU(0.2)(encode)
    i += 1
    # "encoded" is the encoded representation of the input, middle layer of the aue
    encoded = Conv2D(
        filters[i], (4, 4), strides=(1, 1),  name="encoder"
    )(encode)

    i -= 1
    # layer between middle and output layer
    decode = Conv2DTranspose(filters[i], (4, 4), strides=(1, 1), activation="relu")(
        encoded
    )
    decode = LeakyReLU(0.2)(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(decode)
    decode = LeakyReLU(0.2)(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (3, 3), strides=(1, 1),  padding="same"
    )(decode)
    decode = LeakyReLU(0.2)(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(decode)
    decode = LeakyReLU(0.2)(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(decode)
    decode = LeakyReLU(0.2)(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(decode)
    decode = LeakyReLU(0.2)(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (3, 3), strides=(1, 1),  padding="same"
    )(decode)
    i -= 1
    decode = LeakyReLU(0.2)(decode)
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(decode)
    decode = LeakyReLU(0.2)(decode)
    i -= 1
    decode = Conv2DTranspose(
        filters[i], (4, 4), strides=(2, 2),  padding="same"
    )(decode)
    decode = LeakyReLU(0.2)(decode)

    decoded = Conv2D(
        input_shape[-1], (3, 3), activation="sigmoid", padding="same"
    )(decode)

    # this model maps an input to its reconstruction
    autoencoder_single = Model(inputs=input_img, outputs=decoded)
    n_gpus = len(get_available_gpus())
    if n_gpus > 1:
        autoencoder = multi_gpu_model(autoencoder_single, n_gpus)
        print("Autoencoder is trained on %d GPUs" % n_gpus)
    else:
        autoencoder = autoencoder_single
        print("Autoencoder is trained on a single GPU")

    encoder, decoder = split_ae(autoencoder_single)

    # build (aka "compile") the model
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder_single.compile(optimizer="adam", loss="mse")
    return autoencoder, autoencoder_single, encoder, decoder


# def preproc_data(input_shape, batch_size, data_path):
#     print("Preprocessing data, shape:", input_shape)
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         validation_split=.2,
#         rotation_range=180,
#         brightness_range=(0.2, 1),
#         horizontal_flip=True
#     )
#     # print(get_path_to_data(data_path, "train"))
#     train_generator = train_datagen.flow_from_directory(
#         get_path_to_data(data_path, "train"),
#         target_size=input_shape[:2],
#         batch_size=batch_size,
#         class_mode='input',
#         subset="training"
#     )

#     validation_generator = train_datagen.flow_from_directory(
#         get_path_to_data(data_path, "train"),
#         target_size=input_shape[:2],
#         batch_size=batch_size,
#         class_mode='input',
#         subset="validation"
#     )
#     test_generator = ImageDataGenerator(rescale=1./255.0).flow_from_directory(
#         get_path_to_data(data_path, "test"),
#         target_size=input_shape[:2],
#         batch_size=batch_size,
#         class_mode='input',
#         classes=["good"])

#     return train_generator, validation_generator, test_generator

def train_ae(train_file_names, validation_file_names,  args,  target_size=(256, 256, 3)):
    print("Writing logs to %s" % args.logdir)
    train_generator = train_img_generator(
        train_file_names, args.batch_size, target_size)
    validation_generator = train_img_generator(
        validation_file_names, args.batch_size, target_size, preproc=False)

    print("Training Autoencoder for %s feature extraction..." % USE_CASE)
    d = args.ldim
    batch_size = args.batch_size
    filters = args.filters + [d]  # [16, 16, 16, 32, 64, 64, 32, 32, d]

    ae, ae_single, encoder, decoder = build_conv_ae(
        input_shape=target_size, filters=filters)
    # define callbacks for logging and optimized training and ckpt saving

    earlyStopping = EarlyStopping(
        monitor="val_loss", patience=20, verbose=1, mode="min", min_delta=(1/10**5)
    )
    # saves model which was trained on a single cpu since saving the other model threw some kind of error
    checkpoint_format = os.path.join(
        args.ckptdir, 'ep{epoch:04d}-loss{loss:.6f}-val_loss{val_loss:.6f}.h5')

    mcp_save = ModelCheckpointGC(
        checkpoint_format, ae_single, save_best_only=True, verbose=1, monitor="val_loss", mode="min"
    )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=7, verbose=1, mode="min"
    )

    tb = TensorBoard(
        log_dir=args.logdir, histogram_freq=0, write_graph=True, write_images=True
    )

    f = io.StringIO()
    with redirect_stdout(f):
        ae_single.summary()
    summary = f.getvalue()
    print(summary)

    with gfile.GFile(os.path.join(args.logdir, "train-specs.txt"), mode="w") as f:
        f.write("\n".join(("Filters: {}".format(args.filters),
                           "Batch Size: %d" % batch_size,
                           "Latent Space Dim: %d" % d,
                           "Auto Encoder Network %s" % summary)))

    train_length = len(train_file_names)
    val_length = len(validation_file_names)

    ae.fit_generator(
        generator=train_generator,  # needs to produce data infinitely
        epochs=args.epochs,
        # every element once on average
        steps_per_epoch=train_length//batch_size,
        validation_data=validation_generator,
        validation_steps=val_length//batch_size,
        callbacks=[mcp_save, earlyStopping, reduce_lr_loss, tb]
    )

    return ae, ae_single, encoder, decoder


def load(ckpt_path):
    print("Loading Autoencoder for %s feature extraction from directory %s..." % (
        USE_CASE, ckpt_path))

    url = urlparse(ckpt_path)  # [:-1] cuts the \n from the lines
    bucket = "%s://%s" % (url.scheme, url.netloc)

    # [1:] removes leading / to avoid empty string in dir_parts
    dir_parts = os.path.normpath(url.path[1:]).split(os.sep)

    fp = os.path.join("tmp", dir_parts[-1])
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")

    if not os.path.isfile(fp):
        subprocess.call(["gsutil", "-m", "cp", ckpt_path, "tmp"])
    ae = load_model(fp)

    # this needs to be compiled since the untrained, single-GPU model is saved instead of the multi-GPU model
    ae.compile(optimizer="adam", loss="mse")
    encoder, decoder = split_ae(ae)
    return ae, encoder, decoder


def predict_from_generator(model, gen, batch_size):
    predictions = []
    for _ in gen:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(gen))
            except StopIteration:
                pass
        pred = model.predict(np.array(batch))
        # pred[0], because prediction happens on an array, hence result has also length one
        predictions.extend(pred)
        print(len(predictions), "predictions made")
    return predictions


def evaluate(ae, loss_boundary, oc_svm,  X_normal_test, X_anomaly_test, img_dir, target_size):
    # load the data for evaluation purposes
    print("Evaluating feature extractor...")
    # eval_train = model.evaluate_generator(
    #     train_img_generator(X_normal_train, 8), steps=len(X_normal_train))
    batch_size = 64

    eval_test = ae.evaluate_generator(
        train_img_generator(X_normal_test, batch_size, target_size, preproc=False), steps=len(X_normal_test)//batch_size)
    # print("Feature extractor train loss: %f" % eval_train)

    print("Feature extractor  test loss: %f" % eval_test)

    decoded_samples_normal = predict_from_generator(
        ae, img_generator(X_normal_test, 8, target_size), batch_size)

    plot_samples(
        img_generator(X_normal_test, 8, target_size),
        decoded_samples_normal,
        plot_mvtec,
        os.path.join(img_dir, "rec-normals.png"),
    )

    decoded_samples_anomaly = predict_from_generator(
        ae, img_generator(X_anomaly_test, batch_size, target_size), batch_size)

    plot_samples(
        img_generator(X_anomaly_test, batch_size, target_size),
        decoded_samples_anomaly,
        plot_mvtec,
        os.path.join(img_dir, "rec-anomalies.png"),
    )

    metrics = {}

    print("Evaluating the loss based approach")
    metrics["loss"] = eval_loss(model=ae, X_normal=X_normal_test,
                                X_anomaly=X_anomaly_test, loss_boundary=loss_boundary, img_dir=img_dir, target_size=target_size)
    print("Evaluating the OC SVM")

    encoder, _ = split_ae(ae)
    metrics["svm"] = eval_svm(encoder, oc_svm,
                              X_normal_test, X_anomaly_test, img_dir, target_size, batch_size)
    print(metrics)
    return metrics


def get_metrics(TP, TN, FP, FN):
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0

    accuracy = (TN + TP) / (TP + TN + FN + FP)
    TPR = recall
    try:
        TNR = TN / (TN + FP)
    except ZeroDivisionError:
        TNR = 0

    FNR = 1 - TPR
    FPR = 1 - TNR

    try:
        F_1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        F_1 = 0
    try:
        F_2 = 5 * (precision * recall) / (4*precision + recall)
    except ZeroDivisionError:
        F_2 = 0

    try:
        # between -1 and 1
        MCC = (TP*TN - FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    except ZeroDivisionError:
        MCC = 0
    current_metrics = {"precision": precision,
                       "accuracy": accuracy,
                       "recall": recall,
                       "F1_score": F_1,
                       "F2_score": F_2,
                       "TPR": TPR,
                       "TNR": TNR,
                       "FNR": FNR,
                       "FPR": FPR,
                       "MCC": MCC
                       }
    return current_metrics


def ROC_curve(metrics, file_path):
    fig = plt.figure()
    plt.plot([m["FPR"] for m in metrics],
             [m["TPR"] for m in metrics])
    plt.grid()
    plt.xlabel("FALSE POSITIVE RATE")
    plt.ylabel("TRUE POSITIVE RATE")
    savefig(fig, file_path)

    plt.clf()


def loss_per_img(img, rec_img):
    # mean squared error
    return np.sum(np.power(rec_img - img, 2))  # / (np.prod(img.shape) - 1)


def get_losses(model, X_normal, X_anomaly, target_size):
    batch_size = 64
    print("Computing losses for normal samples.")
    samples_normal = img_generator(
        X_normal, len(X_normal), target_size
    )
    print("Reconstructing normal images.")
    decoded_samples_normal = predict_from_generator(
        model, samples_normal, batch_size=batch_size)
    print("Computing normal losses per image")
    normal_losses = np.array([loss_per_img(i, ri)for i, ri in zip(
        img_generator(X_normal, len(X_normal), target_size), decoded_samples_normal)])

    print("Computing losses for anomalous samples.")
    print("Reconstructing anomalous images.")
    decoded_samples_anomaly = predict_from_generator(model, img_generator(
        X_anomaly, len(X_anomaly), target_size
    ), batch_size=batch_size)

    print("Computing anomaly losses per image")
    anomaly_losses = np.array([loss_per_img(i, ri)for i, ri in zip(
        img_generator(X_anomaly, len(X_anomaly), target_size), decoded_samples_anomaly)])
    return normal_losses, anomaly_losses


def eval_loss(model, X_normal, X_anomaly, loss_boundary, img_dir, target_size):
    """anomaly detection based on the loss"""

    # compute losses on the test set
    normal_losses, anomaly_losses = get_losses(model,
                                               X_normal, X_anomaly, target_size)
    bins = 10
    # loss distribution over the normal dataset
    fig = plt.figure()
    label = "Distribution of normal loss values"
    plot_hist(normal_losses, relative=True,
              color="g", bins=bins, label=label)

    label = "Distribution of normal loss values"
    # loss distribution over the anomaly dataset
    plot_hist(anomaly_losses, relative=True,
              color="r", bins=bins, label=label)

    savefig(fig, os.path.join(img_dir, "loss-dist.png"))
    plt.clf()

    # ground truth: negatives = anomaly
    TP = anomaly_losses[anomaly_losses >= loss_boundary]
    FN = anomaly_losses[anomaly_losses < loss_boundary]

    # ground truth: positives = normality
    TN = normal_losses[normal_losses < loss_boundary]
    FP = normal_losses[normal_losses >= loss_boundary]

    # ROC_curve(metrics, os.path.join(img_dir, "loss-ROC.png"))
    return get_metrics(len(TP), len(TN), len(FP), len(FN))


def eval_svm(encoder, oc_svm,   X_normal, X_anomaly, img_dir, target_size, batch_size):

    print("Encoding normal test images to latent space...")
    encoded_normal_imgs_test = encoder.predict_generator(
        train_img_generator(X_normal, batch_size, target_size, preproc=False), steps=len(X_normal)/batch_size)
    print("Encoding anomaly test images to latent space...")
    encoded_anomaly_imgs_test = encoder.predict_generator(
        train_img_generator(X_anomaly, batch_size, target_size, preproc=False), steps=len(X_anomaly)/batch_size)

    # reshape to suitable shape for OCC
    encoded_normal_imgs_test = encoded_normal_imgs_test.reshape(
        -1, np.prod(encoded_normal_imgs_test.shape[1:]))
    encoded_anomaly_imgs_test = encoded_anomaly_imgs_test.reshape(
        - 1, np.prod(encoded_anomaly_imgs_test.shape[1:]))

    score_normal = oc_svm.predict(encoded_normal_imgs_test)
    score_anomaly = oc_svm.predict(encoded_anomaly_imgs_test)

    bins = 10
    # loss distribution over the normal dataset
    fig = plt.figure()
    label = "Distribution of the normal svm score values"
    plot_hist(oc_svm.decision_function(encoded_normal_imgs_test), relative=True,
              color="g", bins=bins, label=label)

    label = "Distribution of normal svm score values"
    # loss distribution over the anomaly dataset
    plot_hist(oc_svm.decision_function(encoded_anomaly_imgs_test), relative=True,
              color="r", bins=bins, label=label)

    savefig(fig, os.path.join(img_dir, "svm-score-dist.png"))
    # clear figure
    plt.clf()

    # Normalities are negatives (no finding)

    # ground truth: anomaly
    TP = len(score_anomaly[score_anomaly == -1])
    FN = len(score_anomaly[score_anomaly == 1])

    # ground truth: Normality
    TN = len(score_normal[score_normal == 1])
    FP = len(score_normal[score_normal == -1])

    # ROC_curve(metrics, os.path.join(img_dir, "svm-ROC.png"))
    return get_metrics(TP, TN, FP, FN)

def train_loss(model, normal_paths, anomaly_paths, target_size):
    print("Training the loss.")
    normal_losses, anomaly_losses = get_losses(
        model, normal_paths, anomaly_paths, target_size)
    # generate new samples of normals and anomalies to determine the best boundary on the train set
    # samples_normal = np.array(
    #    list(img_generator(normal_paths, len(normal_paths), target_size)))  
    # samples_normal_generator = img_generator(normal_paths, len(normal_paths), target_size)
    # samples_anomaly = np.array(
    #   list(img_generator(anomaly_paths, len(anomaly_paths), target_size)))
    
    best_m = -1
    best_boundary = 0
    print("Determining best loss boundary for train set.")
    decisive_metric = "MCC"
    for loss_boundary in normal_losses:
        # ground truth: positives = normality
        TP = normal_losses[normal_losses < loss_boundary]
        FN = normal_losses[normal_losses >= loss_boundary]

        # ground truth: negatives = anomaly
        TN = anomaly_losses[anomaly_losses >= loss_boundary]
        FP = anomaly_losses[anomaly_losses < loss_boundary]
        
        current_metrics = get_metrics(
            len(TP), len(TN), len(FP), len(FN))

        # less or equal since we want the biggest TP_rate (i)
        if current_metrics[decisive_metric] >= best_m:
            best_m = current_metrics[decisive_metric]
            best_boundary = loss_boundary
            print("New best metric:",best_m,"Boundary Loss = ", best_boundary)

    return best_boundary

def train_svm(model, normal_file_names,  target_size, batch_size):
    print("Training the OC SVM")
    print("Encoding train images to latent space...")
    # encode normal instances to latent space
    encoded_normal_imgs_train = model.predict_generator(
        train_img_generator(normal_file_names, batch_size, target_size, preproc=False), steps=len(normal_file_names) / batch_size)  # used later for One Class Classification

    encoded_normal_imgs_train = encoded_normal_imgs_train.reshape(
        - 1, np.prod(encoded_normal_imgs_train.shape[1:]))

    # # encode anomalies to latent space
    # encoded_anomaly_imgs_train = model.predict_generator(
    #     train_img_generator(anomaly_file_names, batch_size, target_size, preproc=False), steps=len(anomaly_file_names)/batch_size)  # used later for One Class Classification
    # encoded_anomaly_imgs_train = encoded_anomaly_imgs_train.reshape(
    #     - 1, np.prod(encoded_anomaly_imgs_train.shape[1:]))

    print("Fitting the OC SVM")
    clf = svm.OneClassSVM()
    # TODO: evaluate different nu values
    clf.fit(encoded_normal_imgs_train)

    # # computes signed distance to the hyperplane
    # score_normal = clf.predict(encoded_normal_imgs_train)
    # score_anomaly = clf.predict(encoded_anomaly_imgs_train)

    # # ground truth: anomaly
    # TP = len(score_anomaly[score_anomaly == -1])
    # FN = len(score_anomaly[score_anomaly == 1])

    # # ground truth: Normality
    # TN = len(score_normal[score_normal == 1])
    # FP = len(score_normal[score_normal == -1])

    # current_metrics = get_metrics(TP, TN, FP, FN)
    # print("SVM train results:", current_metrics)

    return clf


def train_and_evaluate(args):

    global USE_CASE
    USE_CASE = args.use_case

    # define train params
    target_size = (256, 256, 3)

    train_file_names, validation_file_names, normal_test_file_names, anomaly_train_file_names, anomaly_test_file_names = trainer.data_spm.load_data(
        os.path.join(args.datadir, "no_damage.txt"), os.path.join(args.datadir, "damage_encoded.txt"))


    ae_multi, ae, _, _ = train_ae(
        train_file_names, validation_file_names,  args, target_size=target_size)

    # ae, _, _ = load(
    #     "gs://e4u-anomaly-detection/ckpts/AD_spm_20190813_144154/ep0057-loss0.009716-val_loss0.015439.h5")

    loss_boundary = train_loss(
        ae, train_file_names, anomaly_train_file_names, target_size)

    encoder = split_ae(ae)[0]
    oc_svm = train_svm(encoder, train_file_names,
                       target_size, batch_size=args.batch_size)

    # returns metrics dictionary
    metrics = evaluate(ae, loss_boundary, oc_svm, normal_test_file_names,
                       anomaly_test_file_names, args.imgdir, target_size)

    with file_io.FileIO(os.path.join(args.evaldir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
