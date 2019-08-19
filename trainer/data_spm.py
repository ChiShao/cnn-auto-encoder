import base64
import os
from urllib.parse import urlparse

import numpy as np
import tensorflow.io.gfile as gfile

DATA_SET_FRAC = 50


def split_bucket_parts(path):
    url = urlparse(path)  # [:-1] cuts the \n from the lines
    bucket = "%s://%s" % (url.scheme, url.netloc)

    # [1:] removes leading / to avoid empty string in dir_parts
    dir_parts = os.path.normpath(url.path[1:]).split(os.sep)

    return bucket, dir_parts


def get_pipes(lines):
    splitted_lines = np.array([split_bucket_parts(l)[1] for l in lines])
    noc = list(set(splitted_lines[:, 2]))
    return noc


def split_train_test(lines, train_pipes):

    # find line without a train class in it
    tts_lines = int(.8 * len(lines))-1
    i = tts_lines
    for l in lines[tts_lines:]:
        if not any(tp in l for tp in train_pipes):
            break
        i += 1

    train_file_paths = lines[:i]
    test_file_paths = lines[i:]
    # print(len(train_file_paths)/len(lines),
    #   len(test_file_paths)/len(lines))
    return train_file_paths, test_file_paths


def load_data(data_path_normal, data_path_anomaly):

    # get normal data
    with gfile.GFile(data_path_normal, "r") as f:
        lines = f.readlines()
        lines = lines[0::DATA_SET_FRAC]
        lines = [l[:-1] for l in lines]  # remove trailing \n char
    bucket = split_bucket_parts(lines[0])[0]
    pipes = get_pipes(lines)  # pipe names e.g "1-11111 1-11110"
    nop = len(pipes)  # number of pipes

    # 80% of the pipe sections should be used for train and validation, therefor pipe section names are split
    # names of pipes which should be used for training and validation
    train_val_pipes = pipes[:int(.8 * nop)]

    normal_train_file_paths, normal_test_file_paths = split_train_test(
        lines, train_val_pipes)

    # names of pipes which should be used for training and validation
    train_pipes = pipes[:int(.8 * nop)]
    normal_train_file_paths, normal_validation_file_paths = split_train_test(
        lines[:len(normal_train_file_paths)], train_pipes)

    train_ratio = len(normal_train_file_paths)/len(lines)
    val_ratio = len(normal_validation_file_paths)/len(lines)
    test_ratio = len(normal_test_file_paths)/len(lines)

    # get anomaly data
    with gfile.GFile(data_path_anomaly, "r") as f:
        lines = f.readlines()
        lines = lines[0::(DATA_SET_FRAC//10)]
        file_paths = [split_bucket_parts(l[:-1])[1][-1] for l in lines]

    decoded_lines = []
    for l in file_paths:
        rest = os.path.join(*base64.b64decode(
            l[:-4]).decode("utf-8").split(os.sep)[-3:])+".jpg"
        new_path = os.path.join(bucket,  "out", rest)
        decoded_lines.append(new_path)

    anomaly_train_file_paths, anomaly_test_file_paths = split_train_test(
        decoded_lines, train_pipes)

    print("Splitting data:\n%4d Normal Train Samples\n%4d Normal Validation Samples\n%4d Normal Test Samples\n%4d Anomaly Train Samples\n%4d Anomaly Test Samples" % (
        len(normal_train_file_paths), len(normal_validation_file_paths), len(normal_test_file_paths), len(anomaly_train_file_paths), len(anomaly_test_file_paths)))
    print("Train-Val-Test Ratio: %.3f-%.3f-%.3f" %
          (train_ratio, val_ratio, test_ratio))
    return normal_train_file_paths, normal_validation_file_paths, normal_test_file_paths, anomaly_train_file_paths, anomaly_test_file_paths
