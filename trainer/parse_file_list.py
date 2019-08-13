import tensorflow as tf
from tensorflow.io.gfile import GFile
import os
from urllib.parse import urlparse
import numpy as np
import base64


def load_data(data_path_normal, data_path_anomaly):
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
        tts_lines = int(.75 * len(lines))-1
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
    # get normal data

    with GFile(data_path_normal, "r") as f:
        lines = f.readlines()
        lines = [l[:-1] for l in lines]  # remove trailing \n char
    bucket = split_bucket_parts(lines[0])[0]
    pipes = get_pipes(lines)
    nop = len(pipes)  # number of pipes

    tts = int(.75 * nop)  # train test split
    train_pipes, _ = pipes[:tts], pipes[tts:]

    normal_train_file_paths, normal_test_file_paths = split_train_test(
        lines, train_pipes)
    train_validation_split = int(len(normal_train_file_paths)*.8)
    normal_validation_file_paths = normal_train_file_paths[train_validation_split:]
    normal_train_file_paths = normal_train_file_paths[:train_validation_split]

    # get anomaly data
    with GFile(data_path_anomaly, "r") as f:
        lines = f.readlines()
        file_paths = [split_bucket_parts(l[:-1])[1][-1] for l in lines]

    decoded_lines = []
    for l in file_paths:
        rest = os.path.join(*base64.b64decode(
            l[:-4]).decode("utf-8").split(os.sep)[-3:])+".jpg"
        new_path = os.path.join(bucket, "spm2", "out", "out", rest)
        decoded_lines.append(new_path)

    anomaly_train_file_paths, anomaly_test_file_paths = split_train_test(
        decoded_lines, train_pipes)

    print("Splitting data:\n%4d Normal Train Samples\n%4d Normal Validation Samples\n%4d Normal Test Samples\n%4d Anomaly Train Samples\n%4d Anomaly Test Samples" % (
        len(normal_train_file_paths), len(normal_validation_file_paths), len(normal_test_file_paths), len(anomaly_train_file_paths), len(anomaly_test_file_paths)))

    return normal_train_file_paths, normal_validation_file_paths, normal_test_file_paths, anomaly_train_file_paths, anomaly_test_file_paths


# base_path = os.path.join(os.getcwd(), "trainer")
# normal_train_file_paths, normal_validation_file_paths, normal_test_file_paths, anomaly_train_file_paths, anomaly_test_file_paths = load_data(os.path.join(
#     base_path, "no_damage.txt"),                                                                                                                                             os.path.join(base_path, "damage_encoded.txt"))
# print(normal_train_file_paths[:5])
