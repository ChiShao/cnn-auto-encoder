import os
import random

from tensorflow.python.lib.io import file_io


def read_image_file_names(dir_path):
    """Returns a list of absolute file paths for relative dir input with all relevant file names."""
    return [os.path.join(dir_path, p) for p in file_io.list_directory(dir_path)]


def get_path_to_data(data_path, USE_CASE, mode):
    return os.path.join(data_path, USE_CASE, mode)


def load_data(data_path, USE_CASE):
    normal_path = os.path.join(data_path, USE_CASE, "train", "good")
    normal_file_names = read_image_file_names(normal_path)
    random.shuffle(normal_file_names)

    train_validation_split = (len(normal_file_names) // 10) * 8  # equals .85
    train_file_names, validation_file_names = normal_file_names[
        :train_validation_split], normal_file_names[train_validation_split:]

    test_path = os.path.join(data_path, USE_CASE, "test")

    normal_test_file_names = read_image_file_names(
        os.path.join(test_path, "good"))

    anomaly_test_file_names = anomaly_train_file_names = []
    # iterate of test list dir because anomalies are named dynamically
    for i, p in enumerate(filter(lambda x: x != "good", file_io.list_directory(test_path))):
        if i == 0:
            anomaly_test_file_names += read_image_file_names(
                os.path.join(test_path, p))
        else:
            anomaly_train_file_names += read_image_file_names(
                os.path.join(test_path, p))

    print("Splitting data:\n%4d Normal Train Samples\n%4d Normal Validation Samples\n%4d Normal Test Samples\n%4d Anomaly Train Samples\n%4d Anomaly Test Samples" % (
        len(train_file_names), len(validation_file_names), len(normal_test_file_names), len(anomaly_train_file_names), len(anomaly_test_file_names)))

    return train_file_names, validation_file_names, normal_test_file_names, anomaly_train_file_names, anomaly_test_file_names
