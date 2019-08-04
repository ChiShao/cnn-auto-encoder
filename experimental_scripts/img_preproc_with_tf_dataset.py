import os

import tensorflow as tf
from keras.models import load_model


# tf.enable_eager_execution()

USE_CASE = "cable"

ANOMALY_STR = "anomaly"
NORMAL_STR = "normal"
AUTOTUNE = tf.data.experimental.AUTOTUNE

data_path = os.path.join("data", "mvtec_anomaly_detection")


def read_image_file_names(dir_path):
    """Reads images per category for given directory path.
    Returns images as a generator object in intervall[0,1] rgb-format"""
    path_to_dir = os.path.join(dir_path)
    return [os.path.join(path_to_dir, p) for p in os.listdir(dir_path)]


def load_data_paths():
    normal_path = os.path.join(data_path, USE_CASE, "train", "good")
    normal_train_data = read_image_file_names(normal_path)

    test_path = os.path.join(data_path, USE_CASE, "test")
    normal_test_data = read_image_file_names(os.path.join(test_path, "good"))
    anomaly_test_data = []
    for p in os.listdir(test_path):
        if p != "good":
            anomaly_test_data += read_image_file_names(
                os.path.join(test_path, p))
    return normal_train_data, normal_test_data, normal_test_data, anomaly_test_data


X_normal_train, X_normal_validate, X_normal_test, X_anomaly_test = load_data_paths()
print(X_normal_train[0])  # contains only (absolute) file paths to the data
img_path = X_normal_train[0]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

path_ds = tf.data.Dataset.from_tensor_slices(X_normal_train)
print(path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
print(image_ds)
image_ds = tf.data.Dataset.zip((image_ds, image_ds))
print(image_ds)

BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)
ds_iter = ds.make_one_shot_iterator()
next_element = ds_iter.get_next()

model = Input((256,256,3))
model.compile(optimizer="adadelta",loss="mse")

with tf.Session() as sess:
    input_batch, output_batch = sess.run(next_element)
    feature_map_batch = model.train_generator(input_batch,steps=5)
    print(feature_map_batch.shape)