import argparse
import os

import numpy as np
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.optimizers import Adam

from trainer.util import get_anchors, get_classes, data_generator_wrapper

from trainer import util
from trainer.model import create_model
from tensorflow.python.lib.io import file_io

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--model', type=str,
    help='path to model weight file', default="model_data/weights.h5"
)

parser.add_argument(
    '--anchors', type=str,
    help='path to anchor definitions', default="model_data/yolo_anchors.txt"
)

parser.add_argument(
    '--classes', type=str,
    help='path to class definitions', default="model_data/classes.txt"
)

parser.add_argument(
    '--logdir', type=str,
    help='path to log Tensorboard data to', default='logs'
)

parser.add_argument(
    '--train_data', type=str,
    help='path to training data file', default='data/train.txt'
)

parser.add_argument(
    '--gross_epochs', type=int,
    help='num of epochs for gross training', default=50
)

parser.add_argument(
    '--fine_epochs', type=int,
    help='num of epochs for gross training', default=50
)

parser.add_argument(
    '--gross_batch_size', type=int,
    help='batch size for gross training', default=32
)

parser.add_argument(
    '--fine_batch_size', type=int,
    help='batch size for fine training', default=12
)

parser.add_argument(
    '--job-dir', type=str,
    help='save stuff there', default="."
)


parser.add_argument(
    '--output', type=str,
    help='output path for model', default='model/pet_yolo.h5'
)

flags = parser.parse_args()

class_names = get_classes(flags.classes)
num_classes = len(class_names)
anchors = get_anchors(flags.anchors)

input_shape = (416, 416)  # multiple of 32, hw

model = create_model(input_shape, anchors, num_classes,
                     freeze_body=2, weights_path=flags.model)  # make sure you know what you freeze

logging = TensorBoard(log_dir=flags.logdir)
checkpoint_format = os.path.join(
    flags.logdir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
checkpoint = ModelCheckpoint(checkpoint_format,
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.1, patience=10, verbose=1)

val_split = 0.2
with file_io.FileIO(flags.train_data, mode='r') as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
model.compile(optimizer=Adam(lr=1e-3), loss={
    # use custom yolo_loss Lambda layer.
    'yolo_loss': lambda y_true, y_pred: y_pred})

batch_size = flags.gross_batch_size
print('Train on {} samples, val on {} samples, with batch size {}.'.format(
    num_train, num_val, batch_size))
model.fit_generator(util.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=util.data_generator_wrapper(
                        lines[num_train:], batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=flags.gross_epochs,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint, early_stopping])

weights_out = os.path.join(flags.logdir, 'trained_weights_stage_1.h5')
model.save_weights(weights_out)

# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.

for i in range(len(model.layers)):
    model.layers[i].trainable = True
# recompile to apply the change
model.compile(optimizer=Adam(
    lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
print('Unfreeze all of the layers.')

# note that more GPU memory is required after unfreezing the body
batch_size = flags.fine_batch_size
print('Train on {} samples, val on {} samples, with batch size {}.'.format(
    num_train, num_val, batch_size))
model.fit_generator(util.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=util.data_generator_wrapper(
                        lines[num_train:], batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=flags.fine_epochs,
                    initial_epoch=flags.gross_epochs,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])


weights_out = os.path.join(flags.logdir, flags.output)
model.save_weights(weights_out)
