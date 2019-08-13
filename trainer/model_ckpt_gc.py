import os
import warnings

import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.python.lib.io import file_io


class ModelCheckpointGC(ModelCheckpoint):
    """Taken from and modified:
    https://github.com/keras-team/keras/blob/tf-keras/keras/callbacks.py
    """

    def __init__(self, filepath, model, monitor='val_loss', verbose=0,
                save_best_only=False, save_weights_only=False,
                mode='auto', period=1):
        #super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.model_to_save = model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                        'fallback to auto mode.' % (mode),
                        RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            if is_development():
                                self.model_to_save.save(filepath, overwrite=True)
                            else:
                                self.model_to_save.save(filepath.split(
                                    "/")[-1])
                                with file_io.FileIO(filepath.split(
                                        "/")[-1], mode='rb') as input_f:
                                    with file_io.FileIO(filepath, mode='wb+') as output_f:
                                        output_f.write(input_f.read())
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    if is_development():
                        self.model_to_save.save(filepath, overwrite=True)
                    else:
                        self.model_to_save.save(filepath.split(
                            "/")[-1])
                        with file_io.FileIO(filepath.split(
                                "/")[-1], mode='rb') as input_f:
                            with file_io.FileIO(filepath, mode='wb+') as output_f:
                                output_f.write(input_f.read())


def is_development():
    """check if the environment is local or in the gcloud
    created the local variable in bash profile
    export LOCAL_ENV=1

    Returns:
        [boolean] -- True if local env
    """
    try:
        if os.environ['LOCAL_ENV'] == '1':
            return True
        else:
            return False
    except:
        return False
