
import os
import subprocess
from urllib.parse import urlparse

from keras.models import load_model

USE_CASE = "spm"


def load(ckpt_path):
    print("Loading Autoencoder for %s feature extraction from directory %s..." % (
        USE_CASE, ckpt_path))

    url = urlparse(ckpt_path)  # [:-1] cuts the \n from the lines
    bucket = "%s://%s" % (url.scheme, url.netloc)

    # [1:] removes leading / to avoid empty string in dir_parts
    dir_parts = os.path.normpath(url.path[1:]).split(os.sep)

    fp = os.path.join("tmp", dir_parts[-1])


    print(os.listdir())
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")

    if not os.path.isfile(fp):
        subprocess.call(["gsutil", "-m", "cp", ckpt_path, "tmp"])
    ae = load_model(fp)

    # this needs to be compiled since the untrained, single-GPU model is saved instead of the multi-GPU model
    ae.compile(optimizer="adam", loss="mse")
    print(ae.summary())


load("gs://e4u-anomaly-detection/ckpts/AD_spm_20190813_144154/ep0057-loss0.009716-val_loss0.015439.h5")
