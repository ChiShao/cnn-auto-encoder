import argparse

from tensorflow.python.lib.io import file_io

from trainer.model import train_and_evaluate

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument("--use-case", type=str,
                    help="use case, which anomaly should be detected. needs to be dir name under the data dir")

parser.add_argument(
    '--logdir', type=str,
    help='base path to log Tensorboard data to', default='logs'
)
parser.add_argument(
    '--job-dir', type=str,
    help='job directory'
)

parser.add_argument(
    '--evaldir', type=str,
    help='base path to save eval to', default='eval'
)
parser.add_argument(
    '--ckptdir', type=str,
    help='base path to save ckpts to', default='ckpts'
)
parser.add_argument(
    '--imgdir', type=str,
    help='base path to save imgs to', default='imgs'
)
parser.add_argument(
    '--datadir', type=str,
    help='base path to where the data is stored', default='data/mvtec_anomaly_detection'
)
parser.add_argument(
    '--epochs', type=int,
    help='num of epochs for training', default=256
)

parser.add_argument(
    '--batch_size', type=int,
    help='batch size for fine training', default=10
)

parser.add_argument(
    '--filters', nargs="+", type=int,
    help='number of filters for each of the nine layers *without* the latent space dimension',
    default=[16, 16, 16, 32, 64, 64, 32, 32, 32]
)

parser.add_argument(
    '--ldim', type=int,
    help='latent space dimension', default=150
)


args = parser.parse_args()

# create required dirs
file_io.create_dir(args.evaldir)
file_io.create_dir(args.imgdir)
file_io.create_dir(args.ckptdir)
file_io.create_dir(args.logdir)

train_and_evaluate(args)
