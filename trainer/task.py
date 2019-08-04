import argparse
import os

from trainer.model import main

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
    '--ckptdir', type=str,
    help='base path to save ckpts to', default='ckpts'
)
parser.add_argument(
    '--imgdir', type=str,
    help='base path to save imgs to', default='imgs'
)
parser.add_argument(
    '--datadir', type=str,
    help='base path to where the data is stored', default='data'
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
    help='number of filters for each of the nine layers without the latent space dimension', default=[16, 16, 16, 32, 64, 64, 32, 32]
)

parser.add_argument(
    '--ldim', type=int,
    help='latent space dimension', default=164
)


flags = parser.parse_args()

main(flags)
