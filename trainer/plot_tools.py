import matplotlib.pyplot as plt
import types
import numpy as np


def plot_mvtec(data):
    plt.imshow(data)


def hide_axis(subplt, x=True, y=True):
    if x:
        subplt.get_xaxis().set_visible(False)
    if y:
        subplt.get_yaxis().set_visible(False)


def plot_samples(row_one, row_two, f_plot, outfile=""):
    """Plot 8 sample images of the row_one and row_two"""
    plt.figure(figsize=(16, 4))
    # row_one
    n = 8
    for i in range(n):
        try:
            ax = plt.subplot(2, n, i+1)
            hide_axis(ax)
            nxt = next(row_one) if isinstance(
                row_one, types.GeneratorType) else row_one[i]
            f_plot(nxt)  # row_one
        except IndexError:
            pass
        try:
            ax = plt.subplot(2, n, n+i+1)
            hide_axis(ax)
            nxt = next(row_two) if isinstance(
                row_two, types.GeneratorType) else row_two[i]
            f_plot(nxt)  # row_two
        except IndexError:
            pass
    if outfile != "":
        plt.savefig(outfile)


def plot_hist(values, bins=50, relative=False, color="r"):
    t = np.linspace(values.min(), values.max(), bins)
    denominator = len(values) if relative else 1
    hist = np.histogram(values, bins)
    plt.plot(t, hist[0]/denominator, color)
