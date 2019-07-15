import numpy as np

from keras.utils import to_categorical
from keras.datasets import mnist


def get_data(train_split=.7, test_split=.85):
    """retrieves data MNIST data set and rebalances dataset, such that train=.7, test=.15 and validation=.15"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((len(X_train), 28, 28, 1))
    test_len = len(X_test)

    X_test = X_test.reshape((test_len, 28, 28, 1))

    # divide X values bei 255.0 since MNIST data set changed such that pixel values are in [0,255]
    X = np.concatenate((X_train, X_test)) / 255.0
    y = np.concatenate((y_train, y_test))

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # define boundaries for train,validation and test set at .7 and .85 % of the MNIST data set
    x_len = len(X)
    boundaries = [int(x_len * 0.7), int(x_len*0.85)]

    [X_train, X_test, X_validate] = np.split(X, boundaries)

    [y_train, y_test, y_validate] = np.split(y, boundaries)

    # non-anomalies
    zero_indices = np.where(y_train == 0)
    zeros_train = X_train[zero_indices]
    # anomalies
    eights_indices = np.where(y_train == 8)
    eights_train = X_train[eights_indices]

    # one-hot encode target columns
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_validate = to_categorical(y_validate)

    return (X_train, X_test, X_validate), (y_train, y_test, y_validate)


def gauss(x_i, my_i=0, sigma_i2=1):
    # gaussian distribution for one feature
    return np.array((1/np.sqrt(2*np.pi*sigma_i2)) *
                    np.exp(-(x_i-my_i)**2/(2*sigma_i2)))


def my(X):
    return np.array([(1 / len(x)) * np.sum(x) for x in X.T])


def sigma2(X, my):
    # computes sigma squared for each feature
    m = len(X)  # number of data points
    return np.array([np.sum((X[:, i] - my[i]) ** 2)/m for i in range(len(my))])


(X_train, _, _), (_, _, _) = get_data()
X_train = X_train.reshape((-1, 28*28))
my_train = my(X_train)
sigma2_train = sigma2(X_train, my_train)
x = X_train[0]
p_x = np.prod([gauss(x[i], my_train[i], sigma2_train[i])
               for i in range(len(my_train))])  # für 1 x Element X_train

print(p_x, sigma2_train, my_train)
