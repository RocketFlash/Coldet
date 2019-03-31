import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import keras
from imblearn.under_sampling import RandomUnderSampler

import matplotlib
matplotlib.use('agg')


def subsampling_with_stride(array, window_size, stride_size):
    array_size = array.shape[0]
    channels_size = array.shape[1]
    number_of_samples = int(((array_size - window_size)/stride_size) + 1)
    result = np.zeros((number_of_samples, window_size, channels_size))
    for i in range(number_of_samples):
        result[i, :, :] = array[(i*(stride_size)):(i*(stride_size)+window_size), :]
    return np.squeeze(result)


def get_labels(array, window_size, stride_size):
    labels = subsampling_with_stride(array, window_size, stride_size)
    Y = labels[:, -1]
    return Y


def plot_data(title, joints_torques, lebels):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    for i in range(joints_torques.shape[1]):
        ax1.plot(joints_torques[:, i], linewidth=1,
                 label='joint {}'.format(i+1))
    ax2.plot(lebels, linewidth=1, label='collision labels')

    ax1.set_title('Joints positions')
    ax2.set_title('Collision labels')
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax1.grid(True)
    ax2.grid(True)
    plt.show()


def load_data(data_list, sample_size=100, stride=20, to_plot=False, to_log=False):
    data = []
    for di in data_list:
        data.append(np.load(di))
    Xs = [el.item().get('X') for el in data]
    ys = [el.item().get('y') for el in data]
    X_full = np.vstack(Xs)
    y_full = np.vstack(ys)

    if to_plot:
        for Xs_el, ys_el in zip(Xs, ys):
            plot_data('Collisions', Xs_el, ys_el)

    X = subsampling_with_stride(X_full, sample_size, stride)
    Y = get_labels(y_full, sample_size, stride)

    _, counts = np.unique(Y, return_counts=True)

    # Random under sampling
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(
        X.reshape(X.shape[0], X.shape[1]*X.shape[2]), Y)
    X_res = X_res.reshape(X_res.shape[0], X.shape[1], X.shape[2])

    unique, counts = np.unique(y_res, return_counts=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=1)

    # One hot encoding
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_val, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    input_shape = X_train.shape[1:]

    if to_log:
        print('full raw dataset features size {}'.format(X_full.shape))
        print('full raw dataset labels size {}'.format(y_full.shape))
        print('dataset size after sampling {}'.format(X.shape))
        print('labels size after sampling {}'.format(Y.shape))
        print('classes distribution before undersampling: {}'.format(counts))
        print('dataset size after under sampling {}'.format(X_res.shape))
        print('labels size after under sampling {}'.format(y_res.shape))
        print('classes distribution after undersampling: {}'.format(counts))
        print('X_train size: {}'.format(X_train.shape))
        print('y_train size: {}'.format(y_train.shape))
        print('X_val size: {}'.format(X_val.shape))
        print('y_val size: {}'.format(y_val.shape))
        print('X_test size: {}'.format(X_test.shape))
        print('X_test size: {}'.format(y_test.shape))
        print('y_train after one hot size: {}'.format(y_train.shape))
        print('y_val after one hot size: {}'.format(y_val.shape))
        print('y_train after one hot size: {}'.format(y_test.shape))
        print('NN input shape: {}'.format(input_shape))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
