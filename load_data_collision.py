import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filename, num_files=10, num_samples=1000, to_log=False):

    X1s = np.zeros((num_files*num_samples, 128, 32, 21))
    X2s = np.zeros((num_files*num_samples, 1, 7))
    ys = np.zeros((num_files*num_samples, 128, 32))

    for i in range(num_files):
        dt = np.load('dataset/{}{}.npy'.format(filename, i+1))
        X1 = dt.item().get('X1')
        X2 = dt.item().get('X2')
        y = dt.item().get('y')
        X1s[i*num_samples:(i+1)*num_samples, :, :, :] = X1
        X2s[i*num_samples:(i+1)*num_samples, :, :] = X2
        ys[i*num_samples:(i+1)*num_samples, :, :] = y

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        X1s, X2s, ys, test_size=0.2, random_state=42)
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1_train, X2_train, y_train, test_size=0.1, random_state=1)
    y_train = np.expand_dims(y_train, axis=3)
    y_test = np.expand_dims(y_test, axis=3)
    y_val = np.expand_dims(y_val, axis=3)
    if to_log:
        print('X1_train size: {}'.format(X1_train.shape))
        print('X2_train size: {}'.format(X2_train.shape))
        print('y_train size: {}'.format(y_train.shape))
        print('X1_val size: {}'.format(X1_val.shape))
        print('X2_val size: {}'.format(X2_val.shape))
        print('y_val size: {}'.format(y_val.shape))
        print('X1_test size: {}'.format(X1_test.shape))
        print('X2_test size: {}'.format(X2_test.shape))
        print('y_test size: {}'.format(y_test.shape))

    return (X1_train, X2_train, y_train), (X1_val, X2_val, y_val), (X1_test, X2_test, y_test)
