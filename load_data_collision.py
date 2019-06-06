import numpy as np
from sklearn.model_selection import train_test_split
import h5py
# import cv2


def load_data(filename, file_type='hdf5', for_classification=False, grid_size=(20, 8), num_files=10, num_samples=1000, test_size=0.2, val_size=0.1, to_log=False):

    X1s_train, X1s_test, X1s_val = [], [], []
    X2s_train, X2s_test, X2s_val = [], [], []
    ys_train,  ys_test,  ys_val = [], [], []
    for i in range(num_files):
        if to_log:
            print('Read file number{}'.format(i+1))
        
        if file_type=='hdf5':
            with h5py.File('dataset/{}{}.hdf5'.format(filename,i+1), 'r') as f:
                X1 = f['X_jac'][:]
                X2 = f['X_tor'][:]
                y = f['y_col'][:]

        elif file_type=='npy':
            dt = np.load('dataset/{}{}.npy'.format(filename, i+1))
            X1 = dt.item().get('X1')
            X2 = dt.item().get('X2')
            y = dt.item().get('y')

        if for_classification:
            y_class = np.zeros(
                (y.shape[0], grid_size[0], grid_size[1]), dtype=np.int32)
            h, w = y.shape[1], y.shape[2]
            y_step = h/grid_size[0]
            x_step = w/grid_size[1]
            ind_s, ind_y, ind_x = np.nonzero(y)
            ind_y_new = ind_y//y_step
            ind_x_new = ind_x//x_step
            ind_y_new = ind_y_new.astype(int)
            ind_x_new = ind_x_new.astype(int)
            y_class[ind_s, ind_y_new, ind_x_new] = 1
            # cv2.imwrite('y_normal.png', y[0]*255)
            # cv2.imwrite('y_class.png', y_class[0]*255)

        if to_log:
            print('Splitting on train/test')
        X1i_train, X1i_test, X2i_train, X2i_test, yi_train, yi_test = train_test_split(
            X1, X2, y, test_size=test_size, random_state=42)
        X1i_train, X1i_val, X2i_train, X2i_val, yi_train, yi_val = train_test_split(
            X1i_train, X2i_train, yi_train, test_size=val_size, random_state=1)
        X1s_train.append(X1i_train)
        X1s_test.append(X1i_test)
        X2s_train.append(X2i_train)
        X2s_test.append(X2i_test)
        ys_train.append(yi_train)
        ys_test.append(yi_test)
        X1s_val.append(X1i_val)
        X2s_val.append(X2i_val)
        ys_val.append(yi_val)
        if to_log:
            print('Splitting completed')
            print('============================')

    X1_train = np.concatenate(X1s_train, axis=0)
    X1_test = np.concatenate(X1s_test, axis=0)
    X2_train = np.concatenate(X2s_train, axis=0)
    X2_test = np.concatenate(X2s_test, axis=0)
    y_train = np.concatenate(ys_train, axis=0)
    y_test = np.concatenate(ys_test, axis=0)
    X1_val = np.concatenate(X1s_val, axis=0)
    X2_val = np.concatenate(X2s_val, axis=0)
    y_val = np.concatenate(ys_val, axis=0)

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
