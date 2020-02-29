import numpy as np
import keras
from load_data import load_data

from models import RESNET


dataset_path = 'dataset'
data_files = ["data_soft.npy", "data_hard.npy", "data_mixed.npy"]
data_list = ['{}/{}'.format(dataset_path,dt) for dt in data_files]

data_train, data_val, data_test = load_data(data_list)
X_train, y_train = data_train
X_val, y_val = data_val
X_test, y_test = data_test

input_shape = X_train.shape[1:]
print('NN input shape: {}'.format(input_shape))
nb_classes = 3
output_directory = 'output/'
y_true = y_val.astype(np.int64)

model = RESNET(output_directory,input_shape, nb_classes,verbose=1)
model.fit(X_train, y_train, X_val, y_val,y_true)

print('Done!')