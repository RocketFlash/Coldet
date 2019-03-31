import keras 
import numpy as np 
import pandas as pd 
import time
from load_data import load_data

model = keras.models.load_model('output/best_model.hdf5')

dataset_path = 'dataset'
data_files = ["data_soft.npy", "data_hard.npy", "data_mixed.npy"]
data_list = ['{}/{}'.format(dataset_path,dt) for dt in data_files]

data_train, data_val, data_test = load_data(data_list)
X_train, y_train = data_train
X_val, y_val = data_val
X_test, y_test = data_test

y_pred = model.predict(X_test)
# convert the predicted from binary to integer 
y_pred = np.argmax(y_pred , axis=1)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=32,verbose=1)
print('Test score:', score)
print('Test accuracy:', acc)