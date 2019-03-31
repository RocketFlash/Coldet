import keras
import numpy as np
import pandas as pd
import time
from load_data_collision import load_data
import cv2
import tensorflow as tf
import keras.backend as K
import losses

model = keras.models.load_model(
    'output/unet_best_model.hdf5', custom_objects={'dice_and_binary_crossentropy': losses.dice_and_binary_crossentropy, 'dice': losses.dice})


data_train, data_val, data_test = load_data(
    'collision_dataset', num_files=10, num_samples=1000, to_log=True)

print('Data was loaded!')
X1_train, X2_train, y_train = data_train
X1_val, X2_val, y_val = data_val
X1_test, X2_test, y_test = data_test

y_pred = model.predict([X1_test[:100], X2_test[:100]])
# convert the predicted from binary to integer
# y_pred = np.argmax(y_pred, axis=1)
print(y_pred.shape)

for i in range(y_pred.shape[0]):
    cv2.imwrite('output/images/{}_predicted.png'.format(i),
                (y_pred[i]*255).astype(np.uint8))
    cv2.imwrite('output/images/{}_ground_truth.png'.format(i),
                (y_test[i, :, :]*255).astype(np.uint8))
    print('Image number: {}'.format(i))

score, acc = model.evaluate([X1_test, X2_test], y_test,
                            batch_size=32, verbose=1)
print('Test score:', score)
print('Test accuracy:', acc)
