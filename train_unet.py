from models import UNET
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from load_data_collision import load_data

model = UNET('config/unet.yml')
print('Model was created!')

data_train, data_val, data_test = load_data(
    'collision_dataset', file_type='hdf5', num_files=10, num_samples=1000, to_log=True)

print('Data was loaded!')
X1_train, X2_train, y_train = data_train
X1_val, X2_val, y_val = data_val
X1_test, X_test, y_test = data_test


model.fit(X1_train, X2_train, y_train, X1_val, X2_val,
          y_val, batch_size=50, nb_epochs=10000)

print('Done!')
