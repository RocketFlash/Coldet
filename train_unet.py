from unet2 import UNET2
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from load_data_collision import load_data
import losses

metrics = [losses.dice]
loss = [losses.dice_and_binary_crossentropy]
optimizer = Adam()

input_shape = (128, 32, 21)
input_shape_t = (1, 7)

model = UNET2('output/', input_shape, input_shape_t, model_name='dice_and_bce2', scenario=2,
             metrics=metrics, loss=loss, verbose=True)
print('Model was created!')

data_train, data_val, data_test = load_data(
    'collision_dataset', num_files=10, num_samples=1000, to_log=True)

print('Data was loaded!')
X1_train, X2_train, y_train = data_train
X1_val, X2_val, y_val = data_val
X1_test, X_test, y_test = data_test


model.fit(X1_train, X2_train, y_train, X1_val, X2_val,
          y_val, batch_size=2400, nb_epochs=10000)

print('Done!')
