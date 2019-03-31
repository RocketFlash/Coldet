import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers.merge import concatenate, add
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Add, Flatten, Multiply, Concatenate
from keras.models import Model, load_model
import keras
import time
import numpy as np
from sklearn.utils import class_weight
from keras import backend as K
import tensorflow as tf

import matplotlib
matplotlib.use('agg')


class UNET:

    def __init__(self, output_directory, input_jacobians_shape, input_torques_shape, model_name='crossentropy', optimizer=Adam(), scenario=1, n_filters=16, dropout=0.5, batchnorm=True, metrics=['accuracy'], loss=['binary_crossentropy'], verbose=False):
        self.output_directory = output_directory
        self.metrics = metrics
        self.loss = loss
        self.scenario = scenario
        self.model_name = model_name
        self.optimizer = optimizer
        self.model = self.build_model(
            input_jacobians_shape, input_torques_shape, n_filters, dropout, batchnorm)
        if(verbose == True):
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory+'unet_model_init.hdf5')

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def build_model(self, input_jacobians_shape, input_torques_shape, n_filters=16, dropout=0.5, batchnorm=True):

        input_jacobians = Input(shape=input_jacobians_shape, name='input_jac')

        input_torques = Input(shape=input_torques_shape, name='input_tor')

        # Create 2x8 matrix from FFnet and add to each channel in bottleneck
        if self.scenario == 1:
            h1 = Dense(64, activation='relu')(input_torques)
            # h1 = Dropout(dropout)(h1)
            h2 = Dense(128, activation='relu')(h1)
            # h2 = Dropout(dropout)(h2)
            h3 = Dense(128, activation='relu')(h2)
            h3 = Dropout(dropout)(h3)
            h4 = Dense(16, activation='relu')(h3)
        # Create 2x8 matrix from FFnet and add to each channel in bottleneck
        elif self.scenario == 2:
            h1 = Dense(512, activation='relu')(input_torques)
            h1 = Dropout(dropout)(h1)
            h2 = Dense(2048, activation='relu')(h1)
        elif self.scenario == 3:
            h1 = Dense(32, activation='relu')(input_torques)
            h1 = Dropout(dropout)(h1)
            h2 = Dense(64, activation='relu')(h1)
            # h2 = Dropout(dropout)(h2)
            h3 = Dense(64, activation='relu')(h2)
            # h3 = Dropout(dropout)(h3)
            h4 = Dense(1024, activation='relu')(h3)

        # contracting path
        c1 = self.conv2d_block(input_jacobians, n_filters=n_filters*1,
                               kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = self.conv2d_block(p1, n_filters=n_filters*2,
                               kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters=n_filters*4,
                               kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        if self.scenario == 3:
            c4 = self.conv2d_block(p3, n_filters=n_filters*4,
                                   kernel_size=3, batchnorm=batchnorm)
        elif self.scenario == 1 or self.scenario == 2:
            c4 = self.conv2d_block(p3, n_filters=n_filters*8,
                                   kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        # add values to bottleneck
        if self.scenario == 1:
            h4_reshaped = Reshape(tuple(p4.shape[1:3].as_list())+(1,))(h4)
            p4_combined = Add()([p4, h4_reshaped])
            # p4_combined = Multiply()([p4, h4_reshaped])
        elif self.scenario == 2:
            p4_reshaped = Flatten()(p4)
            p4_sum = Add()([p4_reshaped, h2])
            p4_combined = Reshape(tuple(p4.shape[1:].as_list()))(p4_sum)
        elif self.scenario == 3:
            h4_reshaped = Reshape(tuple(p4.shape[1:].as_list()))(h4)
            p4_combined = Concatenate()([p4, h4_reshaped])

        c5 = self.conv2d_block(p4_combined, n_filters=n_filters*16,
                               kernel_size=3, batchnorm=batchnorm)

        # expansive path
        u6 = Conv2DTranspose(n_filters*8, (3, 3),
                             strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters=n_filters*8,
                               kernel_size=3, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters*4, (3, 3),
                             strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters=n_filters*4,
                               kernel_size=3, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters*2, (3, 3),
                             strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters=n_filters*2,
                               kernel_size=3, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters*1, (3, 3),
                             strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters=n_filters*1,
                               kernel_size=3, batchnorm=batchnorm)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(
            inputs=[input_jacobians, input_torques], outputs=[outputs])

        model.compile(optimizer=self.optimizer, loss=self.loss,
                      metrics=self.metrics)

        file_path = self.output_directory + \
            'unet_best_model_{}.hdf5'.format(self.model_name)
        callbacks = [
            EarlyStopping(patience=1000, verbose=1),
            TensorBoard(),
            ReduceLROnPlateau(factor=0.1, patience=10,
                              min_lr=1e-12, verbose=1),
            ModelCheckpoint(filepath=file_path, verbose=1, monitor='loss',
                            save_best_only=True)
        ]
        self.callbacks = callbacks

        return model

    def fit(self, x1_train, x2_train, y_train, x1_val, x2_val, y_val, batch_size=100, nb_epochs=500):

        mini_batch_size = int(min(x1_train.shape[0]/10, batch_size))

        start_time = time.time()

        y_train_reshaped = y_train.reshape(
            (y_train.shape[0]*y_train.shape[1]*y_train.shape[2]))

        print(y_train_reshaped.shape)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(
                                                              y_train_reshaped),
                                                          y_train_reshaped)
        print('class weights: {}'.format(class_weights))
        history = self.model.fit([x1_train, x2_train], y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                                 verbose=self.verbose, validation_data=(
            [x1_val, x2_val], y_val), callbacks=self.callbacks, class_weight=class_weights)

        duration = time.time() - start_time
        print('duration: {}'.format(duration))
        # Plot training & validation accuracy values
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(history.history['dice'])
        plt.plot(history.history['val_dice'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        fig.savefig('output/unet_accuracy.png')   # save the figure to file
        plt.close(fig)

        # Plot training & validation loss values
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(history.history['dice_loss'])
        plt.plot(history.history['val_dice_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        fig.savefig('output/unet_loss.png')   # save the figure to file
        plt.close(fig)

        keras.backend.clear_session()
