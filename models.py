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
from coord import CoordinateChannel2D
import os
import wandb
from wandb.keras import WandbCallback

import time

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from parse_config import parse_train_params

from utils.utils import save_logs



class UNET:

    def __init__(self, config_filename):
        train_params = parse_train_params(config_filename)
        self.output_directory = train_params['output_directory']
        self.metrics = train_params['metrics']
        self.loss = train_params['loss']
        self.scenario = train_params['scenario']
        self.model_name = train_params['model_name']
        self.optimizer = train_params['optimizer']
        input_jacobians_shape = train_params['input_jacobians_shape']
        input_torques_shape = train_params['input_torques_shape']
        self.model = self.build_model(input_jacobians_shape=input_jacobians_shape, 
                                      input_torques_shape=input_torques_shape, 
                                      n_filters=train_params['n_filters'], 
                                      dropout=train_params['dropout'], 
                                      batchnorm=train_params['batchnorm'],
                                      with_coordconv=train_params['with_coordconv'],
                                      with_wandb=train_params['with_wandb'])
        if(train_params['verbose'] == True):
            self.model.summary()
        self.verbose = train_params['verbose']
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.model.save_weights(self.output_directory+self.model_name+'_init.hdf5')

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True, with_coordconv=False, with_wandb=False):
        x = input_tensor
        # first layer
        if with_coordconv:
            x = CoordinateChannel2D()(x)
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        if with_coordconv:
            x = CoordinateChannel2D()(x)
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def build_model(self, input_jacobians_shape, input_torques_shape, n_filters=16, dropout=0.5, batchnorm=True, with_coordconv=False, with_wandb=False):

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
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = self.conv2d_block(p1, n_filters=n_filters*2,
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters=n_filters*4,
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        if self.scenario == 3:
            c4 = self.conv2d_block(p3, n_filters=n_filters*4,
                                   kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)
        elif self.scenario == 1 or self.scenario == 2:
            c4 = self.conv2d_block(p3, n_filters=n_filters*8,
                                   kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)
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
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)

        # expansive path
        u6 = Conv2DTranspose(n_filters*8, (3, 3),
                             strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters=n_filters*8,
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)

        u7 = Conv2DTranspose(n_filters*4, (3, 3),
                             strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters=n_filters*4,
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)

        u8 = Conv2DTranspose(n_filters*2, (3, 3),
                             strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters=n_filters*2,
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)

        u9 = Conv2DTranspose(n_filters*1, (3, 3),
                             strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters=n_filters*1,
                               kernel_size=3, batchnorm=batchnorm, with_coordconv=with_coordconv)

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
        if with_wandb:
            wandb.init(project="coldet")
            callbacks.append(WandbCallback())
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

        self.model.save(os.path.join(wandb.run.dir, "{}_final.h5".format(self.model_name)))

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


class RESNET: 

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, nb_classes):
		n_feature_maps = 64

		input_layer = keras.layers.Input(input_shape)
		
		# BLOCK 1 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu')(output_block_1)

		# BLOCK 2 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu')(output_block_2)

		# BLOCK 3 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

		output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu')(output_block_3)

		# FINAL 
		
		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5' 

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model
	
	def fit(self, x_train, y_train, x_val, y_val,y_true):   
		batch_size = 32
		nb_epochs = 500

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		# save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()



class UNET2:

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
        # Add values 
        if self.scenario == 1:
            h1 = Dense(128, activation='relu')(input_torques)
            # h1 = Dropout(dropout)(h1)
            h2 = Dense(256, activation='relu')(h1)
            # h2 = Dropout(dropout)(h2)
            h3 = Dense(512, activation='relu')(h2)
            h3 = Dropout(dropout)(h3)
            h4 = Dense(1024, activation='relu')(h3)
        elif self.scenario == 2:
            h1 = Dense(512, activation='relu')(input_torques)
            h1 = Dropout(dropout)(h1)
            h2 = Dense(32768, activation='relu')(h1)
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

        if self.scenario == 3:
            c2 = self.conv2d_block(p1, n_filters=n_filters,
                                   kernel_size=3, batchnorm=batchnorm)
        elif self.scenario == 1 or self.scenario == 2:
            c2 = self.conv2d_block(p1, n_filters=n_filters*2,
                                   kernel_size=3, batchnorm=batchnorm)

        c2 = self.conv2d_block(p1, n_filters=n_filters*2,
                               kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        # add values to bottleneck
        if self.scenario == 1:
            h4_reshaped = Reshape(tuple(p2.shape[1:3].as_list())+(1,))(h4)
            p2_combined = Add()([p2, h4_reshaped])
            # p4_combined = Multiply()([p4, h4_reshaped])
        elif self.scenario == 2:
            p2_reshaped = Flatten()(p2)
            p2_sum = Add()([p2_reshaped, h2])
            p2_combined = Reshape(tuple(p2.shape[1:].as_list()))(p2_sum)
        elif self.scenario == 3:
            h4_reshaped = Reshape(tuple(p2.shape[1:].as_list()))(h4)
            p2_combined = Concatenate()([p2, h4_reshaped])

        c3 = self.conv2d_block(p2_combined, n_filters=n_filters*4,
                               kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters*8,
                          kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters=n_filters*16,
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
