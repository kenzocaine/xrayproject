import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization,\
        GlobalMaxPooling2D
from tensorflow.keras.models import Sequential, Model, load_model


class Baseline():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = []

    def train(self, X_train, y_train, X_test, y_test):

        X_train = np.array(X_train)
        X_train = X_train.reshape(len(X_train),
                                  self.input_shape[0],
                                  self.input_shape[1],
                                  self.input_shape[2])

        X_test = np.array(X_test)
        X_test = X_test.reshape(len(X_test),
                                self.input_shape[0],
                                self.input_shape[1],
                                self.input_shape[2])

        print('Starting train..')
        # import pdb; pdb.set_trace()
        self.model.fit(X_train, np.array(y_train),
                       validation_data=(X_test, np.array(y_test)),
                       epochs=40,  # Use early stop in practice
                       batch_size=32,
                       verbose=True)
        return self.model

    def initialize_model(self):
        print('Initializing model...')
        cnn = Sequential()
        cnn.add(tf.keras.layers.Conv2D(filters=32,
                                 kernel_size=3,
                                 activation='relu',
                                 input_shape=self.input_shape))

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        cnn.add(Dropout(0.3))

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        cnn.add(Dropout(0.3))

        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        cnn.add(Dropout(0.3))
        cnn.add(Flatten())

        cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
        cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
        cnn.add(Dropout(0.3))
        cnn.add(tf.keras.layers.BatchNormalization())
        cnn.add(tf.keras.layers.Dense(units=1, activation = "sigmoid"))

        cnn.compile(Adam(lr=0.001),loss='binary_crossentropy',
                    metrics=['accuracy'])
        my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min',
                                                      patience=10)]



        # cnn.compile(loss='binary_crossentropy',
        #              optimizer='adam',
        #              metrics=['accuracy'])
        self.model = cnn

        return cnn
