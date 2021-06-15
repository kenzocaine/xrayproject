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

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min',
                                                      patience=15,
                                                      restore_best_weights=True)
        print('Starting train..')
        # import pdb; pdb.set_trace()
        self.model.fit(X_train, np.array(y_train),
                       validation_data=(X_test, np.array(y_test)),
                       epochs=60,  # Use early stop in practice
                       batch_size=64,
                       verbose=True,
                       callbacks=[early_stop])
        return self.model

    def initialize_model(self):
        print('Initializing model...')
        IMG_SIZE = self.input_shape[0]
        resize_and_rescale = tf.keras.Sequential([tf.keras.
                                                  layers.experimental.
                                                  preprocessing.
                                                  Resizing(IMG_SIZE, IMG_SIZE),
                                                  tf.keras.layers.
                                                  experimental.
                                                  preprocessing.
                                                  Rescaling(1./255)])
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.
            preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.
            preprocessing.RandomRotation((-0.05, 0.05))
        ])
        # input_shape = self.input_shape
        # inputs = tf.keras.Input(shape=input_shape)
        # x = data_augmentation(inputs)

        cnn = Sequential()
        # Add preprocessing
        # Zooming, train on different zooms
        # 
        # cnn.add(resize_and_rescale)
        # cnn.add(data_augmentation)

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

        # outputs = cnn(x)
        # model = tf.keras.Model(inputs, outputs)

        cnn.compile(Adam(lr=0.001),loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.Recall()])

        # cnn.compile(loss='binary_crossentropy',
        #              optimizer='adam',
        #              metrics=['accuracy'])
        self.model = cnn

        return cnn
