import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, concatenate,\
        UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model


class UNET():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = []
        self.channels = input_shape[-1]

    def initialize_unet(self, input_shape):
        print('Initializing model...')

        inputs = Input(input_size)

        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        unet = Model(inputs=[inputs], outputs=[conv10])


        # unet = Sequential()
        # inputs = Input(input_size)

        # unet.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # unet.add(MaxPooling2D(pool_size=(2, 2)))

        # unet.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # unet.add(MaxPooling2D(pool_size=(2, 2)))

        # unet.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        # unet.add(MaxPooling2D(pool_size=(2, 2)))

        # unet.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        # unet.add(MaxPooling2D(pool_size=(2, 2)))

        # unet.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(512, (3, 3), activation='relu', padding='same'))

        # unet.add(concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3))
        # unet.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

        # unet.add(concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3))
        # unet.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

        # unet.add(concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3))
        # unet.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

        # unet.add(concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3))
        # unet.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        # unet.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

        # unet.add(Conv2D(1, (1, 1), activation='sigmoid'))

        unet.compile(Adam(lr=0.001),
                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics=['accuracy'])

        # my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                                  mode='min',
        #                                                  patience=10)]

        self.model = unet

        return unet

    def train(self, X_train, y_train, X_test, y_test):

        X_train = np.array(X_train)
        X_train = X_train.reshape(len(X_train),
                                  self.input_shape[0],
                                  self.input_shape[1],
                                  self.input_shape[2])

        Y_train = np.array(Y_train)
        Y_train = Y_train.reshape(len(Y_train),
                                  self.input_shape[0],
                                  self.input_shape[1],
                                  self.input_shape[2])

        X_test = np.array(X_test)
        X_test = X_test.reshape(len(X_test),
                                self.input_shape[0],
                                self.input_shape[1],
                                self.input_shape[2])

        Y_test = np.array(Y_test)
        Y_test = Y_test.reshape(len(Y_test),
                                self.input_shape[0],
                                self.input_shape[1],
                                self.input_shape[2])

        print('Starting train..')
        # import pdb; pdb.set_trace()

        self.model.fit(X_train, Y_train,
                       epochs=40,  # Use early stop in practice
                       batch_size=32,
                       validation_data=(X_test, Y_test),
                       verbose=True)
        return self.model

    # def unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):

        # inputs = Input(input_size)

        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        # up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        # up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        # up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        # up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        # conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)


        # return Model(inputs=[inputs], outputs=[conv10])
