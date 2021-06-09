import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from xrayproject.utils import load_train, load_masks
from xrayproject.preprocessing import normalize, flip_resize, resize_test
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix

class Segmentation_UNET():
    def __init__(self, input_shape=(224,224,3), output_channels=1):
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.resize_shape =(self.input_shape[0] , self.input_shape[1])
        self.model = []

    def preprocessing(self, images, masks):
        print('Preprocessing...')
        img_p, mask_p, img_flipped, mask_flipped = [], [], [],[]
        for index in range(len(images)):
            img_p_, mask_p_, img_flipped_, mask_flipped_ = flip_resize(images[index], masks[index], self.resize_shape)
            img_p.append(img_p_)
            mask_p.append(mask_p_)
            img_flipped.append(img_flipped_)
            mask_flipped.append(mask_flipped_)
        return img_p, mask_p, img_flipped, mask_flipped

    def train_split(self, images, masks):
        X_train, X_test, Y_train, Y_test = train_test_split(images, masks, test_size=0.25)
        return X_train, X_test, Y_train, Y_test

    def base_model(self):
        base_model = tf.keras.applications.MobileNetV2(input_shape=self.input_shape, include_top=False)

        # # Use the activations of these layers
        # layer_names = [
        #     'block_1_expand_relu',   # 64x64
        #     'block_3_expand_relu',   # 32x32
        #     'block_6_expand_relu',   # 16x16
        #     'block_13_expand_relu',  # 8x8
        #     'block_16_project',      # 4x4
        # ]
        # base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # # Create the feature extraction model
        # down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        # down_stack.trainable = False

        return base_model

    # Note: This is directly copied-and-pasted from the TensorFlow tutorial. I am slightly worried that this may refer to items not defined in this function, causing flow issues...
    def unet_model(self):
        base_model = self.base_model()

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = False

        inputs = tf.keras.layers.Input(shape=self.input_shape)

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            self.output_channels, 2, strides=2,
            padding='same')  #64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def initialize_model(self):
        print('Initializing model...')
        # base_model = tf.keras.applications.MobileNetV2(input_shape=self.input_shape, include_top=False)

        # # Use the activations of these layers
        # layer_names = [
        #     'block_1_expand_relu',   # 64x64
        #     'block_3_expand_relu',   # 32x32
        #     'block_6_expand_relu',   # 16x16
        #     'block_13_expand_relu',  # 8x8
        #     'block_16_project',      # 4x4
        # ]
        # base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # # Create the feature extraction model
        # down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        # down_stack.trainable = False

        model = self.unet_model()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    def train(self, images, masks, targets):
        img_p, mask_p, img_flipped, mask_flipped = self.preprocessing(images, masks)
        X_train, X_test, Y_train, Y_test = self.train_split(img_p, mask_p)

        X_train = np.array(X_train)
        X_train = X_train.reshape(len(X_train), self.input_shape[0], self.input_shape[1], self.input_shape[2])

        Y_train = np.array(Y_train)
        Y_train = Y_train.reshape(len(Y_train), self.input_shape[0], self.input_shape[1], 1)

        X_test = np.array(X_test)
        X_test = X_test.reshape(len(X_test), self.input_shape[0], self.input_shape[1], self.input_shape[2])

        Y_test = np.array(Y_test)
        Y_test = Y_test.reshape(len(Y_test), self.input_shape[0], self.input_shape[1], 1)

        self.model = self.initialize_model()
        print('Starting train..')
        TRAIN_LENGTH = len(X_test)
        BATCH_SIZE = min(5, TRAIN_LENGTH)
        BUFFER_SIZE = 10
        STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
        EPOCHS = 10
        VAL_SUBSPLITS = 5
        VALIDATION_STEPS = len(X_test)//BATCH_SIZE//VAL_SUBSPLITS
        print(Y_train)
        print(Y_test)
        self.model.fit(X_train, Y_train,
                       epochs=EPOCHS,
                       steps_per_epoch=STEPS_PER_EPOCH,
                       validation_steps=VALIDATION_STEPS,
                       validation_data=(X_test, Y_test),
                       verbose=1)

        return self.model

    # def initialize_model_old(self):
    #     print('Initializing model...')
    #     model = tf.keras.models.Sequential()

    #     model.add(tf.keras.layers.Conv2D(16, (2,2), input_shape=self.input_shape, activation="relu"))
    #     #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    #     model.add(tf.keras.layers.Conv2D(16, (2,2) , activation="relu"))
    #     #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    #     model.add(tf.keras.layers.Conv2D(32, (2,2) , activation="relu"))
    #     #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    #     model.add(tf.keras.layers.Flatten())
    #     model.add(tf.keras.layers.Dense(20, activation='relu')) # intermediate layer
    #     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    #     model.compile(loss='binary_crossentropy',
    #                       optimizer='adam',
    #                   metrics=['accuracy'])

    #     return model

    # def train_old(self, images, masks, targets):
    #     img_p, mask_p, img_flipped, mask_flipped = self.preprocessing(images, masks)
    #     X_train, X_test, y_train, y_test = self.train_split(img_p, targets)

    #     X_train = np.array(X_train)
    #     X_train = X_train.reshape(len(X_train), self.input_shape[0], self.input_shape[1], self.input_shape[2])

    #     X_test = np.array(X_test)
    #     X_test = X_test.reshape(len(X_test), self.input_shape[0], self.input_shape[1], self.input_shape[2])

    #     self.model = self.initialize_model()
    #     print('Starting train..')
    #     self.model.fit(X_train, np.array(y_train),
    #                 validation_data = (X_test, np.array(y_test)),
    #                 epochs=1,  # Use early stop in practice
    #                 batch_size=32,
    #                 verbose=1)
    #     return self.model
