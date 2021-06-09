import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from xrayproject.utils import load_train, load_masks
from xrayproject.preprocessing import normalize, flip_resize, resize_test


class Baseline():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.resize_shape =(self.input_shape[0] , self.input_shape[1])
        self.model = []

    def train(self, images, masks, targets):
        img_p, mask_p, img_flipped, mask_flipped = self.preprocessing(images, masks)
        X_train, X_test, y_train, y_test = self.train_split(img_p, targets)

        X_train = np.array(X_train)
        X_train = X_train.reshape(len(X_train), self.input_shape[0], self.input_shape[1], self.input_shape[2])

        X_test = np.array(X_test)
        X_test = X_test.reshape(len(X_test), self.input_shape[0], self.input_shape[1], self.input_shape[2])

        self.model = self.initialize_model()
        print('Starting train..')
        self.model.fit(X_train, np.array(y_train),
                    validation_data = (X_test, np.array(y_test)),
                    epochs=40,  # Use early stop in practice
                    batch_size=32,
                    verbose=1)
        return self.model

    def initialize_model(self):
        print('Initializing model...')
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(16, (2,2), input_shape=self.input_shape, activation="relu"))
        #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 

        model.add(tf.keras.layers.Conv2D(16, (2,2) , activation="relu"))
        #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 

        model.add(tf.keras.layers.Conv2D(32, (2,2) , activation="relu"))
        #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(20, activation='relu')) # intermediate layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                      metrics=['accuracy'])

        return model

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

    def train_split(self, images, targets):
        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.25)
        return X_train, X_test, y_train, y_test

