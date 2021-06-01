import os
import tensorflow as tf


def load_png(index):
    list_of_filenames = []
    for dirname, _, filenames in os.walk(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/')):
        for filename in filenames:
            if filename.endswith('.png'):
                list_of_filenames.append(os.path.join(dirname, filename))
                # print(os.path.join(dirname, filename))
    image = tf.io.read_file(list_of_filenames[index])
    image = tf.io.decode_png(image)
    return image 


if __name__ == '__main__':
    print(load_png(0))
