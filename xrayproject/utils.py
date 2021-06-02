import os
import tensorflow as tf


def load_pngs(n=1, get_all=False, get_target=False):
    list_of_filenames = get_filenames()
    list_of_images = []
    targets = []
    if get_all:
        for file in list_of_filenames:
            image = load_png(file)
            list_of_images.append(image)
            targets.append(file[-5])
        return list_of_images, targets 

    for file in list_of_filenames[0:n]:
        image = load_png(file)
        list_of_images.append(image)
        targets.append(file[-5])
    return list_of_images, targets


def get_filenames():
    list_of_filenames = []
    for dirname, _, filenames in os.walk(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/')):
        for filename in filenames:
            if filename.endswith('.png'):
                list_of_filenames.append(os.path.join(dirname, filename))
    return list_of_filenames 


def load_png(file):
    image = tf.io.read_file(file)
    image = tf.io.decode_png(image)
    return image 

if __name__ == '__main__':
    print(load_png(0))
