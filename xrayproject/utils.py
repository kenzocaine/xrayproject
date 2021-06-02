import os
import tensorflow as tf
import random

def load_pngs(n=1, get_all=False, get_target=False, get_random = True, balanced = True, path = ''):
    # Load png and returns them as list of  img (tensor) , targets (bol)
    # If balanced = True, will attempt to divide n into equal parts of positive and negative samples
    # If random, will choose random images
    # If random = False, will return images with the same order as in raw_data
    # Only set get_all = True if you running on the cloud
    # Keep n < 20 if you dont wanna run into memory problems
    # balanced only works with get_random
    print('hello')
    list_of_filenames = get_filenames(path)
    list_of_images = []
    targets = []
    if get_all:
        for file in list_of_filenames:
            image = load_png(file)
            list_of_images.append(image)
            targets.append(int(file[-5]))
        return list_of_images, targets

    # Extract positive and negative samples
    positive, negative = [], []
    for file in list_of_filenames:
        if int(file[-5]):
            positive.append(file)
        else:
            negative.append(file)

    if get_random:
        if n == 1:
            rand_index = random.randint(0, len(list_of_filenames))
            file = list_of_filenames[rand_index]
            return load_png(file), file[-5]
        if balanced:
            pos = int(n/2)
            neg = int(n - pos)
            rand_list_pos = [random.randint(0, len(positive)) for i in range(pos)]
            rand_list_neg = [random.randint(0, len(negative)) for i in range(neg)]
            for i in rand_list_pos:
                list_of_images.append(load_png(positive[i]))
                targets.append(int(positive[i][-5]))
            for i in rand_list_neg:
                list_of_images.append(load_png(negative[i]))
                targets.append(int(negative[i][-5]))
            return list_of_images, targets
        else:
            rand_list = [random.randint(0, len(list_of_filenames)) for i in range(n)]
            for i in rand_list:
                list_of_images.append(load_png(list_of_filenames[i]))
                targets.append(int(list_of_filenames[i][-5]))
            return list_of_images, targets

    for file in list_of_filenames[0:n]:
        image = load_png(file)
        list_of_images.append(image)
        targets.append(int(file[-5]))
    return list_of_images, targets


def get_filenames(path):
    list_of_filenames = []
    print(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/'))
   # os.walk(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/')):

    for dirname, _, filenames in os.walk(path):
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
