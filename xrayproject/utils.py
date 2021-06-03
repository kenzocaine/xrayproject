import os
import tensorflow as tf
import random
import numpy as np
import PIL
from PIL import Image


def load_pngs(n=1, get_all=False, get_target=False, get_random = True, balanced = True, path = ''):
    # Load png and returns them as list of  img (tensor) , targets (bol)
    # If balanced = True, will attempt to divide n into equal parts of positive and negative samples
    # If random, will choose random images
    # If random = False, will return images with the same order as in raw_data
    # Only set get_all = True if you running on the cloud
    # Keep n < 20 if you dont wanna run into memory problems
    # balanced only works with get_random
    # print('hello')
    list_of_filenames = get_filenames(path)
    list_of_images = []
    targets = []
    ID = []
    if get_all:
        for file in list_of_filenames:
            image = load_png(file)
            list_of_images.append(image)
            targets.append(int(file[-5]))
            ID.append(int(os.path.basename(file).split('_')[1]))
        return list_of_images, targets, ID

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
            return load_png(file), int(file[-5]), os.path.basename(file).split('_')[1]
        if balanced:
            pos = int(n/2)
            neg = int(n - pos)
            rand_list_pos = [random.randint(0, len(positive)) for i in range(pos)]
            rand_list_neg = [random.randint(0, len(negative)) for i in range(neg)]
            for i in rand_list_pos:
                list_of_images.append(load_png(positive[i]))
                targets.append(int(positive[i][-5]))
                ID.append(int(os.path.basename(positive[i]).split('_')[1]))
            for i in rand_list_neg:
                list_of_images.append(load_png(negative[i]))
                targets.append(int(negative[i][-5]))
                ID.append(int(os.path.basename(negative[i]).split('_')[1]))
            return list_of_images, targets, ID
        else:
            rand_list = [random.randint(0, len(list_of_filenames)) for i in range(n)]
            for i in rand_list:
                list_of_images.append(load_png(list_of_filenames[i]))
                targets.append(int(list_of_filenames[i][-5]))
                ID.append(int(os.path.basename(list_of_filenames[i]).split('_')[1]))
            return list_of_images, targets, ID

    for file in list_of_filenames[0:n]:
        image = load_png(file)
        list_of_images.append(image)
        targets.append(int(file[-5]))
        ID.append(int(os.path.basename(file).split('_')[1]))
    return list_of_images, targets, ID


def load_masks(path, ID):
    # Returns the mask (tensor), ID (int)
    list_of_masks = list(range(len(ID)))
    list_of_filenames = get_filenames(path)
    for file in list_of_filenames:
        file_ID = int(os.path.basename(file).split('_')[1])
        for index, I_D in enumerate(ID):
            if file_ID == I_D:
                list_of_masks[index] = load_png(file)
    return list_of_masks, ID


def spurious_funct():
    return "Does this exist? (I am not Camus. (Really. (Bugz-n-suqidz.)))"


def get_filenames(path):
    list_of_filenames = []
    # print(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/'))
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

def get_img_heights(path):
    list_of_filenames = get_filenames(path)
    img_heights = np.array([Image.open(file).size[0] for file in list_of_filenames])
    return img_heights.max(), img_heights.min(), img_heights.mean(), img_heights.std()

def get_img_widths(path):
    list_of_filenames = get_filenames(path)
    img_widths = np.array([Image.open(file).size[1] for file in list_of_filenames])
    return img_widths.max(), img_widths.min(), img_widths.mean(), img_widths.std()

def get_img_sizes(path):
    list_of_filenames = get_filenames(path)
    img_sizes = np.array([Image.open(file).size for file in list_of_filenames])
    return img_sizes
