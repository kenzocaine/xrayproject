import os
import tensorflow as tf
import random
import numpy as np
import PIL
from PIL import Image

def load_masks(n=1, get_all=False, get_random = True, balanced = True, path ='', bucket_name=''):
    # Load png and returns them as list of  img (tensor) , targets (bol)
    # If balanced = True, will attempt to divide n into equal parts of positive and negative samples
    # If random, will choose random images
    # If random = False, will return images with the same order as in raw_data
    # Only set get_all = True if you running on the cloud
    # Keep n < 20 if you dont wanna run into memory problems
    # balanced only works with get_random
    # print('hello')
    print('Using path: ', path)
    print('Using bucket', bucket_name)
    list_of_filenames = get_filenames(path=path, bucket_name=bucket_name)

    assert len(list_of_filenames) != 0, 'List of filenames is empty'
    assert len(list_of_filenames) != n, f'Failed loading filenames.Check your path. Attempted loading {list_of_filenames})' 
    list_of_images = []
    targets = []
    ID = []
    if get_all:
        for file in list_of_filenames:
            image = load_png(file)
            list_of_images.append(image)
            targets.append(int(os.path.basename(file)[:-4].split('_')[2]))
            ID.append(int(os.path.basename(file).split('_')[1]))
        return list_of_images, targets, ID

    # Extract positive and negative samples
    positive, negative = [], []
    for file in list_of_filenames:
        if int(os.path.basename(file)[:-4].split('_')[2]):
            positive.append(file)
        else:
            negative.append(file)
    if get_random:
        if n == 1:
            rand_index = random.randint(0, len(list_of_filenames))
            file = list_of_filenames[rand_index]
            return load_png(file), int(os.path.basename(file).split('_')[2]), os.path.basename(file).split('_')[1]
        if balanced:
            pos = int(n/2)
            neg = int(n - pos)
            rand_list_pos = [random.randint(0, len(positive)) for i in range(pos)]
            rand_list_neg = [random.randint(0, len(negative)) for i in range(neg)]
            for i in rand_list_pos:
                list_of_images.append(load_png(positive[i]))
                targets.append(int(os.path.basename(positive[i]).split('_')[2]))
                ID.append(int(os.path.basename(positive[i]).split('_')[1]))
            for i in rand_list_neg:
                list_of_images.append(load_png(negative[i]))
                targets.append(int(os.path.basename(negative[i]).split('_')[2]))
                ID.append(int(os.path.basename(negative[i]).split('_')[1]))
            return list_of_images, targets, ID
        else:
            rand_list = [random.randint(0, len(list_of_filenames)) for i in range(n)]
            for i in rand_list:
                list_of_images.append(load_png(list_of_filenames[i]))
                targets.append(int(os.path.basename(list_of_filenames[i]).split('_')[2]))
                ID.append(int(os.path.basename(list_of_filenames[i]).split('_')[1]))
            return list_of_images, targets, ID

    for file in list_of_filenames[0:n]:
        image = load_png(file)
        list_of_images.append(image)
        targets.append(int(os.path.basename(file)[:-4].split('_')[2]))
        ID.append(int(os.path.basename(file).split('_')[1]))
    return list_of_images, targets, ID


def load_train(ID, path='', bucket_name='', data='CXR_png'):
    # Returns the mask (tensor), ID (int)
    list_of_masks = list(range(len(ID)))
    list_of_filenames = get_filenames(path=path, bucket_name=bucket_name, data=data)
    for file in list_of_filenames:
        file_ID = int(os.path.basename(file).split('_')[1])
        for index, I_D in enumerate(ID):
            if file_ID == I_D:
                list_of_masks[index] = load_png(file)
    return list_of_masks, ID

def load_test(path):
    # Dont use
    list_of_filenames = get_filenames(path)
    pass

def spurious_funct():
    return "Does this exist? (I am not Camus. (Really. (Bugz-n-suqidz.)))"


def get_filenames(bucket_name='', path='', data='mask'):
    list_of_filenames = []
    assert (len(bucket_name) != 0 or len(path) != 0), 'incorrect path format' 
    # print(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/'))
   # os.walk(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/')):
    if len(bucket_name) != 0:
        print('Generating list of filenames...')
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=f"data/{data}/", delimiter='/') 
        for blob in list(blobs):
            name = 'gs://'+bucket_name + '/' + blob.name
            if name.endswith('.png'):
                list_of_filenames.append(name)
        return list_of_filenames

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png'):
                list_of_filenames.append(os.path.join(dirname, filename))
    return list_of_filenames


def load_png(file):
    if file[0:2] == 'gs':
        print('Loading blob: ', file)
        f = tf.io.gfile.GFile(file, 'rb')
        image = f.read()
        image = tf.io.decode_png(image)
        return image

    print('Loading local file: ', file)
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
