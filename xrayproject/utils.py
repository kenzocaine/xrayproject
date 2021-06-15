import os
import tensorflow as tf
import random
import numpy as np
from PIL import Image
from google.cloud import storage

def load_pngs(n=1, get_all=False, get_random = True, balanced = True, path ='', bucket_name=''):
    # Load png and returns them as list of  img (tensor) , targets (bol)
    # If balanced = True, will attempt to divide n into equal parts of positive and negative samples
    # If random, will choose random images
    # If random = False, will return images with the same order as in raw_data
    # Only set get_all = True if you running on the cloud
    # Keep n < 20 if you dont wanna run into memory problems
    # balanced only works with get_random
    # print('hello')
    if path != '':
        print('Using path: ', path)
    if bucket_name != '':
        print('Using bucket', bucket_name)
    list_of_filenames = get_filenames(path=path, bucket_name=bucket_name)

    assert len(list_of_filenames) != 0, 'List of filenames is empty'
    assert len(list_of_filenames) != n, f'Failed loading filenames.Check your path. Attempted loading {list_of_filenames})' 
    list_of_images = []
    targets = []
    ID = []
    if get_all:
        for file in list_of_filenames:
            image, target, file_ID = load_png(file)
            list_of_images.append(image)
            targets.append(target)
            ID.append(file_ID)
        return list_of_images, targets, ID

    # Extract positive and negative samples
    positive, negative = [], []
    for file in list_of_filenames:
        file_elements = os.path.splitext(os.path.basename(file))[0].split('_')
        if int(file_elements[2]):
            positive.append(file)
        else:
            negative.append(file)
    if get_random:
        if n == 1:
            rand_index = random.randint(0, len(list_of_filenames))
            file = list_of_filenames[rand_index]
            image, target, file_ID = load_png(file)
            return image, target, file_ID
        if balanced:
            pos = int(n/2)
            neg = int(n - pos)
            rand_list_pos = [random.randint(0, len(positive)-1) for i in range(pos)]
            rand_list_neg = [random.randint(0, len(negative)-1) for i in range(neg)]
            for i in rand_list_pos:
                file = positive[i]
                image, target, file_ID = load_png(file)
                list_of_images.append(image)
                targets.append(target)
                ID.append(file_ID)
            for i in rand_list_neg:
                file = negative[i]
                image, target, file_ID = load_png(file)
                list_of_images.append(image)
                targets.append(target)
                ID.append(file_ID)
            return list_of_images, targets, ID
        else:
            rand_list = [random.randint(0, len(list_of_filenames)-1) for i in range(n)]
            for i in rand_list:
                file = list_of_filenames[i]
                image, target, file_ID = load_png(file)
                list_of_images.append(image)
                targets.append(target)
                ID.append(file_ID)
                return list_of_images, targets, ID

    for file in list_of_filenames[0:n]:
        image, target, file_ID = load_png(file)
        list_of_images.append(image)
        targets.append(target)
        ID.append(file_ID)
    return list_of_images, targets, ID


def load_ID(ID, path='', bucket_name='', gfolder='CXR_png'):
    # Returns the mask (tensor), ID (int)
    assert type(ID) == list, 'ID must be a list'
    list_of_images = list(range(len(ID)))
    targets = list(range(len(ID)))
    list_of_filenames = get_filenames(path=path, bucket_name=bucket_name, gfolder=gfolder)

    for file in list_of_filenames:
        file_elements = os.path.splitext(os.path.basename(file))[0].split('_')
        file_ID = int(file_elements[1])
        for index, I_D in enumerate(ID):
            if file_ID == I_D:
                list_of_images[index], targets[index], i = load_png(file)

    return list_of_images, targets, ID


def generate_batches(batch_size = 10,path='', bucket_name='', gfolder='CXR_png', get_all = False):
    list_of_filenames = get_filenames(path = path, bucket_name = bucket_name, gfolder = gfolder) 
    assert len(list_of_filenames) >= batch_size, 'Batch size exceed number of files in folder'
    random.shuffle(list_of_filenames)
    ID = [int(os.path.splitext(os.path.basename(file))[0].split('_')[1]) for file in list_of_filenames]
    if get_all:
        return ID
    if len(list_of_filenames) % batch_size != 0:
        n_batches = int(len(ID) / batch_size) + 1
        print('Length of list is uneven: ', len(list_of_filenames))
        print(f'Returning {n_batches} uneven batches')
    else:
        n_batches = int(len(ID) / batch_size)
        print(f'Even size. Returning {n_batches} number of batches')

    batches = [[] for i in range(n_batches)]
    for batch_index in range(len(batches)):
        for i in range(batch_size):
            if ID:
                batches[batch_index].append(ID.pop())

    return batches


def load_test(path):
    # Dont use
    list_of_filenames = get_filenames(path)
    pass

def spurious_funct():
    return "Does this exist? (I am not Camus. (Really. (Bugz-n-suqidz.)))"


def get_filenames(bucket_name='', path='', gfolder='mask'):
    list_of_filenames = []
    assert (len(bucket_name) != 0 or len(path) != 0), 'incorrect path format' 
    # print(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/'))
   # os.walk(os.path.join(os.path.dirname(__file__),'../raw_data/ChinaSet_AllFiles/CXR_png/')):
    if len(bucket_name) != 0:
        print('Generating list of filenames...')
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=f"data/{gfolder}/", delimiter='/') 
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
    file_elements = os.path.splitext(os.path.basename(file))[0].split('_')
    target = int(file_elements[2])
    ID = int(file_elements[1])
    if file[0:2] == 'gs':
        print('Loading blob: ', file)
        f = tf.io.gfile.GFile(file, 'rb')
        image = f.read()
        image = tf.io.decode_png(image)
        return image, target, ID

    print('Loading local file: ', file)
    image = tf.io.read_file(file)
    image = tf.io.decode_png(image)
    return image, target, ID

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
