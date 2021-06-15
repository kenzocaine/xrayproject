import tensorflow as tf


def augment(images, masks):
    # Augment the train data
    if len(images) != len(masks):
        print('Warning: number of images and masks are not the same')
    print('Augmenting data..')
    images_flipped = [tf.image.flip_left_right(image) for image in images]
    masks_flipped = [tf.image.flip_left_right(mask) for mask in masks]
    return images, masks, images_flipped, masks_flipped


def resize_normalize(images, masks, input_shape=(224, 224)):
    if len(images) != len(masks):
        print('Warning: number of images and masks are not the same')
    print('Normalizing and resizing to shape ', input_shape)
    masks = [tf.image.resize(mask, input_shape) for mask in masks]
    images = [tf.image.resize(image, input_shape) for image in images] 

    images = [tf.cast(image, tf.float32) / 255.0 for image in images]
    masks = [tf.cast(mask, tf.float32) / 255.0 for mask in masks]
    return images, masks


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask


def flip_resize(image, mask, input_shape=(2897, 2499)):
    input_image = tf.image.resize(image, input_shape)
    input_mask = tf.image.resize(mask, input_shape)

    input_image, input_mask = normalize(input_image, input_mask)

    input_image_flipped = tf.image.flip_left_right(input_image)
    input_mask_flipped = tf.image.flip_left_right(input_mask)

    return input_image, input_mask, input_image_flipped, input_mask_flipped


def resize_test(image, mask):
    input_image = tf.image.resize(image, (2897, 2499))
    input_mask = tf.image.resize(mask, (2897, 2499))

    input_image, input_mask = normalize(image, input_mask)
    return input_image, input_mask
