import tensorflow as tf


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
