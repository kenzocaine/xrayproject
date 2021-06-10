def resize_normalize(input_image, input_mask, input_shape=(2897, 2499)):
    resized_image = tf.image.resize(input_image, input_shape)
    resized_mask = tf.image.resize(input_mask, input_shape)
    
    norm_image = tf.cast(resized_image, tf.float32) / 255.0
    norm_mask = tf.cast(resized_mask, tf.float32) / 255.0
    return norm_image, norm_mask