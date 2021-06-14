import numpy as np
import tensorflow as tf


def unet_to_mask_img(raw, mask_model):
    out = []
    for raw_lung_xray in raw:
        img_funct = mask_model.predict(raw_lung_xray[tf.newaxis, ...]).squeeze()
        img_funct = np.sign(img_funct)
        img_funct = (1+np.resize(img_funct, (224, 224, 1)))/2
        img_funct = tf.image.resize(img_funct, (224, 224))
        out.append(img_funct)
    return out
