def unet_to_mask_img(raw_lung_xray, mask_model=seg_model):
    if mask_model == seg_model:
        img_funct = mask_model.model.predict(raw_lung_xray[tf.newaxis, ...]).squeeze()
        img_funct = np.sign(img_funct)
        img_funct = (1+np.resize(img_funct, (224, 224, 1)))/2
        img_funct = tf.image.resize(img_funct, (800, 800))
        raw_lung_xray = tf.image.resize(raw_lung_xray, (800, 800))
        img_funct = img_funct * raw_lung_xray
#        img_funct = img_funct*raw_lung_xray
    else:
        img_funct = tf.image.resize(raw_lung_xray, (128, 128))
        img_funct = mask_model.predict(img_funct[tf.newaxis, ...]).squeeze()
        img_funct = np.sign(img_funct)
        img_funct = (1+np.resize(img_funct, (128, 128, 1)))/2
    return img_funct
