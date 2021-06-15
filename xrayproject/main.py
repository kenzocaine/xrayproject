from google.cloud import storage
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from xrayproject.utils import generate_batches, load_ID
from xrayproject.preprocessing import resize_normalize, augment, enhance, rotate90
from sklearn.model_selection import train_test_split
from xrayproject.to_masks import unet_to_mask_img
from xrayproject.baseline import Baseline
# from xrayproject.baseline import Baseline

BUCKET_NAME = 'wagon-data-627-hedelin'

BUCKET_TRAIN_DATA_PATH = 'data/CXR_png/'
BUCKET_TRAIN_MASK_PATH = 'data/mask/'

MODEL_NAME = 'TB_model'

MODEL_VERSION = 'v1'
STORAGE_LOCATION = 'models/tb_model'

path_image = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
path_masks = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_MASK_PATH}"


def upload_model_to_gcp():

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob_json = bucket.blob(STORAGE_LOCATION+'model_architecture.json')
    blob_h5 = bucket.blob(STORAGE_LOCATION+'model_weights.h5')
    blob_json.upload_from_filename('model_architecture.json')
    blob_h5.upload_from_filename('model_weights.h5')
    print('upload complete!')


def save_model(model, name):
    # json_model = model.to_json()
    # open('model_architecture.json', 'w').write(json_model)
    # saving weights
    # model.save_weights('model_weights.h5', overwrite=True)
    print(f'Saving model {name} locally...')
    model.save(name)
    print('Save complete')

    # upload_model_to_gcp()
    # print(f"uploaded model to gcp cloud storage under \n =>{STORAGE_LOCATION}")

def get_pred_mask(lung_xray, desired_model, lung_threshold=1/2):
    pred_mask = []
    for img in lung_xray:
        pred_lung = (1+np.sign(desired_model.predict(img[tf.newaxis, ...])\
                               .squeeze()-(1-lung_threshold)))/2
        pred_lung = np.resize(pred_lung, (pred_lung.shape[0], pred_lung.shape[1], 1))
        pred_lung = tf.image.resize(pred_lung, (128, 128))
        pred_mask.append(pred_lung)
    return pred_mask

def preprocess_classifier(X_image, targets, seg_model, resize_shape):
    print('Aplying mask...')
    image_seg = []
    temp_resize = (128, 128)
    X_image_norm, X_mask_norm = resize_normalize(X_image, X_image, temp_resize)
    masks = get_pred_mask(X_image_norm, seg_model)
    for i in range(len(X_image_norm)):
        image_seg.append(X_image_norm[i]*masks[i])

    X_image, X_mask, X_image_flipped, X_mask_flipped = augment(image_seg,
                                                               image_seg)
    X_image_rot = rotate90(image_seg)
    X_image = X_image + X_image_flipped + X_image_rot
    # import pdb; pdb.set_trace()

    targets = targets + targets + targets
    # X_image = enhance(X_image)
    # X_image_norm, X_mask_norm = resize_normalize(X_image, X_image, resize_shape)
    # import pdb; pdb.set_trace()

    X_train, X_val, y_train, y_val = train_test_split(np.array(X_image),
                                                      np.array(targets),
                                                      test_size=0.2,
                                                      random_state=42)

    return X_train, X_val, y_train, y_val

def preprocess_classifier_new(images, targets, seg_model, resize_shape):
    print('Applying masks...')
    temp_resize = (128, 128)
    images_norm, mask_norm = resize_normalize(images, images, temp_resize)
    masks = get_pred_mask(images_norm, seg_model)
    image_seg = []
    for i in range(len(images_norm)):
        image_seg.append(images_norm[i]*masks[i])

    image_seg = image_seg * 2
    targets = targets * 2
    image_seg = enhance(image_seg)

    X_train, X_val, y_train, y_val = train_test_split(image_seg,
                                                      targets,
                                                      test_size=0.2,
                                                      random_state=42)
    return X_train, X_val, y_train, y_val

if __name__ == '__main__':
    # Generate batches from mask ID (for training)
    path = '/Users/kimhedelin/code/kenzocaine/xrayproject/raw_data/ChinaSet_AllFiles/CXR_png/'
    batches = generate_batches(batch_size=100,
                               # path=path)
                               bucket_name=BUCKET_NAME,
                               gfolder='mask')
    path = '/Users/kimhedelin/code/kenzocaine/xrayproject/raw_data/ChinaSet_AllFiles/CXR_png/'
    print('Loading seg model...')
    path_to_seg = f'gs://{BUCKET_NAME}/models/seg_model/seg_model_2'
    path_to_seg_local = '/Users/kimhedelin/code/kenzocaine/xrayproject/trained_models/seg_model_2'
    seg_model = load_model(path_to_seg)
    print('Loading complete')
    n_batches_to_load = len(batches)
    batch_number = 1
    input_shape = (128, 128, 3)
    resize_shape = (128, 128)
    print('Initializing classifier model...')
    model_classifier = Baseline(input_shape)
    model_classifier.initialize_model()
    for batch in batches[0:n_batches_to_load]:
        print('Loading batch number: ', batch_number)
        X_image, targets, ID = load_ID(batch,
                                       # path=path)
                                       bucket_name=BUCKET_NAME,
                                       gfolder='CXR_png')
        X_train, X_val, y_train, y_val = preprocess_classifier(X_image,
                                                               targets,
                                                               seg_model,
                                                               resize_shape)
        model_classifier.train(X_train, y_train, X_val, y_val)
        print('Training complete')
        batch_number = batch_number + 1


    save_it = True 
    if save_it:
        name_save = f'gs://{BUCKET_NAME}/{STORAGE_LOCATION}/TB_model_4'
        save_model(model_classifier.model, name_save)
