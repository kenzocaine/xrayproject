from google.cloud import storage
import numpy as np
import joblib
from xrayproject.utils import load_masks, load_train
from xrayproject.preprocessing import flip_resize
from xrayproject.baseline import Baseline

BUCKET_NAME = 'wagon-data-627-hedelin'

BUCKET_TRAIN_DATA_PATH = 'data/CXR_png/'
BUCKET_TRAIN_MASK_PATH = 'data/mask/'

MODEL_NAME = 'baseline'

MODEL_VERSION = 'v1'
STORAGE_LOCATION = 'models/model.joblib'

path_image = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
path_masks = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_MASK_PATH}"

def upload_model_to_gcp():

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.upload_from_filename('model.joblib')


def save_model(model):
    joblib.dump(reg, 'model.joblib')
    print('saved model.joblib locally')

    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n =>{STORAGE_LOCATION}")


if __name__ == '__main__':

    masks, targets, ID = load_masks(10, get_all = False, get_random = True,
                                balanced =True, path = path_masks)
    images, ID = load_train(path_image, ID)
    input_shape = (579, 500, 3)
    model = Baseline(input_shape)
    model.train(images, masks, targets)
    save_model(model)


