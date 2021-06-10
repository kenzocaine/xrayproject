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
STORAGE_LOCATION = 'models/'

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


def save_model(model):
    json_model = model.to_json()
    open('model_architecture.json', 'w').write(json_model)
    # saving weights
    model.save_weights('model_weights.h5', overwrite=True)
    print('saved model locally')

    upload_model_to_gcp()
    print(f"uploaded model to gcp cloud storage under \n =>{STORAGE_LOCATION}")


if __name__ == '__main__':

    masks, targets, ID = load_masks(50, get_all = False, get_random = True,
                                balanced =True, bucket_name = BUCKET_NAME)
    images, ID = load_train(ID, bucket_name = BUCKET_NAME)
    input_shape = (579, 500, 3)
    model = Baseline(input_shape)
    result = model.train(images, masks, targets)
    save_model(result)


