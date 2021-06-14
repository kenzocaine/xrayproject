from google.cloud import storage
import numpy as np
from tensorflow.keras.models import load_model
from xrayproject.utils import generate_batches, load_ID
from xrayproject.preprocessing import resize_normalize, augment
from sklearn.model_selection import train_test_split
from xrayproject.segmentation_zero import Segmentation_UNET
from xrayproject.to_masks import unet_to_mask_img
from xrayproject.baseline import Baseline
# from xrayproject.baseline import Baseline

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


def preprocess_seg(X_mask, X_image, targets):
    X_image, X_mask, X_image_flipped, X_mask_flipped = augment(X_image, X_mask)
    X_image = X_image + X_image_flipped
    X_mask = X_mask + X_mask_flipped
    targets = targets + targets
    # Normalize resize for segementation
    input_shape = (224, 224)
    X_image_norm, X_mask_norm = resize_normalize(X_image, X_mask, input_shape)

    # Split train to train/val for segmentation
    X_train, X_val, y_train, y_val = train_test_split(X_image_norm,
                                                      X_mask_norm,
                                                      test_size=0.2,
                                                      random_state=42)
    return X_train, X_val, y_train, y_val


def preprocess_classifier(X_image, targets, seg_model):
    X_image, X_mask, X_image_flipped, X_mask_flipped = augment(X_image,
                                                               X_image)
    X_image = X_image + X_image_flipped
    targets = targets + targets
    input_shape = (224, 224)
    X_image_norm, X_mask_norm = resize_normalize(X_image, X_image, input_shape)
    print('Aplying mask...')
    masks = unet_to_mask_img(X_image_norm, seg_model)
    print('length mask: ', len(masks))
    print('shape mask: ', np.shape(masks))
    input_shape = (224, 224)
    image_seg = []
    X_image_norm, X_mask_norm = resize_normalize(X_image, X_image, input_shape)
    # import pdb; pdb.set_trace()

    for i in range(len(X_image_norm)):
        image_seg.append(X_image_norm[i]*masks[i])

    X_train, X_val, y_train, y_val = train_test_split(image_seg,
                                                      targets,
                                                      test_size=0.2,
                                                      random_state=42)

    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    # Generate batches from mask ID (for training)
    path = '/Users/kimhedelin/code/kenzocaine/xrayproject/raw_data/ChinaSet_AllFiles/CXR_png/'
    batches = generate_batches(batch_size=100,
                               path = path)
                              # bucket_name=BUCKET_NAME,
                              # gfolder='mask')
    train_seg = False
    train_classifier = True
    if train_seg:

        # Creating segmentation model
        seg_model = Segmentation_UNET()
        seg_model.base_model()
        seg_model.unet_model()
        seg_model.initialize_model()

        # Load batches
        n_batches_to_load = 10
        batch_number = 1
        for batch in batches[0:n_batches_to_load]:
            # Load train mask
            print('Loading batch: ', batch_number)
            X_mask, targets, ID = load_ID(batch,
                                          bucket_name=BUCKET_NAME,
                                          gfolder='mask')

            # Load train image
            X_image, targets, ID = load_ID(batch,
                                           bucket_name=BUCKET_NAME,
                                           gfolder='CXR_png')
            print(targets)
            print('ID: ', ID)
            print('batch: ', batches[0])
            assert len(X_image) == len(X_mask), 'Number of images and masks\
                    are not the same'
            assert len(X_image) == len(targets), 'Mismatched targets'

            X_train, X_val, y_train, y_val = preprocess_seg(X_mask,
                                                            X_image,
                                                            targets)
            print('Input length for train and val', len(X_train), len(y_train))

            seg_model.train(X_train, y_train, X_val, y_val)
            print('Training complete for batch number: ', batch_number)
            batch_number = batch_number + 1

        # Saving seg model
        save_model(seg_model.model, 'seg_model_1')

    # Train classifier
    if train_classifier:
        path = '/Users/kimhedelin/code/kenzocaine/xrayproject/raw_data/ChinaSet_AllFiles/CXR_png/'
        print('Loading seg model...')
        seg_model = load_model('seg_model_1')
        n_batches_to_load = 1
        batch_number = 1
        seg_images = 0
        image_compare = 0
        input_shape = (224, 224, 3)
        print('Initializing classifier model...')
        model_classifier = Baseline(input_shape)
        model_classifier.initialize_model()
        for batch in batches[0:n_batches_to_load]:
            X_image, targets, ID = load_ID(batch,
                                           path=path)
                                           #bucket_name=BUCKET_NAME,
                                           #gfolder='CXR_png')
            X_train, X_val, y_train, y_val = preprocess_classifier(X_image,
                                                                   targets,
                                                                   seg_model)
            import pdb; pdb.set_trace()
            model_classifier.train(X_train, y_train, X_val, y_val)



    # import pdb; pdb.set_trace()

    # pred test
    # Save
    # masks, targets, ID = load_masks(50, get_all = False, get_random = True,
    #                            balanced =True, bucket_name = BUCKET_NAME)
    # images, ID = load_train(ID, bucket_name = BUCKET_NAME)
    # input_shape = (579, 500, 3)
    # model = Baseline(input_shape)
    # result = model.train(images, masks, targets)
    # save_model(result)


