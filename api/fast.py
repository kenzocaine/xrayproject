from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.models import load_model
from xrayproject.preprocessing import flip_resize, resize_normalize
import numpy as np
from tensorflow.keras.models import model_from_json
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

seg_model = load_model('/Users/kimhedelin/code/kenzocaine/xrayproject/trained_models/seg_model_2')
tb_model = load_model('/Users/kimhedelin/code/kenzocaine/xrayproject/trained_models/TB_model_1')

# model = model_from_json(open('model_architecture.json').read())
# model.load_weights('model_weights.h5')# dont forget to compile your model
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

def get_pred_mask(lung_xray, desired_model, lung_threshold=1/2):
    pred_mask = []
    for img in lung_xray:
        pred_lung = (1+np.sign(desired_model.predict(img[tf.newaxis, ...])\
                               .squeeze()-(1-lung_threshold)))/2
        pred_lung = np.resize(pred_lung, (pred_lung.shape[0], pred_lung.shape[1], 1))
        pred_lung = tf.image.resize(pred_lung, (128, 128))
        pred_mask.append(pred_lung)
    return pred_mask

def get_seg_lung(lung_xray):
    temp_resize = (128, 128)

    X_image_norm, X_mask_norm = resize_normalize(lung_xray,
                                                 lung_xray,
                                                 temp_resize)
    mask = get_pred_mask(X_image_norm, seg_model)

    resize_original = (lung_xray[0].shape[1], lung_xray[0].shape[1])

    mask_original = [tf.image.resize(mask[0], resize_original)]
    # print(mask_original[0])
    lung_xray, lung_xray_temp = resize_normalize(lung_xray,
                                                 lung_xray,
                                                 resize_original)

    image_seg = []
    for i in range(len(lung_xray)):
        image_seg.append(lung_xray[i]*mask_original[i])

    return image_seg


def classify(lung_xray, tb_model):
    # print(type(lung_xray))
    temp_resize = (128, 128)
    X_image_norm, X_mask_norm = resize_normalize(lung_xray,
                                                 lung_xray,
                                                 temp_resize)
    mask = get_pred_mask(X_image_norm, seg_model)
    # print(mask[0])

    image_seg = []
    for i in range(len(X_image_norm)):
        image_seg.append(X_image_norm[i]*mask[i])

    print(len(image_seg))
    print(type(image_seg))

    image_pred = np.expand_dims(image_seg[0], axis=0)
    pred = tb_model.predict(image_pred)[0][0]
    return pred




@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/seglung/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image = tf.io.decode_png(contents)

    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    if image.shape[2] == 2:
        image = tf.image.grayscale_to_rgb(image[:,:,:1])
    if image.shape[2] == 4:
        image = image[:,:,:3]

    seg_lung = get_seg_lung([image])
    seg_lung = tf.cast(seg_lung[0]*255.0, tf.uint8)
    # print(seg_lung.shape)
    # seg_lung = tf.image.sobel_edges(tf.expand_dims(seg_lung, 0))
    tf.keras.preprocessing.image.save_img('temp.png', seg_lung)

    # seg_lung_png = tf.io.encode_png(seg_lung)
    # seg_lung_png_bytes = seg_lung_png.numpy()
    # json_compatible_item_data = jsonable_encoder(seg_lung_png_bytes)
    #return JSONResponse(content=json_compatible_item_data)
    return FileResponse('temp.png')
    # return {"filename": file.filename}


@app.post("/predict/")
async def create_predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = tf.io.decode_png(contents)
    # im, ma, im_flipped, ma_flipped = flip_resize(image, image, (579, 500))
    # image_pred = np.expand_dims(im, axis=0)
    # pred = str(model.predict(image_pred))
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    if image.shape[2] == 2:
        image = tf.image.grayscale_to_rgb(image[:,:,:1])
    if image.shape[2] == 4:
        image = image[:,:,:3]

    pred = classify([image], tb_model)

    print('prediction: ', pred)
    print('Image type: ', type(image))
    print('Image shape: ', image.shape) 
    return {f"file {file.filename}": str(pred)}

