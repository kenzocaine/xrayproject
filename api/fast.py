from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from xrayproject.preprocessing import flip_resize
import numpy as np
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

from tensorflow.keras.models import model_from_json

model = model_from_json(open('model_architecture.json').read())
model.load_weights('model_weights.h5')# dont forget to compile your model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)



@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


@app.post("/predict/")
async def create_predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = tf.io.decode_png(contents)
    im, ma, im_flipped, ma_flipped = flip_resize(image, image, (579, 500))
    image_pred = np.expand_dims(im, axis=0)
    pred = str(model.predict(image_pred))
    print('prediction: ', pred)


    print('Image type: ', type(image))
    print('Image shape: ', image.shape) 
    return {f"file {file.filename}": pred}

