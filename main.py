import os
import tempfile

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.classify_model import MNIST_Classify_Model, DataPreprocessing

device = torch.device('cpu')
SAVED_MODEL_PATH = "./model/model.pth"

CLASSIFY_MODEL = MNIST_Classify_Model().to(device)
CLASSIFY_MODEL.load_state_dict(torch.load(SAVED_MODEL_PATH))

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL = 28, 28, 1

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.tobytes()
    return image


class RequestInput(BaseModel):
    input: str


@app.get("/")
async def index():
    return {"Message": ["Hello World"]}


# @app.post("/predict")
# async def predict(request: RequestInput = Depends()):
#     print(request.input)
#     request_input = DataPreprocessing(
#         target_datatype=np.float32,
#         image_width=IMAGE_WIDTH,
#         image_height=IMAGE_HEIGHT,
#         image_channel=IMAGE_CHANNEL,
#     )(image)
#
#     prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
#     prediction = prediction.cpu().detach().numpy()
#     prediction = np.argmax(prediction, axis=1)
#
#     return {"prediction": prediction.tolist()}


@app.post("/predict")
async def predict(image: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(await image.read())
        temp_image_path = temp_image.name

    preprocessed_image = preprocess_image(temp_image_path)
    request_input = DataPreprocessing(
        target_datatype=np.float32,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        image_channel=IMAGE_CHANNEL,
    )(preprocessed_image)

    prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)

    os.remove(temp_image_path)

    return {"prediction": prediction.tolist()}


@app.get("/predict_from_path/{image_path}")
async def predict_from_path(image_path: str = Path(..., description="Путь к изображению")):
    preprocessed_image = preprocess_image(image_path)

    request_input = DataPreprocessing(
        target_datatype=np.float32,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        image_channel=IMAGE_CHANNEL,
    )(preprocessed_image)

    prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)

    return {"prediction": prediction.tolist()}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=1488)
    # uvicorn.run(app, host='127.0.0.1', port=1488)
    # uvicorn.run(app)

# ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# docker run --rm -p  8000:1488 mnist-service

# docker build -t mnist-service .

# проброска томом модель, изображения в двух разных томах
