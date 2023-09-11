import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Depends, UploadFile
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
test_image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (28, 28))
test_image = test_image.tobytes()


class RequestInput(BaseModel):
    input: str


@app.get("/")
async def index():
    return {"Message": ["Hello World"]}


@app.get("/predict")
async def predict(request: RequestInput = Depends()):
    print(request.input)
    request_input = DataPreprocessing(
        target_datatype=np.float32,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        image_channel=IMAGE_CHANNEL,
    )(test_image)

    prediction = CLASSIFY_MODEL(torch.tensor(request_input).to(device))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.argmax(prediction, axis=1)

    return {"prediction": prediction.tolist()}


@app.post("/file/upload-file")
def upload_file(file: UploadFile):
    return file


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=1488)
    # uvicorn.run(app, host='127.0.0.1', port=1488)
    # uvicorn.run(app)

# @app.get("/predict/{string}")
# def read_item(string: str):
#     return {"new_string": 'answer_s' + string, "old_string": string}

# ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# docker run --rm -p  8000:1488 mnist-service

# проброска томом модель, изображения в двух разных томах
# картинка параметр в пути
