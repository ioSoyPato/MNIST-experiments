import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import predictions
from fastapi.staticfiles import StaticFiles

# App
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount the templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def homePage(request:Request):
    return templates.TemplateResponse(request=request,
                                      name="test.html",
                                      context={})


@app.get("/prediction")
async def homePage(request:Request):
    prediction:int = predictions.predictData("./DrawPrediction.png")
    return templates.TemplateResponse(request=request,
                                      name="test.html",
                                      context={"prediccion":prediction,
                                               "barras_probabilidad":"/static/prediction_probabilities.png"})



from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import os

class ImageData(BaseModel):
    image: str

@app.post("/sandia")
async def download_image(image_data:ImageData):
    # Extraer datos de imagen
    image_data_str = image_data.image.split(",")[1]  # Obtener la parte base64
    image = base64.b64decode(image_data_str)  # Decodificar la imagen

    # Definir la ruta donde se guardará la imagen
    folder_path = './'
    os.makedirs(folder_path, exist_ok=True)  # Crear la carpeta si no existe

    # Generar un nombre de archivo único
    file_path = os.path.join(folder_path, f"DrawPrediction.png")

    # Guardar la imagen
    with open(file_path, 'wb') as f:
        f.write(image)

    return JSONResponse(content={"message": "Imagen guardada exitosamente", "file_path": file_path})

