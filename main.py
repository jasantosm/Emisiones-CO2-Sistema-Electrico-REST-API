import re
from fastapi import FastAPI
from typing import List, Dict
from pydantic import BaseModel
from fastapi import status
import numpy as np

from model_predictions import read_test_data, predict

app = FastAPI()

class Data(BaseModel):

    Date: str 
    daily_generacion: float 
    daily_consumo_combustible: float
    daily_demanda: float
    daily_aportes_energia: float
    daily_volumen_util_energia: float
    daily_disponibilidad_real: float
    daily_precio_bolsa: float
    daily_emision_CO2_eq: float

class Test_data(BaseModel):
    test_data: List[Data]
    y_test: List[float]
    k: int

class Predict_data(BaseModel):
    y_pred: List[float]
    x: List[str]
    

@app.get("/model/prediction/", response_model=Predict_data, status_code=status.HTTP_200_OK)
def get_prediction_data():

    y_pred, x = predict()
    return {'y_pred': y_pred, 'x': x}

@app.get("/model/data/test/", response_model=Test_data, status_code=status.HTTP_200_OK)
def get_test_data():

    data, y_test, k = read_test_data()

    data = data.reset_index()

    return {'test_data':data.to_dict(orient='records'), 'y_test':y_test, 'k': k}
